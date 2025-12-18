use anyhow::Result;
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::debug;

use crate::{
    CompletionChunk, CompletionRequest, CompletionResponse, CompletionStream, LLMProvider, Message,
    MessageRole, Usage,
};

#[derive(Clone)]
pub struct GeminiProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    name: String,
}

impl GeminiProvider {
    pub fn new(
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Self> {
        Self::new_with_name(
            "gemini".to_string(),
            api_key,
            model,
            base_url,
            max_tokens,
            temperature,
        )
    }

    pub fn new_with_name(
        name: String,
        api_key: String,
        model: Option<String>,
        base_url: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
    ) -> Result<Self> {
        Ok(Self {
            client: Client::new(),
            api_key,
            model: model.unwrap_or_else(|| "gemini-1.5-pro".to_string()),
            base_url: base_url
                .unwrap_or_else(|| "https://generativelanguage.googleapis.com/v1beta".to_string()),
            max_tokens,
            temperature,
            name,
        })
    }

    fn build_body(&self, request: &CompletionRequest) -> serde_json::Value {
        let (system_instruction, contents) = convert_messages(&request.messages);

        let mut body = json!({
            "contents": contents,
        });

        if let Some(system_instruction) = system_instruction {
            body["systemInstruction"] = system_instruction;
        }

        if request.stream {
            body["stream"] = json!(true);
        }

        let mut generation_config = serde_json::Map::new();
        if let Some(max_tokens) = request.max_tokens.or(self.max_tokens) {
            generation_config.insert(
                "maxOutputTokens".to_string(),
                serde_json::Value::Number(max_tokens.into()),
            );
        }
        if let Some(temperature) = request.temperature.or(self.temperature) {
            generation_config.insert(
                "temperature".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(temperature as f64).unwrap(),
                ),
            );
        }

        if !generation_config.is_empty() {
            body["generationConfig"] = serde_json::Value::Object(generation_config);
        }

        body
    }
}

#[async_trait]
impl LLMProvider for GeminiProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        debug!(
            "Processing Gemini completion request with {} messages",
            request.messages.len()
        );

        let body = self.build_body(&request);
        let url = format!(
            "{}/models/{}:generateContent?key={}",
            self.base_url, self.model, self.api_key
        );

        let response = self.client.post(url).json(&body).send().await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(anyhow::anyhow!(
                "Gemini API error {}: {}",
                status,
                error_text
            ));
        }

        let gemini_response: GeminiResponse = response.json().await?;

        let content = gemini_response
            .candidates
            .unwrap_or_default()
            .into_iter()
            .find_map(|candidate| candidate.content)
            .map(|content| {
                content
                    .parts
                    .into_iter()
                    .filter_map(|p| p.text)
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();

        let usage_meta = gemini_response.usage_metadata.unwrap_or_default();
        let prompt_tokens = usage_meta.prompt_token_count.unwrap_or(0);
        let completion_tokens = usage_meta.candidates_token_count.unwrap_or(0);
        let total_tokens = usage_meta
            .total_token_count
            .unwrap_or(prompt_tokens + completion_tokens);

        let usage = Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        };

        Ok(CompletionResponse {
            content,
            usage,
            model: self.model.clone(),
        })
    }

    async fn stream(&self, request: CompletionRequest) -> Result<CompletionStream> {
        // Gemini's REST API supports streaming, but for now, provide a simple streamed wrapper
        // around the non-streaming call to keep behavior consistent.
        let response = self.complete(request).await?;
        let (tx, rx) = mpsc::channel(1);

        let _ = tx
            .send(Ok(CompletionChunk {
                content: response.content.clone(),
                finished: true,
                tool_calls: None,
                usage: Some(response.usage.clone()),
            }))
            .await;

        Ok(ReceiverStream::new(rx))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn max_tokens(&self) -> u32 {
        self.max_tokens.unwrap_or(16000)
    }

    fn temperature(&self) -> f32 {
        self.temperature.unwrap_or(0.1)
    }
}

#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiContent>,
}

#[derive(Debug, Deserialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiPart {
    text: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct GeminiUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u32>,
}

fn convert_messages(messages: &[Message]) -> (Option<serde_json::Value>, Vec<serde_json::Value>) {
    let mut system_parts = Vec::new();
    let mut contents = Vec::new();

    for msg in messages {
        match msg.role {
            MessageRole::System => system_parts.push(msg.content.clone()),
            MessageRole::User => contents.push(json!({
                "role": "user",
                "parts": [{"text": msg.content}]
            })),
            MessageRole::Assistant => contents.push(json!({
                "role": "model",
                "parts": [{"text": msg.content}]
            })),
        }
    }

    let system_instruction = if system_parts.is_empty() {
        None
    } else {
        Some(json!({
            "parts": system_parts.into_iter().map(|text| json!({ "text": text })).collect::<Vec<_>>()
        }))
    };

    (system_instruction, contents)
}
