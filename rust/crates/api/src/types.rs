use std::collections::BTreeMap;

use runtime::{pricing_for_model, TokenUsage, UsageCostEstimate};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MessageRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<InputMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
    /// OpenAI-compatible tuning parameters. Optional — omitted from payload when None.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Reasoning effort level for OpenAI-compatible reasoning models (e.g. `o4-mini`).
    /// Accepted values: `"low"`, `"medium"`, `"high"`. Omitted when `None`.
    /// Silently ignored by backends that do not support it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Provider-specific OpenAI-compatible request body parameters. These are
    /// copied into the final JSON payload after core fields are populated so
    /// users can opt into gateway features such as `web_search_options`,
    /// `parallel_tool_calls`, or custom local-server switches without waiting
    /// for first-class typed fields. Core protocol keys are protected and cannot
    /// be overridden through this map.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_body: BTreeMap<String, Value>,
}

impl MessageRequest {
    #[must_use]
    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputMessage {
    pub role: String,
    pub content: Vec<InputContentBlock>,
}

impl InputMessage {
    #[must_use]
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![InputContentBlock::Text { text: text.into() }],
        }
    }

    #[must_use]
    pub fn user_tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            role: "user".to_string(),
            content: vec![InputContentBlock::ToolResult {
                tool_use_id: tool_use_id.into(),
                content: vec![ToolResultContentBlock::Text {
                    text: content.into(),
                }],
                is_error,
            }],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentBlock {
    Text {
        text: String,
    },
    Thinking {
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Vec<ToolResultContentBlock>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
    /// An assistant block replayed exactly as the server sent it.
    ///
    /// Server-side tool blocks have no client-side counterpart to rebuild, but
    /// they still belong in the history: resuming a `pause_turn` requires the
    /// prior content to come back unchanged. Serialized untagged so the original
    /// object goes out as-is rather than nested under another `type`.
    #[serde(untagged)]
    Passthrough(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolResultContentBlock {
    Text { text: String },
    Json { value: Value },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    Auto,
    Any,
    Tool { name: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub role: String,
    pub content: Vec<OutputContentBlock>,
    pub model: String,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub request_id: Option<String>,
}

impl MessageResponse {
    #[must_use]
    pub fn total_tokens(&self) -> u32 {
        self.usage.total_tokens()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OutputContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    RedactedThinking {
        data: Value,
    },
    /// A block type this build does not model, kept **verbatim**.
    ///
    /// The Anthropic wire format is open: server-side tools add block types
    /// (`server_tool_use`, `web_search_tool_result`, `mcp_tool_use`, ...) and
    /// compatible providers add their own. Without this arm serde rejects the
    /// **whole response** over one unrecognized block, so a single server-tool
    /// call fails the entire request.
    ///
    /// The original JSON is retained rather than discarded because "the client
    /// need not execute it" is not the same as "the client need not send it
    /// back": a `pause_turn` response has to be replayed with its content intact
    /// for the server to resume the turn.
    Unknown(Value),
}

/// The block shapes this build understands.
///
/// Split out so [`OutputContentBlock`] can keep an escape hatch for everything
/// else while these still deserialize strictly - a known block that is missing a
/// field is an error, never silently demoted to [`OutputContentBlock::Unknown`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum KnownOutputContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    Thinking {
        #[serde(default)]
        thinking: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: Value,
    },
}

impl OutputContentBlock {
    /// The wire `type` of this block, whether or not it is modeled.
    #[must_use]
    pub fn kind(&self) -> &str {
        match self {
            Self::Text { .. } => "text",
            Self::ToolUse { .. } => "tool_use",
            Self::Thinking { .. } => "thinking",
            Self::RedactedThinking { .. } => "redacted_thinking",
            Self::Unknown(raw) => raw.get("type").and_then(Value::as_str).unwrap_or_default(),
        }
    }

    fn into_known(self) -> Result<KnownOutputContentBlock, Value> {
        match self {
            Self::Text { text } => Ok(KnownOutputContentBlock::Text { text }),
            Self::ToolUse { id, name, input } => {
                Ok(KnownOutputContentBlock::ToolUse { id, name, input })
            }
            Self::Thinking {
                thinking,
                signature,
            } => Ok(KnownOutputContentBlock::Thinking {
                thinking,
                signature,
            }),
            Self::RedactedThinking { data } => {
                Ok(KnownOutputContentBlock::RedactedThinking { data })
            }
            Self::Unknown(raw) => Err(raw),
        }
    }
}

impl From<KnownOutputContentBlock> for OutputContentBlock {
    fn from(value: KnownOutputContentBlock) -> Self {
        match value {
            KnownOutputContentBlock::Text { text } => Self::Text { text },
            KnownOutputContentBlock::ToolUse { id, name, input } => {
                Self::ToolUse { id, name, input }
            }
            KnownOutputContentBlock::Thinking {
                thinking,
                signature,
            } => Self::Thinking {
                thinking,
                signature,
            },
            KnownOutputContentBlock::RedactedThinking { data } => Self::RedactedThinking { data },
        }
    }
}

impl Serialize for OutputContentBlock {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.clone().into_known() {
            Ok(known) => known.serialize(serializer),
            Err(raw) => raw.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for OutputContentBlock {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        use serde::de::Error as _;

        let raw = Value::deserialize(deserializer)?;
        let kind = raw
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::custom("content block is missing a string `type`"))?;
        if matches!(kind, "text" | "tool_use" | "thinking" | "redacted_thinking") {
            // Deserialize strictly: a malformed known block must still fail.
            return serde_json::from_value::<KnownOutputContentBlock>(raw)
                .map(Into::into)
                .map_err(D::Error::custom);
        }
        Ok(Self::Unknown(raw))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

impl Usage {
    #[must_use]
    pub const fn total_tokens(&self) -> u32 {
        self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
    }

    #[must_use]
    pub const fn token_usage(&self) -> TokenUsage {
        TokenUsage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
            cache_creation_input_tokens: self.cache_creation_input_tokens,
            cache_read_input_tokens: self.cache_read_input_tokens,
        }
    }

    #[must_use]
    pub fn estimated_cost_usd(&self, model: &str) -> UsageCostEstimate {
        let usage = self.token_usage();
        pricing_for_model(model).map_or_else(
            || usage.estimate_cost_usd(),
            |pricing| usage.estimate_cost_usd_with_pricing(pricing),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageStartEvent {
    pub message: MessageResponse,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MessageDeltaEvent {
    pub delta: MessageDelta,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageDelta {
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockStartEvent {
    pub index: u32,
    pub content_block: OutputContentBlock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentBlockDeltaEvent {
    pub index: u32,
    pub delta: ContentBlockDelta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlockDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    ThinkingDelta { thinking: String },
    SignatureDelta { signature: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentBlockStopEvent {
    pub index: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageStopEvent {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart(MessageStartEvent),
    MessageDelta(MessageDeltaEvent),
    ContentBlockStart(ContentBlockStartEvent),
    ContentBlockDelta(ContentBlockDeltaEvent),
    ContentBlockStop(ContentBlockStopEvent),
    MessageStop(MessageStopEvent),
}

#[cfg(test)]
mod tests {
    use runtime::format_usd;
    use serde_json::json;

    use super::{InputContentBlock, MessageResponse, OutputContentBlock, Usage};

    /// A non-streaming response carrying a server-side block alongside ordinary
    /// ones must parse, keep every block in order, and keep the unknown one
    /// byte-for-byte so it can be replayed.
    #[test]
    fn non_streaming_response_preserves_an_unmodeled_server_block() {
        let body = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "anthropic/glm-5.3",
            "content": [
                { "type": "text", "text": "checking" },
                {
                    "type": "server_tool_use",
                    "id": "call_d8176ee9",
                    "name": "webReader",
                    "input": { "return_format": "text", "url": "https://example.com" }
                },
                { "type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a"} }
            ],
            "stop_reason": "tool_use"
        });

        let response: MessageResponse =
            serde_json::from_value(body.clone()).expect("server tool block must not fail parsing");

        assert_eq!(response.content.len(), 3, "no block may be dropped");
        assert!(matches!(
            response.content[0],
            OutputContentBlock::Text { .. }
        ));
        assert!(matches!(
            response.content[2],
            OutputContentBlock::ToolUse { .. }
        ));
        let OutputContentBlock::Unknown(raw) = &response.content[1] else {
            panic!(
                "the server block must be preserved: {:?}",
                response.content[1]
            );
        };
        assert_eq!(raw, &body["content"][1], "preserved verbatim");
        assert_eq!(response.content[1].kind(), "server_tool_use");

        // And it round-trips: re-serializing yields the original object, which is
        // what a `pause_turn` continuation has to send back.
        assert_eq!(
            serde_json::to_value(&response.content[1]).expect("serialize"),
            body["content"][1]
        );
    }

    /// The catch-all must not become a swallow-all: a *known* block missing a
    /// required field is a real error, not an unknown block.
    #[test]
    fn a_malformed_known_block_still_fails_instead_of_becoming_unknown() {
        let error = serde_json::from_value::<OutputContentBlock>(json!({
            "type": "tool_use",
            "id": "toolu_1"
        }))
        .expect_err("tool_use without name/input must be rejected");
        assert!(
            error.to_string().contains("name"),
            "the error should name the missing field: {error}"
        );
    }

    /// A block with no `type` at all is malformed, not "unknown".
    #[test]
    fn a_block_without_a_type_is_rejected() {
        serde_json::from_value::<OutputContentBlock>(json!({ "text": "hi" }))
            .expect_err("a block without `type` must be rejected");
    }

    /// Replaying history must put the original object back on the wire, not a
    /// wrapper — the server has to see exactly what it sent.
    #[test]
    fn replayed_history_carries_the_original_server_block() {
        let raw = json!({
            "type": "server_tool_use",
            "id": "call_1",
            "name": "webReader",
            "input": { "url": "https://example.com" }
        });

        let serialized = serde_json::to_value(InputContentBlock::Passthrough(raw.clone()))
            .expect("serialize passthrough");

        assert_eq!(serialized, raw, "replayed untouched, not re-wrapped");
    }

    #[test]
    fn usage_total_tokens_includes_cache_tokens() {
        let usage = Usage {
            input_tokens: 10,
            cache_creation_input_tokens: 2,
            cache_read_input_tokens: 3,
            output_tokens: 4,
        };

        assert_eq!(usage.total_tokens(), 19);
        assert_eq!(usage.token_usage().total_tokens(), 19);
    }

    #[test]
    fn message_response_estimates_cost_from_model_usage() {
        let response = MessageResponse {
            id: "msg_cost".to_string(),
            kind: "message".to_string(),
            role: "assistant".to_string(),
            content: Vec::new(),
            model: "claude-sonnet-4-20250514".to_string(),
            stop_reason: Some("end_turn".to_string()),
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1_000_000,
                cache_creation_input_tokens: 100_000,
                cache_read_input_tokens: 200_000,
                output_tokens: 500_000,
            },
            request_id: None,
        };

        let cost = response.usage.estimated_cost_usd(&response.model);
        assert_eq!(format_usd(cost.total_cost_usd()), "$54.6750");
        assert_eq!(response.total_tokens(), 1_800_000);
    }

    #[test]
    fn input_content_block_thinking_serializes_with_snake_case_type() {
        // given
        let block = InputContentBlock::Thinking {
            thinking: "pondering".to_string(),
            signature: Some("sig_123".to_string()),
        };

        // when
        let serialized = serde_json::to_value(&block).unwrap();
        let deserialized: InputContentBlock = serde_json::from_value(json!({
            "type": "thinking",
            "thinking": "pondering",
            "signature": "sig_123"
        }))
        .unwrap();

        // then
        assert_eq!(
            serialized,
            json!({
                "type": "thinking",
                "thinking": "pondering",
                "signature": "sig_123"
            })
        );
        assert_eq!(deserialized, block);
    }
}
