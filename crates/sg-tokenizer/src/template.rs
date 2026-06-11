//! `PromptBuilder`: a hand-port of the Gemma 4 chat template (plan 01).
//!
//! Source of truth: `docs/reference/gemma-4-chat-template.jinja`, extracted
//! from the GGUF (`tokenizer.chat_template`). Byte-exact parity with jinja2
//! rendering is enforced by `tests/template.rs` against fixtures from
//! `tools/gen_template_fixtures.py`. Quirks of the original (comma placement,
//! `<|"|>` string quoting, brace imbalance when a schema omits `type`) are
//! reproduced, not fixed.
//!
//! Out of scope, by server design: multimodal content items (the API rejects
//! images), Google-native `tool_responses` embedded on assistant messages
//! (tool results arrive as `Tool`-role messages, OpenAI style), and
//! content-parts arrays (the server flattens text blocks before calling
//! this).

use serde_json::Value;

/// `<|"|>`, the escape/quote token strings are wrapped in.
pub const QUOTE: &str = "<|\"|>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
    /// A tool result (OpenAI-style `role: "tool"` message); consumed by the
    /// preceding assistant message's forward scan.
    Tool,
}

/// One tool invocation recorded on an assistant message.
#[derive(Debug, Clone)]
pub struct ToolCallMsg {
    /// Correlation id (`tool_calls[].id`), matched against
    /// [`ChatMessage::tool_call_id`].
    pub id: Option<String>,
    pub name: String,
    /// Either a JSON object (rendered in Gemma's argument format) or a
    /// pre-serialized `Value::String` inserted verbatim.
    pub arguments: Value,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
    /// Reasoning text (`reasoning` / `reasoning_content`); rendered as a
    /// thought channel only on post-last-user assistant messages that also
    /// carry tool calls.
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallMsg>,
    /// On `Tool` messages: which call this result answers.
    pub tool_call_id: Option<String>,
    /// On `Tool` messages: explicit function name (fallback when no
    /// `tool_call_id` matches).
    pub tool_name: Option<String>,
}

impl ChatMessage {
    pub fn text(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            tool_name: None,
        }
    }
}

/// One tool declaration (the `function` object of an OpenAI-shaped tool).
#[derive(Debug, Clone)]
pub struct ToolDecl {
    pub name: String,
    pub description: String,
    /// JSON-schema-ish parameters object; `Value::Null` when absent.
    pub parameters: Value,
    /// Optional response declaration (`function.response`).
    pub response: Option<Value>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PromptOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrevType {
    None,
    Think,
    Tool,
    ToolCall,
    ToolResponse,
}

/// Render the full conversation prefix, byte-identical to
/// `apply_chat_template` with the GGUF's template.
pub fn render_prompt(messages: &[ChatMessage], tools: &[ToolDecl], opts: PromptOptions) -> String {
    let mut out = String::from("<bos>");
    let mut prev = PrevType::None;

    let mut msgs = messages;
    let first_is_system = msgs.first().is_some_and(|m| m.role == ChatRole::System);
    if opts.enable_thinking || !tools.is_empty() || first_is_system {
        out.push_str("<|turn>system\n");
        if opts.enable_thinking {
            out.push_str("<|think|>\n");
            prev = PrevType::Think;
        }
        if first_is_system {
            out.push_str(msgs[0].content.trim());
            msgs = &msgs[1..];
        }
        for tool in tools {
            out.push_str("<|tool>");
            out.push_str(render_declaration(tool).trim());
            out.push_str("<tool|>");
            prev = PrevType::Tool;
        }
        out.push_str("<turn|>\n");
    }

    let last_user_idx = msgs
        .iter()
        .rposition(|m| m.role == ChatRole::User)
        .map_or(-1, |i| i as i64);

    for (idx, message) in msgs.iter().enumerate() {
        if message.role == ChatRole::Tool {
            continue;
        }
        prev = PrevType::None;
        let role = match message.role {
            ChatRole::Assistant => "model",
            ChatRole::User => "user",
            ChatRole::System => "system",
            ChatRole::Tool => unreachable!(),
        };

        // Consecutive assistant messages continue one model turn.
        let prev_non_tool = msgs[..idx].iter().rev().find(|m| m.role != ChatRole::Tool);
        let continues_turn = message.role == ChatRole::Assistant
            && prev_non_tool.is_some_and(|m| m.role == ChatRole::Assistant);
        if !continues_turn {
            out.push_str("<|turn>");
            out.push_str(role);
            out.push('\n');
        }

        if let Some(reasoning) = &message.reasoning
            && idx as i64 > last_user_idx
            && !message.tool_calls.is_empty()
        {
            out.push_str("<|channel>thought\n");
            out.push_str(reasoning);
            out.push_str("\n<channel|>");
        }

        for tc in &message.tool_calls {
            out.push_str("<|tool_call>call:");
            out.push_str(&tc.name);
            out.push('{');
            match &tc.arguments {
                Value::Object(map) => {
                    for (i, (key, value)) in dictsort(map).into_iter().enumerate() {
                        if i > 0 {
                            out.push(',');
                        }
                        out.push_str(key);
                        out.push(':');
                        format_argument(value, false, &mut out);
                    }
                }
                Value::String(raw) => out.push_str(raw),
                _ => {}
            }
            out.push_str("}<tool_call|>");
        }
        if !message.tool_calls.is_empty() {
            prev = PrevType::ToolCall;
        }

        // Forward-scan consecutive Tool messages for this call's results.
        let mut rendered_response = false;
        if !message.tool_calls.is_empty() {
            for follow in msgs[idx + 1..]
                .iter()
                .take_while(|m| m.role == ChatRole::Tool)
            {
                let mut name = follow.tool_name.as_deref().unwrap_or("unknown");
                if let Some(id) = &follow.tool_call_id
                    && let Some(tc) = message
                        .tool_calls
                        .iter()
                        .find(|tc| tc.id.as_ref() == Some(id))
                {
                    name = &tc.name;
                }
                out.push_str("<|tool_response>response:");
                out.push_str(name);
                out.push_str("{value:");
                format_argument(&Value::String(follow.content.clone()), false, &mut out);
                out.push_str("}<tool_response|>");
                rendered_response = true;
                prev = PrevType::ToolResponse;
            }
        }

        let captured = if message.role == ChatRole::Assistant {
            strip_thinking(&message.content)
        } else {
            message.content.trim().to_owned()
        };
        out.push_str(&captured);
        let has_content = !captured.trim().is_empty();

        if prev == PrevType::ToolCall && !rendered_response {
            // Calls without results yet: leave the turn open at the response
            // marker so generation resumes after tools run.
            out.push_str("<|tool_response>");
        } else if !rendered_response || has_content {
            // (Template: `not (tool responses rendered and no content)`.)
            out.push_str("<turn|>\n");
        }
    }

    if opts.add_generation_prompt && prev != PrevType::ToolResponse && prev != PrevType::ToolCall {
        out.push_str("<|turn>model\n");
        if !opts.enable_thinking {
            // Pre-filled empty thought channel steers the model straight to
            // the answer.
            out.push_str("<|channel>thought\n<channel|>");
        }
    }
    out
}

/// Remove `<|channel>…<channel|>` spans from assistant history, then trim
/// (the template's `strip_thinking` macro).
pub fn strip_thinking(text: &str) -> String {
    let mut result = String::new();
    for part in text.split("<channel|>") {
        match part.split_once("<|channel>") {
            Some((before, _)) => result.push_str(before),
            None => result.push_str(part),
        }
    }
    result.trim().to_owned()
}

/// `format_function_declaration`: Gemma's bespoke schema serialization.
fn render_declaration(tool: &ToolDecl) -> String {
    let mut out = format!(
        "declaration:{}{{description:{QUOTE}{}{QUOTE}",
        tool.name, tool.description
    );
    let params = &tool.parameters;
    if truthy(params) {
        out.push_str(",parameters:{");
        if let Some(props) = params.get("properties").filter(|v| truthy(v)) {
            out.push_str("properties:{");
            if let Value::Object(map) = props {
                format_parameters(map, false, &mut out);
            }
            out.push_str("},");
        }
        if let Some(Value::Array(required)) = params.get("required").filter(|v| truthy(v)) {
            out.push_str("required:[");
            push_quoted_list(required, &mut out);
            out.push_str("],");
        }
        if let Some(ty) = params.get("type").filter(|v| truthy(v)) {
            out.push_str("type:");
            out.push_str(QUOTE);
            out.push_str(&upper(ty));
            out.push_str(QUOTE);
            out.push('}');
        }
    }
    if let Some(resp) = &tool.response {
        out.push_str(",response:{");
        if let Some(Value::String(desc)) = resp.get("description").filter(|v| truthy(v)) {
            out.push_str("description:");
            out.push_str(QUOTE);
            out.push_str(desc);
            out.push_str(QUOTE);
            out.push(',');
        }
        if resp.get("type").is_some_and(|t| upper(t) == "OBJECT") {
            out.push_str("type:");
            out.push_str(QUOTE);
            out.push_str("OBJECT");
            out.push_str(QUOTE);
            out.push('}');
        }
    }
    out.push('}');
    out
}

/// `format_parameters`: one property-map level of the schema serialization.
fn format_parameters(props: &serde_json::Map<String, Value>, filter_keys: bool, out: &mut String) {
    const STANDARD_KEYS: [&str; 5] = ["description", "type", "properties", "required", "nullable"];
    let mut first = true;
    for (key, value) in dictsort(props) {
        if filter_keys && STANDARD_KEYS.contains(&key.as_str()) {
            continue;
        }
        if !first {
            out.push(',');
        }
        first = false;
        out.push_str(key);
        out.push_str(":{");

        let mut add_comma = false;
        let comma = |out: &mut String, add_comma: &mut bool| {
            if *add_comma {
                out.push(',');
            } else {
                *add_comma = true;
            }
        };
        let ty = value.get("type").map(upper).unwrap_or_default();

        if let Some(Value::String(desc)) = value.get("description").filter(|v| truthy(v)) {
            out.push_str("description:");
            out.push_str(QUOTE);
            out.push_str(desc);
            out.push_str(QUOTE);
            add_comma = true;
        }
        if ty == "STRING" {
            if let Some(en) = value.get("enum").filter(|v| truthy(v)) {
                comma(out, &mut add_comma);
                out.push_str("enum:");
                format_argument(en, true, out);
            }
        } else if ty == "ARRAY"
            && let Some(Value::Object(items)) = value.get("items").filter(|v| truthy(v))
        {
            comma(out, &mut add_comma);
            out.push_str("items:{");
            let mut ifirst = true;
            for (ikey, ivalue) in dictsort(items) {
                if ivalue.is_null() {
                    continue;
                }
                if !ifirst {
                    out.push(',');
                }
                ifirst = false;
                match ikey.as_str() {
                    "properties" => {
                        out.push_str("properties:{");
                        if let Value::Object(map) = ivalue {
                            format_parameters(map, false, out);
                        }
                        out.push('}');
                    }
                    "required" => {
                        out.push_str("required:[");
                        if let Value::Array(req) = ivalue {
                            push_quoted_list(req, out);
                        }
                        out.push(']');
                    }
                    "type" => {
                        out.push_str("type:");
                        match ivalue {
                            Value::String(_) => {
                                format_argument(&Value::String(upper(ivalue)), true, out);
                            }
                            Value::Array(types) => {
                                let uppered: Vec<Value> =
                                    types.iter().map(|t| Value::String(upper(t))).collect();
                                format_argument(&Value::Array(uppered), true, out);
                            }
                            other => format_argument(other, true, out),
                        }
                    }
                    _ => {
                        out.push_str(ikey);
                        out.push(':');
                        format_argument(ivalue, true, out);
                    }
                }
            }
            out.push('}');
        }
        if value.get("nullable").is_some_and(truthy) {
            comma(out, &mut add_comma);
            out.push_str("nullable:true");
        }
        if ty == "OBJECT" {
            match value.get("properties") {
                Some(Value::Object(map)) => {
                    comma(out, &mut add_comma);
                    out.push_str("properties:{");
                    format_parameters(map, false, out);
                    out.push('}');
                }
                _ => {
                    // The schema node itself is treated as the property map,
                    // skipping the standard keys.
                    if let Value::Object(map) = value {
                        comma(out, &mut add_comma);
                        out.push_str("properties:{");
                        format_parameters(map, true, out);
                        out.push('}');
                    }
                }
            }
            if let Some(Value::Array(req)) = value.get("required").filter(|v| truthy(v)) {
                comma(out, &mut add_comma);
                out.push_str("required:[");
                push_quoted_list(req, out);
                out.push(']');
            }
        }
        comma(out, &mut add_comma);
        out.push_str("type:");
        out.push_str(QUOTE);
        out.push_str(&ty);
        out.push_str(QUOTE);
        out.push('}');
    }
}

/// `format_argument`: values in Gemma's argument syntax. Strings are wrapped
/// in `<|"|>`; object keys are wrapped only when `escape_keys`.
pub(crate) fn format_argument(value: &Value, escape_keys: bool, out: &mut String) {
    match value {
        Value::String(s) => {
            out.push_str(QUOTE);
            out.push_str(s);
            out.push_str(QUOTE);
        }
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Object(map) => {
            out.push('{');
            for (i, (key, v)) in dictsort(map).into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if escape_keys {
                    out.push_str(QUOTE);
                    out.push_str(key);
                    out.push_str(QUOTE);
                } else {
                    out.push_str(key);
                }
                out.push(':');
                format_argument(v, escape_keys, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                format_argument(v, escape_keys, out);
            }
            out.push(']');
        }
        // Python `str(None)`.
        Value::Null => out.push_str("None"),
        Value::Number(n) => {
            let s = n.to_string();
            out.push_str(&s);
            // Python floats always show a decimal point (str(2.0) == "2.0").
            // serde prints whole f64s the same way, so nothing extra to do —
            // but exotic magnitudes (1e30) format differently in Python and
            // are not supported (golden corpus stays in sane ranges).
        }
    }
}

fn push_quoted_list(items: &[Value], out: &mut String) {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(QUOTE);
        match item {
            Value::String(s) => out.push_str(s),
            other => format_argument(other, true, out),
        }
        out.push_str(QUOTE);
    }
}

/// jinja `dictsort`: case-insensitive, stable.
fn dictsort(map: &serde_json::Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by_key(|(k, _)| k.to_lowercase());
    entries
}

/// Python truthiness for the JSON values the template branches on.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// jinja `| upper` on a (string) value.
fn upper(v: &Value) -> String {
    match v {
        Value::String(s) => s.to_uppercase(),
        other => {
            let mut s = String::new();
            format_argument(other, true, &mut s);
            s.to_uppercase()
        }
    }
}
