//! Streaming parser for model output: splits the token stream into content,
//! thought-channel, and tool-call events (plan 01; plan 05 maps these onto
//! the API's SSE blocks).
//!
//! Wire format (pinned by the chat template and the `response_schema` in
//! `tokenizer_config.json`):
//!   `<|channel>thought\n…<channel|>`      reasoning span
//!   `<|tool_call>call:NAME{ARGS}<tool_call|>`  one tool invocation
//!   `<|tool_response>`                    model requests tool results
//!   `<turn|>` / `<eos>`                   end of turn
//! ARGS use Gemma's argument syntax: unquoted keys, `<|"|>`-quoted strings,
//! `true`/`false`/`None`, numbers, nested `{}`/`[]`.
//!
//! Channel boundaries are detected by *token id*, never by matching marker
//! strings in decoded text — BPE-produced lookalike text can't fake a
//! control token.

use serde_json::Value;

use crate::detok::DetokBuffer;
use crate::tokenizer::{DecodeError, Tokenizer};
use crate::vocab::Vocab;

/// One parsed event. Deltas are incremental and UTF-8 complete.
#[derive(Debug, Clone, PartialEq)]
pub enum TurnEvent {
    ContentDelta(String),
    ThinkingDelta(String),
    ToolCall {
        name: String,
        arguments: Value,
    },
    /// A `<|tool_call>` block that didn't parse; raw text preserved so the
    /// caller can surface it.
    InvalidToolCall {
        raw: String,
        error: String,
    },
    /// `<|tool_response>`: the model expects tool results before continuing.
    AwaitingToolResponse,
    /// `<turn|>` or `<eos>`.
    EndOfTurn,
}

#[derive(Debug)]
enum State {
    Content,
    /// After `<|channel>`, reading the header line (e.g. `thought`).
    ChannelHeader {
        header: String,
    },
    /// Inside a channel body until `<channel|>`.
    Channel,
    /// Inside `<|tool_call>…<tool_call|>`, accumulating raw text.
    ToolCall {
        raw: String,
    },
}

/// Incremental token-stream parser for one model turn (or several — state
/// resets at each `EndOfTurn`).
#[derive(Debug)]
pub struct TurnParser {
    state: State,
    detok: DetokBuffer,
    channel_open: u32,
    channel_close: u32,
    tool_call_open: u32,
    tool_call_close: u32,
    tool_response_open: u32,
    eot: u32,
    eos: u32,
}

impl TurnParser {
    pub fn new(vocab: &Vocab) -> Self {
        let id = |piece: &str| {
            vocab
                .id_of(piece)
                .unwrap_or_else(|| panic!("vocab is missing the {piece} control token"))
        };
        Self {
            state: State::Content,
            detok: DetokBuffer::new(),
            channel_open: id("<|channel>"),
            channel_close: id("<channel|>"),
            tool_call_open: id("<|tool_call>"),
            tool_call_close: id("<tool_call|>"),
            tool_response_open: id("<|tool_response>"),
            eot: vocab.eot(),
            eos: vocab.eos(),
        }
    }

    /// Feed one sampled token; parsed events are appended to `events`.
    pub fn push(
        &mut self,
        tokenizer: &Tokenizer,
        id: u32,
        events: &mut Vec<TurnEvent>,
    ) -> Result<(), DecodeError> {
        // Structural tokens switch state regardless of where they appear —
        // the model owns the framing, the parser just follows.
        if id == self.eot || id == self.eos {
            self.end_segment(events);
            events.push(TurnEvent::EndOfTurn);
            return Ok(());
        }
        if id == self.tool_call_open {
            self.end_segment(events);
            self.state = State::ToolCall { raw: String::new() };
            return Ok(());
        }
        if id == self.tool_call_close {
            if let State::ToolCall { raw } = std::mem::replace(&mut self.state, State::Content) {
                let mut raw = raw;
                self.detok.flush(&mut raw);
                events.push(parse_tool_call(&raw));
            }
            return Ok(());
        }
        if id == self.channel_open {
            self.end_segment(events);
            self.state = State::ChannelHeader {
                header: String::new(),
            };
            return Ok(());
        }
        if id == self.channel_close {
            self.end_segment(events);
            self.state = State::Content;
            return Ok(());
        }
        if id == self.tool_response_open {
            self.end_segment(events);
            events.push(TurnEvent::AwaitingToolResponse);
            return Ok(());
        }

        // Ordinary text: decode into the current segment.
        let mut text = String::new();
        tokenizer.decode_streaming(id, &mut self.detok, &mut text)?;
        if text.is_empty() {
            return Ok(());
        }
        match &mut self.state {
            State::Content => events.push(TurnEvent::ContentDelta(text)),
            State::Channel => events.push(TurnEvent::ThinkingDelta(text)),
            State::ToolCall { raw } => raw.push_str(&text),
            State::ChannelHeader { header } => {
                header.push_str(&text);
                if let Some(nl) = header.find('\n') {
                    // Body text may arrive in the same token as the header
                    // newline.
                    let body = header[nl + 1..].to_owned();
                    self.state = State::Channel;
                    if !body.is_empty() {
                        events.push(TurnEvent::ThinkingDelta(body));
                    }
                }
            }
        }
        Ok(())
    }

    /// Flush at end of stream (aborted generation): pending partial UTF-8
    /// degrades to U+FFFD, an unterminated tool call surfaces as invalid.
    pub fn finish(&mut self, events: &mut Vec<TurnEvent>) {
        self.end_segment(events);
        if let State::ToolCall { raw } = std::mem::replace(&mut self.state, State::Content)
            && !raw.is_empty()
        {
            events.push(TurnEvent::InvalidToolCall {
                raw,
                error: "unterminated tool call".into(),
            });
        }
        self.state = State::Content;
    }

    /// Close out the current text segment across a structural boundary.
    fn end_segment(&mut self, events: &mut Vec<TurnEvent>) {
        let mut tail = String::new();
        self.detok.flush(&mut tail);
        if tail.is_empty() {
            return;
        }
        match &mut self.state {
            State::Content => events.push(TurnEvent::ContentDelta(tail)),
            State::Channel | State::ChannelHeader { .. } => {
                events.push(TurnEvent::ThinkingDelta(tail));
            }
            State::ToolCall { raw } => raw.push_str(&tail),
        }
    }
}

/// Parse `call:NAME{ARGS}` into a [`TurnEvent`].
fn parse_tool_call(raw: &str) -> TurnEvent {
    match try_parse_tool_call(raw) {
        Ok((name, arguments)) => TurnEvent::ToolCall { name, arguments },
        Err(error) => TurnEvent::InvalidToolCall {
            raw: raw.to_owned(),
            error,
        },
    }
}

fn try_parse_tool_call(raw: &str) -> Result<(String, Value), String> {
    let rest = raw.strip_prefix("call:").ok_or("missing `call:` prefix")?;
    let brace = rest.find('{').ok_or("missing `{` after function name")?;
    let name = rest[..brace].trim();
    if name.is_empty() {
        return Err("empty function name".into());
    }
    let mut p = ArgParser {
        input: &rest[brace..],
        pos: 0,
    };
    let args = p.object()?;
    p.skip_ws();
    if p.pos != p.input.len() {
        return Err(format!("trailing input after arguments at byte {}", p.pos));
    }
    Ok((name.to_owned(), args))
}

/// Recursive-descent parser for Gemma's argument syntax (the inverse of
/// `template::format_argument` with `escape_keys = false`).
struct ArgParser<'a> {
    input: &'a str,
    pos: usize,
}

const QUOTE: &str = "<|\"|>";

impl ArgParser<'_> {
    fn rest(&self) -> &str {
        &self.input[self.pos..]
    }

    fn skip_ws(&mut self) {
        self.pos += self.rest().len() - self.rest().trim_start().len();
    }

    fn eat(&mut self, prefix: &str) -> bool {
        if self.rest().starts_with(prefix) {
            self.pos += prefix.len();
            true
        } else {
            false
        }
    }

    fn object(&mut self) -> Result<Value, String> {
        if !self.eat("{") {
            return Err(format!("expected `{{` at byte {}", self.pos));
        }
        let mut map = serde_json::Map::new();
        self.skip_ws();
        if self.eat("}") {
            return Ok(Value::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.key()?;
            self.skip_ws();
            if !self.eat(":") {
                return Err(format!(
                    "expected `:` after key {key:?} at byte {}",
                    self.pos
                ));
            }
            let value = self.value()?;
            map.insert(key, value);
            self.skip_ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("}") {
                return Ok(Value::Object(map));
            }
            return Err(format!("expected `,` or `}}` at byte {}", self.pos));
        }
    }

    /// Bare key (model-emitted keys are unquoted), or a `<|"|>`-quoted one.
    fn key(&mut self) -> Result<String, String> {
        if self.rest().starts_with(QUOTE) {
            return self.quoted();
        }
        let end = self
            .rest()
            .find([':', ',', '{', '}'])
            .ok_or_else(|| format!("unterminated key at byte {}", self.pos))?;
        let key = self.rest()[..end].trim();
        if key.is_empty() {
            return Err(format!("empty key at byte {}", self.pos));
        }
        let key = key.to_owned();
        self.pos += end;
        Ok(key)
    }

    fn value(&mut self) -> Result<Value, String> {
        self.skip_ws();
        if self.rest().starts_with(QUOTE) {
            return Ok(Value::String(self.quoted()?));
        }
        if self.rest().starts_with('{') {
            return self.object();
        }
        if self.eat("[") {
            let mut items = Vec::new();
            self.skip_ws();
            if self.eat("]") {
                return Ok(Value::Array(items));
            }
            loop {
                items.push(self.value()?);
                self.skip_ws();
                if self.eat(",") {
                    continue;
                }
                if self.eat("]") {
                    return Ok(Value::Array(items));
                }
                return Err(format!("expected `,` or `]` at byte {}", self.pos));
            }
        }
        // Bare scalar up to the next structural character.
        let end = self
            .rest()
            .find([',', '}', ']'])
            .unwrap_or(self.rest().len());
        let input: &str = self.input;
        let word = input[self.pos..self.pos + end].trim();
        self.pos += end;
        match word {
            "true" => Ok(Value::Bool(true)),
            "false" => Ok(Value::Bool(false)),
            "None" | "null" => Ok(Value::Null),
            "" => Err(format!("empty value at byte {}", self.pos)),
            w => {
                if let Ok(i) = w.parse::<i64>() {
                    Ok(Value::Number(i.into()))
                } else if let Ok(f) = w.parse::<f64>()
                    && let Some(n) = serde_json::Number::from_f64(f)
                {
                    Ok(Value::Number(n))
                } else {
                    // Unquoted enum-ish word: accept as string rather than
                    // failing the whole call.
                    Ok(Value::String(w.to_owned()))
                }
            }
        }
    }

    /// `<|"|>…<|"|>`; the closing quote is the next occurrence (the format
    /// has no escaping inside strings — mirrors the reference regex).
    fn quoted(&mut self) -> Result<String, String> {
        debug_assert!(self.rest().starts_with(QUOTE));
        self.pos += QUOTE.len();
        let end = self
            .rest()
            .find(QUOTE)
            .ok_or_else(|| format!("unterminated string at byte {}", self.pos))?;
        let s = self.rest()[..end].to_owned();
        self.pos += end + QUOTE.len();
        Ok(s)
    }
}
