//! BPE engine, special matching, and streaming-detok tests on a tiny
//! synthetic vocab (no model file needed; the real-vocab anchor is
//! `parity.rs`).

use sg_gguf::meta::{MetaArray, MetaValue, Metadata};
use sg_tokenizer::{DetokBuffer, SpecialTokens, Tokenizer, TurnEvent, TurnParser, VocabError};

const NORMALS: &[&str] = &[
    "a", "b", "c", "x", "y", "z", "ab", "bc", "abc", "xy", "xyz", "aa", "▁", "▁a", "\n",
];
const SPECIALS: &[&str] = &[
    "<s>",
    "<ss>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<|\"|>",
];
const MERGES: &[&str] = &["b c", "a b", "ab c", "x y", "xy z", "a a", "▁ a"];

/// Byte tokens sit at ids 4..260, normals after that.
fn id_of_byte(b: u8) -> u32 {
    4 + b as u32
}

fn id_of(piece: &str) -> u32 {
    let i = NORMALS
        .iter()
        .chain(SPECIALS)
        .position(|p| *p == piece)
        .unwrap_or_else(|| panic!("{piece} not in tiny vocab"));
    260 + i as u32
}

fn tiny_meta() -> Metadata {
    let mut pieces: Vec<String> = ["<pad>", "<eos>", "<bos>", "<unk>"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let mut types = vec![3i32; 4];
    for b in 0..=255u8 {
        pieces.push(format!("<0x{b:02X}>"));
        types.push(6);
    }
    for p in NORMALS {
        pieces.push((*p).to_owned());
        types.push(1);
    }
    for p in SPECIALS {
        pieces.push((*p).to_owned());
        types.push(3);
    }

    let mut m = Metadata::default();
    let mut kv = |k: &str, v: MetaValue| assert!(m.insert(k.into(), v));
    kv("tokenizer.ggml.model", MetaValue::String("gemma4".into()));
    kv(
        "tokenizer.ggml.scores",
        MetaValue::Array(MetaArray::F32(vec![-1000.0; pieces.len()])),
    );
    kv(
        "tokenizer.ggml.tokens",
        MetaValue::Array(MetaArray::String(pieces)),
    );
    kv(
        "tokenizer.ggml.token_type",
        MetaValue::Array(MetaArray::I32(types)),
    );
    kv(
        "tokenizer.ggml.merges",
        MetaValue::Array(MetaArray::String(
            MERGES.iter().map(|s| s.to_string()).collect(),
        )),
    );
    kv("tokenizer.ggml.bos_token_id", MetaValue::U32(2));
    kv("tokenizer.ggml.eos_token_id", MetaValue::U32(1));
    kv("tokenizer.ggml.padding_token_id", MetaValue::U32(0));
    kv("tokenizer.ggml.unknown_token_id", MetaValue::U32(3));
    kv("tokenizer.ggml.add_bos_token", MetaValue::Bool(false));
    kv("tokenizer.ggml.add_space_prefix", MetaValue::Bool(false));
    m
}

fn tok() -> Tokenizer {
    Tokenizer::from_metadata(&tiny_meta()).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn merges_cascade_in_rank_order() {
    let t = tok();
    // (x,y) then (xy,z).
    assert_eq!(t.encode("xyz", SpecialTokens::Plain), [id_of("xyz")]);
    // For "abc", (b,c) has rank 0 and wins over (a,b); no (a,bc) merge exists.
    assert_eq!(
        t.encode("abc", SpecialTokens::Plain),
        [id_of("a"), id_of("bc")]
    );
}

#[test]
fn equal_rank_merges_leftmost_first() {
    let t = tok();
    assert_eq!(
        t.encode("aaaa", SpecialTokens::Plain),
        [id_of("aa"), id_of("aa")]
    );
    // Leftmost (a,a) wins, leaving [aa, a] — not [a, aa].
    assert_eq!(
        t.encode("aaa", SpecialTokens::Plain),
        [id_of("aa"), id_of("a")]
    );
}

#[test]
fn spaces_normalize_to_underscore_piece() {
    let t = tok();
    assert_eq!(
        t.encode("a a", SpecialTokens::Plain),
        [id_of("a"), id_of("▁a")]
    );
    // A literal ▁ in the input behaves like a space.
    assert_eq!(
        t.encode("a▁a", SpecialTokens::Plain),
        [id_of("a"), id_of("▁a")]
    );
}

#[test]
fn unknown_chars_fall_back_to_bytes_before_merging() {
    let t = tok();
    // 'é' = C3 A9.
    assert_eq!(
        t.encode("é", SpecialTokens::Plain),
        [id_of_byte(0xC3), id_of_byte(0xA9)]
    );
    assert_eq!(
        t.encode("aé", SpecialTokens::Plain),
        [id_of("a"), id_of_byte(0xC3), id_of_byte(0xA9)]
    );
}

#[test]
fn special_matching_is_leftmost_longest_and_mode_gated() {
    let t = tok();
    let got = t.encode("x<ss>y", SpecialTokens::Match);
    assert_eq!(got, [id_of("x"), id_of("<ss>"), id_of("y")]);
    // "<ss>" must win over its prefix "<s>".
    assert!(!got.contains(&id_of("<s>")));
    assert_eq!(
        t.encode("<s>x", SpecialTokens::Match),
        [id_of("<s>"), id_of("x")]
    );

    // Plain mode: '<', 's', '>' aren't vocab pieces, so they byte-fall back —
    // and crucially no control id appears.
    let plain = t.encode("x<ss>y", SpecialTokens::Plain);
    assert!(!plain.contains(&id_of("<ss>")) && !plain.contains(&id_of("<s>")));
    assert_eq!(
        plain,
        [
            id_of("x"),
            id_of_byte(b'<'),
            id_of_byte(b's'),
            id_of_byte(b's'),
            id_of_byte(b'>'),
            id_of("y"),
        ]
    );
}

#[test]
fn decode_reverses_normalization_bytes_and_specials() {
    let t = tok();
    for text in ["a a", "abc xyz", "aé", "x<ss>y", "line\nbreak"] {
        let ids = t.encode(text, SpecialTokens::Match);
        assert_eq!(t.decode(&ids).unwrap(), *text, "{text:?}");
    }
    assert_eq!(t.decode(&[2, id_of("a")]).unwrap(), "<bos>a");
    assert!(t.decode(&[99_999]).is_err());
}

#[test]
fn streaming_detok_withholds_partial_code_points() {
    let t = tok();
    let mut buf = DetokBuffer::new();
    let mut out = String::new();
    // 👍 = F0 9F 91 8D as four byte tokens.
    for (i, b) in [0xF0u8, 0x9F, 0x91, 0x8D].into_iter().enumerate() {
        t.decode_streaming(id_of_byte(b), &mut buf, &mut out)
            .unwrap();
        if i < 3 {
            assert!(out.is_empty(), "emitted partial UTF-8 at byte {i}");
            assert!(buf.has_pending());
        }
    }
    assert_eq!(out, "👍");
    assert!(!buf.has_pending());
}

#[test]
fn streaming_detok_replaces_broken_runs() {
    let t = tok();

    // Incomplete run flushed at end of generation.
    let mut buf = DetokBuffer::new();
    let mut out = String::new();
    t.decode_streaming(id_of_byte(0xF0), &mut buf, &mut out)
        .unwrap();
    t.flush_streaming(&mut buf, &mut out);
    assert_eq!(out, "\u{FFFD}");

    // Incomplete run terminated by a normal token.
    let mut buf = DetokBuffer::new();
    let mut out = String::new();
    t.decode_streaming(id_of_byte(0xF0), &mut buf, &mut out)
        .unwrap();
    t.decode_streaming(id_of("a"), &mut buf, &mut out).unwrap();
    assert_eq!(out, "\u{FFFD}a");

    // Outright invalid byte is replaced immediately.
    let mut buf = DetokBuffer::new();
    let mut out = String::new();
    t.decode_streaming(id_of_byte(0xFF), &mut buf, &mut out)
        .unwrap();
    assert_eq!(out, "\u{FFFD}");
}

#[test]
fn construction_validates_the_payload() {
    // Wrong model string.
    let mut m = Metadata::default();
    m.insert(
        "tokenizer.ggml.model".into(),
        MetaValue::String("llama".into()),
    );
    assert!(matches!(
        Tokenizer::from_metadata(&m),
        Err(VocabError::WrongModel(_))
    ));

    // add_bos_token = true contradicts the hard-coded encode behavior.
    let mut m = Metadata::default();
    for (k, v) in tiny_meta().iter() {
        let v = if k == "tokenizer.ggml.add_bos_token" {
            MetaValue::Bool(true)
        } else {
            v.clone()
        };
        m.insert(k.to_owned(), v);
    }
    assert!(matches!(
        Tokenizer::from_metadata(&m),
        Err(VocabError::UnsupportedFlag { .. })
    ));

    // A merge referencing absent pieces.
    let mut m = Metadata::default();
    for (k, v) in tiny_meta().iter() {
        let v = if k == "tokenizer.ggml.merges" {
            MetaValue::Array(MetaArray::String(vec!["a z".into()]))
        } else {
            v.clone()
        };
        m.insert(k.to_owned(), v);
    }
    assert!(matches!(
        Tokenizer::from_metadata(&m),
        Err(VocabError::MergePieceMissing { .. })
    ));
}

fn feed(t: &Tokenizer, parser: &mut TurnParser, text: &str, events: &mut Vec<TurnEvent>) {
    for id in t.encode(text, SpecialTokens::Match) {
        parser.push(t, id, events).unwrap();
    }
}

fn collect_text(events: &[TurnEvent], want_thinking: bool) -> String {
    events
        .iter()
        .filter_map(|e| match e {
            TurnEvent::ThinkingDelta(s) if want_thinking => Some(s.as_str()),
            TurnEvent::ContentDelta(s) if !want_thinking => Some(s.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn turn_parser_splits_thinking_content_and_tool_calls() {
    let t = tok();
    let mut parser = TurnParser::new(t.vocab());
    let mut events = Vec::new();

    feed(
        &t,
        &mut parser,
        "<|channel>thought\nplan it<channel|>",
        &mut events,
    );
    feed(&t, &mut parser, "answer ok", &mut events);
    feed(
        &t,
        &mut parser,
        "<|tool_call>call:get_weather{city:<|\"|>Oulu, {FI}<|\"|>,days:3,\
         opts:{dry:true,ratio:2.5},list:[1,None,metric],empty:{}}<tool_call|>",
        &mut events,
    );
    feed(&t, &mut parser, "<turn|>", &mut events);
    parser.finish(&mut events);

    assert_eq!(collect_text(&events, true), "plan it");
    assert_eq!(collect_text(&events, false), "answer ok");
    let call = events
        .iter()
        .find(|e| matches!(e, TurnEvent::ToolCall { .. }))
        .expect("tool call event");
    let TurnEvent::ToolCall { name, arguments } = call else {
        unreachable!()
    };
    assert_eq!(name, "get_weather");
    assert_eq!(
        *arguments,
        serde_json::json!({
            "city": "Oulu, {FI}",
            "days": 3,
            "opts": {"dry": true, "ratio": 2.5},
            "list": [1, null, "metric"],
            "empty": {},
        })
    );
    assert_eq!(events.last(), Some(&TurnEvent::EndOfTurn));
}

#[test]
fn turn_parser_reports_tool_response_marker_and_bad_calls() {
    let t = tok();
    let mut parser = TurnParser::new(t.vocab());
    let mut events = Vec::new();

    feed(
        &t,
        &mut parser,
        "<|tool_call>call:ping{}<tool_call|>",
        &mut events,
    );
    feed(&t, &mut parser, "<|tool_response>", &mut events);
    assert!(events.contains(&TurnEvent::AwaitingToolResponse));
    assert!(events.iter().any(|e| matches!(
        e,
        TurnEvent::ToolCall { name, arguments }
            if name == "ping" && *arguments == serde_json::json!({})
    )));

    let mut events = Vec::new();
    feed(
        &t,
        &mut parser,
        "<|tool_call>nonsense<tool_call|>",
        &mut events,
    );
    assert!(
        matches!(&events[0], TurnEvent::InvalidToolCall { error, .. } if error.contains("call:")),
        "{events:?}"
    );

    // Unterminated call surfaces at finish().
    let mut events = Vec::new();
    feed(&t, &mut parser, "<|tool_call>call:ping{", &mut events);
    parser.finish(&mut events);
    assert!(
        matches!(&events[0], TurnEvent::InvalidToolCall { error, .. } if error.contains("unterminated")),
        "{events:?}"
    );
}

#[test]
fn turn_parser_keeps_lookalike_text_as_content() {
    let t = tok();
    let mut parser = TurnParser::new(t.vocab());
    let mut events = Vec::new();
    // Encoded Plain, the marker string is ordinary bytes — must stay content.
    for id in t.encode("see <|tool_call> here", SpecialTokens::Plain) {
        parser.push(&t, id, &mut events).unwrap();
    }
    parser.finish(&mut events);
    assert_eq!(collect_text(&events, false), "see <|tool_call> here");
    assert!(
        events
            .iter()
            .all(|e| matches!(e, TurnEvent::ContentDelta(_)))
    );
}

#[test]
fn vocab_exposes_the_stop_tokens() {
    let t = tok();
    assert_eq!(t.vocab().bos(), 2);
    assert_eq!(t.vocab().eos(), 1);
    assert_eq!(t.vocab().eot(), id_of("<turn|>"));
}
