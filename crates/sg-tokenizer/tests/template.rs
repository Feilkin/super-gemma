//! Chat-template parity: `render_prompt` must reproduce jinja2 rendering of
//! the GGUF template byte-for-byte on the golden corpus
//! (`tools/gen_template_fixtures.py`). Pure string work — runs everywhere,
//! no model file needed.

use serde_json::Value;
use sg_tokenizer::template::{
    ChatMessage, ChatRole, PromptOptions, ToolCallMsg, ToolDecl, render_prompt,
};

fn to_message(v: &Value) -> ChatMessage {
    let role = match v["role"].as_str().unwrap() {
        "system" | "developer" => ChatRole::System,
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        other => panic!("unknown role {other}"),
    };
    let tool_calls = v["tool_calls"]
        .as_array()
        .map(|calls| {
            calls
                .iter()
                .map(|c| ToolCallMsg {
                    id: c["id"].as_str().map(str::to_owned),
                    name: c["function"]["name"].as_str().unwrap().to_owned(),
                    arguments: c["function"]["arguments"].clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    ChatMessage {
        role,
        content: v["content"].as_str().unwrap_or_default().to_owned(),
        reasoning: v["reasoning"].as_str().map(str::to_owned),
        tool_calls,
        tool_call_id: v["tool_call_id"].as_str().map(str::to_owned),
        tool_name: v["name"].as_str().map(str::to_owned),
    }
}

fn to_tool(v: &Value) -> ToolDecl {
    let f = &v["function"];
    ToolDecl {
        name: f["name"].as_str().unwrap().to_owned(),
        description: f["description"].as_str().unwrap().to_owned(),
        parameters: f.get("parameters").cloned().unwrap_or(Value::Null),
        response: f.get("response").cloned(),
    }
}

#[test]
fn renders_byte_identical_to_jinja() {
    let fixture = include_str!("fixtures/template_parity.jsonl");
    let mut lines = fixture.lines();
    let header = lines.next().expect("header");
    eprintln!("fixture: {header}");

    for (i, line) in lines.enumerate() {
        let case: Value = serde_json::from_str(line).expect("fixture line");
        let messages: Vec<ChatMessage> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(to_message)
            .collect();
        let tools: Vec<ToolDecl> = case["tools"]
            .as_array()
            .map(|ts| ts.iter().map(to_tool).collect())
            .unwrap_or_default();
        let opts = PromptOptions {
            add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap(),
            enable_thinking: case["enable_thinking"].as_bool().unwrap(),
        };

        let got = render_prompt(&messages, &tools, opts);
        let want = case["rendered"].as_str().unwrap();
        assert_eq!(got, want, "case {i}: prompt mismatch");
    }
}
