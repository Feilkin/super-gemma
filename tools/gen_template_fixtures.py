#!/usr/bin/env python3
"""Generate chat-template parity fixtures (plan 01).

Renders a corpus of (messages, tools, options) cases through the Gemma 4
chat template with jinja2 — configured exactly like transformers'
`apply_chat_template` (trim_blocks, lstrip_blocks) — and writes JSONL
fixtures that the Rust `PromptBuilder` must match byte-for-byte.

Pinned environment (same venv as gen_tokenizer_fixtures.py):
    ~/.venvs/super-gemma/bin/pip install jinja2
Run from the repo root:
    ~/.venvs/super-gemma/bin/python tools/gen_template_fixtures.py

Inputs:  docs/reference/gemma-4-chat-template.jinja (extracted from the GGUF)
Outputs: crates/sg-tokenizer/tests/fixtures/template_parity.jsonl
"""

import json
from pathlib import Path

import jinja2

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "docs/reference/gemma-4-chat-template.jinja"
OUT = REPO / "crates/sg-tokenizer/tests/fixtures/template_parity.jsonl"


def msg(role, content, **kw):
    return {"role": role, "content": content, **kw}


def call(name, arguments, id=None):
    c = {"function": {"name": name, "arguments": arguments}}
    if id is not None:
        c["id"] = id
    return c


def tool(name, description, parameters=None, response=None):
    fn = {"name": name, "description": description}
    if parameters is not None:
        fn["parameters"] = parameters
    if response is not None:
        fn["response"] = response
    return {"type": "function", "function": fn}


WEATHER_TOOL = tool(
    "get_weather",
    "Get current weather for a city.",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["metric", "imperial"], "description": "Unit system"},
            "days": {"type": "integer", "description": "Forecast days", "nullable": True},
        },
        "required": ["city"],
    },
)

EDIT_TOOL = tool(
    "edit_file",
    "Apply edits to a file.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
            "edits": {
                "type": "array",
                "description": "Edit operations",
                "items": {
                    "type": "object",
                    "properties": {
                        "old": {"type": "string", "description": "Text to replace"},
                        "new": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["old", "new"],
                },
            },
            "options": {
                "type": "object",
                "description": "Extra options",
                "properties": {
                    "dry_run": {"type": "boolean", "description": "Validate only"},
                    "Backup": {"type": "boolean"},
                },
                "required": ["dry_run"],
            },
        },
        "required": ["path", "edits"],
    },
    response={"description": "Edit results", "type": "object"},
)

# items.type as a list; object schema without explicit properties (the
# filter_keys branch); enum of strings.
ODD_TOOL = tool(
    "odd_tool",
    "Schema corner cases.",
    {
        "type": "object",
        "properties": {
            "mixed": {"type": "array", "description": "Mixed list", "items": {"type": ["string", "integer"]}},
            "bare_obj": {
                "type": "object",
                "description": "Object sans properties key",
                "loose_field": {"type": "string", "description": "Inline prop"},
                "Zeta": {"type": "integer"},
            },
            "mode": {"type": "string", "enum": ["a", "B", "c"]},
        },
        "required": ["mode"],
    },
)

NOPARAM_TOOL = tool("ping", "Liveness check.")

CASES = [
    # Basic shapes.
    dict(messages=[msg("user", "Hello!")], add_generation_prompt=True),
    dict(messages=[msg("user", "Hello!")], add_generation_prompt=False),
    dict(messages=[msg("system", "You are terse."), msg("user", "Hi")], add_generation_prompt=True),
    dict(
        messages=[
            msg("system", "  You are terse.  \n"),
            msg("user", " padded "),
            msg("assistant", "Sure."),
            msg("user", "More?\nSecond line."),
        ],
        add_generation_prompt=True,
    ),
    # developer role counts as system.
    dict(messages=[msg("developer", "Dev rules."), msg("user", "Go")], add_generation_prompt=True),
    # Thinking on/off.
    dict(messages=[msg("user", "Think hard.")], add_generation_prompt=True, enable_thinking=True),
    dict(
        messages=[msg("system", "Sys."), msg("user", "Q")],
        add_generation_prompt=True,
        enable_thinking=True,
    ),
    # Assistant history containing thought spans (strip_thinking).
    dict(
        messages=[
            msg("user", "Q1"),
            msg("assistant", "<|channel>thought\nhidden reasoning<channel|>Visible answer."),
            msg("user", "Q2"),
        ],
        add_generation_prompt=True,
    ),
    dict(
        messages=[
            msg("user", "Q"),
            msg("assistant", "pre <|channel>think a<channel|>mid<|channel>think b<channel|> post"),
            msg("user", "again"),
        ],
        add_generation_prompt=True,
    ),
    # Consecutive assistant messages: continuation suppresses the turn header.
    dict(
        messages=[
            msg("user", "Q"),
            msg("assistant", "Part one."),
            msg("assistant", "Part two."),
            msg("user", "ok"),
        ],
        add_generation_prompt=True,
    ),
    # Tools declared, no calls yet.
    dict(messages=[msg("user", "Weather in Oulu?")], tools=[WEATHER_TOOL], add_generation_prompt=True),
    dict(
        messages=[msg("system", "Use tools."), msg("user", "Hi")],
        tools=[WEATHER_TOOL, EDIT_TOOL, ODD_TOOL, NOPARAM_TOOL],
        add_generation_prompt=True,
    ),
    dict(
        messages=[msg("user", "thinking + tools")],
        tools=[NOPARAM_TOOL],
        add_generation_prompt=True,
        enable_thinking=True,
    ),
    # A call awaiting its result: turn stays open at <|tool_response>.
    dict(
        messages=[
            msg("user", "Weather in Oulu?"),
            msg("assistant", "", tool_calls=[call("get_weather", {"city": "Oulu", "units": "metric"})]),
        ],
        tools=[WEATHER_TOOL],
        add_generation_prompt=True,
    ),
    # Call + result + final answer (id-based name resolution).
    dict(
        messages=[
            msg("user", "Weather in Oulu?"),
            msg(
                "assistant",
                "",
                tool_calls=[call("get_weather", {"units": "metric", "city": "Oulu"}, id="c1")],
            ),
            msg("tool", '{"temp_c": -7, "sky": "clear"}', tool_call_id="c1"),
            msg("assistant", "It is -7 °C and clear."),
            msg("user", "and tomorrow?"),
        ],
        tools=[WEATHER_TOOL],
        add_generation_prompt=True,
    ),
    # Two calls, two results; name via id match and via explicit "name".
    # (A tool message with neither — unresolvable name — crashes the original
    # template ('None' + str), so the server must always resolve one.)
    dict(
        messages=[
            msg("user", "Compare Oulu and Turku"),
            msg(
                "assistant",
                "",
                tool_calls=[
                    call("get_weather", {"city": "Oulu"}, id="a"),
                    call("get_weather", {"city": "Turku"}, id="b"),
                ],
            ),
            msg("tool", "snow", tool_call_id="a"),
            msg("tool", "rain", name="get_weather"),
        ],
        tools=[WEATHER_TOOL],
        add_generation_prompt=True,
    ),
    # Result plus same-message content: turn closes normally.
    dict(
        messages=[
            msg("user", "Q"),
            msg("assistant", "Done: ok", tool_calls=[call("ping", {})]),
            msg("tool", "pong", name="ping"),
        ],
        tools=[NOPARAM_TOOL],
        add_generation_prompt=True,
    ),
    # Argument soup: nesting, arrays, numbers, bools, null, unicode, dictsort.
    dict(
        messages=[
            msg("user", "args"),
            msg(
                "assistant",
                "",
                tool_calls=[
                    call(
                        "edit_file",
                        {
                            "path": "/tmp/ä öß.txt",
                            "edits": [
                                {"old": "a{b}", "new": "c:d,e", "count": 2},
                                {"old": "x", "new": "", "count": 0},
                            ],
                            "options": {"dry_run": True, "Backup": False, "ratio": 0.5, "note": None},
                            "Zz": [1, -2, 3.5, "four", True, None],
                            "aA": "case sort test",
                        },
                        id="e1",
                    )
                ],
            ),
            msg("tool", "3 edits applied", tool_call_id="e1"),
        ],
        tools=[EDIT_TOOL],
        add_generation_prompt=True,
    ),
    # Pre-serialized string arguments pass through verbatim.
    dict(
        messages=[
            msg("user", "raw"),
            msg("assistant", "", tool_calls=[call("ping", 'payload:<|"|>raw string<|"|>')]),
        ],
        tools=[NOPARAM_TOOL],
        add_generation_prompt=True,
    ),
    # Reasoning rendering: only after the last user message AND with calls.
    dict(
        messages=[
            msg("user", "Q"),
            msg(
                "assistant",
                "",
                reasoning="I should call the tool.",
                tool_calls=[call("ping", {})],
            ),
        ],
        tools=[NOPARAM_TOOL],
        add_generation_prompt=True,
    ),
    dict(
        messages=[
            msg("user", "Q1"),
            msg("assistant", "answer", reasoning="suppressed: before last user"),
            msg("user", "Q2"),
        ],
        add_generation_prompt=True,
    ),
    dict(
        messages=[
            msg("user", "Q"),
            msg("assistant", "no calls so no channel", reasoning="suppressed: no tool_calls"),
        ],
        add_generation_prompt=False,
    ),
    # Mid-conversation system message keeps the system role header.
    dict(
        messages=[
            msg("user", "Q"),
            msg("assistant", "A"),
            msg("system", "New rules."),
            msg("user", "Q2"),
        ],
        add_generation_prompt=True,
    ),
    # Unicode + specials-looking content.
    dict(
        messages=[
            msg("system", "Répondez 简洁に."),
            msg("user", "user text with <turn|> and <|tool_call> lookalikes 🤔"),
            msg("assistant", "emoji 👍 answer"),
            msg("user", "next"),
        ],
        add_generation_prompt=True,
    ),
]


def main() -> None:
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    template = env.from_string(TEMPLATE.read_text(encoding="utf-8"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"generator": "tools/gen_template_fixtures.py", "cases": len(CASES)}) + "\n")
        for case in CASES:
            rendered = template.render(
                messages=case["messages"],
                tools=case.get("tools"),
                add_generation_prompt=case.get("add_generation_prompt", False),
                enable_thinking=case.get("enable_thinking", False),
                bos_token="<bos>",
            )
            record = {
                "messages": case["messages"],
                "tools": case.get("tools"),
                "add_generation_prompt": case.get("add_generation_prompt", False),
                "enable_thinking": case.get("enable_thinking", False),
                "rendered": rendered,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
