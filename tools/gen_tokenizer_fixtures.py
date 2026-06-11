#!/usr/bin/env python3
"""Generate the tokenizer parity fixtures (plan 01).

Encodes a deterministic ~12k-line corpus with the HF `tokenizers` runtime and
the official `google/gemma-4-31B-it` tokenizer.json, writing JSONL
{"text", "ids"} fixtures that `sg-tokenizer`'s parity test must match 100 %.

Pinned environment (see docs/reference/README.md conventions):
    python3 -m venv ~/.venvs/super-gemma
    ~/.venvs/super-gemma/bin/pip install tokenizers
Run from the repo root:
    ~/.venvs/super-gemma/bin/python tools/gen_tokenizer_fixtures.py

Inputs:  models/hf-gemma-4-31b-it/tokenizer.json  (hf download google/gemma-4-31B-it)
Outputs: crates/sg-tokenizer/tests/fixtures/encode_parity.jsonl

Never hand-edit the output; regenerate. The corpus is generated with a fixed
RNG seed, so the file is reproducible byte-for-byte for a given tokenizers
version (recorded in the header line).
"""

import json
import random
import sys
from pathlib import Path

import tokenizers
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parent.parent
TOKENIZER_JSON = REPO / "models/hf-gemma-4-31b-it/tokenizer.json"
OUT = REPO / "crates/sg-tokenizer/tests/fixtures/encode_parity.jsonl"

SEED = 0x5EED_6E44


def fixed_samples() -> list[str]:
    s = [
        # --- plain text, several scripts ---
        "The quick brown fox jumps over the lazy dog.",
        "Sphinx of black quartz, judge my vow!",
        "hello", " hello", "hello ", "  hello  world  ",
        "Hello, World! How are you today?",
        "Übergrößenträger äußerst öffentlich",
        "Ça va très bien, merci — à bientôt !",
        "El niño comió piña y jalapeños.",
        "Zażółć gęślą jaźń",
        "Příliš žluťoučký kůň úpěl ďábelské ódy",
        "Съешь же ещё этих мягких французских булок, да выпей чаю",
        "Γαζέες καὶ μυρτιὲς δὲν θὰ βρῶ πιὰ στὸ χρυσαφὶ ξέφωτο",
        "בְּרֵאשִׁית בָּרָא אֱלֹהִים",
        "صِف خَلقَ خَودِ كَمِثلِ الشَمسِ إِذ بَزَغَت",
        "我能吞下玻璃而不伤身体。",
        "私はガラスを食べられます。それは私を傷つけません。",
        "나는 유리를 먹을 수 있어요. 그래도 아프지 않아요",
        "मैं काँच खा सकता हूँ और मुझे उससे कोई चोट नहीं पहुंचती।",
        "ฉันกินกระจกได้ แต่มันไม่ทำให้ฉันเจ็บ",
        "Tôi có thể ăn thủy tinh mà không hại gì.",
        "ვეპხის ტყაოსანი შოთა რუსთაველი",
        "Mogę jeść szkło, i mi nie szkodzi.",
        # --- emoji & symbols ---
        "👍", "🤷🏽‍♀️", "👨‍👩‍👧‍👦 family", "flags 🇫🇮🇯🇵🇧🇷",
        "emoji soup 🐍🦀🔥✨🎉🚀💯",
        "math: ∀x∈ℝ: ⌈x⌉ ≥ ⌊x⌋, ∑ᵢ aᵢ ≠ ∅, √(-1) = i",
        "box ┌─┬─┐ │ ├─┼─┤ └─┴─┘ blocks ░▒▓█",
        "zalgo: ḩ̸̢̛e̵̟̔l̷̡̓l̸̮̈o̶͚̾",
        "Ligatures: ﬁ ﬂ ﬃ; fullwidth: ＨＥＬＬＯ　ＷＯＲＬＤ",
        "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕄𝕒𝕥𝕙 𝒮𝒸𝓇𝒾𝓅𝓉 𝚖𝚘𝚗𝚘",
        "rare plane 1: 𐀀𐀁𐀂 (Linear B), 𓀀𓀁 (hieroglyphs), 🜁🜂🜃",
        # --- whitespace pathologies ---
        " ", "  ", "   ", "\t", "\t\t", "\n", "\n\n", "\r\n", "\r",
        " \t \n \t ", "a b", "a  b", "a   b", "a\tb", "a\nb", "a\r\nb",
        "    indented four", "\tindented tab", "trailing spaces   ",
        "many                         spaces",
        "\n\n\n\n\n\n\n\n\n\n\n\n", "\t\t\t\t\t\t\t\t\t\t",
        "mixed \t\n \t\n whitespace",
        " nbsp and em-space and​zwsp",
        # --- control chars & oddities ---
        "null\x00byte", "bell\x07char", "esc\x1bseq", "del\x7fchar",
        "combining: é à ô ñ ü",
        "bidi: ‮REVERSED‬ done",
        # --- special-token strings appearing in user text ---
        "<bos>", "<eos>", "<pad>", "<unk>", "<mask>",
        "<|turn>user\nhello<turn|>",
        "<|channel>thought\nsome reasoning<channel|>",
        "<|tool_call>call:get_weather{\"city\": \"Helsinki\"}<tool_call|>",
        "<|tool_response>{\"ok\": true}<tool_response|>",
        "text with <|think|> marker and <|\"|> escape",
        "almost special: <|turn >, < |turn>, <turn|, |turn>, <<|turn>>",
        "<|image|> <|audio|> <|video|> <image|> <audio|>",
        "<|nonexistent|> <fake> <|> <||>",
        # --- code ---
        'fn main() { println!("hello, world"); }',
        "pub async fn read_at(&self, buf: &mut [u8], off: u64) -> io::Result<usize> {",
        "let x: Vec<u32> = (0..60).map(|i| if i % 6 == 5 { 4 } else { 16 }).collect();",
        "#[derive(Debug, Clone, PartialEq)]",
        "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "lambda *args, **kwargs: functools.reduce(operator.add, args, 0)",
        "SELECT t.name, COUNT(*) FROM tensors t GROUP BY 1 HAVING COUNT(*) > 10;",
        "if (ptr == NULL) { return -EINVAL; } /* classic */",
        "template<typename T> constexpr auto&& fwd = std::forward<T>;",
        '{"key": "value", "nested": {"list": [1, 2.5, -3e10, null, true]}}',
        "<html><body onload=\"alert('xss')\">&amp;&lt;&gt;</body></html>",
        "git commit -m \"fix: handle \\\"quoted\\\" args\" && git push",
        "s/foo/bar/g; awk '{print $1}' | xargs -0 rm --",
        "0x4655_4747 0b1010 0o777 1_000_000 6.022e23 .5f64 1e-6",
        "регистрация_пользователя = функция(имя, пароль)",
        "变量名 = 函数(参数一, 参数二)  # 中文注释",
        # --- long tokens / repetition ---
        "a" * 200, "ab" * 100, "▁" * 50, "the " * 80,
        "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
        "ThisIsAVeryLongCamelCaseIdentifierThatGoesOnAndOnAndOn",
        "snake_case_name_with_many_many_many_segments_indeed",
        "https://example.com/path/to/resource?query=value&other=1#fragment",
        "user.name+tag@sub.domain.example.co.uk",
        "",  # empty string
    ]
    return s


CODE_IDENTS = [
    "buffer", "offset", "tensor", "layer", "head", "cache", "token", "ctx",
    "stride", "queue", "fence", "submit", "decode", "prefill", "rope", "norm",
]
WORDS = (
    "the of and to in a is that it for as was with be by on not he this are "
    "or his from at which but have an had they you were her all she there "
    "would their we him been has when who will no more if out so up said what "
    "its about than into them can only other time new some could these two "
    "may first then do any like my now over such our man me even most made"
).split()
UNICODE_RANGES = [
    (0x0020, 0x007E),   # ASCII
    (0x00A1, 0x024F),   # Latin-1/Extended
    (0x0370, 0x03FF),   # Greek
    (0x0400, 0x04FF),   # Cyrillic
    (0x0590, 0x05F4),   # Hebrew
    (0x0600, 0x06FF),   # Arabic
    (0x0900, 0x097F),   # Devanagari
    (0x0E00, 0x0E5B),   # Thai
    (0x3040, 0x30FF),   # Kana
    (0x4E00, 0x9FFF),   # CJK
    (0xAC00, 0xD7A3),   # Hangul
    (0x1F300, 0x1F64F), # emoji
    (0x10000, 0x100FA), # Linear B (byte-fallback territory)
]


def generated_samples(rng: random.Random, n: int) -> list[str]:
    out = []
    for _ in range(n):
        kind = rng.randrange(6)
        if kind == 0:  # word salad
            k = rng.randrange(1, 30)
            sep = rng.choice([" ", "  ", " ", " ", "\t"])
            out.append(sep.join(rng.choice(WORDS) for _ in range(k)))
        elif kind == 1:  # code-ish line
            a, b, c = (rng.choice(CODE_IDENTS) for _ in range(3))
            n1, n2 = rng.randrange(0, 1 << 16), rng.randrange(0, 1 << 8)
            tpl = rng.choice([
                f"let {a}_{b} = {c}[{n1}] >> {n2};",
                f"{a}.{b}({c}, {n1}, 0x{n2:x})",
                f"if {a} != {n1} {{ {b}.push({c}); }}",
                f"for {a} in 0..{n2} {{ {b} += {c}[{a}]; }}",
                f"assert_eq!({a}.{b}(), Some({n1}));",
            ])
            indent = rng.choice(["", "    ", "        ", "\t"])
            out.append(indent + tpl)
        elif kind == 2:  # random unicode from one range
            lo, hi = rng.choice(UNICODE_RANGES)
            k = rng.randrange(1, 40)
            out.append("".join(chr(rng.randrange(lo, hi + 1)) for _ in range(k)))
        elif kind == 3:  # mixed-range unicode with spaces
            k = rng.randrange(2, 25)
            chars = []
            for _ in range(k):
                lo, hi = rng.choice(UNICODE_RANGES)
                chars.append(chr(rng.randrange(lo, hi + 1)))
                if rng.random() < 0.2:
                    chars.append(" ")
            out.append("".join(chars))
        elif kind == 4:  # whitespace torture
            k = rng.randrange(1, 20)
            out.append("".join(rng.choice([" ", "\t", "\n", "x", "▁", "."]) for _ in range(k)))
        else:  # numbers & punctuation soup
            k = rng.randrange(1, 25)
            out.append("".join(rng.choice("0123456789.,;:!?()[]{}+-*/=<>|&^%$#@~`'\"\\ ") for _ in range(k)))
    return out


def main() -> None:
    if not TOKENIZER_JSON.exists():
        sys.exit(f"missing {TOKENIZER_JSON}; run: hf download google/gemma-4-31B-it "
                 "tokenizer.json tokenizer_config.json --local-dir models/hf-gemma-4-31b-it")
    tok = Tokenizer.from_file(str(TOKENIZER_JSON))
    rng = random.Random(SEED)
    corpus = fixed_samples() + generated_samples(rng, 12_000)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        header = {
            "generator": "tools/gen_tokenizer_fixtures.py",
            "tokenizers_version": tokenizers.__version__,
            "seed": SEED,
            "lines": len(corpus),
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for text in corpus:
            ids = tok.encode(text, add_special_tokens=False).ids
            f.write(json.dumps({"text": text, "ids": ids}, ensure_ascii=False) + "\n")
    print(f"wrote {len(corpus)} cases to {OUT}")


if __name__ == "__main__":
    main()
