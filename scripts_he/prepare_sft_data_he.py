"""
Builds Hebrew-only SFT JSONLs in the bare-list format required by
tasks/customjson.py (each line is a JSON list of {role, content},
alternating user/assistant starting with user).

Hebrew SFT sources:
  - dicta-il/HeQ           (CC-BY 4.0)   reading-comprehension QA
  - tau/parashoot          (CC-BY 4.0)   first Hebrew QA dataset
  - shacharbinyamin/dolly-15k-translated-to-hebrew  (CC-BY-SA 3.0)
  - dicta-il/OpenHermes-2.5-Hebrew  (Apache 2.0)   multi-turn chats
  - HebAI/alpaca-he         (CC-BY-NC-4.0)  -- only with --include-nc

Also generates ~200 hand-templated Hebrew identity rows so the SFT model
has a coherent persona without bleeding English idiom from translation.

Outputs (under $NANOCHAT_BASE_DIR):
    sft_data_he.train.jsonl
    sft_data_he.val.jsonl       (5% held-out)
    identity_conversations_he.jsonl

Run:
    python -m scripts_he.prepare_sft_data_he             # commercial-safe
    python -m scripts_he.prepare_sft_data_he --include-nc  # adds Alpaca-HE
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

from nanochat.common import get_base_dir
from nanochat_he.text_utils import strip_nikkud

SYSTEM_PROMPT_HE = "אתה עוזר מועיל, אמין וידידותי שמדבר עברית."


def _emit(messages: list[dict]) -> list[dict] | None:
    """
    Normalize to bare-list format and validate.
    customjson.py requires alternating user/assistant starting at index 0.
    System prompts get folded into the first user content.
    Returns the validated list or None if invalid.
    """
    cleaned: list[dict] = []
    system_text = None
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        content = strip_nikkud(content.strip())
        if not content:
            continue
        if role == "system":
            system_text = content
            continue
        if role in ("human", "user"):
            cleaned.append({"role": "user", "content": content})
        elif role in ("gpt", "assistant", "bot"):
            cleaned.append({"role": "assistant", "content": content})

    if len(cleaned) < 2:
        return None

    if cleaned[0]["role"] != "user":
        return None

    if system_text:
        cleaned[0]["content"] = f"{system_text}\n\n{cleaned[0]['content']}"

    for i, m in enumerate(cleaned):
        expected = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected:
            return None

    if cleaned[-1]["role"] != "assistant":
        cleaned = cleaned[:-1]
        if len(cleaned) < 2:
            return None

    return cleaned


def load_heq():
    from datasets import load_dataset
    try:
        ds = load_dataset("dicta-il/HeQ", split="train")
    except Exception as e:
        print(f"[warn] HeQ load failed: {e}")
        return
    for row in ds:
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", {}) or {}
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        if not answer_texts:
            continue
        answer = answer_texts[0]
        user_msg = f"קרא את הקטע הבא וענה על השאלה.\n\nקטע:\n{context}\n\nשאלה: {question}"
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ])
        if result:
            yield result


def load_parashoot():
    from datasets import load_dataset
    try:
        ds = load_dataset("tau/parashoot", split="train")
    except Exception as e:
        print(f"[warn] ParaShoot load failed: {e}")
        return
    for row in ds:
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", {}) or {}
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []
        if not answer_texts:
            continue
        answer = answer_texts[0]
        user_msg = f"על פי הקטע הבא, ענה על השאלה.\n\nקטע:\n{context}\n\nשאלה: {question}"
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ])
        if result:
            yield result


def load_dolly_he():
    from datasets import load_dataset
    try:
        ds = load_dataset("shacharbinyamin/dolly-15k-translated-to-hebrew", split="train")
    except Exception as e:
        print(f"[warn] Dolly-HE load failed: {e}")
        return
    for row in ds:
        instruction = row.get("instruction") or row.get("Instruction") or ""
        context = row.get("context") or row.get("Context") or ""
        response = row.get("response") or row.get("Response") or ""
        if not instruction or not response:
            continue
        user_content = instruction if not context else f"{instruction}\n\nהקשר:\n{context}"
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ])
        if result:
            yield result


def load_openhermes_he():
    from datasets import load_dataset
    try:
        ds = load_dataset("dicta-il/OpenHermes-2.5-Hebrew", split="train")
    except Exception as e:
        print(f"[warn] OpenHermes-2.5-Hebrew load failed: {e}")
        return
    for row in ds:
        conversations = row.get("conversations") or row.get("messages") or []
        messages = [{"role": "system", "content": SYSTEM_PROMPT_HE}]
        for turn in conversations:
            role = turn.get("from") or turn.get("role") or ""
            content = turn.get("value") or turn.get("content") or ""
            messages.append({"role": role, "content": content})
        result = _emit(messages)
        if result:
            yield result


def load_alpaca_he():
    from datasets import load_dataset
    try:
        ds = load_dataset("HebAI/alpaca-he", split="train")
    except Exception as e:
        print(f"[warn] Alpaca-HE load failed: {e}")
        return
    for row in ds:
        instruction = row.get("instruction", "")
        input_text = row.get("input", "")
        output = row.get("output", "")
        if not instruction or not output:
            continue
        user_content = f"{instruction}\n{input_text}".strip()
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ])
        if result:
            yield result


IDENTITY_NAME = "ננו-צ׳אט"
IDENTITY_CREATOR = "פרויקט nanochat המקורי של אנדריי קרפתי, מותאם לעברית"

_NAME_PROMPTS = [
    "מה שמך?",
    "איך קוראים לך?",
    "אתה יכול להציג את עצמך?",
    "מי אתה?",
    "מה השם שלך?",
    "תציג את עצמך בבקשה.",
    "האם יש לך שם?",
    "איך אקרא לך?",
]
_CREATOR_PROMPTS = [
    "מי יצר אותך?",
    "מי בנה אותך?",
    "מי פיתח אותך?",
    "באיזו חברה אתה?",
    "מי עומד מאחוריך?",
    "איפה נוצרת?",
]
_CAPABILITY_PROMPTS = [
    "מה אתה יכול לעשות?",
    "במה אתה מתמחה?",
    "באילו תחומים אתה יכול לעזור?",
    "מה היכולות שלך?",
    "במה תוכל לעזור לי?",
    "אתה יודע עברית?",
]
_LANG_PROMPTS = [
    "באיזו שפה אתה מדבר?",
    "אתה מבין עברית?",
    "אתה דובר עברית?",
    "אפשר לדבר איתך בעברית?",
]


def gen_identity_rows() -> list[list[dict]]:
    rows: list[list[dict]] = []
    name_answers = [
        f"שמי {IDENTITY_NAME}, אני מודל שפה שמדבר עברית ונועד לעזור לך.",
        f"קוראים לי {IDENTITY_NAME}. אני כאן לעזור לך בעברית.",
        f"אני {IDENTITY_NAME}, עוזר שיחה ידידותי בעברית.",
        f"שמי {IDENTITY_NAME} — נעים מאוד.",
    ]
    creator_answers = [
        f"אני מבוסס על {IDENTITY_CREATOR}.",
        f"נבניתי כחלק מ{IDENTITY_CREATOR}.",
        f"מאחוריי עומד {IDENTITY_CREATOR}.",
    ]
    capability_answers = [
        "אני יכול לענות על שאלות, להסביר נושאים, לעזור בכתיבה, לתרגם ולנהל שיחה כללית בעברית.",
        "אני מתמחה בשיחה בעברית: שאלות ידע כללי, סיכומים, הסברים והדרכה.",
        "אפשר לבקש ממני להסביר רעיון, לכתוב טקסט, לענות על שאלה או פשוט לשוחח.",
    ]
    lang_answers = [
        "כן, אני מדבר עברית באופן שוטף.",
        "השפה העיקרית שלי היא עברית.",
        "כמובן, נשמח לדבר בעברית.",
    ]

    rng = random.Random(0)
    for prompt in _NAME_PROMPTS:
        for ans in name_answers:
            rows.append([{"role": "user", "content": prompt},
                         {"role": "assistant", "content": ans}])
    for prompt in _CREATOR_PROMPTS:
        for ans in creator_answers:
            rows.append([{"role": "user", "content": prompt},
                         {"role": "assistant", "content": ans}])
    for prompt in _CAPABILITY_PROMPTS:
        for ans in capability_answers:
            rows.append([{"role": "user", "content": prompt},
                         {"role": "assistant", "content": ans}])
    for prompt in _LANG_PROMPTS:
        for ans in lang_answers:
            rows.append([{"role": "user", "content": prompt},
                         {"role": "assistant", "content": ans}])

    rng.shuffle(rows)
    return rows


def write_jsonl(rows: list[list[dict]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(include_nc: bool, val_frac: float, seed: int):
    base_dir = Path(get_base_dir())
    out_train = base_dir / "sft_data_he.train.jsonl"
    out_val = base_dir / "sft_data_he.val.jsonl"
    out_identity = base_dir / "identity_conversations_he.jsonl"

    print(f"Output dir: {base_dir}")
    rng = random.Random(seed)
    records: list[list[dict]] = []

    print("Loading HeQ ...")
    records.extend(load_heq())
    print(f"  cumulative: {len(records):,}")

    print("Loading ParaShoot ...")
    records.extend(load_parashoot())
    print(f"  cumulative: {len(records):,}")

    print("Loading Dolly-HE ...")
    records.extend(load_dolly_he())
    print(f"  cumulative: {len(records):,}")

    print("Loading OpenHermes-2.5-HE ...")
    records.extend(load_openhermes_he())
    print(f"  cumulative: {len(records):,}")

    if include_nc:
        print("Loading Alpaca-HE (NC license) ...")
        records.extend(load_alpaca_he())
        print(f"  cumulative: {len(records):,}")
    else:
        print("Skipping Alpaca-HE (pass --include-nc to include).")

    if not records:
        sys.exit("ERROR: no Hebrew SFT examples loaded. Check HF connectivity / credentials.")

    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_frac))
    val_records = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, out_train)
    write_jsonl(val_records, out_val)

    identity_rows = gen_identity_rows()
    write_jsonl(identity_rows, out_identity)

    print()
    print(f"Wrote {len(train_records):,} train rows -> {out_train}")
    print(f"Wrote {len(val_records):,} val   rows -> {out_val}")
    print(f"Wrote {len(identity_rows):,} identity rows -> {out_identity}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Hebrew-only SFT data")
    parser.add_argument("--include-nc", action="store_true",
                        help="Include non-commercial datasets (Alpaca-HE)")
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction of data to hold out for validation (default 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(include_nc=args.include_nc, val_frac=args.val_frac, seed=args.seed)
