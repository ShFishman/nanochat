"""
Builds Hebrew-only SFT JSONLs in the bare-list format required by
tasks/customjson.py (each line is a JSON list of {role, content},
alternating user/assistant starting with user).

Hebrew SFT sources (all verified live, public, no gating):
  - Etelis/HeQ_v1                          (CC-BY 4.0)  ~30k SQuAD-style QA
  - imvladikon/parashoot                   (CC-BY 4.0)  ~3k SQuAD-style QA
  - yuvalav/hebrew-qa                      (CC-BY 4.0)  ~30k alpaca-style QA
  - CohereLabs/aya_collection_language_split (Apache 2.0) Hebrew instruction data

Also generates ~200 hand-templated Hebrew identity rows so the SFT model
has a coherent persona without bleeding English idiom from translation.

Outputs (under $NANOCHAT_BASE_DIR):
    sft_data_he.train.jsonl
    sft_data_he.val.jsonl       (5% held-out)
    identity_conversations_he.jsonl

Run:
    python -m scripts_he.prepare_sft_data_he
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


def _parse_answers_field(answers):
    """HeQ_v1 / parashoot answers may be dict or JSON string. Return first answer text."""
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except Exception:
            return None
    if isinstance(answers, dict):
        texts = answers.get("text") or []
        if texts and isinstance(texts, list):
            return texts[0]
    return None


def load_heq():
    """Etelis/HeQ_v1: SQuAD-style Hebrew QA. Note: column names are Capitalized."""
    from datasets import load_dataset
    try:
        ds = load_dataset("Etelis/HeQ_v1", split="train")
    except Exception as e:
        print(f"[warn] HeQ_v1 load failed: {e}")
        return
    for row in ds:
        context = row.get("Context") or row.get("context") or ""
        question = row.get("Question") or row.get("question") or ""
        answer = _parse_answers_field(row.get("Answers") or row.get("answers"))
        if not answer or row.get("Is_Impossible"):
            continue
        user_msg = f"קרא את הקטע הבא וענה על השאלה.\n\nקטע:\n{context}\n\nשאלה: {question}"
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ])
        if result:
            yield result


def load_parashoot():
    """imvladikon/parashoot: SQuAD-style Hebrew QA. ~1.8k train rows."""
    from datasets import load_dataset
    try:
        ds = load_dataset("imvladikon/parashoot", split="train")
    except Exception as e:
        print(f"[warn] ParaShoot load failed: {e}")
        return
    for row in ds:
        context = row.get("context", "")
        question = row.get("question", "")
        answer = _parse_answers_field(row.get("answers"))
        if not answer:
            continue
        user_msg = f"על פי הקטע הבא, ענה על השאלה.\n\nקטע:\n{context}\n\nשאלה: {question}"
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": answer},
        ])
        if result:
            yield result


def load_hebrew_qa():
    """yuvalav/hebrew-qa: ~30k Alpaca-style Hebrew QA pairs (instruction/input/output)."""
    from datasets import load_dataset
    try:
        ds = load_dataset("yuvalav/hebrew-qa", split="train")
    except Exception as e:
        print(f"[warn] yuvalav/hebrew-qa load failed: {e}")
        return
    for row in ds:
        instruction = (row.get("instruction") or "").strip()
        input_text = (row.get("input") or "").strip()
        output = (row.get("output") or "").strip()
        if not instruction or not output:
            continue
        user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ])
        if result:
            yield result


def load_aya_hebrew():
    """CohereLabs/aya_collection_language_split, config='hebrew'. inputs/targets columns."""
    from datasets import load_dataset
    try:
        ds = load_dataset("CohereLabs/aya_collection_language_split", "hebrew", split="train")
    except Exception as e:
        # Fall back to test split if train missing for this config
        print(f"[warn] aya hebrew train split failed: {e}; trying test split")
        try:
            ds = load_dataset("CohereLabs/aya_collection_language_split", "hebrew", split="test")
        except Exception as e2:
            print(f"[warn] aya hebrew load failed entirely: {e2}")
            return
    for row in ds:
        inputs = (row.get("inputs") or "").strip()
        targets = (row.get("targets") or "").strip()
        if not inputs or not targets:
            continue
        result = _emit([
            {"role": "system", "content": SYSTEM_PROMPT_HE},
            {"role": "user", "content": inputs},
            {"role": "assistant", "content": targets},
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


def prepare(val_frac: float, seed: int):
    base_dir = Path(get_base_dir())
    out_train = base_dir / "sft_data_he.train.jsonl"
    out_val = base_dir / "sft_data_he.val.jsonl"
    out_identity = base_dir / "identity_conversations_he.jsonl"

    print(f"Output dir: {base_dir}")
    rng = random.Random(seed)
    records: list[list[dict]] = []

    print("Loading Etelis/HeQ_v1 ...")
    records.extend(load_heq())
    print(f"  cumulative: {len(records):,}")

    print("Loading imvladikon/parashoot ...")
    records.extend(load_parashoot())
    print(f"  cumulative: {len(records):,}")

    print("Loading yuvalav/hebrew-qa ...")
    records.extend(load_hebrew_qa())
    print(f"  cumulative: {len(records):,}")

    print("Loading CohereLabs/aya_collection_language_split (hebrew) ...")
    records.extend(load_aya_hebrew())
    print(f"  cumulative: {len(records):,}")

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
    parser.add_argument("--val-frac", type=float, default=0.05,
                        help="Fraction of data to hold out for validation (default 0.05)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(val_frac=args.val_frac, seed=args.seed)
