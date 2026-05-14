"""
Hebrew text normalization helpers.

Nikkud and cantillation marks (te'amim) are Unicode combining marks in
the range U+0591 .. U+05C7. They are mostly absent from modern Hebrew
text (used only in liturgical / academic contexts) and inflate the
tokenizer's effective vocabulary without adding signal for chat use.
We strip them entirely before tokenization.

Range U+0591 .. U+05C7 covers:
    U+0591..U+05BD  Hebrew accents (te'amim) and cantillation marks
    U+05BF          Hebrew point Rafe
    U+05C1..U+05C2  Shin dot, Sin dot
    U+05C4..U+05C5  Hebrew marks (upper/lower dot)
    U+05C7          Hebrew point Qamats Qatan

Hebrew consonants U+05D0..U+05EA and punctuation (Maqaf U+05BE,
Geresh U+05F3, Gershayim U+05F4) are preserved.
"""

import re
import unicodedata

_NIKKUD_RE = re.compile("[֑-ׇ]")


def strip_nikkud(text: str) -> str:
    """Remove all Hebrew diacritical marks from `text`.

    Applies NFC normalization first so that pre-composed forms (rare for
    Hebrew but possible after datasets concat) decompose predictably.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    return _NIKKUD_RE.sub("", text)
