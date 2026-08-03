"""
Answer extraction and scoring.

This is deliberately separate from the model calls. The parser is versioned
because PREREGISTRATION.md requires it to be frozen after dev-split work, and
because a parser tuned on map or router responses would invalidate those
splits. Every stored row carries the parser_version that produced it.

PARSER_VERSION history:
  v0.1  Day 1 smoke parser. Handles the observed `answer: C` and JSON
        `{"answer": "C"}` forms plus a bare-letter fallback. This will be
        re-frozen after the dev split; do not treat v0.1 as final.
"""

from __future__ import annotations

import re

PARSER_VERSION = "v0.1"

# Matches: answer: C | "answer": "C" | answer:C | Answer : c
_ANSWER_FIELD = re.compile(r'answer\s*["\']?\s*:\s*["\']?\s*([A-D])\b', re.IGNORECASE)
# Fallback: a lone A-D standing on its own (e.g. the model just prints "C")
_BARE_LETTER = re.compile(r'\b([A-D])\b')

# Math: prefer a \boxed{...} value, else fall back to the last number in the text.
_BOXED = re.compile(r'\\boxed\{([^}]*)\}')
_NUMBER = re.compile(r'-?\$?\s*[\d,]*\.?\d+')


class ParseStatus:
    OK = "ok"
    UNPARSEABLE = "unparseable"
    EMPTY = "empty"


def extract_mcq_answer(text: str) -> tuple[str | None, str]:
    """
    Return (letter, parse_status). letter is one of A-D or None.

    Preference order: the explicit answer field first, then a bare letter.
    A response that yields no letter is UNPARSEABLE, which the pre-registration
    scores as incorrect (never dropped).
    """
    if not isinstance(text, str) or not text.strip():
        return None, ParseStatus.EMPTY

    m = _ANSWER_FIELD.search(text)
    if m:
        return m.group(1).upper(), ParseStatus.OK

    m = _BARE_LETTER.search(text)
    if m:
        return m.group(1).upper(), ParseStatus.OK

    return None, ParseStatus.UNPARSEABLE


def score_mcq(raw_response: str, gold_letter: str) -> dict:
    """
    Score one multiple-choice response against the gold letter.

    Returns a dict with the normalized answer, parse status, and correctness.
    Unparseable or empty responses are scored incorrect, per the pre-registration.
    """
    gold = gold_letter.strip().upper()
    predicted, status = extract_mcq_answer(raw_response)
    correct = status == ParseStatus.OK and predicted == gold
    return {
        "normalized_answer": predicted,
        "parse_status": status,
        "parser_version": PARSER_VERSION,
        "gold": gold,
        "correct": bool(correct),
    }


def _normalize_number(s: str) -> str | None:
    """Strip commas, currency, and whitespace; drop a trailing .0. Returns None if empty."""
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    if not s:
        return None
    # normalize 21.0 -> 21, keep genuine decimals like 3.5
    if s.endswith(".0"):
        s = s[:-2]
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def extract_math_answer(text: str) -> tuple[str | None, str]:
    """
    Return (normalized_number, parse_status). Prefers \\boxed{...}, then the last
    number in the response. Unparseable if no number is found.
    """
    if not isinstance(text, str) or not text.strip():
        return None, ParseStatus.EMPTY

    m = _BOXED.search(text)
    if m:
        norm = _normalize_number(m.group(1))
        if norm is not None:
            return norm, ParseStatus.OK

    numbers = _NUMBER.findall(text)
    if numbers:
        norm = _normalize_number(numbers[-1])
        if norm is not None:
            return norm, ParseStatus.OK

    return None, ParseStatus.UNPARSEABLE


def score_math(raw_response: str, gold_number: str) -> dict:
    """Score one GSM8K response against the gold numeric answer."""
    gold = _normalize_number(gold_number)
    predicted, status = extract_math_answer(raw_response)
    correct = status == ParseStatus.OK and predicted is not None and predicted == gold
    return {
        "normalized_answer": predicted,
        "parse_status": status,
        "parser_version": PARSER_VERSION,
        "gold": gold,
        "correct": bool(correct),
    }
