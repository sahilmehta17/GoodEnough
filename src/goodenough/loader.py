"""
Item loader. Reads the frozen split files and turns each row into an Item the
runner can send to both models.

The rendered prompt is identical for both deployments. The only thing that
differs at call time is the local-only enable_thinking flag, which lives in the
client, not in the prompt. This is what keeps the semantic prompt byte-identical
across models (PREREGISTRATION.md section 10).

No network, no pandas. Just reads committed JSONL.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from . import config

FROZEN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "frozen")

TASK_MCQ = "mcq"
TASK_MATH = "math"


@dataclass
class Item:
    item_id: str
    dataset: str      # "mmlu" | "gsm8k"
    subject: str      # MMLU subject, or "gsm8k"
    split: str        # "dev" | "map" | "router"
    task_type: str    # TASK_MCQ | TASK_MATH
    prompt: str       # the semantic prompt, identical to both models
    gold: str         # gold letter (MCQ) or gold number string (math)
    max_tokens: int
    gold_steps: int | None = None  # GSM8K only


def _render_mmlu(row: dict) -> str:
    lines = [row["question"]]
    for letter, choice in zip(config_letters(), row["choices"]):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append(config.MCQ_INSTRUCTION)
    return "\n".join(lines)


def config_letters():
    return ["A", "B", "C", "D"]


def _render_gsm(row: dict) -> str:
    return f"{row['question']}\n\n{config.MATH_INSTRUCTION}"


def _to_item(row: dict) -> Item:
    if row["dataset"] == "mmlu":
        return Item(
            item_id=row["item_id"],
            dataset="mmlu",
            subject=row["subject"],
            split=row["split"],
            task_type=TASK_MCQ,
            prompt=_render_mmlu(row),
            gold=row["gold_letter"],
            max_tokens=config.MAX_TOKENS_MCQ,
        )
    if row["dataset"] == "gsm8k":
        return Item(
            item_id=row["item_id"],
            dataset="gsm8k",
            subject="gsm8k",
            split=row["split"],
            task_type=TASK_MATH,
            prompt=_render_gsm(row),
            gold=row["gold_number"],
            max_tokens=config.MAX_TOKENS_MATH,
            gold_steps=row.get("gold_steps"),
        )
    raise ValueError(f"unknown dataset: {row['dataset']}")


def load_frozen(dataset: str, split: str) -> list[Item]:
    """dataset in {mmlu, gsm8k}, split in {dev, map, router}."""
    path = os.path.join(FROZEN_DIR, f"{dataset}_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/freeze_splits.py and commit data/frozen/ first."
        )
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(_to_item(json.loads(line)))
    return items
