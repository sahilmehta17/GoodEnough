"""
Day 1 smoke test: one paired item through the complete measurement path.

Purpose (from CLAUDE.md build order): prove that a single item can go local ->
hosted -> stored -> scored -> one paired row, and that nothing in the plumbing
is broken, BEFORE building the full dataset runner. This is the Tuesday tripwire.

The item below is a FIXTURE, not a benchmark item. It exists only to exercise
the path. It is not part of any frozen split and must never enter the map or
router analysis. Real items come from the MMLU loader in the Day 2 runner.

Run this on the machine where llama-server is live and GROQ_API_KEY is set:

    python scripts/day1_smoke.py

Success looks like: two rows written, both parsed, a printed paired summary,
and finish_reason "stop" on the local call (proving thinking mode is off).
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

# Make src/ importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import config, clients, scoring, store  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "day1_smoke.sqlite")

# A deliberately easy, unambiguous fixture. Gold answer is B.
FIXTURE = {
    "item_id": "SMOKE-0001",
    "question": "Into which body of water does the Nile River ultimately flow?",
    "options": {"A": "Red Sea", "B": "Mediterranean Sea",
                "C": "Persian Gulf", "D": "Arabian Sea"},
    "gold": "B",
}


def build_semantic_prompt(item: dict) -> str:
    """Identical for both models. No deployment controls live in here."""
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return f"{item['question']}\n{opts}\n\n{config.MCQ_INSTRUCTION}"


def to_row(call: clients.CallResult, item: dict, semantic_prompt: str) -> store.ResultRow:
    scored = scoring.score_mcq(call.raw_response, item["gold"])
    return store.ResultRow(
        dataset="fixture",
        split="smoke",
        item_id=item["item_id"],
        model_role=call.model_role,
        model_id_requested=call.model_id_requested,
        model_id_returned=call.model_id_returned,
        semantic_prompt=semantic_prompt,
        rendered_input=call.rendered_input,
        raw_response=call.raw_response,
        normalized_answer=scored["normalized_answer"],
        parser_version=scored["parser_version"],
        parse_status=scored["parse_status"],
        correct=None if call.error else scored["correct"],
        error=call.error,
        retries=call.retries,
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        latency_ms_uncached=call.latency_ms,
        cache_hit=False,
        finish_reason=call.finish_reason,
        run_date=datetime.now(timezone.utc).isoformat(),
        seed=config.SEED,
    )


def main() -> int:
    config.load_dotenv()
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    semantic_prompt = build_semantic_prompt(FIXTURE)
    print("Semantic prompt (identical to both models):")
    print("  " + semantic_prompt.replace("\n", "\n  "))
    print()

    print("Calling local model...")
    local = clients.call_local(semantic_prompt, max_tokens=config.MAX_TOKENS_MCQ)
    print("Calling hosted model...")
    hosted = clients.call_hosted(semantic_prompt, max_tokens=config.MAX_TOKENS_MCQ)

    conn = store.connect(DB_PATH)
    for call in (local, hosted):
        row = to_row(call, FIXTURE, semantic_prompt)
        store.insert_row(conn, row)

    # Paired summary
    print("\n" + "=" * 64)
    print("PAIRED RESULT")
    print("=" * 64)
    for name, call in (("LOCAL ", local), ("HOSTED", hosted)):
        if call.error:
            print(f"{name}  ERROR (http={call.http_status}): {call.error}")
            continue
        scored = scoring.score_mcq(call.raw_response, FIXTURE["gold"])
        print(f"{name}  answer={scored['normalized_answer']} "
              f"correct={scored['correct']} "
              f"parse={scored['parse_status']} "
              f"finish={call.finish_reason} "
              f"out_tok={call.output_tokens} "
              f"lat={call.latency_ms:.0f}ms "
              f"retries={call.retries}")
        print(f"         model_returned={call.model_id_returned!r} "
              f"raw={call.raw_response.strip()[:80]!r}")

    print(f"\nGold answer: {FIXTURE['gold']}")
    print(f"Rows written to: {os.path.relpath(DB_PATH)}")

    # Tripwire checks
    problems = []
    if local.error:
        problems.append("local call failed")
    elif local.finish_reason != "stop":
        problems.append(f"local finish_reason is {local.finish_reason!r}, expected 'stop' "
                        "(thinking mode may still be on)")
    if hosted.error:
        problems.append("hosted call failed")

    print("\n" + ("TRIPWIRE PASS: paired path works end to end."
                  if not problems else "TRIPWIRE ISSUES:"))
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
