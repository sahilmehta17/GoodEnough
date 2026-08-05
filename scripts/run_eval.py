"""
The evaluation runner. Sends frozen items to both models, scores them, and
writes one row per (item, model) to SQLite.

Design points that matter:

- Resumable. A finished item is one that already has a non-error row in the DB.
  Re-running skips finished work, so a run interrupted at any point continues
  cleanly. This is how a multi-day run survives daily rate caps and laptop sleep.

- Budget-aware. The hosted (Groq) side has a ~100k token-per-day free cap. The
  runner tracks tokens already spent TODAY (from the DB) and stops making hosted
  calls before it blows the cap, telling you to resume tomorrow. Local calls are
  free and unbounded.

- Never drops an item. A model error is written as an error row (scored as
  incorrect per the pre-registration) and retried on the next run.

Usage
-----
    # smoke: first 3 items, both models
    python scripts/run_eval.py --dataset mmlu --split map --limit 3

    # full local pass (free, ~40 min for all 800 map items)
    python scripts/run_eval.py --dataset mmlu --split map --models local

    # hosted pass until the daily budget is hit, then resume tomorrow
    python scripts/run_eval.py --dataset mmlu --split map --models hosted
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import config, clients, loader, scoring, store  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")

# Safety margin under Groq's 100k TPD so we never get daily-cap 429 storms.
HOSTED_DAILY_TOKEN_BUDGET = 90_000


def _score(item: loader.Item, raw: str) -> dict:
    if item.task_type == loader.TASK_MCQ:
        return scoring.score_mcq(raw, item.gold)
    return scoring.score_math(raw, item.gold)


def _already_done(conn, item: loader.Item, role: str) -> bool:
    """A finished item has a row with no error for this (item, role)."""
    row = conn.execute(
        "SELECT 1 FROM results WHERE dataset=? AND split=? AND item_id=? "
        "AND model_role=? AND error IS NULL LIMIT 1",
        (item.dataset, item.split, item.item_id, role),
    ).fetchone()
    return row is not None


def _hosted_tokens_today(conn) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)),0) "
        "FROM results WHERE model_role='hosted' AND substr(run_date,1,10)=?",
        (today,),
    ).fetchone()
    return int(row[0])


def _call(role: str, item: loader.Item) -> clients.CallResult:
    if role == "local":
        return clients.call_local(item.prompt, item.max_tokens)
    return clients.call_hosted(item.prompt, item.max_tokens)


def _row(call: clients.CallResult, item: loader.Item) -> store.ResultRow:
    if call.error:
        scored = {"normalized_answer": None, "parse_status": None,
                  "parser_version": scoring.PARSER_VERSION, "correct": None}
    else:
        scored = _score(item, call.raw_response)
    return store.ResultRow(
        dataset=item.dataset, split=item.split, item_id=item.item_id,
        model_role=call.model_role, model_id_requested=call.model_id_requested,
        model_id_returned=call.model_id_returned, semantic_prompt=item.prompt,
        rendered_input=call.rendered_input, raw_response=call.raw_response,
        normalized_answer=scored["normalized_answer"], parser_version=scored["parser_version"],
        parse_status=scored["parse_status"],
        correct=None if call.error else scored["correct"],
        error=call.error, retries=call.retries,
        input_tokens=call.input_tokens, output_tokens=call.output_tokens,
        latency_ms_uncached=call.latency_ms, cache_hit=False,
        finish_reason=call.finish_reason,
        run_date=datetime.now(timezone.utc).isoformat(), seed=config.SEED,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["mmlu", "gsm8k"])
    ap.add_argument("--split", required=True, choices=["dev", "map", "router"])
    ap.add_argument("--models", default="local,hosted",
                    help="comma list: local, hosted, or both")
    ap.add_argument("--limit", type=int, default=0, help="only first N items (0 = all)")
    ap.add_argument("--hosted-budget", type=int, default=HOSTED_DAILY_TOKEN_BUDGET)
    args = ap.parse_args()

    config.load_dotenv()
    roles = [r.strip() for r in args.models.split(",") if r.strip()]

    items = loader.load_frozen(args.dataset, args.split)
    if args.limit:
        items = items[: args.limit]

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = store.connect(DB_PATH)

    hosted_today = _hosted_tokens_today(conn)
    print(f"{args.dataset}/{args.split}: {len(items)} items, models={roles}")
    if "hosted" in roles:
        print(f"Hosted tokens already spent today: {hosted_today:,} / {args.hosted_budget:,}")

    counts = {"local": {"done": 0, "skip": 0, "err": 0},
              "hosted": {"done": 0, "skip": 0, "err": 0, "budget_stop": 0}}

    last_done = -1
    rate_stop = False
    for n, item in enumerate(items, 1):
        for role in roles:
            if _already_done(conn, item, role):
                counts[role]["skip"] += 1
                continue
            if role == "hosted" and hosted_today >= args.hosted_budget:
                counts["hosted"]["budget_stop"] += 1
                continue

            call = _call(role, item)

            # A long Groq cooldown: stop cleanly, do not record the item so it is
            # retried next run, do not sleep through the cooldown.
            if role == "hosted" and call.error and call.error.startswith("RATE_LIMIT_STOP"):
                rate_stop = True
                print(f"  Groq issued a long cooldown at item {n} ({call.error}). "
                      f"Stopping cleanly; re-run later to resume.")
                break

            store.insert_row(conn, _row(call, item))

            if call.error:
                counts[role]["err"] += 1
            else:
                counts[role]["done"] += 1
                if role == "hosted":
                    hosted_today += (call.input_tokens or 0) + (call.output_tokens or 0)

        if rate_stop:
            counts["hosted"]["budget_stop"] += (len(items) - n + 1)
            break

        # Only print when work was actually done since the last line, so a
        # budget stop does not spam identical lines that look like a hang.
        total_done = sum(counts[r]["done"] for r in roles)
        if (n % 25 == 0 or n == len(items)) and total_done != last_done:
            last_done = total_done
            msg = f"  [{n}/{len(items)}] "
            for role in roles:
                c = counts[role]
                msg += f"{role}: done={c['done']} skip={c['skip']} err={c['err']} "
            if "hosted" in roles:
                msg += f"| hosted_tokens_today={hosted_today:,}"
            print(msg)

        # If the hosted budget is spent and there is no free local work left in
        # this run, stop early instead of iterating no-op skips to the end.
        if "local" not in roles and hosted_today >= args.hosted_budget:
            counts["hosted"]["budget_stop"] += (len(items) - n)
            print(f"  hosted budget reached at item {n}; {len(items) - n} items "
                  f"remain for tomorrow.")
            break

    print("\nDone this run.")
    for role in roles:
        c = counts[role]
        line = f"  {role}: {c['done']} new, {c['skip']} already done, {c['err']} errored"
        if role == "hosted" and c["budget_stop"]:
            line += f", {c['budget_stop']} skipped (daily budget). Resume tomorrow."
        print(line)

    if "hosted" in roles and counts["hosted"]["budget_stop"]:
        print("\nHosted daily budget reached. Re-run the same command tomorrow to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
