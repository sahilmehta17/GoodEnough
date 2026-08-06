"""
Build the cost report.

Prices and times the four deployable routing policies (always_local,
always_hosted, map_based, cascade) on the router split, plus reports the
oracle as an accuracy ceiling with no cost or latency (it is not a real
policy). Reuses analysis.evaluate_router_policies for the policy logic and
its hosted_item_ids output; this script only attaches a dollar price and a
latency to each item the policy already decided to route.

Also reports a project-wide cost sanity check: total hosted dollars spent
across every dataset and split evaluated so far, independent of the router
split or any policy, plus the four quantities CLAUDE.md asks for (actual
cash spent, hosted list-price-equivalent, local incremental spend, local
machine occupancy).

Writes reports/cost.md and reports/cost.json. Read-only on the database, so
it is safe to run while collection continues. A router split with no paired
hosted rows yet is reported as "not yet available", never a zero.

Usage
-----
    python scripts/build_cost.py
    python scripts/build_cost.py --db path/to/other.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import analysis, config  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MARGIN = 0.10  # PREREGISTRATION.md section 3

DEPLOYABLE_POLICIES = ["always_local", "always_hosted", "map_based", "cascade"]
ALL_POLICIES = DEPLOYABLE_POLICIES + ["oracle"]


def subject_of(item_id: str) -> str:
    parts = item_id.split("/")
    return parts[1] if parts[0] == "mmlu" else parts[0]


def item_dollars(input_tokens, output_tokens, price_in_per_m, price_out_per_m) -> float:
    """Dollar cost of one call. Missing token counts price as zero, not an error."""
    it = input_tokens or 0
    ot = output_tokens or 0
    return (it * price_in_per_m + ot * price_out_per_m) / 1_000_000.0


def map_verdicts(conn: sqlite3.Connection) -> dict:
    """Recompute each subject's non-inferiority verdict from the map split."""
    rows = conn.execute(
        "SELECT item_id, model_role, correct FROM results "
        "WHERE dataset='mmlu' AND split='map' AND error IS NULL").fetchall()
    by_item: dict[str, dict[str, int]] = {}
    for item_id, role, correct in rows:
        if correct is not None:
            by_item.setdefault(item_id, {})[role] = int(correct)
    by_subject: dict[str, dict[str, list]] = {}
    for item_id, roles in by_item.items():
        if "local" in roles and "hosted" in roles:
            s = by_subject.setdefault(subject_of(item_id), {"local": [], "hosted": []})
            s["local"].append(roles["local"])
            s["hosted"].append(roles["hosted"])
    verdicts = {}
    for subj, d in by_subject.items():
        res = analysis.slice_result(d["local"], d["hosted"], MARGIN, with_bootstrap=False)
        verdicts[subj] = res["verdict"]
    return verdicts


def router_items_with_metrics(conn: sqlite3.Connection) -> tuple[list[dict], dict]:
    """
    Per router-split item, paired (both models answered, no error): the policy
    inputs evaluate_router_policies needs, plus a metrics dict keyed by
    item_id carrying local/hosted latency and hosted token counts, which the
    policy logic itself has no reason to know about.
    """
    rows = conn.execute(
        "SELECT item_id, model_role, correct, parse_status, latency_ms_uncached, "
        "input_tokens, output_tokens FROM results "
        "WHERE dataset='mmlu' AND split='router' AND error IS NULL AND cache_hit=0"
    ).fetchall()

    by_item: dict[str, dict[str, dict]] = {}
    for item_id, role, correct, parse_status, latency_ms, in_tok, out_tok in rows:
        by_item.setdefault(item_id, {})[role] = {
            "correct": correct, "parse_status": parse_status,
            "latency_ms": latency_ms, "input_tokens": in_tok, "output_tokens": out_tok,
        }

    items, metrics = [], {}
    for item_id, roles in by_item.items():
        if "local" not in roles or "hosted" not in roles:
            continue
        local, hosted = roles["local"], roles["hosted"]
        if local["correct"] is None or hosted["correct"] is None:
            continue
        items.append({
            "item_id": item_id,
            "subject": subject_of(item_id),
            "local_correct": int(local["correct"]),
            "local_parse_ok": (local["parse_status"] == "ok"),
            "hosted_correct": int(hosted["correct"]),
        })
        metrics[item_id] = {
            "local_latency_ms": local["latency_ms"],
            "hosted_latency_ms": hosted["latency_ms"],
            "hosted_input_tokens": hosted["input_tokens"],
            "hosted_output_tokens": hosted["output_tokens"],
        }
    return items, metrics


def policy_economics(policy_key: str, policy: dict, metrics: dict, all_item_ids: list) -> dict:
    """
    Dollars and wall-clock seconds for one policy result from
    evaluate_router_policies. Oracle is not deployable: no cost, no latency.

    Cascade is the one subtlety: it always runs local first, so every item
    pays local latency, and the escalated subset ALSO pays hosted latency on
    top (double-charged in time, not in dollars, since local is free).
    """
    if policy_key == "oracle":
        return {"total_usd": None, "usd_per_1k_requests": None,
                "total_seconds": None, "median_seconds_per_request": None}

    n = policy["n"]
    hosted_ids = set(policy["hosted_item_ids"])

    dollars_total = sum(
        item_dollars(metrics[iid]["hosted_input_tokens"], metrics[iid]["hosted_output_tokens"],
                     config.HOSTED_PRICE_INPUT_PER_M, config.HOSTED_PRICE_OUTPUT_PER_M)
        for iid in hosted_ids
    )

    per_item_ms = []
    for iid in all_item_ids:
        local_ms = metrics[iid]["local_latency_ms"] or 0.0
        hosted_ms = metrics[iid]["hosted_latency_ms"] or 0.0
        is_hosted = iid in hosted_ids
        if policy_key == "cascade":
            per_item_ms.append(local_ms + (hosted_ms if is_hosted else 0.0))
        else:
            per_item_ms.append(hosted_ms if is_hosted else local_ms)

    total_seconds = sum(per_item_ms) / 1000.0
    median_seconds = (statistics.median(per_item_ms) / 1000.0) if per_item_ms else None
    return {
        "total_usd": dollars_total,
        "usd_per_1k_requests": (dollars_total / n * 1000.0) if n else None,
        "total_seconds": total_seconds,
        "median_seconds_per_request": median_seconds,
    }


def project_hosted_totals(conn: sqlite3.Connection) -> dict:
    """Cost sanity check: total hosted dollars across every dataset and split."""
    n, total_in, total_out = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) "
        "FROM results WHERE model_role='hosted' AND error IS NULL AND cache_hit=0"
    ).fetchone()
    dollars = item_dollars(total_in, total_out,
                           config.HOSTED_PRICE_INPUT_PER_M, config.HOSTED_PRICE_OUTPUT_PER_M)
    return {"hosted_calls": n, "total_input_tokens": total_in, "total_output_tokens": total_out,
            "hosted_list_price_equivalent_usd": dollars}


def project_local_occupancy_seconds(conn: sqlite3.Connection) -> float | None:
    """Total wall-clock seconds the local model has occupied the machine, project-wide."""
    row = conn.execute(
        "SELECT SUM(latency_ms_uncached) FROM results "
        "WHERE model_role='local' AND error IS NULL AND cache_hit=0 "
        "AND latency_ms_uncached IS NOT NULL"
    ).fetchone()
    return (row[0] / 1000.0) if row and row[0] is not None else None


def project_local_incremental_spend(conn: sqlite3.Connection) -> float:
    """
    Priced the same way as the hosted total, through
    config.LOCAL_PRICE_INPUT_PER_M / LOCAL_PRICE_OUTPUT_PER_M rather than a
    hardcoded 0.0, so a future change to those constants is not silently
    ignored here. They are 0.0 today; the result is 0.0 by computation, not
    by assumption.
    """
    total_in, total_out = conn.execute(
        "SELECT COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) "
        "FROM results WHERE model_role='local' AND error IS NULL AND cache_hit=0"
    ).fetchone()
    return item_dollars(total_in, total_out,
                        config.LOCAL_PRICE_INPUT_PER_M, config.LOCAL_PRICE_OUTPUT_PER_M)


def pooled_latency_medians(conn: sqlite3.Connection) -> dict:
    """
    Median local and hosted latency pooled across every evaluated item so far
    (not just the router split, which is too small on its own right now).
    This is what the latency-inversion prose is computed from.
    """
    out = {}
    for role in ("local", "hosted"):
        lat = [r[0] for r in conn.execute(
            "SELECT latency_ms_uncached FROM results WHERE model_role=? "
            "AND error IS NULL AND cache_hit=0 AND latency_ms_uncached IS NOT NULL",
            (role,)).fetchall()]
        out[role] = statistics.median(lat) if lat else None
    return out


def latency_inversion(medians: dict) -> dict | None:
    local_ms, hosted_ms = medians.get("local"), medians.get("hosted")
    if not local_ms or not hosted_ms:
        return None
    if hosted_ms < local_ms:
        return {"faster_role": "hosted", "slower_role": "local",
                "multiplier": local_ms / hosted_ms}
    if local_ms < hosted_ms:
        return {"faster_role": "local", "slower_role": "hosted",
                "multiplier": hosted_ms / local_ms}
    return {"faster_role": None, "slower_role": None, "multiplier": 1.0}


def build(db_path: str) -> dict:
    if not os.path.exists(db_path):
        print(f"No database at {db_path}.")
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    verdicts = map_verdicts(conn)
    items, metrics = router_items_with_metrics(conn)
    router_n = len(items)

    router_result = None
    if router_n > 0:
        all_ids = [i["item_id"] for i in items]
        policies = analysis.evaluate_router_policies(items, verdicts)
        econ = {k: policy_economics(k, v, metrics, all_ids) for k, v in policies.items()}
        router_result = {"n": router_n, "policies": policies, "economics": econ}

    project_totals = project_hosted_totals(conn)
    project_totals["local_machine_occupancy_seconds"] = project_local_occupancy_seconds(conn)
    project_totals["actual_cash_spent_usd"] = 0.0  # Groq free plan
    project_totals["local_incremental_api_spend_usd"] = project_local_incremental_spend(conn)

    inversion = latency_inversion(pooled_latency_medians(conn))
    conn.close()

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pricing": {
            "hosted_input_per_m_usd": config.HOSTED_PRICE_INPUT_PER_M,
            "hosted_output_per_m_usd": config.HOSTED_PRICE_OUTPUT_PER_M,
            "source": "https://console.groq.com/docs/models",
            "date_read": "2026-08-06",
        },
        "project_totals": project_totals,
        "latency_inversion": inversion,
        "router_split": router_result,
    }


def write_reports(report: dict):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "cost.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    lines = ["# Cost\n"]

    pt = report["project_totals"]
    lines.append("## Project totals (every dataset and split evaluated so far)\n")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    lines.append(f"| actual cash spent | ${pt['actual_cash_spent_usd']:.2f} |")
    lines.append(f"| hosted list-price-equivalent | ${pt['hosted_list_price_equivalent_usd']:.4f} |")
    lines.append(f"| local incremental API spend | ${pt['local_incremental_api_spend_usd']:.2f} |")
    occ = pt["local_machine_occupancy_seconds"]
    occ_s = f"{occ:.1f}" if occ is not None else "not yet available"
    lines.append(f"| local machine occupancy (seconds) | {occ_s} |")
    lines.append(f"\nHosted calls counted: {pt['hosted_calls']:,} "
                 f"({pt['total_input_tokens']:,} input tokens, "
                 f"{pt['total_output_tokens']:,} output tokens). Actual cash spent is "
                 "zero because collection runs on Groq's free plan; the list-price-"
                 "equivalent is what those same tokens would cost at Groq's published "
                 "on-demand rate. Local incremental API spend is zero by construction, "
                 "not because local inference has no cost; it is not economically free, "
                 "it draws no metered API dollars.\n")

    inv = report["latency_inversion"]
    lines.append("\n## Latency inversion\n")
    if inv is None:
        lines.append("Not yet available: need at least one local and one hosted "
                     "latency observation.\n")
    elif inv["faster_role"] is None:
        lines.append("Local and hosted medians are equal in the data collected so far.\n")
    else:
        lines.append(f"Pooled across every item evaluated so far, {inv['faster_role']} "
                     f"responds faster than {inv['slower_role']} by "
                     f"{inv['multiplier']:.1f}x (median end-to-end latency).\n")

    rs = report["router_split"]
    lines.append("\n## Routing policy cost and latency (router split)\n")
    if rs is None:
        lines.append("Not yet available: no paired router-split item has both a "
                     "local and a hosted result yet.\n")
    else:
        lines.append(f"n = {rs['n']} paired router-split items.\n")
        lines.append("| policy | accuracy | total $ | $ per 1,000 requests | "
                     "total seconds | median seconds/request |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for k in ALL_POLICIES:
            if k not in rs["policies"]:
                continue
            p, e = rs["policies"][k], rs["economics"][k]
            acc = f"{p['accuracy']:.3f}"
            usd = "-" if e["total_usd"] is None else f"{e['total_usd']:.4f}"
            usd_1k = "-" if e["usd_per_1k_requests"] is None else f"{e['usd_per_1k_requests']:.2f}"
            secs = "-" if e["total_seconds"] is None else f"{e['total_seconds']:.1f}"
            med = "-" if e["median_seconds_per_request"] is None else f"{e['median_seconds_per_request']:.2f}"
            lines.append(f"| {k} | {acc} | {usd} | {usd_1k} | {secs} | {med} |")
        lines.append("\nOracle has no cost or latency: it is a perfect-knowledge upper "
                     "bound on accuracy, not a deployable policy.\n")

    with open(os.path.join(REPORTS_DIR, "cost.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH, help="path to results.sqlite")
    args = ap.parse_args()

    report = build(args.db)
    if report is None:
        return 1
    write_reports(report)

    pt = report["project_totals"]
    print(f"Hosted list-price-equivalent so far: ${pt['hosted_list_price_equivalent_usd']:.4f} "
          f"({pt['hosted_calls']} calls)")
    occ = pt["local_machine_occupancy_seconds"]
    print(f"Local machine occupancy: {occ:.1f}s" if occ is not None else
          "Local machine occupancy: not yet available")
    inv = report["latency_inversion"]
    if inv and inv["faster_role"]:
        print(f"Latency inversion: {inv['faster_role']} faster than {inv['slower_role']} "
              f"by {inv['multiplier']:.1f}x (pooled median)")
    rs = report["router_split"]
    router_msg = "not yet available" if rs is None else f"{rs['n']} paired items"
    print(f"Router split: {router_msg}")
    print("\nWrote reports/cost.md and reports/cost.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
