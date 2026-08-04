"""
Build the map from the results database.

Reads every paired result (items where BOTH models produced an error-free
answer), computes per-subject accuracy, the accuracy gap with a confidence
interval, and the non-inferiority verdict, then writes:

    reports/map.md    human-readable table
    reports/map.json  machine-readable results

Also reports the disagreement (discordance) rate from the dev split and a
cost/latency summary. Read-only on the database; safe to run while collection
continues, though the map is only final once every split is complete.

Usage
-----
    python scripts/build_map.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import analysis, config  # noqa: E402
from goodenough import datasets_config as dc  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MARGIN = 0.10  # PREREGISTRATION.md section 3


def subject_of(item_id: str) -> str:
    parts = item_id.split("/")
    if parts[0] == "mmlu":
        return parts[1]
    return parts[0]  # gsm8k


def load_paired(conn, dataset: str, split: str):
    """
    Return {subject: {"local": [...], "hosted": [...]}} for items where both
    models have an error-free row, aligned by item_id.
    """
    rows = conn.execute(
        "SELECT item_id, model_role, correct FROM results "
        "WHERE dataset=? AND split=? AND error IS NULL",
        (dataset, split),
    ).fetchall()

    by_item: dict[str, dict[str, int]] = {}
    for item_id, role, correct in rows:
        if correct is None:
            continue
        by_item.setdefault(item_id, {})[role] = int(correct)

    out: dict[str, dict[str, list]] = {}
    for item_id, roles in by_item.items():
        if "local" in roles and "hosted" in roles:
            subj = subject_of(item_id)
            slot = out.setdefault(subj, {"local": [], "hosted": []})
            slot["local"].append(roles["local"])
            slot["hosted"].append(roles["hosted"])
    return out


def cost_latency(conn, dataset: str, split: str):
    """Median/mean latency per model and total hosted tokens over a split."""
    summary = {}
    for role in ("local", "hosted"):
        lat = [r[0] for r in conn.execute(
            "SELECT latency_ms_uncached FROM results WHERE dataset=? AND split=? "
            "AND model_role=? AND error IS NULL AND latency_ms_uncached IS NOT NULL",
            (dataset, split, role)).fetchall()]
        toks = conn.execute(
            "SELECT COALESCE(SUM(COALESCE(input_tokens,0)+COALESCE(output_tokens,0)),0) "
            "FROM results WHERE dataset=? AND split=? AND model_role=? AND error IS NULL",
            (dataset, split, role)).fetchone()[0]
        summary[role] = {
            "n": len(lat),
            "latency_ms_median": (statistics.median(lat) if lat else None),
            "latency_ms_mean": (statistics.mean(lat) if lat else None),
            "total_tokens": int(toks),
        }
    return summary


def build():
    if not os.path.exists(DB_PATH):
        print(f"No database at {DB_PATH}. Run scripts/run_eval.py first.")
        return None
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    # Disagreement from dev
    dev = load_paired(conn, "mmlu", "dev")
    dev_local = [v for s in dev.values() for v in s["local"]]
    dev_hosted = [v for s in dev.values() for v in s["hosted"]]
    dev_disc = analysis.discordance(dev_local, dev_hosted) if dev_local else None

    # The map
    mp = load_paired(conn, "mmlu", "map")
    slices = []
    for subj in dc.MMLU_SUBJECTS:
        s = mp.get(subj)
        if not s or not s["local"]:
            slices.append({"subject": subj, "primary": subj in dc.MMLU_PRIMARY,
                           "n": 0, "verdict": "no_data"})
            continue
        res = analysis.slice_result(s["local"], s["hosted"], MARGIN)
        res["subject"] = subj
        res["primary"] = subj in dc.MMLU_PRIMARY
        slices.append(res)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "margin": MARGIN,
        "local_model": config.LOCAL_MODEL_ID,
        "hosted_model": config.HOSTED_MODEL_ID,
        "dev_discordance": dev_disc,
        "map_slices": slices,
        "map_cost_latency": cost_latency(conn, "mmlu", "map"),
    }
    conn.close()
    return report


def write_reports(report: dict):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "map.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    lines = []
    lines.append("# GoodEnough map\n")
    lines.append(f"Local: `{report['local_model']}`  Hosted: `{report['hosted_model']}`  "
                 f"Margin: {report['margin']:.2f}\n")
    dd = report["dev_discordance"]
    if dd:
        lines.append(f"Dev-set disagreement rate: {dd['rate']:.3f} "
                     f"({dd['discordant']}/{dd['n']} items). This drives how many "
                     f"items are needed for a confident verdict.\n")

    lines.append("\n## MMLU non-inferiority map\n")
    lines.append("delta = local accuracy minus hosted accuracy. "
                 "CI is the one-sided 95% bound. Verdict is at the "
                 f"{report['margin']:.2f} margin.\n")
    lines.append("\n| subject | | n | local | hosted | delta | 95% CI | verdict |")
    lines.append("|---|---|---:|---:|---:|---:|:---:|---|")
    for s in report["map_slices"]:
        star = "P" if s.get("primary") else ""
        if s.get("verdict") == "no_data":
            lines.append(f"| {s['subject']} | {star} | 0 | | | | | no data yet |")
            continue
        lines.append(
            f"| {s['subject']} | {star} | {s['n']} | {s['acc_local']:.2f} | "
            f"{s['acc_hosted']:.2f} | {s['delta']:+.3f} | "
            f"[{s['ci_lower']:+.3f}, {s['ci_upper']:+.3f}] | **{s['verdict']}** |")
    lines.append("\nP marks the two primary slices named before data collection.\n")

    cl = report["map_cost_latency"]
    lines.append("\n## Cost and latency (MMLU map)\n")
    lines.append("| model | n | median latency (ms) | mean latency (ms) | total tokens |")
    lines.append("|---|---:|---:|---:|---:|")
    for role in ("local", "hosted"):
        c = cl[role]
        med = f"{c['latency_ms_median']:.0f}" if c["latency_ms_median"] else "-"
        mean = f"{c['latency_ms_mean']:.0f}" if c["latency_ms_mean"] else "-"
        lines.append(f"| {role} | {c['n']} | {med} | {mean} | {c['total_tokens']:,} |")
    lines.append("\nLocal token cost is zero incremental API spend; the tokens column "
                 "for local reflects local compute only, not money.\n")

    with open(os.path.join(REPORTS_DIR, "map.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    report = build()
    if report is None:
        return 1
    write_reports(report)

    # Console summary
    print("Dev disagreement:",
          f"{report['dev_discordance']['rate']:.3f}" if report["dev_discordance"] else "n/a")
    print("\nMMLU map:")
    for s in report["map_slices"]:
        if s.get("verdict") == "no_data":
            print(f"  {s['subject']:26} no data yet")
            continue
        star = "*" if s.get("primary") else " "
        print(f" {star}{s['subject']:26} n={s['n']:<4} "
              f"local={s['acc_local']:.2f} hosted={s['acc_hosted']:.2f} "
              f"delta={s['delta']:+.3f} CI=[{s['ci_lower']:+.3f},{s['ci_upper']:+.3f}] "
              f"-> {s['verdict']}")
    print("\nWrote reports/map.md and reports/map.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
