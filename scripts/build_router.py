"""
Build the router report.

Derives the routing policy from the map (which subjects the small model was
judged non-inferior on), then evaluates five policies on the held-out router
split: always-local, always-hosted, map-based, local-first cascade, and the
oracle upper bound. Reports each policy's accuracy and how many paid hosted
calls it made.

Writes reports/router.md and reports/router.json. Read-only on the database.

Usage
-----
    python scripts/build_router.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import analysis  # noqa: E402
from goodenough import datasets_config as dc  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MARGIN = 0.10


def subject_of(item_id: str) -> str:
    parts = item_id.split("/")
    return parts[1] if parts[0] == "mmlu" else parts[0]


def map_verdicts(conn) -> dict:
    """Recompute each subject's non-inferiority verdict from the map split."""
    rows = conn.execute(
        "SELECT item_id, model_role, correct FROM results "
        "WHERE dataset='mmlu' AND split='map' AND error IS NULL").fetchall()
    by_item = {}
    for item_id, role, correct in rows:
        if correct is not None:
            by_item.setdefault(item_id, {})[role] = int(correct)
    by_subject = {}
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


def router_items(conn) -> list[dict]:
    """Per router-split item: local correctness + parse status, hosted correctness."""
    rows = conn.execute(
        "SELECT item_id, model_role, correct, parse_status FROM results "
        "WHERE dataset='mmlu' AND split='router' AND error IS NULL").fetchall()
    by_item = {}
    for item_id, role, correct, parse_status in rows:
        by_item.setdefault(item_id, {})[role] = (correct, parse_status)
    items = []
    for item_id, roles in by_item.items():
        if "local" in roles and "hosted" in roles:
            lc, lp = roles["local"]
            hc, _ = roles["hosted"]
            if lc is None or hc is None:
                continue
            items.append({
                "subject": subject_of(item_id),
                "local_correct": int(lc),
                "local_parse_ok": (lp == "ok"),
                "hosted_correct": int(hc),
            })
    return items


def build():
    if not os.path.exists(DB_PATH):
        print(f"No database at {DB_PATH}.")
        return None
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    verdicts = map_verdicts(conn)
    items = router_items(conn)
    conn.close()

    policies = analysis.evaluate_router_policies(items, verdicts)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "margin": MARGIN,
        "map_verdicts": verdicts,
        "router_n": len(items),
        "policies": policies,
    }


def write_reports(report: dict):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "router.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    lines = ["# Router policies\n",
             f"Evaluated on the held-out router split ({report['router_n']} items). "
             "hosted_calls is the number of paid calls (local is free). "
             "Oracle is a perfect-knowledge ceiling, not a deployable policy.\n"]
    lines.append("| policy | accuracy | hosted calls | notes |")
    lines.append("|---|---:|---:|---|")
    order = ["always_local", "always_hosted", "map_based", "cascade", "oracle"]
    notes = {
        "always_local": "free, never calls hosted",
        "always_hosted": "most expensive, single-model ceiling",
        "map_based": "local where the map judged non-inferior",
        "cascade": "escalate only on local parse failure",
        "oracle": "upper bound, not real",
    }
    p = report["policies"]
    for k in order:
        if k not in p:
            continue
        hc = p[k]["hosted_calls"]
        hc_s = "-" if hc is None else f"{hc}"
        extra = notes[k]
        if k == "cascade" and "escalation_rate" in p[k]:
            extra += f" (escalated {p[k]['escalation_rate']*100:.0f}%)"
        lines.append(f"| {k} | {p[k]['accuracy']:.3f} | {hc_s} | {extra} |")

    with open(os.path.join(REPORTS_DIR, "router.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    report = build()
    if report is None:
        return 1
    if report["router_n"] == 0:
        print("No paired router-split data yet (need both models on the router split).")
        return 0
    write_reports(report)
    print(f"Router split: {report['router_n']} paired items")
    for k, v in report["policies"].items():
        hc = "-" if v["hosted_calls"] is None else v["hosted_calls"]
        print(f"  {k:14} acc={v['accuracy']:.3f} hosted_calls={hc}")
    print("\nWrote reports/router.md and reports/router.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
