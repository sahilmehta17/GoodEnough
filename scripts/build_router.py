"""
Build the router report.

Derives the routing policy from the map (which subjects the small model was
judged non-inferior on), then evaluates five policies on the held-out router
split: always-local, always-hosted, map-based, local-first cascade, and the
oracle upper bound. Reports each policy's accuracy and how many paid hosted
calls it made.

Leads with the item-level breakdown, because that is what bounds routing: only
items the local model gets right and the hosted model gets wrong can put any
policy above always-hosted. The oracle minus always-hosted gap is reported with
a paired bootstrap interval and is never reported without it.

Writes reports/router.md and reports/router.json. Read-only on the database.

Usage
-----
    python scripts/build_router.py
    python scripts/build_router.py --db path/to/other.sqlite
"""

from __future__ import annotations

import argparse
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

# Paired bootstrap for the oracle gap: resampled by item, 10,000 resamples,
# fixed seed. tail=0.025 makes the reported interval a two-sided 95%.
#
# This does NOT match PREREGISTRATION.md section 8, and an earlier version of
# this comment wrongly claimed it did. Section 8 fixes the paired bootstrap at
# one-sided 95% (tail=0.05) as the cross-check on the per-slice non-inferiority
# delta, and that is what analysis.slice_result still uses. The oracle gap is a
# different quantity: section 13 mandates it only as an upper bound and assigns
# it no level, so it is post-hoc, has no privileged direction, and is reported
# two-sided at 95%. Same function, deliberately different tail.
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 42
BOOTSTRAP_TAIL = 0.025

# Printed next to the numbers so the level and sidedness travel with them.
INTERVAL_LABEL = "two-sided 95%"


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


def oracle_gap(items: list[dict]) -> dict | None:
    """
    Oracle accuracy minus always-hosted accuracy, with a paired bootstrap
    interval resampled by item.

    Per item the oracle is right whenever either model is right, so the paired
    difference (oracle - hosted) is 1 exactly on the items where local was right
    and hosted was wrong, and 0 everywhere else. That makes this gap the ceiling
    on what any routing policy can gain over always-hosted, and it is the reason
    the local_only count leads this report.
    """
    if not items:
        return None
    hosted = [i["hosted_correct"] for i in items]
    oracle = [1 if (i["local_correct"] or i["hosted_correct"]) else 0 for i in items]
    boot = analysis.bootstrap_paired(oracle, hosted, iters=BOOTSTRAP_ITERS,
                                     seed=BOOTSTRAP_SEED, tail=BOOTSTRAP_TAIL)
    n = len(items)
    return {
        "oracle_accuracy": sum(oracle) / n,
        "always_hosted_accuracy": sum(hosted) / n,
        "gap": (sum(oracle) - sum(hosted)) / n,
        "ci_lower": boot["lower"],
        "ci_upper": boot["upper"],
        "ci_convention": INTERVAL_LABEL,
        "ci_tail": BOOTSTRAP_TAIL,
        "n": n,
        "bootstrap_iters": BOOTSTRAP_ITERS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def build(db_path: str):
    if not os.path.exists(db_path):
        print(f"No database at {db_path}; skipping. See reports/ for the "
              "committed results (README, Reproduce).")
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    verdicts = map_verdicts(conn)
    items = router_items(conn)
    conn.close()

    policies = analysis.evaluate_router_policies(items, verdicts)
    breakdown = analysis.outcome_breakdown([i["local_correct"] for i in items],
                                           [i["hosted_correct"] for i in items])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "margin": MARGIN,
        "map_verdicts": verdicts,
        "router_n": len(items),
        "item_breakdown": breakdown,
        "oracle_gap": oracle_gap(items),
        "policies": policies,
    }


def _breakdown_section(report: dict) -> list[str]:
    """
    The four-way item split, first in the report. Everything below it is
    downstream of these four counts.
    """
    b = report.get("item_breakdown")
    if not b or not b["n"]:
        return []
    n = b["n"]

    lines = ["\n## Item-level breakdown\n"]
    lines.append(f"Every one of the {n} paired items on the held-out router split, "
                 "by which model got it right.\n")
    lines.append("\n| outcome | items | share |")
    lines.append("|---|---:|---:|")
    lines.append(f"| both correct | {b['both_correct']} | {b['both_correct']/n:.3f} |")
    lines.append(f"| hosted correct, local incorrect | {b['hosted_only']} | "
                 f"{b['hosted_only']/n:.3f} |")
    lines.append(f"| **local correct, hosted incorrect** | **{b['local_only']}** | "
                 f"**{b['local_only']/n:.3f}** |")
    lines.append(f"| neither correct | {b['neither_correct']} | "
                 f"{b['neither_correct']/n:.3f} |")

    decided = b["both_correct"] + b["hosted_only"] + b["neither_correct"]
    lines.append(
        f"\n**{b['local_only']} of {n}.** That row is every item where the local 1.7B "
        "model was right and the hosted 70B model was wrong. It is the only row where "
        f"routing can put accuracy above always-hosted, so those {b['local_only']} "
        "items are the entire accuracy budget available to any policy in this study. "
        f"The remaining {decided} items score identically under every policy that "
        f"routes them to the model that was already right: {b['both_correct']} both "
        f"models get right, {b['hosted_only']} only the hosted model gets right, and "
        f"{b['neither_correct']} neither gets right.\n")
    lines.append(
        f"\nRouting still has a cost argument: sending an item to the local model "
        "costs no metered API dollars. What these counts bound is the accuracy "
        f"argument. Against always-hosted, a policy gains on at most {b['local_only']} "
        f"items and loses on every one of the {b['hosted_only']} hosted-correct, "
        "local-incorrect items it decides to keep local.\n")
    return lines


def _oracle_gap_section(report: dict) -> list[str]:
    """The oracle ceiling, never printed without its interval."""
    g = report.get("oracle_gap")
    if not g:
        return []
    lines = ["\n## Oracle gap over always-hosted\n"]
    lines.append(
        f"Oracle accuracy {g['oracle_accuracy']:.3f} minus always-hosted accuracy "
        f"{g['always_hosted_accuracy']:.3f} = **{g['gap']:+.4f}**, "
        f"{INTERVAL_LABEL} paired bootstrap CI "
        f"[{g['ci_lower']:+.4f}, {g['ci_upper']:+.4f}] "
        f"({g['bootstrap_iters']:,} resamples by item, seed {g['bootstrap_seed']}, "
        f"n = {g['n']}).\n")
    lines.append(
        f"\nThat interval is {INTERVAL_LABEL}, not the one-sided 95% used for the "
        "per-slice non-inferiority intervals in reports/map.md. The oracle gap is not "
        "named in PREREGISTRATION.md section 8; section 13 mandates it only as an "
        "upper bound and fixes no level for it. It is a post-hoc quantity with no "
        "privileged direction to test against, so it is reported two-sided. "
        "reports/map.md carries the full convention for both classes.\n")
    lines.append(
        f"\nThe oracle needs per-item knowledge of which model is right, so it is a "
        "ceiling rather than a policy. On this model pair at this sample size, the "
        f"ceiling on any routing policy is between {g['ci_lower']*100:.1f} and "
        f"{g['ci_upper']*100:.1f} percentage points over always-hosted. The gap is "
        "reported only with that interval; the point estimate on its own is not a "
        "result.\n")
    return lines


def write_reports(report: dict):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    # newline="\n" on both writes below: reports go out with LF on every
    # platform, matching .gitattributes (eol=lf). Without it Windows writes CRLF
    # and a rebuild in a fresh clone shows every report as modified in git status
    # with no content change.
    with open(os.path.join(REPORTS_DIR, "router.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)

    lines = ["# Router policies\n"]
    lines.extend(_breakdown_section(report))
    lines.extend(_oracle_gap_section(report))

    lines.append("\n## Policy comparison\n")
    lines.append(f"Evaluated on the same held-out router split ({report['router_n']} "
                 "items). hosted_calls is the number of paid calls (local is free). "
                 "Oracle is a perfect-knowledge ceiling, not a deployable policy.\n")
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

    with open(os.path.join(REPORTS_DIR, "router.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH, help="path to results.sqlite")
    args = ap.parse_args()

    report = build(args.db)
    if report is None:
        # Missing database is the expected state in a fresh clone, not a
        # failure. Exit 0 so the README's one-command rebuild does not die
        # at the first builder and skip the remaining four.
        return 0
    if report["router_n"] == 0:
        print("No paired router-split data yet (need both models on the router split).")
        return 0
    write_reports(report)
    print(f"Router split: {report['router_n']} paired items")
    b = report["item_breakdown"]
    print(f"  both correct={b['both_correct']} hosted_only={b['hosted_only']} "
          f"local_only={b['local_only']} neither={b['neither_correct']}")
    g = report["oracle_gap"]
    if g:
        print(f"  oracle - always_hosted = {g['gap']:+.4f} "
              f"CI=[{g['ci_lower']:+.4f},{g['ci_upper']:+.4f}] ({INTERVAL_LABEL})")
    for k, v in report["policies"].items():
        hc = "-" if v["hosted_calls"] is None else v["hosted_calls"]
        print(f"  {k:14} acc={v['accuracy']:.3f} hosted_calls={hc}")
    print("\nWrote reports/router.md and reports/router.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
