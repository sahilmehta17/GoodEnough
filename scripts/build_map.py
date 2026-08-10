"""
Build the map from the results database.

Reads every paired result (items where BOTH models produced an error-free
answer), computes per-subject accuracy, the accuracy gap with a confidence
interval, and the non-inferiority verdict, then writes:

    reports/map.md    human-readable table
    reports/map.json  machine-readable results

Also reports the disagreement (discordance) rate from the dev split, the
unparseable rate per subject per model, the verdict at each predeclared
sensitivity margin, a difficulty correlation across slices, and a cost/latency
summary. Read-only on the database; safe to run while collection continues,
though the map is only final once every split is complete.

Usage
-----
    python scripts/build_map.py
    python scripts/build_map.py --db path/to/other.sqlite
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
from goodenough import datasets_config as dc  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MARGIN = 0.10  # PREREGISTRATION.md section 3

# Section 3 also asks for sensitivity at 5 and 15 points. Same interval, same
# analysis.verdict; only the margin moves.
SENSITIVITY_MARGINS = [0.05, 0.10, 0.15]

# Bootstrap settings for the difficulty correlation, matching the paired
# bootstrap used elsewhere: 10,000 resamples, fixed seed. tail=0.025 makes the
# reported interval a two-sided 95%.
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 42
BOOTSTRAP_TAIL = 0.025

# A subject is called out in the unparseable prose only above this rate. Below
# it the count is still in the table; this only decides what gets a sentence.
UNPARSEABLE_NOTABLE_RATE = 0.10

MODEL_ROLES = ("local", "hosted")


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


def unparseable_rates(conn, dataset: str, split: str) -> dict:
    """
    Per subject, per model role: how often the frozen parser could not read the
    model's response. PREREGISTRATION.md section 12 requires this reported per
    model as its own result.

    The denominator is responses that came back without an API error. API
    failures are a separate reliability quantity (section 12 again), so they
    are counted as 'errors' here rather than folded into the unparseable rate.
    """
    rows = conn.execute(
        "SELECT item_id, model_role, parse_status, error, correct FROM results "
        "WHERE dataset=? AND split=?",
        (dataset, split),
    ).fetchall()

    out: dict[str, dict[str, dict]] = {}
    for item_id, role, parse_status, error, correct in rows:
        slot = out.setdefault(subject_of(item_id), {}).setdefault(
            role, {"unparseable": 0, "unparseable_scored_incorrect": 0,
                   "n": 0, "errors": 0})
        if error is not None:
            slot["errors"] += 1
            continue
        slot["n"] += 1
        if parse_status == "unparseable":
            slot["unparseable"] += 1
            # Section 12 says an unparseable response scores incorrect. Counted
            # rather than assumed, so the report states what the rows actually say.
            if correct == 0:
                slot["unparseable_scored_incorrect"] += 1

    for roles in out.values():
        for slot in roles.values():
            slot["rate"] = (slot["unparseable"] / slot["n"]) if slot["n"] else None
    return out


def sensitivity_table(slices: list[dict], margins: list[float]) -> list[dict]:
    """
    Each slice's verdict at every predeclared margin. Reuses analysis.verdict on
    the interval already computed for that slice: nothing is re-estimated, only
    re-classified, so a slice cannot disagree with itself across the table.
    """
    out = []
    for s in slices:
        scored = s.get("verdict") != "no_data"
        out.append({
            "subject": s["subject"],
            "primary": bool(s.get("primary")),
            "verdicts": {
                f"{m:.2f}": (analysis.verdict(s["ci_lower"], s["ci_upper"], m)
                             if scored else "no_data")
                for m in margins
            },
        })
    return out


def difficulty_correlation(slices: list[dict]) -> dict | None:
    """
    Across slices, does the gap track how hard the slice is? x is hosted
    accuracy (the difficulty proxy), y is delta. A positive r means the local
    model falls further behind on the slices the hosted model also finds harder,
    which is the direction difficulty-based routing assumes.

    One point per slice, so n is the slice count, not the item count. That is
    the whole reason this is underpowered and reported as such.
    """
    scored = [s for s in slices if s.get("verdict") != "no_data"]
    if len(scored) < 2:
        return None
    xs = [s["acc_hosted"] for s in scored]
    ys = [s["delta"] for s in scored]
    r = analysis.pearson(xs, ys)
    ci = analysis.bootstrap_pearson(xs, ys, iters=BOOTSTRAP_ITERS,
                                    seed=BOOTSTRAP_SEED, tail=BOOTSTRAP_TAIL)
    lo, hi = ci["lower"], ci["upper"]
    return {
        "r": r,
        "ci_lower": lo,
        "ci_upper": hi,
        "n_slices": len(scored),
        "bootstrap_iters": BOOTSTRAP_ITERS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "resamples_used": ci["iters_used"],
        "spans_zero": (lo is not None and hi is not None and lo <= 0.0 <= hi),
    }


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


def build(db_path: str):
    if not os.path.exists(db_path):
        print(f"No database at {db_path}. Run scripts/run_eval.py first.")
        return None
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

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
        "sensitivity_margins": SENSITIVITY_MARGINS,
        "local_model": config.LOCAL_MODEL_ID,
        "hosted_model": config.HOSTED_MODEL_ID,
        "dev_discordance": dev_disc,
        "map_slices": slices,
        "map_sensitivity": sensitivity_table(slices, SENSITIVITY_MARGINS),
        "map_unparseable": unparseable_rates(conn, "mmlu", "map"),
        "difficulty_correlation": difficulty_correlation(slices),
        "map_cost_latency": cost_latency(conn, "mmlu", "map"),
    }
    conn.close()
    return report


def _sensitivity_section(report: dict) -> list[str]:
    """Verdict for every slice at each predeclared margin (section 3)."""
    rows = report.get("map_sensitivity") or []
    if not rows:
        return []
    margins = report["sensitivity_margins"]
    keys = [f"{m:.2f}" for m in margins]

    lines = ["\n## Sensitivity to the margin\n"]
    lines.append("The primary margin is "
                 f"{report['margin']:.2f}. PREREGISTRATION.md section 3 also asks for "
                 f"{' and '.join(f'{m*100:.0f}' for m in margins if m != report['margin'])} "
                 "points. Same interval per slice, reclassified; nothing is re-estimated.\n")
    header = " | ".join(f"margin {float(k)*100:.0f}pp" for k in keys)
    lines.append(f"\n| subject | | {header} |")
    lines.append("|---|---|" + "---|" * len(keys))
    for r in rows:
        star = "P" if r["primary"] else ""
        cells = " | ".join(r["verdicts"][k] for k in keys)
        lines.append(f"| {r['subject']} | {star} | {cells} |")

    # Count movement across margins, so any claim about stability is computed.
    moved = [r["subject"] for r in rows
             if len({r["verdicts"][k] for k in keys}) > 1]
    if moved:
        lines.append(f"\n{len(moved)} of {len(rows)} slices change verdict across these "
                     f"margins: {', '.join(moved)}.\n")
    else:
        lines.append("\nNo slice changes verdict across these margins.\n")
    return lines


def _unparseable_section(report: dict) -> list[str]:
    """
    Unparseable count and rate per subject per model (section 12), plus prose
    generated from those same counts.
    """
    up = report.get("map_unparseable") or {}
    if not up:
        return []
    subjects = [s for s in dc.MMLU_SUBJECTS if s in up]

    lines = ["\n## Unparseable responses (MMLU map)\n"]
    lines.append("An unparseable response scores incorrect and is never dropped "
                 "(PREREGISTRATION.md section 12). The denominator is responses that "
                 "returned without an API error; API failures are counted separately "
                 "in the errors columns.\n")
    lines.append("\n| subject | hosted unparseable | hosted rate | local unparseable | "
                 "local rate | hosted errors | local errors |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for subj in subjects:
        cells = []
        for role in ("hosted", "local"):
            d = up[subj].get(role)
            if d is None:
                cells.append(("-", "-"))
                continue
            rate = "-" if d["rate"] is None else f"{d['rate']:.2f}"
            cells.append((f"{d['unparseable']}/{d['n']}", rate))
        errs = [str(up[subj].get(role, {}).get("errors", 0)) for role in ("hosted", "local")]
        lines.append(f"| {subj} | {cells[0][0]} | {cells[0][1]} | {cells[1][0]} | "
                     f"{cells[1][1]} | {errs[0]} | {errs[1]} |")

    clean = [s for s in subjects
             if all(up[s][r]["unparseable"] == 0 for r in up[s])]
    dirty = [s for s in subjects if s not in clean]
    lines.append(f"\n{len(clean)} of the {len(subjects)} subjects are clean on both "
                 "models: zero unparseable responses from either.")
    if dirty:
        detail = "; ".join(
            f"{s} (" + ", ".join(f"{r} {up[s][r]['unparseable']}/{up[s][r]['n']}"
                                 for r in ("hosted", "local") if r in up[s]) + ")"
            for s in dirty)
        lines.append(f"The exceptions are {detail}.")

    both_high = [s for s in subjects
                 if all(up[s][r]["rate"] is not None
                        and up[s][r]["rate"] >= UNPARSEABLE_NOTABLE_RATE
                        for r in ("hosted", "local") if r in up[s])
                 and {"hosted", "local"}.issubset(up[s])]
    for s in both_high:
        h, l = up[s]["hosted"], up[s]["local"]
        total_unp = h["unparseable"] + l["unparseable"]
        scored_zero = (h["unparseable_scored_incorrect"]
                       + l["unparseable_scored_incorrect"])
        scoring = (f"All {total_unp} of those responses scored incorrect"
                   if scored_zero == total_unp
                   else f"{scored_zero} of those {total_unp} responses scored incorrect")
        lines.append(
            f"\n{s} is unparseable at {h['rate']:.0%} on the hosted 70B model and "
            f"{l['rate']:.0%} on the local 1.7B model. A failure rate that high on "
            "both models is a harness limitation on that subject, in prompt format "
            "and answer extraction, not a capability difference between the models. "
            f"{scoring}, so the {s} accuracies above are a floor for both models, not "
            "an estimate of what either model knows.")
    if dirty:
        lines.append("\nThese rates are reported, not corrected. The parser was frozen "
                     "on the dev split before any map response was seen; editing it "
                     "against map data would invalidate the map (PREREGISTRATION.md "
                     "section 12, CLAUDE.md).\n")
    return lines


def _correlation_section(report: dict) -> list[str]:
    """Hosted accuracy against delta across slices, with a bootstrap interval."""
    dcorr = report.get("difficulty_correlation")
    if not dcorr or dcorr["r"] is None:
        return []
    lines = ["\n## Difficulty and the size of the gap\n"]
    lines.append(
        f"Pearson correlation between hosted accuracy and delta across the "
        f"{dcorr['n_slices']} slices: r = {dcorr['r']:+.3f}, "
        f"95% bootstrap CI [{dcorr['ci_lower']:+.3f}, {dcorr['ci_upper']:+.3f}] "
        f"({dcorr['bootstrap_iters']:,} resamples, seed {dcorr['bootstrap_seed']}).\n")
    direction = ("the local model falls further behind on the slices the hosted model "
                 "also finds harder" if dcorr["r"] > 0 else
                 "the local model falls further behind on the slices the hosted model "
                 "finds easier")
    zero_clause = (f"and contains zero, so the correlation is not distinguishable from "
                   f"zero at n = {dcorr['n_slices']}"
                   if dcorr["spans_zero"] else
                   f"though it excludes zero at n = {dcorr['n_slices']}")
    lines.append(
        f"\nThis is underpowered. One point per slice means n = {dcorr['n_slices']}, "
        f"and the interval is correspondingly wide {zero_clause}. The sign is the "
        f"direction difficulty-based routing would predict: {direction}. That direction "
        "is worth recording and is not evidence for the mechanism at this sample size.\n")
    return lines


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

    lines.extend(_sensitivity_section(report))
    lines.extend(_unparseable_section(report))
    lines.extend(_correlation_section(report))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH, help="path to results.sqlite")
    args = ap.parse_args()

    report = build(args.db)
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
