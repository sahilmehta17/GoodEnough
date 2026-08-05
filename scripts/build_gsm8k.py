"""
Build the GSM8K difficulty report.

Joins each answer's correctness (from the results DB) to the problem's gold
reasoning-step count (from the frozen files), then for each model fits how
correctness falls as steps rise: a logistic slope with a bootstrap confidence
interval, plus a plain accuracy-by-step-count table.

Writes reports/gsm8k.md and reports/gsm8k.json. Read-only on the database.

Usage
-----
    python scripts/build_gsm8k.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import analysis, loader  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def build():
    if not os.path.exists(DB_PATH):
        print(f"No database at {DB_PATH}.")
        return None

    steps_by_item = {it.item_id: it.gold_steps for it in loader.load_frozen("gsm8k", "map")}

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT item_id, model_role, correct FROM results "
        "WHERE dataset='gsm8k' AND split='map' AND error IS NULL",
    ).fetchall()
    conn.close()

    per_model = {"local": {"steps": [], "correct": []},
                 "hosted": {"steps": [], "correct": []}}
    for item_id, role, correct in rows:
        if correct is None or role not in per_model:
            continue
        steps = steps_by_item.get(item_id)
        if steps is None:
            continue
        per_model[role]["steps"].append(steps)
        per_model[role]["correct"].append(int(correct))

    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    for role, d in per_model.items():
        if not d["steps"]:
            result["models"][role] = {"n": 0}
            continue
        b0, slope = analysis.logistic_fit(d["steps"], d["correct"])
        ci = analysis.bootstrap_slope(d["steps"], d["correct"])
        result["models"][role] = {
            "n": len(d["steps"]),
            "overall_acc": sum(d["correct"]) / len(d["correct"]),
            "slope_per_step": slope,
            "slope_ci": [ci["lower"], ci["upper"]],
            "accuracy_by_steps": analysis.bucket_accuracy(d["steps"], d["correct"]),
        }
    return result


def write_reports(result: dict):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(os.path.join(REPORTS_DIR, "gsm8k.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    lines = ["# GSM8K difficulty\n",
             "How each model's accuracy falls as a problem needs more reasoning steps.",
             "Slope is the change in log-odds of a correct answer per extra step; "
             "a more negative slope means faster degradation.\n"]

    lines.append("| model | n | overall acc | slope per step | 90% CI |")
    lines.append("|---|---:|---:|---:|:---:|")
    for role in ("local", "hosted"):
        m = result["models"].get(role, {})
        if not m.get("n"):
            lines.append(f"| {role} | 0 | | | no data yet |")
            continue
        lo, hi = m["slope_ci"]
        ci = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "n/a"
        lines.append(f"| {role} | {m['n']} | {m['overall_acc']:.2f} | "
                     f"{m['slope_per_step']:+.3f} | {ci} |")

    # Accuracy-by-steps table, both models side by side
    all_steps = set()
    for role in ("local", "hosted"):
        m = result["models"].get(role, {})
        all_steps.update(int(k) for k in m.get("accuracy_by_steps", {}))
    if all_steps:
        lines.append("\n## Accuracy by reasoning steps\n")
        lines.append("| steps | local n | local acc | hosted n | hosted acc |")
        lines.append("|---:|---:|---:|---:|---:|")
        for s in sorted(all_steps):
            lm = result["models"].get("local", {}).get("accuracy_by_steps", {}).get(s) \
                or result["models"].get("local", {}).get("accuracy_by_steps", {}).get(str(s))
            hm = result["models"].get("hosted", {}).get("accuracy_by_steps", {}).get(s) \
                or result["models"].get("hosted", {}).get("accuracy_by_steps", {}).get(str(s))
            ln = f"{lm['n']}" if lm else "-"
            la = f"{lm['acc']:.2f}" if lm else "-"
            hn = f"{hm['n']}" if hm else "-"
            ha = f"{hm['acc']:.2f}" if hm else "-"
            lines.append(f"| {s} | {ln} | {la} | {hn} | {ha} |")

    with open(os.path.join(REPORTS_DIR, "gsm8k.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    result = build()
    if result is None:
        return 1
    write_reports(result)
    for role in ("local", "hosted"):
        m = result["models"].get(role, {})
        if not m.get("n"):
            print(f"  {role}: no data yet")
            continue
        lo, hi = m["slope_ci"]
        print(f"  {role}: n={m['n']} acc={m['overall_acc']:.2f} "
              f"slope/step={m['slope_per_step']:+.3f} CI=[{lo:+.3f},{hi:+.3f}]")
    print("\nWrote reports/gsm8k.md and reports/gsm8k.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
