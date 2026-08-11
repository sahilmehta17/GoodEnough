"""
Build the GSM8K difficulty report.

Joins each answer's correctness (from the results DB) to the problem's gold
reasoning-step count (from the frozen files), then for each model fits how
correctness falls as steps rise: a logistic slope with a bootstrap confidence
interval, plus a plain accuracy-by-step-count table.

A logistic slope needs both outcomes to occur often enough to be estimated. A
model that is right on nearly every item gives a near-separable fit, where the
slope is driven by one or two responses and its interval runs to a boundary.
Such a slope is not interpretable, so it is suppressed from the report text and
kept in the JSON behind an explicit flag. The rule is a fixed threshold on the
rarer outcome, applied to every model the same way (see MIN_MINORITY_OUTCOMES).

Writes reports/gsm8k.md and reports/gsm8k.json. Read-only on the database.

Usage
-----
    python scripts/build_gsm8k.py
    python scripts/build_gsm8k.py --db path/to/other.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import analysis, loader  # noqa: E402

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# Fewer than this many of the rarer outcome (errors, or correct answers) and the
# logistic slope for that model is reported as not interpretable rather than as
# a number. Applied identically to local and hosted.
MIN_MINORITY_OUTCOMES = 5

# Bootstrap for the logistic slope. PREREGISTRATION.md section 8 fixes the
# paired bootstrap at 10,000 resamples with a fixed seed, so the slope uses the
# same three numbers rather than analysis.bootstrap_slope's smaller default.
# tail=0.05 puts a one-sided 95% bound at each end; the two ends together are a
# two-sided 90% interval. Same numbers either way, which is why the convention
# has to be stated wherever the interval is printed.
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 42
BOOTSTRAP_TAIL = 0.05

# Printed wherever a slope interval appears, so the level and the sidedness
# travel with the numbers instead of having to be inferred. The long form
# carries the equivalence and goes in prose; the short form fits a table header,
# where the prose directly above it supplies the equivalence.
INTERVAL_LABEL = "two-sided 90%, equivalently one-sided 95% bounds"
INTERVAL_LABEL_SHORT = "two-sided 90%"


def build(db_path: str):
    if not os.path.exists(db_path):
        print(f"No database at {db_path}.")
        return None

    steps_by_item = {it.item_id: it.gold_steps for it in loader.load_frozen("gsm8k", "map")}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
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
        ci = analysis.bootstrap_slope(d["steps"], d["correct"], iters=BOOTSTRAP_ITERS,
                                      seed=BOOTSTRAP_SEED, tail=BOOTSTRAP_TAIL)
        n = len(d["steps"])
        n_correct = sum(d["correct"])
        n_incorrect = n - n_correct
        minority = min(n_correct, n_incorrect)
        interpretable = minority >= MIN_MINORITY_OUTCOMES
        result["models"][role] = {
            "n": n,
            "overall_acc": n_correct / n,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "slope_per_step": slope,
            "slope_ci": [ci["lower"], ci["upper"]],
            "slope_ci_convention": INTERVAL_LABEL,
            "slope_ci_tail": BOOTSTRAP_TAIL,
            "bootstrap_iters": BOOTSTRAP_ITERS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            # The slope stays in the JSON either way. This flag says whether it
            # may be read as an estimate; reports/gsm8k.md honours it.
            "slope_interpretable": interpretable,
            "slope_not_interpretable_reason": (
                None if interpretable else
                f"only {minority} of {n} responses fall in the rarer outcome "
                f"({n_incorrect} incorrect, {n_correct} correct); below the "
                f"{MIN_MINORITY_OUTCOMES}-response floor the logistic fit is near "
                "separable and the slope is set by those few responses"),
            "min_minority_outcomes": MIN_MINORITY_OUTCOMES,
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

    lines.append(
        f"\n**Interval convention.** Each slope interval below is {INTERVAL_LABEL}. "
        "Those are two descriptions of one computation, not two computations: the "
        f"bootstrap takes the {BOOTSTRAP_TAIL:.0%} and "
        f"{1.0 - BOOTSTRAP_TAIL:.0%} percentiles of the resampled slopes, so each end "
        "is a one-sided 95% bound and the pair spans a two-sided 90% interval. The "
        "one-sided reading is the level PREREGISTRATION.md section 8 fixes. "
        f"{BOOTSTRAP_ITERS:,} resamples by item, seed {BOOTSTRAP_SEED}.\n")

    lines.append(f"\n| model | n | correct | incorrect | overall acc | slope per step "
                 f"| {INTERVAL_LABEL_SHORT} CI |")
    lines.append("|---|---:|---:|---:|---:|---:|:---:|")
    for role in ("local", "hosted"):
        m = result["models"].get(role, {})
        if not m.get("n"):
            lines.append(f"| {role} | 0 | | | | | no data yet |")
            continue
        if m.get("slope_interpretable"):
            lo, hi = m["slope_ci"]
            slope_cell = f"{m['slope_per_step']:+.3f}"
            ci_cell = f"[{lo:+.3f}, {hi:+.3f}]" if lo is not None else "n/a"
        else:
            slope_cell = "not estimable"
            ci_cell = "not estimable"
        lines.append(f"| {role} | {m['n']} | {m['n_correct']} | {m['n_incorrect']} | "
                     f"{m['overall_acc']:.2f} | {slope_cell} | {ci_cell} |")

    suppressed = [role for role in ("local", "hosted")
                  if result["models"].get(role, {}).get("n")
                  and not result["models"][role].get("slope_interpretable")]
    for role in suppressed:
        m = result["models"][role]
        lines.append(
            f"\nNo slope is reported for {role}. It answered {m['n_correct']} of "
            f"{m['n']} items correctly (accuracy {m['overall_acc']:.2f}), leaving "
            f"{m['n_incorrect']} errors across the whole split. That is too few errors "
            "to estimate how accuracy changes with step count: the logistic fit is "
            "near separable, so any slope it produces is set by those "
            f"{m['n_incorrect']} responses and its interval runs to a boundary. The "
            f"floor used here is {m['min_minority_outcomes']} responses in the rarer "
            "outcome, applied to both models. The fitted value is kept in "
            "reports/gsm8k.json flagged `slope_interpretable: false`, and it should "
            "not be read as an estimate.\n")

    # The floor is a post-hoc analysis decision. Disclose it in the report itself
    # so a reader does not have to open the pre-registration to find out.
    unaffected = []
    for role in ("local", "hosted"):
        m = result["models"].get(role, {})
        if m.get("n") and m.get("slope_interpretable"):
            unaffected.append((role, min(m["n_correct"], m["n_incorrect"])))
    if any(result["models"].get(r, {}).get("n") for r in ("local", "hosted")):
        clears = "; ".join(f"{role} has {k} and clears it" for role, k in unaffected)
        lines.append(
            f"\n**Disclosure.** The {MIN_MINORITY_OUTCOMES}-response floor used above "
            "was not pre-registered. It was chosen after data collection, once the "
            "hosted fit was seen to be near separable, so it is a post-hoc analysis "
            "decision (recorded in the PREREGISTRATION.md amendment log). It is applied "
            "identically to both models rather than to hosted alone, and it is a rule "
            "about how many responses fall in the rarer outcome, not about which model "
            "produced them."
            + (f" It changes no reported slope for the models that meet it: {clears}.\n"
               if unaffected else "\n"))

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH, help="path to results.sqlite")
    args = ap.parse_args()

    result = build(args.db)
    if result is None:
        return 1
    write_reports(result)
    for role in ("local", "hosted"):
        m = result["models"].get(role, {})
        if not m.get("n"):
            print(f"  {role}: no data yet")
            continue
        head = (f"  {role}: n={m['n']} acc={m['overall_acc']:.2f} "
                f"errors={m['n_incorrect']}")
        if not m["slope_interpretable"]:
            print(f"{head} slope=not estimable (too few errors)")
            continue
        lo, hi = m["slope_ci"]
        print(f"{head} slope/step={m['slope_per_step']:+.3f} "
              f"CI=[{lo:+.3f},{hi:+.3f}] ({INTERVAL_LABEL})")
    print("\nWrote reports/gsm8k.md and reports/gsm8k.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
