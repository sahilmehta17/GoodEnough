"""
Build the paper figures.

Each figure is emitted as a CSV first, then a vector PDF is rendered from
that CSV, as two separate steps. A figure tweak should never re-run
analysis, and the CSVs are useful artifacts in their own right.

fig1_forest            per-slice MMLU non-inferiority. Sourced from
                       reports/map.json; statistics are not recomputed here.
fig2_cost_frontier     accuracy vs dollars per routing policy, sourced from
                       reports/cost.json.
fig3_latency_frontier  accuracy vs wall-clock seconds per routing policy,
                       sourced from reports/cost.json. Same y-axis limits
                       and policy colours as fig2, so the cost/latency
                       inversion reads at a glance across the pair.
fig4_gsm8k_slope       accuracy vs GSM8K reasoning-step count, local next to
                       hosted, paired by item so both columns describe the
                       same items. Only emitted once GSM8K map has paired
                       local+hosted data; this groups and averages
                       (analysis.bucket_accuracy), it does not re-fit the
                       logistic slope reported in reports/gsm8k.json.

matplotlib is dev-only (requirements-dev.txt): this script is not on the
runtime/analysis path and reports/*.json remain plain JSON either way.

Requires reports/map.json and reports/cost.json to already exist (run
build_map.py and build_cost.py first). Read-only on the database for fig4.

Usage
-----
    python scripts/build_figures.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import analysis, loader  # noqa: E402

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "results.sqlite")

POLICY_ORDER = ["always_local", "always_hosted", "map_based", "cascade", "oracle"]
POLICY_COLORS = {
    "always_local": "#4C72B0",
    "always_hosted": "#DD8452",
    "map_based": "#55A868",
    "cascade": "#C44E52",
    "oracle": "#7F7F7F",
}


def _configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "pdf.fonttype": 42,  # embed as real text, not paths
        "axes.titlesize": 11,
        "axes.labelsize": 11,
    })
    matplotlib.pyplot = pyplot
    return matplotlib


# --------------------------------------------------------------------------
# fig1: forest plot of per-slice non-inferiority, from reports/map.json
# --------------------------------------------------------------------------

def fig1_rows(map_report: dict) -> list[dict]:
    rows = []
    for s in map_report["map_slices"]:
        if s.get("verdict") == "no_data":
            continue
        rows.append({
            "slice": s["subject"],
            "is_primary": 1 if s.get("primary") else 0,
            "n_pairs": s["n"],
            "delta_pp": s["delta"] * 100.0,
            "ci_lower_pp": s["ci_lower"] * 100.0,
            "ci_upper_pp": s["ci_upper"] * 100.0,
            "verdict": s["verdict"],
        })
    # Primary slices first, as named before data collection; exploratory
    # slices keep the map's own order after that.
    rows.sort(key=lambda r: (0 if r["is_primary"] else 1))
    return rows


def render_fig1(rows: list[dict], margin_pp: float, csv_path: str, pdf_path: str):
    _write_csv(csv_path, rows,
               ["slice", "is_primary", "n_pairs", "delta_pp", "ci_lower_pp",
                "ci_upper_pp", "verdict"])
    if not rows:
        return
    plt = _configure_matplotlib().pyplot

    fig, ax = plt.subplots(figsize=(6.0, 0.45 * len(rows) + 1.0))
    ys = list(range(len(rows)))[::-1]
    verdict_colors = {"non_inferior": "#55A868", "below_margin": "#C44E52",
                      "inconclusive": "#8172B2", "no_data": "#7F7F7F"}
    for y, r in zip(ys, rows):
        color = verdict_colors.get(r["verdict"], "#333333")
        lo, hi, delta = r["ci_lower_pp"], r["ci_upper_pp"], r["delta_pp"]
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, zorder=2)
        marker = "s" if r["is_primary"] else "o"
        ax.scatter([delta], [y], color=color, marker=marker, s=40, zorder=3)

    ax.axvline(-margin_pp, linestyle="--", color="#333333", linewidth=1)
    ax.text(-margin_pp, len(rows) - 0.4, "predeclared margin", rotation=90,
           va="top", ha="right", fontsize=9)

    labels = [f"{'* ' if r['is_primary'] else ''}{r['slice']}" for r in rows]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("accuracy_local - accuracy_hosted (percentage points)")
    ax.margins(y=0.05)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# fig2 / fig3: cost and latency frontiers, from reports/cost.json
# --------------------------------------------------------------------------

def frontier_rows(cost_report: dict) -> list[dict]:
    rs = cost_report.get("router_split")
    if not rs:
        return []
    rows = []
    for k in POLICY_ORDER:
        if k not in rs["policies"]:
            continue
        p, e = rs["policies"][k], rs["economics"][k]
        rows.append({
            "policy": k,
            "accuracy_pp": p["accuracy"] * 100.0,
            "total_usd": e["total_usd"],
            "usd_per_1k_requests": e["usd_per_1k_requests"],
            "total_seconds": e["total_seconds"],
            "median_seconds_per_request": e["median_seconds_per_request"],
            "deployable": 0 if k == "oracle" else 1,
        })
    return rows


def render_frontier(rows: list[dict], value_key: str, value_label: str,
                    csv_columns: list[str], csv_path: str, pdf_path: str,
                    y_limits: tuple[float, float]):
    _write_csv(csv_path, rows, csv_columns)
    if not rows:
        return
    plt = _configure_matplotlib().pyplot

    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    deployable = [r for r in rows if r["deployable"] and r[value_key] is not None]
    oracle = [r for r in rows if not r["deployable"]]

    for r in deployable:
        ax.scatter([r[value_key]], [r["accuracy_pp"]], color=POLICY_COLORS.get(r["policy"], "#333"),
                   s=50, zorder=3, label=r["policy"])
    for r in oracle:
        ax.axhline(r["accuracy_pp"], linestyle=":", color=POLICY_COLORS.get(r["policy"], "#333"),
                   linewidth=1.5, label=f"{r['policy']} (not deployable)")

    ax.set_xlabel(value_label)
    ax.set_ylabel("accuracy (percentage points)")
    ax.set_ylim(*y_limits)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _shared_accuracy_ylimits(rows: list[dict]) -> tuple[float, float]:
    if not rows:
        return (0.0, 100.0)
    accs = [r["accuracy_pp"] for r in rows]
    lo, hi = min(accs), max(accs)
    pad = max(2.0, (hi - lo) * 0.15)
    return (max(0.0, lo - pad), min(100.0, hi + pad))


# --------------------------------------------------------------------------
# fig4: GSM8K difficulty slope, paired local vs hosted by reasoning steps
# --------------------------------------------------------------------------

def paired_gsm8k_by_step(conn: sqlite3.Connection) -> list[dict]:
    """
    Items in the GSM8K map split where BOTH local and hosted produced an
    error-free, scored answer, grouped by gold reasoning-step count. This is
    a paired view (same items for both columns), unlike gsm8k.json's two
    independent per-role buckets, so it needs its own small join here rather
    than reading gsm8k.json directly.
    """
    steps_by_item = {it.item_id: it.gold_steps for it in loader.load_frozen("gsm8k", "map")}
    rows = conn.execute(
        "SELECT item_id, model_role, correct FROM results "
        "WHERE dataset='gsm8k' AND split='map' AND error IS NULL").fetchall()
    by_item: dict[str, dict[str, int]] = {}
    for item_id, role, correct in rows:
        if correct is not None:
            by_item.setdefault(item_id, {})[role] = int(correct)

    steps, local_correct, hosted_correct = [], [], []
    for item_id, roles in by_item.items():
        if "local" not in roles or "hosted" not in roles:
            continue
        s = steps_by_item.get(item_id)
        if s is None:
            continue
        steps.append(s)
        local_correct.append(roles["local"])
        hosted_correct.append(roles["hosted"])

    if not steps:
        return []

    local_by_step = analysis.bucket_accuracy(steps, local_correct)
    hosted_by_step = analysis.bucket_accuracy(steps, hosted_correct)
    out = []
    for s in sorted(local_by_step):
        if s not in hosted_by_step:
            continue
        out.append({
            "reasoning_steps": s,
            "n_items": local_by_step[s]["n"],
            "local_accuracy_pp": local_by_step[s]["acc"] * 100.0,
            "hosted_accuracy_pp": hosted_by_step[s]["acc"] * 100.0,
        })
    return out


def render_fig4(rows: list[dict], csv_path: str, pdf_path: str):
    _write_csv(csv_path, rows,
               ["reasoning_steps", "n_items", "local_accuracy_pp", "hosted_accuracy_pp"])
    if not rows:
        return
    plt = _configure_matplotlib().pyplot

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    xs = [r["reasoning_steps"] for r in rows]
    ax.plot(xs, [r["local_accuracy_pp"] for r in rows], marker="o",
           color=POLICY_COLORS["always_local"], label="local")
    ax.plot(xs, [r["hosted_accuracy_pp"] for r in rows], marker="s",
           color=POLICY_COLORS["always_hosted"], label="hosted")
    ax.set_xlabel("gold reasoning steps")
    ax.set_ylabel("accuracy (percentage points)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------

def _write_csv(path: str, rows: list[dict], columns: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c) for c in columns})


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    import json
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build(db_path: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    map_report = _load_json(os.path.join(REPORTS_DIR, "map.json"))
    cost_report = _load_json(os.path.join(REPORTS_DIR, "cost.json"))

    if map_report is None:
        print("No reports/map.json. Not yet available: run scripts/build_map.py first.")
    else:
        rows = fig1_rows(map_report)
        render_fig1(rows, map_report["margin"] * 100.0,
                   os.path.join(FIGURES_DIR, "fig1_forest.csv"),
                   os.path.join(FIGURES_DIR, "fig1_forest.pdf"))
        print(f"fig1_forest: {len(rows)} slices")

    if cost_report is None:
        print("No reports/cost.json. Not yet available: run scripts/build_cost.py first.")
    else:
        frontier = frontier_rows(cost_report)
        ylim = _shared_accuracy_ylimits(frontier)
        if not frontier:
            print("fig2/fig3: not yet available (router split has no paired hosted data yet)")
        render_frontier(
            frontier, "total_usd", "total dollars",
            ["policy", "accuracy_pp", "total_usd", "usd_per_1k_requests", "deployable"],
            os.path.join(FIGURES_DIR, "fig2_cost_frontier.csv"),
            os.path.join(FIGURES_DIR, "fig2_cost_frontier.pdf"), ylim)
        render_frontier(
            frontier, "total_seconds", "total wall-clock seconds",
            ["policy", "accuracy_pp", "total_seconds", "median_seconds_per_request", "deployable"],
            os.path.join(FIGURES_DIR, "fig3_latency_frontier.csv"),
            os.path.join(FIGURES_DIR, "fig3_latency_frontier.pdf"), ylim)
        if frontier:
            print(f"fig2/fig3: {len(frontier)} policies")

    if not os.path.exists(db_path):
        print(f"No database at {db_path}. fig4 not yet available.")
        return
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    fig4_rows = paired_gsm8k_by_step(conn)
    conn.close()
    render_fig4(fig4_rows, os.path.join(FIGURES_DIR, "fig4_gsm8k_slope.csv"),
               os.path.join(FIGURES_DIR, "fig4_gsm8k_slope.pdf"))
    if fig4_rows:
        print(f"fig4_gsm8k_slope: {len(fig4_rows)} step buckets")
    else:
        print("fig4_gsm8k_slope: not yet available "
             "(no GSM8K map item has both a local and a hosted result yet)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH, help="path to results.sqlite")
    args = ap.parse_args()
    build(args.db)
    print(f"\nWrote CSVs and PDFs into {os.path.relpath(FIGURES_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
