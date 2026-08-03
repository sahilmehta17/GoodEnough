"""
Freeze the benchmark splits. Run once, on a machine with internet, BEFORE any
model is called against benchmark items. This is the gate the .cursor rule and
PREREGISTRATION.md require: no benchmark call until the frozen splits are committed.

Dependency-free by design. It uses Hugging Face's dataset "rows" HTTP API and
the Python standard library only. No pandas, pyarrow, or datasets, which also
means it runs on locked-down machines where those compiled libraries are blocked.

What it does
------------
1. Reads the 8 MMLU subjects (test + validation) and GSM8K test over HTTP.
2. Deterministically selects items into dev / map / router under seed 42.
3. Writes item CONTENT (not just IDs) to data/frozen/*.jsonl, so the runner
   never needs the network again and the exact items are reproducible.
4. Writes data/frozen/MANIFEST.json with source info, per-slice counts, seed,
   and a sha256 of each frozen file.

Usage
-----
    python scripts/freeze_splits.py --dry-run   # fetch counts only, write nothing
    python scripts/freeze_splits.py             # write frozen files + manifest

Determinism
-----------
For each split the FULL population is pulled, then a seeded shuffle of the row
order selects items. Identical on any machine. Nothing here depends on hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from goodenough import datasets_config as dc  # noqa: E402

FROZEN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "frozen")
ROWS_API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the rows API caps a page at 100


def _get_json(url: str, max_retries: int = 5) -> dict:
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "goodenough-freeze/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise RuntimeError(f"failed to fetch {url}: {exc!r}") from exc
            time.sleep(2 ** attempt)
    return {}


def _rows_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    q = urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    return f"{ROWS_API}?{q}"


def fetch_count(dataset: str, config: str, split: str) -> int:
    data = _get_json(_rows_url(dataset, config, split, 0, 1))
    return int(data.get("num_rows_total", 0))


def fetch_split(dataset: str, config: str, split: str) -> list[dict]:
    """Return every row's data dict, in original dataset order."""
    rows: list[dict] = []
    offset = 0
    total = None
    while True:
        data = _get_json(_rows_url(dataset, config, split, offset, PAGE))
        batch = data.get("rows", [])
        if total is None:
            total = int(data.get("num_rows_total", 0))
        rows.extend(r["row"] for r in batch)
        offset += PAGE
        if not batch or offset >= total:
            break
        time.sleep(0.2)  # be polite to the API
    return rows


def _seeded_order(n: int, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mmlu_row(item: dict, subject: str, split: str, orig_index: int) -> dict:
    gold_letter = dc.MMLU_CHOICES[int(item["answer"])]
    return {
        "dataset": "mmlu",
        "subject": subject,
        "split": split,
        "item_id": f"mmlu/{subject}/{split}/{orig_index}",
        "question": item["question"],
        "choices": list(item["choices"]),
        "gold_letter": gold_letter,
    }


def _gsm_row(item: dict, split: str, orig_index: int) -> dict:
    solution = item["answer"]
    return {
        "dataset": "gsm8k",
        "subject": "gsm8k",
        "split": split,
        "item_id": f"gsm8k/{split}/{orig_index}",
        "question": item["question"],
        "gold_number": solution.split("####")[-1].strip(),
        "gold_steps": solution.count("<<"),   # calculator annotations = steps
        "solution": solution,
    }


def freeze_mmlu(dry_run: bool):
    map_rows, router_pool, dev_rows = [], [], []
    counts = {}

    for subject in dc.MMLU_SUBJECTS:
        if dry_run:
            test_n = fetch_count(dc.MMLU_HF_PATH, subject, "test")
            val_n = fetch_count(dc.MMLU_HF_PATH, subject, "validation")
            order = _seeded_order(test_n, dc.SEED)
            n_map = len(order[: dc.MMLU_MAP_PER_SUBJECT])
            n_router = len(order[dc.MMLU_MAP_PER_SUBJECT:
                                 dc.MMLU_MAP_PER_SUBJECT + dc.MMLU_ROUTER_PER_SUBJECT])
            counts[subject] = {"test_available": test_n, "validation_available": val_n,
                               "map": n_map, "router_candidate": n_router}
            continue

        test = fetch_split(dc.MMLU_HF_PATH, subject, "test")
        val = fetch_split(dc.MMLU_HF_PATH, subject, "validation")
        order = _seeded_order(len(test), dc.SEED)
        map_idx = order[: dc.MMLU_MAP_PER_SUBJECT]
        router_idx = order[dc.MMLU_MAP_PER_SUBJECT:
                           dc.MMLU_MAP_PER_SUBJECT + dc.MMLU_ROUTER_PER_SUBJECT]

        for i in map_idx:
            map_rows.append(_mmlu_row(test[i], subject, "map", i))
        for i in router_idx:
            router_pool.append(_mmlu_row(test[i], subject, "router", i))
        for i in range(len(val)):
            dev_rows.append(_mmlu_row(val[i], subject, "dev", i))

        counts[subject] = {"test_available": len(test), "validation_available": len(val),
                           "map": len(map_idx), "router_candidate": len(router_idx)}
        print(f"  fetched {subject}: test={len(test)} val={len(val)}")

    router_rows = []
    if not dry_run:
        r_order = _seeded_order(len(router_pool), dc.SEED)
        router_rows = [router_pool[i] for i in r_order[: dc.MMLU_ROUTER_TOTAL_CAP]]
        _write_jsonl(os.path.join(FROZEN_DIR, "mmlu_dev.jsonl"), dev_rows)
        _write_jsonl(os.path.join(FROZEN_DIR, "mmlu_map.jsonl"), map_rows)
        _write_jsonl(os.path.join(FROZEN_DIR, "mmlu_router.jsonl"), router_rows)

    return {"per_subject": counts,
            "totals": {"dev": len(dev_rows), "map": len(map_rows), "router": len(router_rows)}}


def freeze_gsm8k(dry_run: bool):
    if dry_run:
        test_n = fetch_count(dc.GSM8K_HF_PATH, dc.GSM8K_HF_CONFIG, "test")
        return {"test_available": test_n,
                "totals": {"dev": min(dc.GSM8K_DEV_N, test_n),
                           "map": min(dc.GSM8K_MAP_N, max(0, test_n - dc.GSM8K_DEV_N))}}

    test = fetch_split(dc.GSM8K_HF_PATH, dc.GSM8K_HF_CONFIG, "test")
    order = _seeded_order(len(test), dc.SEED)
    dev_idx = order[: dc.GSM8K_DEV_N]
    map_idx = order[dc.GSM8K_DEV_N: dc.GSM8K_DEV_N + dc.GSM8K_MAP_N]
    dev_rows = [_gsm_row(test[i], "dev", i) for i in dev_idx]
    map_rows = [_gsm_row(test[i], "map", i) for i in map_idx]
    _write_jsonl(os.path.join(FROZEN_DIR, "gsm8k_dev.jsonl"), dev_rows)
    _write_jsonl(os.path.join(FROZEN_DIR, "gsm8k_map.jsonl"), map_rows)
    print(f"  fetched gsm8k: test={len(test)}")
    return {"test_available": len(test),
            "totals": {"dev": len(dev_rows), "map": len(map_rows)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch counts only, write nothing")
    args = ap.parse_args()

    if not args.dry_run:
        os.makedirs(FROZEN_DIR, exist_ok=True)

    print(f"Seed: {dc.SEED}")
    print(f"MMLU subjects ({len(dc.MMLU_SUBJECTS)}): {', '.join(dc.MMLU_SUBJECTS)}")
    print(f"Primary: {dc.MMLU_PRIMARY}")
    print("Fetching over the Hugging Face rows API (no pandas/pyarrow)...\n")

    mmlu = freeze_mmlu(args.dry_run)
    gsm = freeze_gsm8k(args.dry_run)

    print("\nMMLU per-subject:")
    for subj, c in mmlu["per_subject"].items():
        tag = " [PRIMARY]" if subj in dc.MMLU_PRIMARY else ""
        small = "  <-- small: whole test used as map" if c["test_available"] < dc.MMLU_MAP_PER_SUBJECT else ""
        print(f"  {subj:26} test={c['test_available']:<5} val={c['validation_available']:<4} "
              f"map={c['map']:<4} router_cand={c['router_candidate']}{tag}{small}")

    print(f"\nMMLU totals: dev={mmlu['totals']['dev']}  map={mmlu['totals']['map']}  "
          f"router={mmlu['totals']['router']}")
    print(f"GSM8K: dev={gsm['totals']['dev']}  map={gsm['totals']['map']} "
          f"(of {gsm['test_available']} test items)")

    if args.dry_run:
        print("\nDRY RUN: nothing written. Re-run without --dry-run to freeze.")
        return 0

    files = ["mmlu_dev.jsonl", "mmlu_map.jsonl", "mmlu_router.jsonl",
             "gsm8k_dev.jsonl", "gsm8k_map.jsonl"]
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": dc.SEED,
        "source": "huggingface datasets-server rows API",
        "mmlu_source": dc.MMLU_HF_PATH,
        "gsm8k_source": f"{dc.GSM8K_HF_PATH}:{dc.GSM8K_HF_CONFIG}",
        "mmlu_per_subject": mmlu["per_subject"],
        "mmlu_totals": mmlu["totals"],
        "gsm8k": gsm,
        "files": {f: _sha256(os.path.join(FROZEN_DIR, f)) for f in files},
    }
    with open(os.path.join(FROZEN_DIR, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nFrozen. Wrote {len(files)} split files + MANIFEST.json to data/frozen/")
    print("Commit data/frozen/ before running any benchmark call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
