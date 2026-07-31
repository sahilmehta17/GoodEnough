#!/usr/bin/env python3
"""
goodenough Day 0 gate.

Three checks that must pass before the project starts:
  A. Local throughput. Can a full pass finish in a reasonable window?
  B. Hosted acceptance. Does the provider behave as documented?
  C. Token budget. Does the run fit inside the daily ceiling, and over how many days?

Zero dependencies. Standard library only.

Prerequisites
-------------
1. llama-server running:
     llama-server -hf Qwen/Qwen3-1.7B-GGUF:Q8_0 --port 8080 --seed 42
2. Groq API key:
     Windows PowerShell:  $env:GROQ_API_KEY="gsk_..."
     bash:                export GROQ_API_KEY=gsk_...

Usage
-----
     python day0_gate.py
     python day0_gate.py --hosted-calls 100     # full acceptance test
     python day0_gate.py --skip-hosted          # local only
"""

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

LOCAL_URL = "http://localhost:8080/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
SEED = 42

# Groq free plan, llama-3.3-70b-versatile, verified 2026-07-30.
GROQ_TPD = 100_000
GROQ_RPD = 1_000
GROQ_RPM = 30

# Planned run: 8 slices x 100 MMLU + 150 GSM8K + ~150 dev items.
PLANNED_MMLU_ITEMS = 800
PLANNED_GSM8K_ITEMS = 150
PLANNED_DEV_ITEMS = 150

MCQ_SUFFIX = (
    'Please show your choice in the answer field with only the choice letter, '
    'e.g., "answer": "C".'
)

# MMLU-shaped probes. Deliberately varied in length.
PROBES = [
    "Which of the following is the capital of Australia?\nA. Sydney\nB. Melbourne\nC. Canberra\nD. Perth",
    "The Tropic of Cancer lies at approximately which latitude?\nA. 0 degrees\nB. 23.5 degrees N\nC. 23.5 degrees S\nD. 66.5 degrees N",
    "Which river flows through Cairo?\nA. Niger\nB. Congo\nC. Nile\nD. Zambezi",
    "A valid syllogism requires that:\nA. both premises are negative\nB. the middle term is distributed at least once\nC. the conclusion is universal\nD. all terms are singular",
    "If P implies Q and Q is false, what follows?\nA. P is true\nB. P is false\nC. nothing follows\nD. Q implies P",
]

LOCAL_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
    "seed": SEED,
    "max_tokens": 64,
}

HOSTED_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "seed": SEED,
    "max_completion_tokens": 64,
}


def load_dotenv(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def post(url, payload, headers=None, timeout=180):
    body = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            return time.perf_counter() - t0, r.status, dict(r.headers), json.loads(raw)
    except urllib.error.HTTPError as e:
        return time.perf_counter() - t0, e.code, dict(e.headers), {"error": e.read().decode()[:400]}
    except Exception as e:
        return time.perf_counter() - t0, 0, {}, {"error": repr(e)}


def build_messages(probe):
    return [{"role": "user", "content": f"{probe}\n\n{MCQ_SUFFIX}"}]


def build_payload(probe, provider):
    if provider == "local":
        return {
            "model": "local",
            "messages": build_messages(probe),
            **LOCAL_SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    if provider == "hosted":
        return {
            "model": GROQ_MODEL,
            "messages": build_messages(probe),
            **HOSTED_SAMPLING,
        }
    raise ValueError(f"unknown provider: {provider}")


# ---------------------------------------------------------------- check A

def check_local(n_calls):
    print("\n" + "=" * 68)
    print("CHECK A: local throughput")
    print("=" * 68)

    lats, out_toks, in_toks, failures = [], [], [], 0

    print(f"  warm-up (3 calls, discarded)")
    for i in range(3):
        post(LOCAL_URL, build_payload(PROBES[i % len(PROBES)], "local"))

    print(f"  measuring {n_calls} calls")
    for i in range(n_calls):
        probe = PROBES[i % len(PROBES)]
        lat, status, _, data = post(LOCAL_URL, build_payload(probe, "local"))
        if status != 200 or "choices" not in data:
            failures += 1
            print(f"    [{i+1}] FAIL status={status} {str(data)[:120]}")
            continue
        lats.append(lat)
        u = data.get("usage", {})
        in_toks.append(u.get("prompt_tokens", 0))
        out_toks.append(u.get("completion_tokens", 0))
        if i == 0:
            txt = data["choices"][0]["message"]["content"]
            print(f"    sample response: {txt[:160]!r}")
        sys.stdout.write(f"\r    {i+1}/{n_calls}  last={lat:.2f}s")
        sys.stdout.flush()
    print()

    if not lats:
        print("  RESULT: FAIL. No successful local calls. Is llama-server running on :8080?")
        return None

    med = statistics.median(lats)
    mean_out = statistics.mean(out_toks) if out_toks else 0
    tps = mean_out / med if med else 0
    total_items = PLANNED_MMLU_ITEMS + PLANNED_GSM8K_ITEMS + PLANNED_DEV_ITEMS
    est_hours = med * total_items / 3600

    print(f"\n  failures:            {failures}/{n_calls}")
    print(f"  median latency:      {med:.2f}s")
    print(f"  mean input tokens:   {statistics.mean(in_toks):.0f}")
    print(f"  mean output tokens:  {mean_out:.0f}")
    print(f"  generation tok/s:    {tps:.1f}")
    print(f"  est. full pass:      {est_hours:.1f}h for {total_items} items")

    if est_hours > 4:
        print(f"\n  RESULT: WARN. Full pass exceeds 4h. Cut to 6 slices BEFORE starting.")
    else:
        print(f"\n  RESULT: PASS.")
    return {"median_latency": med, "tps": tps, "est_hours": est_hours,
            "mean_in": statistics.mean(in_toks), "mean_out": mean_out}


# ---------------------------------------------------------------- check B

def check_hosted(n_calls, api_key):
    print("\n" + "=" * 68)
    print("CHECK B: hosted acceptance")
    print("=" * 68)

    headers = {"Authorization": f"Bearer {api_key}"}
    lats, in_toks, out_toks = [], [], []
    failures, rate_limited = 0, 0
    last_headers, model_returned, seed_ok = {}, None, None

    interval = 60.0 / GROQ_RPM + 0.15
    print(f"  {n_calls} calls, pacing at {interval:.2f}s to stay under {GROQ_RPM} RPM")

    for i in range(n_calls):
        probe = PROBES[i % len(PROBES)]
        payload = build_payload(probe, "hosted")
        lat, status, hdrs, data = post(GROQ_URL, payload, headers)

        if status == 429:
            rate_limited += 1
            print(f"\n    [{i+1}] 429 rate limited. retry-after={hdrs.get('retry-after')}")
            time.sleep(float(hdrs.get("retry-after", 5)))
            continue
        if status != 200 or "choices" not in data:
            failures += 1
            print(f"\n    [{i+1}] FAIL status={status} {str(data)[:200]}")
            if failures >= 3:
                print("  aborting after 3 failures")
                break
            continue

        last_headers = hdrs
        lats.append(lat)
        u = data.get("usage", {})
        in_toks.append(u.get("prompt_tokens", 0))
        out_toks.append(u.get("completion_tokens", 0))
        if model_returned is None:
            model_returned = data.get("model")
            seed_ok = "seed" in str(data.get("system_fingerprint", "")) or True
            print(f"\n    sample: {data['choices'][0]['message']['content'][:120]!r}")
        sys.stdout.write(f"\r    {i+1}/{n_calls}  last={lat:.2f}s")
        sys.stdout.flush()
        time.sleep(interval)
    print()

    if not lats:
        print("  RESULT: FAIL. No successful hosted calls.")
        return None

    print(f"\n  failures:            {failures}")
    print(f"  rate limited:        {rate_limited}")
    print(f"  median latency:      {statistics.median(lats):.2f}s")
    print(f"  mean input tokens:   {statistics.mean(in_toks):.0f}")
    print(f"  mean output tokens:  {statistics.mean(out_toks):.0f}")
    print(f"  model requested:     {GROQ_MODEL}")
    print(f"  model returned:      {model_returned}")
    if model_returned != GROQ_MODEL:
        print("    WARN: returned model id differs from requested. Record this in the manifest.")

    print("\n  rate limit headers:")
    for k in sorted(last_headers):
        if k.lower().startswith("x-ratelimit"):
            print(f"    {k}: {last_headers[k]}")

    print("\n  ACTION: confirm `seed` reproducibility manually by rerunning one prompt twice")
    print("          and diffing the output. Record yes/no in PREREGISTRATION.md.")
    print("\n  RESULT: PASS." if failures == 0 else "\n  RESULT: WARN, see failures above.")
    return {"median_latency": statistics.median(lats),
            "mean_in": statistics.mean(in_toks), "mean_out": statistics.mean(out_toks)}


# ---------------------------------------------------------------- check C

def check_budget(hosted):
    print("\n" + "=" * 68)
    print("CHECK C: token budget")
    print("=" * 68)

    if hosted:
        mmlu_per = hosted["mean_in"] + hosted["mean_out"]
    else:
        mmlu_per = 220
        print("  (hosted skipped, using 220 tok/item estimate)")

    gsm_per = mmlu_per + 300  # step-by-step output
    mmlu_tok = (PLANNED_MMLU_ITEMS + PLANNED_DEV_ITEMS) * mmlu_per
    gsm_tok = PLANNED_GSM8K_ITEMS * gsm_per
    total = mmlu_tok + gsm_tok
    days = total / GROQ_TPD
    req_days = (PLANNED_MMLU_ITEMS + PLANNED_DEV_ITEMS + PLANNED_GSM8K_ITEMS) / GROQ_RPD

    print(f"  MMLU + dev:          {PLANNED_MMLU_ITEMS + PLANNED_DEV_ITEMS} items x {mmlu_per:.0f} tok = {mmlu_tok:,.0f}")
    print(f"  GSM8K:               {PLANNED_GSM8K_ITEMS} items x {gsm_per:.0f} tok = {gsm_tok:,.0f}")
    print(f"  TOTAL:               {total:,.0f} tokens")
    print(f"\n  Groq TPD ceiling:    {GROQ_TPD:,}")
    print(f"  days (token-bound):  {days:.1f}")
    print(f"  days (request-bound):{req_days:.1f}")
    print(f"\n  BINDING CONSTRAINT:  {'tokens per day' if days > req_days else 'requests per day'}")
    print(f"\n  => the runner MUST be resumable across {max(1, int(days) + 1)} days.")
    if days > 5:
        print("\n  RESULT: WARN. Over 5 days. Cut to 6 slices.")
    else:
        print("\n  RESULT: PASS.")


# ---------------------------------------------------------------- main

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-calls", type=int, default=20)
    ap.add_argument("--hosted-calls", type=int, default=20)
    ap.add_argument("--skip-hosted", action="store_true")
    args = ap.parse_args()

    print("goodenough :: Day 0 gate")
    print("All three checks must pass before week 1 begins.")

    local = check_local(args.local_calls)

    hosted = None
    if not args.skip_hosted:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            print("\nCHECK B: SKIPPED. GROQ_API_KEY not set.")
        else:
            hosted = check_hosted(args.hosted_calls, key)

    check_budget(hosted)

    print("\n" + "=" * 68)
    print("NEXT")
    print("=" * 68)
    print("  1. Record the local file hash, llama.cpp commit, CPU model, thread")
    print("     count, RAM and seed in PREREGISTRATION.md section 9.")
    print("  2. Confirm hosted seed reproducibility and record yes/no.")
    print("  3. If any check WARNed, cut to 6 slices now, before writing code.")
    print("  4. Then and only then: day 1, one paired item end to end.\n")


if __name__ == "__main__":
    main()
