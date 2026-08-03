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
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LOCAL_URL = "http://localhost:8080/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
SEED = 42

# Groq free plan, llama-3.3-70b-versatile, verified 2026-07-30.
GROQ_TPD = 100_000
GROQ_RPD = 1_000
GROQ_RPM = 30

# Planned run: 8 slices x 100 MMLU + 150 GSM8K + ~150 dev + 150 router items.
PLANNED_MMLU_ITEMS = 800
PLANNED_GSM8K_ITEMS = 150
PLANNED_DEV_ITEMS = 150
PLANNED_ROUTER_ITEMS = 150

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


TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


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


def post_with_retry(
    url,
    payload,
    headers,
    max_attempts=3,
    post_fn=post,
    sleep_fn=time.sleep,
):
    result = None
    for attempt in range(max_attempts):
        result = post_fn(url, payload, headers)
        _, status, response_headers, _ = result
        if status not in TRANSIENT_STATUSES:
            return result
        if attempt + 1 < max_attempts:
            delay = float(response_headers.get("retry-after", 2 ** attempt))
            sleep_fn(delay)
    return result


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


ANSWER_RE = re.compile(
    r'\banswer\s*["\']?\s*:\s*["\']?\s*([A-D])\b', re.IGNORECASE
)


def extract_mcq_answer(text):
    if not isinstance(text, str):
        return None
    match = ANSWER_RE.search(text)
    return match.group(1).upper() if match else None


def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_local(
    timeout=180,
    poll_interval=1,
    get_fn=None,
    sleep_fn=time.sleep,
):
    get_fn = get_fn or (lambda: get_json("http://localhost:8080/health"))
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            health = get_fn()
            if health.get("status") == "ok":
                return health
        except Exception as exc:
            last_error = exc
        sleep_fn(poll_interval)
    raise TimeoutError(f"local model did not become ready: {last_error!r}")


def summarize_calls(calls):
    latencies = [call["latency_seconds"] for call in calls]
    mean_output = statistics.mean(call["output_tokens"] for call in calls)
    median_latency = statistics.median(latencies)
    return {
        "successful_calls": len(calls),
        "median_e2e_latency_seconds": median_latency,
        "mean_input_tokens": statistics.mean(
            call["input_tokens"] for call in calls
        ),
        "mean_output_tokens": mean_output,
        "e2e_output_tokens_per_second": mean_output / median_latency,
    }


def write_json_atomic(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)


# ---------------------------------------------------------------- check A

def check_local(n_calls):
    print("\n" + "=" * 68)
    print("CHECK A: local throughput")
    print("=" * 68)

    print("  waiting for local /health")
    wait_for_local()

    try:
        props = get_json("http://localhost:8080/props")
    except Exception as exc:
        props = {"error": repr(exc)}

    calls = []
    failures = 0

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
        message = data["choices"][0].get("message", {})
        txt = message.get("content")
        if not extract_mcq_answer(txt):
            failures += 1
            print(f"    [{i+1}] FAIL unparseable content={txt!r}")
            continue
        u = data.get("usage", {})
        calls.append(
            {
                "latency_seconds": lat,
                "input_tokens": u.get("prompt_tokens", 0),
                "output_tokens": u.get("completion_tokens", 0),
            }
        )
        if i == 0:
            print(f"    sample response: {txt[:160]!r}")
        sys.stdout.write(f"\r    {i+1}/{n_calls}  last={lat:.2f}s")
        sys.stdout.flush()
    print()

    if not calls:
        print("  RESULT: FAIL. No successful local calls. Is llama-server running on :8080?")
        return {
            "status": "fail",
            "failures": failures,
            "requested_calls": n_calls,
            "props": props,
        }

    summary = summarize_calls(calls)
    total_items = (
        PLANNED_MMLU_ITEMS
        + PLANNED_GSM8K_ITEMS
        + PLANNED_DEV_ITEMS
        + PLANNED_ROUTER_ITEMS
    )
    est_hours = summary["median_e2e_latency_seconds"] * total_items / 3600
    status = "warn" if failures > 0 or est_hours > 4 else "pass"

    print(f"\n  failures:            {failures}/{n_calls}")
    print(f"  median latency:      {summary['median_e2e_latency_seconds']:.2f}s")
    print(f"  mean input tokens:   {summary['mean_input_tokens']:.0f}")
    print(f"  mean output tokens:  {summary['mean_output_tokens']:.0f}")
    print(f"  end-to-end output tok/s: {summary['e2e_output_tokens_per_second']:.1f}")
    print(f"  est. full pass:      {est_hours:.1f}h for {total_items} items")

    if status == "warn":
        if failures > 0:
            print(f"\n  RESULT: WARN. {failures} failed or unparseable call(s).")
        else:
            print(
                "\n  RESULT: WARN. Full pass exceeds 4h. Cut to 6 slices BEFORE starting."
            )
    else:
        print(f"\n  RESULT: PASS.")
    return {
        **summary,
        "status": status,
        "failures": failures,
        "requested_calls": n_calls,
        "est_hours": est_hours,
        "props": props,
    }

# ---------------------------------------------------------------- check B

def _hosted_message_content(data):
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    return message.get("content")


def check_hosted(n_calls, api_key):
    print("\n" + "=" * 68)
    print("CHECK B: hosted acceptance")
    print("=" * 68)

    headers = {"Authorization": f"Bearer {api_key}"}
    calls = []
    failures, rate_limited = 0, 0
    last_headers, model_returned = {}, None
    repeatability = None

    interval = 60.0 / GROQ_RPM + 0.15
    print(f"  {n_calls} calls, pacing at {interval:.2f}s to stay under {GROQ_RPM} RPM")

    for i in range(n_calls):
        probe = PROBES[i % len(PROBES)]
        payload = build_payload(probe, "hosted")
        attempts = 2 if i == 0 else 1
        attempt_results = []
        for _ in range(attempts):
            attempt_results.append(
                post_with_retry(GROQ_URL, payload, headers)
            )
            time.sleep(interval)

        if i == 0 and len(attempt_results) == 2:
            first_data = attempt_results[0][3]
            second_data = attempt_results[1][3]
            first_content = _hosted_message_content(first_data)
            second_content = _hosted_message_content(second_data)
            repeatability = {
                "same_content": first_content == second_content,
                "first_system_fingerprint": first_data.get("system_fingerprint"),
                "second_system_fingerprint": second_data.get("system_fingerprint"),
            }
            print(
                f"\n    repeatability: same_content={repeatability['same_content']} "
                f"fp=({repeatability['first_system_fingerprint']!r}, "
                f"{repeatability['second_system_fingerprint']!r})"
            )

        abort = False
        for lat, status, hdrs, data in attempt_results:
            if status == 429:
                rate_limited += 1
                failures += 1
                print(f"\n    [{i+1}] 429 after retries")
            elif status != 200 or "choices" not in data:
                failures += 1
                print(f"\n    [{i+1}] FAIL status={status} {str(data)[:200]}")
            else:
                last_headers = hdrs
                u = data.get("usage", {})
                calls.append(
                    {
                        "latency_seconds": lat,
                        "input_tokens": u.get("prompt_tokens", 0),
                        "output_tokens": u.get("completion_tokens", 0),
                    }
                )
                if model_returned is None:
                    model_returned = data.get("model")
                    content = _hosted_message_content(data) or ""
                    print(f"\n    sample: {content[:120]!r}")
                sys.stdout.write(f"\r    {i+1}/{n_calls}  last={lat:.2f}s")
                sys.stdout.flush()

            if failures >= 3:
                print("  aborting after 3 failures")
                abort = True
                break
        if abort:
            break
    print()

    if not calls:
        print("  RESULT: FAIL. No successful hosted calls.")
        return {
            "status": "fail",
            "failures": failures,
            "rate_limited": rate_limited,
            "requested_calls": n_calls,
            "model_requested": GROQ_MODEL,
            "repeatability": repeatability,
        }

    summary = summarize_calls(calls)
    status = "pass" if failures == 0 else "warn"
    rate_limit_headers = {
        k: last_headers[k]
        for k in sorted(last_headers)
        if k.lower().startswith("x-ratelimit")
    }

    print(f"\n  failures:            {failures}")
    print(f"  rate limited:        {rate_limited}")
    print(f"  median latency:      {summary['median_e2e_latency_seconds']:.2f}s")
    print(f"  mean input tokens:   {summary['mean_input_tokens']:.0f}")
    print(f"  mean output tokens:  {summary['mean_output_tokens']:.0f}")
    print(f"  model requested:     {GROQ_MODEL}")
    print(f"  model returned:      {model_returned}")
    if model_returned != GROQ_MODEL:
        print("    WARN: returned model id differs from requested. Record this in the manifest.")

    print("\n  rate limit headers:")
    for k, value in rate_limit_headers.items():
        print(f"    {k}: {value}")

    print("\n  repeatability evidence recorded; determinism is not required for PASS.")
    print("\n  RESULT: PASS." if status == "pass" else "\n  RESULT: WARN, see failures above.")
    return {
        **summary,
        "status": status,
        "failures": failures,
        "rate_limited": rate_limited,
        "requested_calls": n_calls,
        "model_requested": GROQ_MODEL,
        "model_returned": model_returned,
        "rate_limit_headers": rate_limit_headers,
        "repeatability": repeatability,
    }


# ---------------------------------------------------------------- check C

def check_budget(hosted):
    print("\n" + "=" * 68)
    print("CHECK C: token budget")
    print("=" * 68)

    if hosted and hosted.get("status") in ("pass", "warn"):
        mmlu_per = hosted["mean_input_tokens"] + hosted["mean_output_tokens"]
    else:
        mmlu_per = 220
        print("  (hosted skipped, using 220 tok/item estimate)")

    planned_mcq = PLANNED_MMLU_ITEMS + PLANNED_DEV_ITEMS + PLANNED_ROUTER_ITEMS
    gsm_per = mmlu_per + 300  # step-by-step output
    mmlu_tok = planned_mcq * mmlu_per
    gsm_tok = PLANNED_GSM8K_ITEMS * gsm_per
    total = mmlu_tok + gsm_tok
    days = total / GROQ_TPD
    req_days = (planned_mcq + PLANNED_GSM8K_ITEMS) / GROQ_RPD
    binding = "tokens per day" if days > req_days else "requests per day"
    status = "warn" if days > 5 else "pass"

    print(
        f"  MMLU + dev + router: {planned_mcq} items x {mmlu_per:.0f} tok = {mmlu_tok:,.0f}"
    )
    print(f"  GSM8K:               {PLANNED_GSM8K_ITEMS} items x {gsm_per:.0f} tok = {gsm_tok:,.0f}")
    print(f"  TOTAL:               {total:,.0f} tokens")
    print(f"\n  Groq TPD ceiling:    {GROQ_TPD:,}")
    print(f"  days (token-bound):  {days:.1f}")
    print(f"  days (request-bound):{req_days:.1f}")
    print(f"\n  BINDING CONSTRAINT:  {binding}")
    print(f"\n  => the runner MUST be resumable across {max(1, int(days) + 1)} days.")
    if status == "warn":
        print("\n  RESULT: WARN. Over 5 days. Cut to 6 slices.")
    else:
        print("\n  RESULT: PASS.")
    return {
        "status": status,
        "mmlu_tokens_per_item": mmlu_per,
        "planned_mcq_items": planned_mcq,
        "planned_router_items": PLANNED_ROUTER_ITEMS,
        "total_tokens": total,
        "token_bound_days": days,
        "request_bound_days": req_days,
        "binding": binding,
    }


def overall_status(local, hosted, budget):
    statuses = [local.get("status"), budget.get("status")]
    if hosted.get("status") not in ("skipped", None):
        statuses.append(hosted.get("status"))
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


# ---------------------------------------------------------------- main

def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-calls", type=int, default=20)
    ap.add_argument("--hosted-calls", type=int, default=20)
    ap.add_argument("--skip-hosted", action="store_true")
    ap.add_argument("--output", help="write a machine-readable Day 0 JSON report")
    ap.add_argument(
        "--thread-count",
        type=int,
        default=None,
        help="llama.cpp thread count used for this run",
    )
    args = ap.parse_args()

    print("goodenough :: Day 0 gate")
    print("All three checks must pass before week 1 begins.")

    local = check_local(args.local_calls)

    hosted = {"status": "skipped", "reason": "skip-hosted"}
    if not args.skip_hosted:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            print("\nCHECK B: SKIPPED. GROQ_API_KEY not set.")
            hosted = {"status": "skipped", "reason": "GROQ_API_KEY not set"}
        else:
            hosted = check_hosted(args.hosted_calls, key)

    budget = check_budget(hosted)
    report = {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "thread_count": args.thread_count,
        "local": local,
        "hosted": hosted,
        "budget": budget,
        "status": overall_status(local, hosted, budget),
    }
    if args.output:
        write_json_atomic(args.output, report)
        print(f"\n  wrote report: {args.output}")

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
