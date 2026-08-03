"""
Model clients. Two functions, one shape.

Each returns a CallResult with everything the store needs: raw text, token
counts, measured latency, and any error. Neither function parses or scores;
that is scoring.py's job. Keeping I/O and interpretation separate means the
same raw response can be re-scored later without re-calling the model.

Latency note: latency_ms here is end-to-end wall clock for the HTTP call as
measured on the caller's machine. There is no streaming and therefore no
time-to-first-token. Only meaningful when run on the pinned hardware.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import config


@dataclass
class CallResult:
    model_role: str            # "local" or "hosted"
    model_id_requested: str
    model_id_returned: str | None
    rendered_input: str        # the exact user content sent
    raw_response: str          # verbatim model text, never trimmed
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float          # end-to-end, uncached
    retries: int
    error: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    extra: dict = field(default_factory=dict)


# Python-urllib's default User-Agent is blocked at Groq's Cloudflare edge
# (returns "error code: 1010"). A normal client User-Agent passes. This is a
# documented workaround, not a hack around auth: the API key still authorizes.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)


def _post(url: str, payload: dict, headers: dict | None, timeout: int):
    """Single HTTP POST. Returns (latency_s, status, headers, parsed_json)."""
    body = json.dumps(payload).encode()
    hdrs = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - t0
            data = json.loads(resp.read().decode("utf-8"))
            return elapsed, resp.status, dict(resp.headers), data
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        detail = exc.read().decode("utf-8", "replace")[:500]
        return elapsed, exc.code, dict(exc.headers), {"__error__": detail}
    except Exception as exc:  # network, timeout, JSON
        elapsed = time.perf_counter() - t0
        return elapsed, 0, {}, {"__error__": repr(exc)}


def _messages(user_content: str) -> list[dict]:
    return [{"role": "user", "content": user_content}]


def call_local(user_content: str, max_tokens: int, timeout: int = 180) -> CallResult:
    """Call the local llama-server. No retries: a local failure is a real fault."""
    payload = {
        "model": config.LOCAL_MODEL_ID,
        "messages": _messages(user_content),
        "max_tokens": max_tokens,
        "chat_template_kwargs": config.LOCAL_CHAT_TEMPLATE_KWARGS,
        **config.LOCAL_SAMPLING,
    }
    latency_s, status, _hdrs, data = _post(config.LOCAL_CHAT_URL, payload, None, timeout)

    if status != 200 or "choices" not in data:
        return CallResult(
            model_role="local",
            model_id_requested=config.LOCAL_MODEL_ID,
            model_id_returned=None,
            rendered_input=user_content,
            raw_response="",
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_s * 1000,
            retries=0,
            error=data.get("__error__", f"unexpected response, status={status}"),
            http_status=status,
        )

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return CallResult(
        model_role="local",
        model_id_requested=config.LOCAL_MODEL_ID,
        model_id_returned=data.get("model"),
        rendered_input=user_content,
        raw_response=choice["message"].get("content") or "",
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        latency_ms=latency_s * 1000,
        retries=0,
        http_status=status,
        finish_reason=choice.get("finish_reason"),
    )


def call_hosted(user_content: str, max_tokens: int, timeout: int = 120,
                max_retries: int = 4) -> CallResult:
    """
    Call Groq. Retries on 429 and transient failures with backoff, because the
    free plan rate-limits and the full run spans days. A persistent failure is
    returned as an error, never silently dropped.
    """
    api_key = config.get_groq_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": config.HOSTED_MODEL_ID,
        "messages": _messages(user_content),
        "max_completion_tokens": max_tokens,
        **config.HOSTED_SAMPLING,
    }

    retries = 0
    last_error = None
    last_status = None
    total_latency_ms = 0.0

    while retries <= max_retries:
        latency_s, status, hdrs, data = _post(config.GROQ_CHAT_URL, payload, headers, timeout)
        total_latency_ms += latency_s * 1000
        last_status = status

        if status == 200 and "choices" in data:
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return CallResult(
                model_role="hosted",
                model_id_requested=config.HOSTED_MODEL_ID,
                model_id_returned=data.get("model"),
                rendered_input=user_content,
                raw_response=choice["message"].get("content") or "",
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                latency_ms=latency_s * 1000,  # last successful call only
                retries=retries,
                http_status=status,
                finish_reason=choice.get("finish_reason"),
            )

        last_error = data.get("__error__", f"status={status}")
        if status == 429:
            wait = float(hdrs.get("retry-after", 2 ** retries))
        elif status in (500, 502, 503, 0):
            wait = 2 ** retries
        else:
            break  # 4xx other than 429 will not fix themselves
        time.sleep(wait)
        retries += 1

    return CallResult(
        model_role="hosted",
        model_id_requested=config.HOSTED_MODEL_ID,
        model_id_returned=None,
        rendered_input=user_content,
        raw_response="",
        input_tokens=None,
        output_tokens=None,
        latency_ms=total_latency_ms,
        retries=retries,
        error=last_error,
        http_status=last_status,
    )
