"""
Single source of truth for the frozen experiment configuration.

Every value here is pinned in PREREGISTRATION.md section 9. Nothing in the
codebase should hardcode a model name, endpoint, or sampling parameter. If a
value needs to change, it changes here and in the pre-registration, and if
evaluation data already exists the change is an amendment, not an edit.
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------
# Local deployment (runs on the user's machine, llama.cpp / llama-server)
# --------------------------------------------------------------------------

LOCAL_BASE_URL = "http://localhost:8080"
LOCAL_CHAT_URL = f"{LOCAL_BASE_URL}/v1/chat/completions"
LOCAL_HEALTH_URL = f"{LOCAL_BASE_URL}/health"

# Identifier we record in the store. The local server does not care about the
# model field, but we want an honest label in every row.
LOCAL_MODEL_ID = "unsloth/Qwen3-1.7B-GGUF:Q4_K_M"

# Qwen's recommended non-thinking sampling. Greedy decoding is explicitly
# discouraged by the model card, so temperature is nonzero on purpose.
LOCAL_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 1.5,
    "seed": 42,
}

# The one deployment control that differs from hosted. It disables Qwen3's
# reasoning trace so the model answers directly. Passed inside the request body
# so the semantic prompt stays byte-identical to the hosted request.
LOCAL_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


# --------------------------------------------------------------------------
# Hosted deployment (Groq, free plan)
# --------------------------------------------------------------------------

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
HOSTED_MODEL_ID = "llama-3.3-70b-versatile"

# Groq uses max_completion_tokens rather than max_tokens. The reference model
# is not a reasoning model, so no thinking switch is sent.
HOSTED_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.8,
    "seed": 42,
}

# Verified free-plan limits for llama-3.3-70b-versatile (Groq docs, 2026-07-30).
# TPD is the binding constraint, which is why the full runner must be resumable.
GROQ_RPM = 30
GROQ_RPD = 1_000
GROQ_TPD = 100_000


# --------------------------------------------------------------------------
# Output length, pinned per task family
# --------------------------------------------------------------------------

MAX_TOKENS_MCQ = 512     # MMLU answers are a few tokens; this only caps runaways
MAX_TOKENS_MATH = 1024   # GSM8K needs room for step-by-step reasoning

SEED = 42


# --------------------------------------------------------------------------
# Pricing, for the cost report only. Not used by anything that runs a call.
# --------------------------------------------------------------------------

# Groq pricing for llama-3.3-70b-versatile, USD per 1M tokens.
# Read from https://console.groq.com/docs/models on 2026-08-06 (groq.com/pricing
# itself is a client-rendered page that did not yield pricing text to a fetch;
# console.groq.com/docs/models is Groq's own docs and states the same figures
# independently reported by third-party Groq pricing trackers on the same date).
HOSTED_PRICE_INPUT_PER_M = 0.59
HOSTED_PRICE_OUTPUT_PER_M = 0.79

# Local inference is zero incremental API spend. It is not economically free;
# say so in the report text.
LOCAL_PRICE_INPUT_PER_M = 0.0
LOCAL_PRICE_OUTPUT_PER_M = 0.0


# --------------------------------------------------------------------------
# Prompt templates, identical across deployments (PREREGISTRATION.md section 10)
# --------------------------------------------------------------------------

MCQ_INSTRUCTION = (
    'Please show your choice in the answer field with only the choice letter, '
    'e.g., "answer": "C".'
)

MATH_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def get_groq_api_key() -> str:
    """Read the Groq key from the environment. Never logged or printed."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Put it in .env (which is gitignored) or "
            "export it in the shell before running any hosted call."
        )
    return key


def load_dotenv(path: str = ".env") -> None:
    """
    Minimal .env loader so scripts pick up GROQ_API_KEY without a dependency.
    Values already in the environment win, so it never clobbers a real export.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
