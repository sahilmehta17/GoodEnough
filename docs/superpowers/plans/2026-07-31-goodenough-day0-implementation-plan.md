# GoodEnough Day 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile the pinned experiment, freeze dataset membership, and produce passing local and hosted feasibility evidence before any benchmark evaluation.

**Architecture:** Keep the Day 0 gate dependency-free and split its behavior into pure request-building and parsing helpers plus small network orchestration functions. Use a separate Hugging Face dataset script to create a committed immutable manifest, so benchmark code consumes frozen item assignments rather than selecting data dynamically.

**Tech Stack:** Python 3.11+, standard-library `unittest`, llama.cpp Chat Completions API, Groq Chat Completions API, Hugging Face `datasets==5.0.0`, JSON and JSONL artifacts.

## Global Constraints

- Deadline remains three weeks; Day 0 must finish before benchmark calls.
- Semantic prompt text must be byte-identical across deployments.
- Local thinking is disabled with `chat_template_kwargs={"enable_thinking": false}`, never `/no_think` prompt text.
- Local MCQ sampling is temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, seed 42, and 64 maximum output tokens.
- Hosted MCQ sampling is temperature 0.7, top-p 0.8, seed 42, and 64 maximum completion tokens.
- Diagnostic probes and Day 0 feasibility calls are not benchmark evaluation data.
- Existing user changes in `scripts/day0_gate.py` and `scripts/probe.py` must be preserved.
- No dashboard, streaming, learned router, additional model, or unrelated refactor.

## File Map

- Modify `scripts/day0_gate.py`: request contracts, health wait, parsing checks, retries, metrics, seed check, and JSON report.
- Preserve `scripts/probe.py`: existing diagnostic evidence; do not delete or silently rewrite it during Day 0.
- Create `tests/test_day0_gate.py`: dependency-free unit tests for gate behavior.
- Modify `PREREGISTRATION.md`: actual Q4_K_M deployment, exact request settings, statistical construction, and amendment record.
- Create `requirements.txt`: pin only the dataset loader needed to freeze MMLU membership.
- Create `scripts/freeze_mmlu_splits.py`: deterministic split assignment and manifest writer.
- Create `tests/test_freeze_mmlu_splits.py`: synthetic-data tests for counts, disjointness, and determinism.
- Create `data/splits/mmlu_manifest.jsonl`: immutable item assignments.
- Create `data/splits/mmlu_manifest_summary.json`: dataset revision and split counts.
- Create `artifacts/day0/t4.json`, `artifacts/day0/t8.json`, and `artifacts/day0/final.json`: machine-readable feasibility evidence.

---

### Task 1: Freeze request construction with tests

**Files:**
- Create: `tests/test_day0_gate.py`
- Modify: `scripts/day0_gate.py:51-102`

**Interfaces:**
- Consumes: a probe string and provider name `local` or `hosted`.
- Produces: `build_messages(probe: str) -> list[dict[str, str]]` and `build_payload(probe: str, provider: str) -> dict`.

- [ ] **Step 1: Write failing request-contract tests**

```python
import unittest

from scripts import day0_gate


class RequestContractTests(unittest.TestCase):
    def test_semantic_prompt_does_not_contain_local_control_text(self):
        messages = day0_gate.build_messages("Question")
        self.assertNotIn("/no_think", messages[0]["content"])
        self.assertIn('"answer": "C"', messages[0]["content"])

    def test_local_payload_pins_qwen_sampling_and_disables_thinking(self):
        payload = day0_gate.build_payload("Question", "local")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["top_k"], 20)
        self.assertEqual(payload["min_p"], 0)
        self.assertEqual(payload["presence_penalty"], 1.5)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(
            payload["chat_template_kwargs"], {"enable_thinking": False}
        )

    def test_hosted_payload_uses_only_supported_pinned_fields(self):
        payload = day0_gate.build_payload("Question", "hosted")
        self.assertEqual(payload["model"], "llama-3.3-70b-versatile")
        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["top_p"], 0.8)
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["max_completion_tokens"], 64)
        for unsupported in (
            "top_k",
            "min_p",
            "presence_penalty",
            "chat_template_kwargs",
        ):
            self.assertNotIn(unsupported, payload)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and verify the current code fails**

Run: `python -m unittest tests.test_day0_gate.RequestContractTests -v`

Expected: failures because the current prompt contains `/no_think` and `build_payload` does not exist.

- [ ] **Step 3: Implement the exact request builders**

Replace the prompt and sampling definitions with:

```python
MCQ_SUFFIX = (
    'Please show your choice in the answer field with only the choice letter, '
    'e.g., "answer": "C".'
)

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
```

Update local and hosted call sites to call `build_payload` instead of assembling payloads inline.

- [ ] **Step 4: Run request-contract tests**

Run: `python -m unittest tests.test_day0_gate.RequestContractTests -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit the request contract**

```powershell
git add scripts/day0_gate.py tests/test_day0_gate.py
git commit -m "fix: pin Day 0 request contracts"
```

---

### Task 2: Make local readiness and parseability real gate conditions

**Files:**
- Modify: `tests/test_day0_gate.py`
- Modify: `scripts/day0_gate.py:84-159`

**Interfaces:**
- Consumes: local health responses and completion message content.
- Produces: `extract_mcq_answer(text: str) -> str | None` and `wait_for_local(...) -> dict`.

- [ ] **Step 1: Add failing parser and health-wait tests**

```python
from unittest.mock import Mock


class GateValidationTests(unittest.TestCase):
    def test_extracts_documented_answer_forms(self):
        self.assertEqual(day0_gate.extract_mcq_answer('answer: C'), "C")
        self.assertEqual(day0_gate.extract_mcq_answer('{"answer": "b"}'), "B")
        self.assertEqual(day0_gate.extract_mcq_answer('Answer : "D"'), "D")

    def test_rejects_empty_reasoning_only_and_unrelated_text(self):
        self.assertIsNone(day0_gate.extract_mcq_answer(""))
        self.assertIsNone(day0_gate.extract_mcq_answer("I considered every option"))

    def test_wait_for_local_survives_loading_then_returns_health(self):
        get_fn = Mock(
            side_effect=[
                RuntimeError("connection refused"),
                {"status": "loading"},
                {"status": "ok"},
            ]
        )
        result = day0_gate.wait_for_local(
            timeout=5,
            poll_interval=0,
            get_fn=get_fn,
            sleep_fn=lambda _: None,
        )
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(get_fn.call_count, 3)
```

- [ ] **Step 2: Run the new tests and verify failure**

Run: `python -m unittest tests.test_day0_gate.GateValidationTests -v`

Expected: failures because both helpers are undefined.

- [ ] **Step 3: Implement answer extraction and health polling**

```python
import re

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
```

Call `wait_for_local()` once at the start of `check_local`. During measured calls, count a response with empty content or no parsed answer as a failure and print its raw content safely.

- [ ] **Step 4: Run all gate tests**

Run: `python -m unittest tests.test_day0_gate -v`

Expected: all tests pass.

- [ ] **Step 5: Commit real gate validation**

```powershell
git add scripts/day0_gate.py tests/test_day0_gate.py
git commit -m "fix: validate local readiness and answers"
```

---

### Task 3: Add honest metrics and an atomic Day 0 artifact

**Files:**
- Modify: `tests/test_day0_gate.py`
- Modify: `scripts/day0_gate.py:107-306`
- Create at runtime: `artifacts/day0/t4.json`
- Create at runtime: `artifacts/day0/t8.json`
- Create at runtime: `artifacts/day0/final.json`

**Interfaces:**
- Consumes: per-call latency and usage records plus CLI metadata.
- Produces: `summarize_calls(calls: list[dict]) -> dict` and `write_json_atomic(path: str, value: dict) -> None`.

- [ ] **Step 1: Add failing summary and artifact tests**

```python
import json
import tempfile
from pathlib import Path


class ArtifactTests(unittest.TestCase):
    def test_summary_labels_end_to_end_throughput_honestly(self):
        summary = day0_gate.summarize_calls(
            [
                {"latency_seconds": 2.0, "input_tokens": 60, "output_tokens": 8},
                {"latency_seconds": 1.0, "input_tokens": 80, "output_tokens": 10},
            ]
        )
        self.assertEqual(summary["median_e2e_latency_seconds"], 1.5)
        self.assertEqual(summary["mean_output_tokens"], 9)
        self.assertEqual(summary["e2e_output_tokens_per_second"], 6)
        self.assertNotIn("generation_tokens_per_second", summary)

    def test_atomic_writer_creates_readable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            day0_gate.write_json_atomic(path, {"status": "pass"})
            self.assertEqual(json.loads(path.read_text()), {"status": "pass"})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
```

- [ ] **Step 2: Run artifact tests and verify failure**

Run: `python -m unittest tests.test_day0_gate.ArtifactTests -v`

Expected: failures because both helpers are undefined.

- [ ] **Step 3: Implement summaries and atomic output**

```python
from pathlib import Path


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
```

Add CLI options `--output` and `--thread-count`. Build one report containing schema version, timestamp, local `/props`, thread count, local summary, hosted summary or skipped status, budget estimate, and overall status. Replace the printed `generation tok/s` label with `end-to-end output tok/s`.

Increase planned request accounting to include `PLANNED_ROUTER_ITEMS = 150` and include it in both token and request estimates.

- [ ] **Step 4: Run all gate tests and inspect help**

Run: `python -m unittest tests.test_day0_gate -v`

Expected: all tests pass.

Run: `python scripts/day0_gate.py --help`

Expected: help includes `--output` and `--thread-count`.

- [ ] **Step 5: Commit artifact support**

```powershell
git add scripts/day0_gate.py tests/test_day0_gate.py
git commit -m "feat: record reproducible Day 0 evidence"
```

---

### Task 4: Verify hosted retries and seed behavior

**Files:**
- Modify: `tests/test_day0_gate.py`
- Modify: `scripts/day0_gate.py:164-233`

**Interfaces:**
- Consumes: the existing `post` result tuple and response headers.
- Produces: `post_with_retry(...)` and a hosted report containing repeatability and system fingerprints.

- [ ] **Step 1: Add a failing same-item retry test**

```python
class HostedReliabilityTests(unittest.TestCase):
    def test_rate_limit_retries_the_same_payload(self):
        calls = []

        def post_fn(url, payload, headers, timeout=180):
            calls.append(payload)
            if len(calls) == 1:
                return 0.1, 429, {"retry-after": "0"}, {"error": "limited"}
            return 0.2, 200, {}, {
                "model": day0_gate.GROQ_MODEL,
                "choices": [{"message": {"content": "answer: C"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 4},
                "system_fingerprint": "fp_1",
            }

        result = day0_gate.post_with_retry(
            day0_gate.GROQ_URL,
            {"item": "same"},
            {"Authorization": "Bearer test"},
            post_fn=post_fn,
            sleep_fn=lambda _: None,
        )
        self.assertEqual(result[1], 200)
        self.assertEqual(calls, [{"item": "same"}, {"item": "same"}])
```

- [ ] **Step 2: Run the retry test and verify failure**

Run: `python -m unittest tests.test_day0_gate.HostedReliabilityTests -v`

Expected: failure because `post_with_retry` is undefined.

- [ ] **Step 3: Implement bounded retry behavior**

```python
TRANSIENT_STATUSES = {429, 500, 502, 503, 504}


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
```

Use this helper for hosted acceptance calls. Make two identical requests for the first probe and record:

```python
{
    "same_content": first_content == second_content,
    "first_system_fingerprint": first.get("system_fingerprint"),
    "second_system_fingerprint": second.get("system_fingerprint"),
}
```

Repeatability is recorded as evidence, not forced to true. Delete the existing hard-coded `or True` expression.

- [ ] **Step 4: Run all gate tests**

Run: `python -m unittest tests.test_day0_gate -v`

Expected: all tests pass.

- [ ] **Step 5: Commit hosted reliability behavior**

```powershell
git add scripts/day0_gate.py tests/test_day0_gate.py
git commit -m "fix: verify hosted retries and repeatability"
```

---

### Task 5: Reconcile and amend the pre-registration

**Files:**
- Modify: `PREREGISTRATION.md:75-110`

**Interfaces:**
- Consumes: the approved design and verified running-server metadata.
- Produces: a complete pinned deployment contract and dated amendment entry.

- [ ] **Step 1: Replace the obsolete local configuration**

Use this exact configuration block, with the thread count updated in Task 7 from the measured winner:

```text
model repo:     unsloth/Qwen3-1.7B-GGUF
model file:     Qwen3-1.7B-Q4_K_M.gguf
quantization:   Q4_K_M
file hash:      B139949C5BD74937AD8ED8C8CF3D9FFB1E99C866C823204DC42C0D91FA181897
runtime:        llama.cpp b10173-e9fa0781f
backend:        CPU, 0 GPU layers
context:        4096
parallel slots: 1
mode:           non-thinking via chat_template_kwargs enable_thinking=false
sampling:       temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5
seed:           42
max tokens:     MMLU 64; GSM8K 512 if retained
hardware:       Intel Core Ultra 7 258V
```

Add a hosted block that pins temperature 0.7, top-p 0.8, seed 42, and 64 maximum completion tokens for MMLU. State that Groq determinism is best effort and the system fingerprint is recorded.

- [ ] **Step 2: Define the exact one-sided statistical bounds**

Replace the vague exact-interval sentence with the approved Clopper-Pearson and Bonferroni construction from the design document. State explicitly that equality with `-delta` is inconclusive because the hypothesis requires the bound to exceed the margin.

- [ ] **Step 3: Add the amendment record**

Append:

```markdown
| 2026-07-31 | Replaced unavailable official Q8_0 deployment with verified Unsloth Q4_K_M deployment; disabled thinking through the chat template; fully specified hosted sampling and exact interval construction | Day 0 probes showed the official Q8 configuration was not the deployment that produced valid non-thinking output. These probes were feasibility diagnostics, not benchmark evaluation data. | No benchmark results exist; future results use the amended configuration. |
```

- [ ] **Step 4: Scan for stale configuration and placeholders**

Run: `rg -n "Q8_0|/no_think|<fill|pending" PREREGISTRATION.md scripts/day0_gate.py`

Expected: no obsolete configuration or placeholder remains. A textual statement saying prompt text must not contain `/no_think` is permitted.

- [ ] **Step 5: Commit the amended contract**

```powershell
git add PREREGISTRATION.md
git commit -m "docs: amend pinned evaluation configuration"
```

---

### Task 6: Generate the immutable MMLU manifest

**Files:**
- Create: `requirements.txt`
- Create: `scripts/freeze_mmlu_splits.py`
- Create: `tests/test_freeze_mmlu_splits.py`
- Create at runtime: `data/splits/mmlu_manifest.jsonl`
- Create at runtime: `data/splits/mmlu_manifest_summary.json`

**Interfaces:**
- Consumes: dictionaries shaped as `subject -> source split -> list of MMLU rows`.
- Produces: `assign_splits(pools: dict) -> list[dict]` and two committed manifest artifacts.

- [ ] **Step 1: Pin the dataset loader**

Create `requirements.txt`:

```text
datasets==5.0.0
```

- [ ] **Step 2: Write failing deterministic-assignment tests**

```python
import unittest

from scripts.freeze_mmlu_splits import SUBJECTS, assign_splits


def row(subject, index):
    return {
        "question": f"{subject} question {index}",
        "choices": ["A", "B", "C", "D"],
        "answer": index % 4,
    }


class SplitAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.pools = {
            subject: {
                "dev": [row(subject, i) for i in range(5)],
                "validation": [row(subject, i + 5) for i in range(25)],
                "test": [row(subject, i + 100) for i in range(130)],
            }
            for subject in SUBJECTS
        }

    def test_assignments_are_deterministic_disjoint_and_sized(self):
        first = assign_splits(self.pools)
        second = assign_splits(self.pools)
        self.assertEqual(first, second)
        self.assertEqual(len({item["item_id"] for item in first}), len(first))
        self.assertEqual(sum(i["project_split"] == "dev" for i in first), 150)
        self.assertEqual(sum(i["project_split"] == "router" for i in first), 150)
        for subject in SUBJECTS:
            map_count = sum(
                i["project_split"] == "map" and i["subject"] == subject
                for i in first
            )
            self.assertLessEqual(map_count, 100)
```

- [ ] **Step 3: Run the split test and verify failure**

Run: `python -m unittest tests.test_freeze_mmlu_splits -v`

Expected: import failure because the split script does not exist.

- [ ] **Step 4: Implement deterministic item identity and assignment**

Use these constants and signatures:

```python
DATASET_ID = "cais/mmlu"
DATASET_REVISION = "ea9505acd1c0964d6d5c00d208d0fdf8e4810e6d"
SPLIT_SEED = 42
DEV_TARGET = 150
ROUTER_TARGET = 150
MAP_CAP_PER_SUBJECT = 100
SUBJECTS = (
    "high_school_geography",
    "formal_logic",
    "nutrition",
    "marketing",
    "miscellaneous",
    "college_mathematics",
    "professional_law",
    "high_school_psychology",
)


def stable_rank(*parts):
    text = "|".join(str(part) for part in (SPLIT_SEED, *parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def item_id(subject, source_split, source_index, row):
    canonical = json.dumps(
        {
            "dataset_revision": DATASET_REVISION,
            "subject": subject,
            "source_split": source_split,
            "source_index": source_index,
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
```

`assign_splits` must:

1. Select ten ranked dev or validation rows per subject.
2. Fill the remaining dev target from all unused dev and validation rows by stable rank.
3. Allocate 19 router items to the first six alphabetically sorted subjects and 18 to the final two, totaling 150.
4. Select router items from test by stable rank.
5. Select up to 100 remaining test items per subject for map.
6. Emit records ordered by project split, subject, and stable rank.

Each record stores dataset ID, revision, subject, source split, source index, item ID, project split, project order, and a SHA-256 content hash. It does not duplicate question text or answers.

- [ ] **Step 5: Run split tests**

Run: `python -m unittest tests.test_freeze_mmlu_splits -v`

Expected: all tests pass.

- [ ] **Step 6: Add the Hugging Face loader and writers**

For each subject, load dev, validation, and test at the pinned revision with:

```python
load_dataset(DATASET_ID, subject, revision=DATASET_REVISION)
```

Write JSONL atomically, then write a summary containing dataset ID, revision, seed, subject list, total counts by project split, and per-subject counts.

- [ ] **Step 7: Install and generate the frozen artifacts**

Run: `python -m pip install -r requirements.txt`

Expected: `datasets==5.0.0` installs successfully.

Run: `python scripts/freeze_mmlu_splits.py`

Expected: manifest and summary files are created, with 150 dev items and 150 router items.

- [ ] **Step 8: Re-run the generator and prove byte stability**

Run twice: `python scripts/freeze_mmlu_splits.py`

Run: `git diff --exit-code -- data/splits/mmlu_manifest.jsonl data/splits/mmlu_manifest_summary.json`

Expected: no diff after regeneration.

- [ ] **Step 9: Commit the frozen membership**

```powershell
git add requirements.txt scripts/freeze_mmlu_splits.py tests/test_freeze_mmlu_splits.py data/splits/mmlu_manifest.jsonl data/splits/mmlu_manifest_summary.json
git commit -m "feat: freeze MMLU evaluation splits"
```

---

### Task 7: Execute and freeze Day 0 evidence

**Files:**
- Modify: `PREREGISTRATION.md`
- Create: `artifacts/day0/t4.json`
- Create: `artifacts/day0/t8.json`
- Create: `artifacts/day0/final.json`

**Interfaces:**
- Consumes: the repaired gate, running llama.cpp server, and existing `.env` Groq key.
- Produces: selected thread count, hosted acceptance evidence, and a complete frozen Day 0 configuration.

- [ ] **Step 1: Run the complete unit suite**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass without contacting either model endpoint.

- [ ] **Step 2: Run the four-thread local gate**

Start llama.cpp:

```powershell
llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M --port 8080 --seed 42 --ctx-size 4096 -ngl 0 -t 4 --jinja -np 1
```

In the project terminal:

```powershell
python scripts/day0_gate.py --skip-hosted --thread-count 4 --output artifacts/day0/t4.json
```

Expected: zero request failures, zero parse failures, and overall local status pass.

- [ ] **Step 3: Run the eight-thread local gate**

Restart llama.cpp with `-t 8`, keeping every other argument identical.

Run:

```powershell
python scripts/day0_gate.py --skip-hosted --thread-count 8 --output artifacts/day0/t8.json
```

Expected: zero request failures, zero parse failures, and overall local status pass.

- [ ] **Step 4: Select the pinned thread count**

Select the configuration with lower median end-to-end latency when both have zero failures. If their medians differ by less than 5%, select four threads to reduce contention. Record both measurements and the rule in `PREREGISTRATION.md`.

- [ ] **Step 5: Run hosted acceptance once with the winner**

Keep the winning local server running and execute:

```powershell
python scripts/day0_gate.py --thread-count WINNER --output artifacts/day0/final.json
```

Replace `WINNER` with the measured integer, either 4 or 8.

Expected: local pass, hosted successful calls, requested and returned model IDs recorded, rate-limit headers captured when present, and repeatability evidence recorded without claiming guaranteed determinism.

- [ ] **Step 6: Finish the pre-registration configuration**

Record the winning thread count, machine RAM reported by Windows Settings or Task Manager, the two local median latencies, hosted returned model ID, hosted system fingerprint, and repeatability result. Do not change hypotheses, subjects, margin, splits, or scoring rules.

- [ ] **Step 7: Run final verification**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

Run: `rg -n "<fill|pending|Q8_0|mode:.*no_think" PREREGISTRATION.md scripts/day0_gate.py`

Expected: no unresolved configuration remains.

Run: `git status --short`

Expected: only the pre-existing diagnostic `scripts/probe.py` may remain untracked; every Day 0 deliverable is tracked.

- [ ] **Step 8: Commit Day 0 evidence**

```powershell
git add PREREGISTRATION.md artifacts/day0/t4.json artifacts/day0/t8.json artifacts/day0/final.json
git commit -m "results: freeze Day 0 feasibility evidence"
```

Day 0 is complete only after this commit. The next plan begins with one dev item processed end to end through local inference, hosted inference, raw storage, parsing, and paired scoring.
