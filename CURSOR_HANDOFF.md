# GoodEnough Cursor Handover

Updated: 2026-07-31

## Read This First

This document transfers the complete working context for GoodEnough into Cursor. Do not restart project selection, broaden the scope, or replace the approved design.

Read these files completely before editing code, in this order:

1. `CURSOR_HANDOFF.md`
2. `CLAUDE.md`
3. `PREREGISTRATION.md`
4. `docs/superpowers/specs/2026-07-31-goodenough-learning-implementation-design.md`
5. `docs/superpowers/plans/2026-07-31-goodenough-day0-implementation-plan.md`
6. `scripts/day0_gate.py`
7. `scripts/probe.py`

The design document is the approved system design. The Day 0 plan is the executable plan. `CLAUDE.md` contains the original scope guardrails, but its obsolete Q8 assumptions must be reconciled through the approved pre-registration amendment.

## User and Collaboration Style

The user is Sahil Mehta, an AI and full-stack engineer with about 1.5 years of experience. His working stack is Node and TypeScript on Express, React with TypeScript and Vite, and Python with FastAPI.

GoodEnough is being built for two reasons:

1. Add a rigorous applied AI engineering project to Sahil's resume.
2. Learn the underlying topics through implementation.

The deadline is strictly three weeks. Work quickly and keep explanations attached to the code being written.

The agreed collaboration mode is:

1. Briefly explain the concept required by the current task.
2. Explain why the selected design fits GoodEnough.
3. Implement alongside Sahil.
4. Run a focused test.
5. Explain the observed result and continue.

Do not provide detached tutorials or generate the entire project without explaining it. Do not repeatedly ask Sahil to choose among already-settled options.

Hard writing preference: never use em dashes.

## Project Question

GoodEnough asks:

> Where does a quantized 1.7B model running on a consumer laptop CPU stay within a predeclared accuracy margin of a fixed hosted 70B model?

It evaluates subject-level capability, uncertainty, latency, token use, and routing policies. It is a case study of two pinned deployment configurations, not a claim about all local and hosted models.

The primary non-inferiority margin is 10 percentage points. Sensitivity is reported at 5 and 15 points.

Primary MMLU subjects:

- `high_school_geography`
- `formal_logic`

Exploratory subjects:

- `nutrition`
- `marketing`
- `miscellaneous`
- `college_mathematics`
- `professional_law`
- `high_school_psychology`

GSM8K is secondary and the first major feature cut if the schedule slips.

## Why This Project Was Chosen

Earlier research concluded that Sahil's portfolio already contains several application-layer agent and retrieval projects. His missing signals are measured latency, cost awareness, systems behavior, and rigorous deployment comparison.

The same research found that his job-search bottleneck is primarily top of funnel and recommended resume positioning and referrals over another large project. Sahil clarified that GoodEnough is specifically a learning and resume artifact, not a replacement for job-search activity.

GoodEnough is more aligned with applied and product AI work than the two abandoned ideas:

- `least-priv-cloud`, an RL cloud environment, became too complex and targeted frontier-lab work.
- A speculative-decoding implementation targeted inference infrastructure roles rather than product AI roles.

Do not reopen either project direction.

## Current Git State

Active branch:

```text
codex/goodenough-day0
```

Recent commits:

```text
5323615 Plan GoodEnough Day 0 implementation
6a400cd Document GoodEnough learning implementation design
f159b8c Pre-registration, project constraints, Day 0 gate
```

The branch was created from `main` after the design and plan commits. `origin/main` is behind those two documentation commits.

Existing uncommitted user work must be preserved:

- `scripts/day0_gate.py` is modified with a small `.env` loader.
- `scripts/probe.py` is an untracked diagnostic script that compared thinking-control variants.

Do not reset, discard, overwrite, or silently absorb those changes. Modify `scripts/day0_gate.py` as planned while retaining the `.env` loader. Preserve `scripts/probe.py` unless Sahil explicitly chooses to remove or commit it later.

## Verified Local Deployment

The working local deployment is:

```text
model repo:     unsloth/Qwen3-1.7B-GGUF
model file:     Qwen3-1.7B-Q4_K_M.gguf
quantization:   Q4_K_M
file size:      1,107,409,472 bytes
sha256:         B139949C5BD74937AD8ED8C8CF3D9FFB1E99C866C823204DC42C0D91FA181897
llama.cpp:      b10173-e9fa0781f
backend:        CPU, -ngl 0
context:        4096
parallel slots: 1
CPU:            Intel Core Ultra 7 258V
seed:           42
chat template:  Jinja
thinking:       disabled with chat_template_kwargs enable_thinking=false
```

RAM still needs to be recorded from Windows Settings or Task Manager.

Known server command for four threads:

```powershell
llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M --port 8080 --seed 42 --ctx-size 4096 -ngl 0 -t 4 --jinja -np 1
```

The eight-thread comparison changes only `-t 4` to `-t 8`.

The `.env` file contains `GROQ_API_KEY`, is covered by `.gitignore`, and is not tracked. Never print, stage, or copy its value.

## Diagnostic Evidence Already Observed

The first gate attempt began while llama.cpp was still loading. It received a 503 and was interrupted. This was a startup race, not a model failure.

A later local-only run completed successfully:

```text
successful calls:       20 of 20
parseable sample:       answer: C
median client latency:  2.34 seconds
mean input tokens:      70
mean output tokens:     8
projected 1,100 items:  0.7 hours using that short-MCQ estimate
estimated hosted use:   287,000 tokens over about 3 quota days
```

The gate printed `3.6 generation tok/s`, but that label is wrong. It divided mean output tokens by total request latency. llama.cpp server logs reported actual token evaluation rates mostly around 30 to 58 tokens per second. The repaired gate must call its client-derived number end-to-end output throughput, not generation throughput.

The eight-thread server was started and served valid responses, but no controlled t4 versus t8 client summary has been frozen yet. Hosted acceptance has not been completed in the supplied output.

## Settled Thinking-Mode Decision

Three variants were tested:

- Variant A: local `chat_template_kwargs={"enable_thinking": false}`
- Variant B: append `/no_think` to the prompt
- Variant C: no switch

Variant C placed reasoning in `reasoning_content`, left normal content empty, and consumed the token cap. Variants A and B worked. Variant A was selected because it preserves byte-identical semantic prompt text across local and hosted deployments.

The current gate is still wrong. It appends `/no_think` inside `MCQ_SUFFIX` and sends that prompt to both deployments. Task 1 fixes this.

## Pinned Request Contract

Semantic MCQ suffix for both deployments:

```text
Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".
```

Local request:

```text
temperature:      0.7
top_p:            0.8
top_k:            20
min_p:            0
presence_penalty: 1.5
seed:             42
max_tokens:       64
chat_template_kwargs.enable_thinking: false
```

Hosted request:

```text
model:                 llama-3.3-70b-versatile
temperature:           0.7
top_p:                 0.8
seed:                  42
max_completion_tokens: 64
```

Groq currently does not support presence penalty on its models and does not expose top-k or min-p in the Chat Completions schema. These are documented deployment differences, not prompt differences.

If GSM8K survives the cut order, both deployments use a 512-token output cap for it.

## Frozen Data Design

No benchmark model call may occur before the item manifest is generated and committed.

The dataset is pinned to:

```text
dataset:  cais/mmlu
revision: ea9505acd1c0964d6d5c00d208d0fdf8e4810e6d
seed:     42
```

Project splits:

- Dev: approximately 150 official dev and validation items. Prompt and parser work is allowed here. Dev results are not reported.
- Map: up to 100 official test items per subject. These determine the accuracy map.
- Router: 150 subject-balanced official test items held out for routing evaluation.

Each item belongs to exactly one project split. A frozen JSONL manifest stores the source split and source index rather than duplicating benchmark question text.

`college_mathematics` has only 100 official test items. After its router reservation, fewer than 100 remain for map. Report the achieved count honestly.

## Paired Statistical Design

For an evaluated item, the two models produce one of four outcomes:

| Local | Hosted | Outcome |
|---|---|---|
| Correct | Correct | Agreement |
| Correct | Wrong | Local win |
| Wrong | Correct | Hosted win |
| Wrong | Wrong | Agreement |

If `b` is local wins, `c` is hosted wins, and `n` is paired items:

```text
observed accuracy difference = (b - c) / n
```

The primary non-inferiority calculation is a conservative one-sided 95% lower bound:

1. Compute a 97.5% one-sided Clopper-Pearson lower bound for the local-win probability.
2. Compute a 97.5% one-sided Clopper-Pearson upper bound for the hosted-win probability.
3. Subtract the hosted upper bound from the local lower bound.
4. The Bonferroni allocation guarantees at least 95% coverage for the resulting lower difference bound.

A separate one-sided upper bound reverses the construction.

At margin `delta`:

- Non-inferior if the lower bound is greater than `-delta`.
- Below margin if the upper bound is less than `-delta`.
- Inconclusive otherwise.

Equality at the boundary is inconclusive. The paired bootstrap resamples whole paired rows 10,000 times with a fixed seed and is reported as a secondary check. If methods disagree, report both.

## Failure Policy

- Poll local `/health` before warm-ups.
- Retry transient connection failures and 5xx responses with bounded exponential backoff.
- Honor Groq `Retry-After` for 429 and retry the same item.
- Never retry a parsing failure.
- Preserve every raw response and error.
- Persistent API failure after retries scores incorrect and is separately marked as a reliability failure.
- Never silently drop an item.
- An interrupted run must resume without repeating completed calls.

## Three-Week Delivery Contract

### Remaining Day 0

Repair the gate, amend the pre-registration, freeze splits, compare t4 and t8, run hosted acceptance once, and commit Day 0 evidence.

### Week 1

One paired dev item end to end, SQLite cache and replay, resumable runner, parser built on dev, three-seed dev check, and public methodology README.

### Week 2

Frozen map run, exact bounds, paired bootstrap, classifications, result table, primary chart, limitations, and a resume-ready README.

### Week 3

Routing policies, optional local-first cascade, minimal Express proxy, and final writeup.

Cut work in this order if behind:

1. GSM8K
2. Local-first cascade
3. Express proxy
4. Two exploratory MMLU slices, stopping at six total slices

Never cut frozen splits, raw evidence, paired confidence bounds, or the final writeup.

## Publication Decision

Do not build an interactive dashboard within the three-week scope.

The primary public presentation is the GitHub README containing:

- One headline result table
- One primary chart with confidence bounds
- Latency and token-use measurements
- Deployment manifest
- Method and limitations
- Links to JSON, JSONL, and CSV result artifacts
- Reproduction instructions

If the full project finishes early, a static GitHub Pages report may reuse committed artifacts. It must not introduce a database, backend, new evaluation logic, or duplicated source of truth.

## Current Day 0 Defects

The approved plan already defines the test-first repairs. The important defects are:

1. `/no_think` is still embedded in the semantic prompt.
2. `top_k=20` and `min_p=0` are omitted from local requests, so llama.cpp defaults differ from the pre-registration.
3. Hosted temperature is currently 0.0 rather than the approved 0.7 contract.
4. `seed_ok` contains `or True` and cannot test anything.
5. A 200 response with empty or unparseable content can pass the gate.
6. Startup does not wait for `/health`.
7. A 429 skips a call instead of retrying the same item.
8. The throughput metric is mislabeled.
9. Router calls are missing from the quota estimate.
10. The pre-registration still describes obsolete Q8 and `/no_think` settings.
11. The promised frozen split manifest does not exist.

## Exact Next Task

Execute only Task 1 from:

```text
docs/superpowers/plans/2026-07-31-goodenough-day0-implementation-plan.md
```

Task 1 freezes request construction with standard-library `unittest` tests. Follow the test-first sequence exactly:

1. Explain semantic prompt versus deployment configuration in a few paragraphs.
2. Create the failing request-contract tests.
3. Run them and show the expected failure.
4. Implement `build_messages` and `build_payload` minimally.
5. Run the focused tests.
6. Review the diff with Sahil.
7. Commit only Task 1 files after verification.

Do not call Groq during Task 1. Do not run MMLU. Local diagnostic calls are unnecessary for Task 1.

## Cursor Model and Carrier Recommendation

Cursor is a good daily carrier for GoodEnough because Sahil wants explanations beside implementation, quick file navigation, and short test loops. Cursor's official CLI page currently lists GPT-5.6 Sol variants. Use the native Cursor model picker for Sol rather than assuming a custom OpenAI API key can provide the full agent experience. Cursor's official API-key documentation says custom OpenAI keys are limited to standard chat models, while specialized features may continue using Cursor-hosted models.

Recommended selection:

- Use GPT-5.6 Sol with high reasoning for request contracts, statistics, caching semantics, code review, and final interpretation.
- Enable a fast Sol variant when available for shorter edit and test loops.
- Use a cheaper or faster model only for mechanical formatting or repetitive changes.

Keep commits small. Never let the agent silently rewrite the experiment because a result is inconvenient.

Official references:

- Cursor CLI and current model picker: https://cursor.com/en-US/cli
- Cursor project rules: https://docs.cursor.com/context/rules
- Cursor custom API keys: https://docs.cursor.com/settings/api-keys

## Copy-Paste Opening Prompt for Cursor

```text
Read CURSOR_HANDOFF.md, CLAUDE.md, PREREGISTRATION.md, the approved design, and the Day 0 implementation plan completely before editing anything.

We are on branch codex/goodenough-day0. Preserve the existing uncommitted .env loader in scripts/day0_gate.py and preserve scripts/probe.py. Never print or stage .env.

Use the learning workflow: briefly explain the concept, implement the smallest test-first step, run the focused test, and explain the result. Do not redo project research or broaden scope.

Start with Task 1 only from docs/superpowers/plans/2026-07-31-goodenough-day0-implementation-plan.md. Do not call Groq or run benchmark data. Show me the failing tests before implementing the request builders.
```
