# GoodEnough Learning Implementation Design

Date: 2026-07-31

## Goal

Build GoodEnough into a resume-ready applied AI engineering project within a strict three-week deadline while learning the concepts through implementation.

GoodEnough measures where a quantized 1.7B model running on a consumer laptop CPU stays within a predeclared accuracy margin of a hosted 70B model. It then evaluates routing policies that use the resulting capability map.

The project is successful when Sahil can both show the artifact and explain the important design decisions, failure modes, and statistical conclusions in an interview.

## Working Style

Implementation uses learning-first vertical slices:

1. Explain the concept needed for the current component.
2. Explain why the selected design fits this project.
3. Implement the smallest complete behavior.
4. Run one focused test or demonstration.
5. Continue to the next behavior.

Explanations stay attached to current implementation work. Separate tutorials, optional notebooks, speculative abstractions, and unrelated refactors are out of scope.

## Architecture

The system has five layers.

### Model clients

Local llama.cpp and hosted Groq clients return a shared response shape while retaining provider-specific metadata and raw responses.

### Evaluation pipeline

The pipeline loads a frozen item, constructs one semantic prompt, invokes both deployments, parses their answers, scores them, and emits paired results.

### Persistence

SQLite stores immutable request and response records. Cache identity includes model identity, semantic prompt, sampling parameters, deployment-specific parameters, and seed. Cached results reproduce a completed call without another model invocation.

### Analysis

Analysis computes per-slice accuracy, paired accuracy differences, conservative exact confidence bounds, a paired bootstrap check, latency summaries, token usage, and the declared three-status classification.

### Routing and API

Routing consumes the completed capability map. It never influences evaluation. The minimal HTTP proxy is built last and only if earlier milestones remain on schedule.

## Data Flow

Each benchmark item follows one immutable path:

```text
frozen item
  -> semantic prompt
  -> local and hosted request configurations
  -> raw responses stored
  -> versioned parser
  -> normalized answers
  -> paired scored result
  -> aggregate analysis
```

The semantic prompt is byte-identical across deployments. Deployment-specific controls remain outside it.

- Local inference disables thinking with `chat_template_kwargs={"enable_thinking": false}`.
- Local sampling explicitly supplies every pinned parameter.
- Hosted sampling explicitly supplies every supported pinned parameter.
- Each record stores requested and returned model IDs.
- Raw responses are retained even when parsing or scoring fails.

## Day 0 Configuration Repair

Before benchmark calls, the repository must be reconciled with the configuration already proven by diagnostic probes:

- Model repository: `unsloth/Qwen3-1.7B-GGUF`
- File: `Qwen3-1.7B-Q4_K_M.gguf`
- SHA-256: `B139949C5BD74937AD8ED8C8CF3D9FFB1E99C866C823204DC42C0D91FA181897`
- llama.cpp build: `b10173-e9fa0781f`
- Context size: 4096
- Parallel slots: 1
- CPU: Intel Core Ultra 7 258V
- GPU layers: 0
- Seed: 42
- Thinking: disabled through the local chat-template parameter, not prompt text

Pinned multiple-choice requests use these fields:

- Local: temperature 0.7, top-p 0.8, top-k 20, min-p 0, presence penalty 1.5, seed 42, and 64 maximum output tokens.
- Hosted: temperature 0.7, top-p 0.8, seed 42, and 64 maximum completion tokens.

Groq does not support presence penalty on its current models and does not expose top-k or min-p in the Chat Completions request schema, so those local-model controls are recorded deployment differences rather than prompt differences. GSM8K, if retained, uses 512 maximum output tokens on both deployments; every other supported sampling field remains the same.

The pre-registration amendment records the move from the abandoned official Q8 configuration to the working Unsloth Q4_K_M configuration. Diagnostic probes and Day 0 feasibility calls are identified as non-benchmark data.

The repaired gate must:

- Poll local health before warm-up calls.
- Reject empty or unparseable responses.
- Keep the semantic prompt free of `/no_think`.
- Supply all pinned sampling parameters explicitly.
- Distinguish end-to-end request latency from llama.cpp generation throughput.
- Include router calls in quota estimates.
- Verify hosted acceptance and seed behavior without a hard-coded success value.
- Produce a small machine-readable Day 0 manifest.

Four-thread and eight-thread local runs use the same probes and configuration. The faster stable result becomes the pinned thread count. The hosted acceptance check runs once to conserve quota.

## Frozen Dataset Design

Before any benchmark request, a committed manifest records dataset revision, source split, subject, item ID, project split, and deterministic order.

### Dev

Approximately 150 items come only from official MMLU dev and validation data. Dev items are used for prompt selection, parser construction, discordance estimates, and the three-seed variance check. Dev results are not reported as evaluation results.

### Map

Map contains up to 100 remaining official test items per subject. Map results determine the per-subject non-inferiority classification. Prompts and parser behavior cannot change in response to map outputs.

### Router

Router contains 150 subject-balanced items reserved from official test data before map evaluation. It is used only for held-out routing-policy evaluation.

When a subject does not contain enough test items for both its router allocation and 100 map items, every remaining item is used for map and the achieved sample size is reported. An item belongs to exactly one project split.

## Paired Statistical Design

For each item, the paired outcome is one of:

| Local | Hosted | Paired outcome |
|---|---|---|
| Correct | Correct | Agreement |
| Correct | Wrong | Local win |
| Wrong | Correct | Hosted win |
| Wrong | Wrong | Agreement |

If `b` is the number of local wins, `c` is the number of hosted wins, and `n` is the number of paired items, the observed accuracy difference is:

```text
(b - c) / n
```

The primary non-inferiority calculation is a one-sided 95% lower confidence bound. It uses a 97.5% one-sided Clopper-Pearson lower bound for the local-win probability and a 97.5% one-sided Clopper-Pearson upper bound for the hosted-win probability. The Bonferroni allocation makes their difference a conservative bound with at least 95% coverage:

- Lower difference bound equals the lower local-win bound minus the upper hosted-win bound.
- A separate one-sided 95% upper difference bound reverses the construction, using an upper local-win bound minus a lower hosted-win bound.

For margin `delta`, classification is:

- Non-inferior when the lower difference bound is greater than `-delta`.
- Below margin when the upper difference bound is less than `-delta`.
- Inconclusive otherwise.

The lower and upper bounds are each one-sided 95% statements; they are not presented as a jointly covered two-sided 95% interval. The secondary check resamples paired item rows with replacement 10,000 times using a fixed seed and reports corresponding one-sided bounds. If the exact and bootstrap methods disagree, both are reported.

Tests cover hand-verifiable cases, zero discordance, all discordance in one direction, identical model results, and classification exactly at the margin.

## Failure Handling

- Local startup waits for the health endpoint instead of treating model loading as an evaluation failure.
- Transient connection failures and 5xx responses use bounded exponential backoff.
- Groq 429 responses honor `Retry-After` and retry the same item.
- Parsing failures are not retried.
- Every raw response and error is stored.
- Persistent API failure after the retry limit scores incorrect and is also recorded as a reliability failure.
- No item is silently dropped.
- An interrupted run resumes from stored results without repeating completed calls.

## Testing Strategy

- Unit tests use saved local and hosted response fixtures and never require network access.
- Parser tests cover accepted formats, empty output, malformed output, and reasoning-only responses.
- Cache tests prove exact replay and cache-key sensitivity to every pinned request parameter.
- Runner tests simulate transient failures, rate limits, permanent failures, and interruption.
- Statistical tests use hand-calculated paired tables and deterministic bootstrap seeds.
- One opt-in integration test exercises the local llama.cpp server.
- One manually invoked hosted smoke test protects free-plan quota.

## Three-Week Schedule

### Remaining Day 0

- Amend and reconcile the pre-registration.
- Freeze the exact statistical construction.
- Generate and commit the split manifest.
- Repair the Day 0 gate.
- Compare four and eight local threads.
- Run hosted acceptance once.
- Freeze the deployment manifest.

### Week 1: working evaluation system

- Process one dev item through both models, parsing, scoring, and storage.
- Add SQLite cache and exact replay.
- Add the resumable runner, retries, and quota handling.
- Preserve raw responses and errors.
- Construct and freeze the parser on dev only.
- Run the three-seed dev check.
- Publish a README with methodology and results marked pending.

Week 1 tripwire: an interrupted run must resume without repeating completed calls.

### Week 2: evidence

- Run the frozen map.
- Calculate paired differences and exact confidence bounds.
- Run the bootstrap check.
- Classify each slice.
- Generate the main chart and result tables.
- Write limitations and explain inconclusive results.

Week 2 tripwire: the repository must be resume-ready without the proxy.

### Week 3: product layer

- Evaluate always-local, always-hosted, map-based, and oracle routing.
- Add the local-first cascade only if on schedule.
- Build the minimal Express proxy last.
- Complete the technical writeup and reproducibility instructions.

## Cut Order

If the schedule slips, cut work in this order:

1. GSM8K
2. Local-first cascade
3. Express proxy
4. Two exploratory MMLU slices, stopping at the six-slice project floor

Frozen splits, raw-response preservation, paired confidence intervals, and the final writeup are never cut.

## Definition of Done

The minimum resume-ready result is:

- A public reproducible repository
- At least six MMLU slices evaluated with paired intervals
- Every slice classified as non-inferior, below margin, or inconclusive
- Raw and aggregate result artifacts
- Measured local latency and hosted token usage
- A clear deployment manifest
- A README explaining the question, method, results, limitations, and reproduction steps

Week 3 routing and proxy work strengthen the product story but do not determine whether the project succeeded.
