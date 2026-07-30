# goodenough

Measuring where a quantized 1.7B model on a laptop CPU meets a predeclared accuracy margin relative to a fixed hosted 70B model, then evaluating routing policies against that map.

**Read `PREREGISTRATION.md` before writing code. It is frozen. Everything below serves it.**

---

## THE MOST IMPORTANT RULE

**This specification is frozen. Do not add features.**

Two prior versions of this project were abandoned because every review pass added required machinery until a three-week scope became untenable. That failure mode is the primary risk here, not the code.

When you discover a problem during implementation, classify it as exactly one of three things and say which:

1. **Correctness defect.** It makes a reported number wrong. Fix it.
2. **Documented limitation.** Write it in the README's limitations section. Do not fix it.
3. **Future work.** Add one line to the future work list. Do not fix it, do not scaffold for it.

**Never silently promote a discovery into a new requirement.** If you believe something genuinely must be added, stop and ask rather than building it.

Things that are explicitly OUT of scope and must not be built:
- Streaming / SSE
- A learned prompt classifier
- Semantic caching
- A third model tier
- Mutation testing
- LLM-as-judge scoring
- BANKING77
- A web dashboard
- Full OpenAI API compatibility beyond the fields the runner actually uses
- Any additional dataset, model, or provider

---

## Build order. The HTTP proxy comes LAST.

The HTTP layer is the easy part. The risk is in model invocation, prompt formatting, token limits, parsing, caching, quota behavior, and reproducible scoring. Surface those on day one.

**Day 1: one vertical slice, no HTTP.** A script that takes one MMLU item, calls local, calls hosted, stores both raw responses, parses and scores both, emits one paired result row.

**Day 2: cache and replay.** Cache keyed on `model + prompt + sampling params + seed`. Token counts, end-to-end latency, SQLite persistence.

> **TRIPWIRE.** By end of day 2, one paired benchmark item must round-trip through the complete measurement path and reproduce exactly from cache. Not a proxy request. A scored paired row. If that does not exist, stop and report it.

**Day 3: the full runner.** Iterate items, respect rate limits, resume after the daily token ceiling, retry with backoff.

**Day 4: public repo.** README with hypothesis, margin, named primary slices, method, "results pending." `PREREGISTRATION.md` and the split file committed **before any evaluation call**.

**Week 1 weekend: dev split.** Prompts, parser, discordance estimate, 3-seed variance check. Compute required n. Freeze slice count. Start the map run.

**Week 2: the map.** Scoring, aggregation, intervals, three-status classification, chart. GSM8K step-count slope, secondary and cuttable.

**Week 3: proxy, router, writeup.**

---

## Hard constraints

**Resumability is not optional.** Groq free plan gives `llama-3.3-70b-versatile` 1K requests/day but only **100K tokens/day**. The full run is ~287K tokens, so it spans 3 to 4 days. TPD binds before RPD. The runner must stop at the ceiling and resume tomorrow from cache without redoing work. Build this on day 3, not when you first hit a 429.

**Cached tokens do not count toward Groq limits.** The cache is load-bearing.

**Never silently drop an item.** Unparseable responses score incorrect and the unparseable rate is reported. Persistent API failures score incorrect and reliability is reported separately.

**Never write to a frozen file after data exists.** `PREREGISTRATION.md`, the split file, and the subject list are immutable once the first evaluation call is made. Changes go in the amendment log.

**Parser is built on dev only.** A parser tuned by inspecting map-split responses invalidates the map. If parsing fails on map data, that is a documented unparseable rate, not a reason to edit the parser.

---

## Stack

- **Runner, scoring, statistics:** Python
- **Proxy (week 3 only):** Node/TypeScript, Express
- **Storage:** SQLite
- **Local inference:** llama.cpp `llama-server`, OpenAI-shaped endpoint on `localhost:8080`
- **Hosted inference:** Groq, `llama-3.3-70b-versatile`

Keep dependencies minimal. No frameworks beyond what is listed.

---

## Data model

Every item, every model, one row. Never overwrite; append and version.

```
dataset, version, split, item_id
model_role            (local | hosted)
model_id_requested, model_id_returned
semantic_prompt, rendered_input
raw_response
normalized_answer, parser_version, parse_status
correct               (bool)
error, retries
input_tokens, output_tokens
latency_ms_uncached
cache_hit             (bool)
run_date, seed
```

---

## Statistics

- **Primary:** exact matched-binary interval on discordant pairs, one-sided 95%
- **Check:** paired bootstrap by item, 10,000 resamples, fixed seed
- Report planned n, observed discordance, and achieved interval width as three separate numbers
- Three statuses per slice: non-inferior / below margin / inconclusive
- Two primary slices carry the headline. The other six are exploratory with unadjusted intervals.

Do not compute a headline cost ratio. Report four separate quantities: actual cash spent, hosted list-price-equivalent, local incremental API spend ($0), and local machine occupancy in wall-clock seconds.

---

## Latency measurement

End-to-end request latency only. **No TTFT** (there is no streaming, so there is no observed first-token time).

Controls: 5 warm-up requests discarded, sequential only, fixed item order, cache hits excluded from latency stats, `max_tokens` fixed, background CPU load recorded, retry and error latency reported separately.

p95 from 100 requests is set by five observations. Per-slice latency is descriptive. Compute p95 only on the pooled set.

---

## Definition of done

**Week 2 floor.** Public repo, map over at least 6 slices with intervals and three-status classification, README explaining it. **If only this ships, the project succeeded.**

**Week 3.** The above plus four routing policies, oracle bound, held-out evaluation, full writeup with the manifest.

**If behind:** cut slices, cut GSM8K, cut the cascade policy. **Never cut** the frozen splits, the predeclared margin, or the confidence intervals. Those are the entire value of the project.

---

## Expected results, written in advance

The map-based router may route almost everything hosted. At 1.7B versus 70B with a 10-point margin, few or no slices may qualify. A mostly-inconclusive map is the honest expected outcome at achievable n.

**Both are legitimate results and are pre-committed as such.** Do not tune anything to avoid them.

---

## Tone for the README

Narrow and defensible. This is a case study of two pinned deployment configurations, not a new routing method. Cite FrugalGPT (arXiv 2305.05176), RouteLLM (arXiv 2406.18665), LLMRouterBench (arXiv 2601.07206), and the 2026 routing survey (arXiv 2603.04445). Note explicitly that RouteLLM trained partly on MMLU with gold labels, so this benchmark choice sits inside their evaluation domain.

The contribution: nobody in that literature benchmarks a quantized consumer-CPU deployment with measured wall-clock latency on the actual machine. That part is ours. Nothing more.
