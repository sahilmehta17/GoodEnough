# GoodEnough

**Where is a 1.7B model running on a laptop CPU good enough to replace a hosted 70B model?**

Most teams pick one model and pay for it on every request. The obvious saving is to send easy requests to a small cheap model and hard ones to a large one, but almost nobody ships that, because you cannot easily prove the cheap path did not quietly get worse. GoodEnough measures the boundary directly: it runs a quantized 1.7B model on a consumer CPU and a fixed hosted 70B model over the same benchmark questions, and for each slice reports whether the local model is **non-inferior within a margin chosen before looking at the data**, below that margin, or inconclusive at the sample size available.

This is a reproducible case study of two pinned deployment configurations, not a new routing method. See [prior art](#prior-art).

## The claim, stated so it can be wrong

On at least one predeclared benchmark slice, the lower bound of a one-sided 95% confidence interval for `accuracy_local - accuracy_hosted` exceeds `-10 percentage points`, where the margin was fixed before any evaluation data was observed.

Non-inferior nowhere is a publishable result. Non-inferior everywhere means the benchmark is too easy and is reported as such. Mostly inconclusive is the honest expected outcome at small samples and is reported with a power analysis. No outcome leaves the project with nothing to show. The full design, frozen in advance, is in [PREREGISTRATION.md](PREREGISTRATION.md).

## Status

| Phase | State |
|---|---|
| Day 0: environment, pinned config, feasibility gate | done |
| Day 1: paired-item measurement path | done |
| Day 2: MMLU loader, frozen splits, resumable runner | next |
| Week 2: the map (per-slice non-inferiority) | pending |
| Week 3: routing policies and write-up | pending |

## Pinned configuration

| | |
|---|---|
| Local | unsloth `Qwen3-1.7B-GGUF` Q4_K_M, llama.cpp, CPU only (`-ngl 0`), non-thinking mode |
| Hosted | Groq `llama-3.3-70b-versatile` (free plan) |
| Hardware | Intel Core Ultra 7 258V (Lunar Lake), 4P + 4LP-E, 32 GB |
| Benchmarks | MMLU (8 subjects) and GSM8K, public labels, deterministic scoring |

Exact hashes, sampling parameters, and seeds are frozen in [PREREGISTRATION.md](PREREGISTRATION.md) section 9.

## Reproduce

Requires Python 3.11+, [llama.cpp](https://github.com/ggml-org/llama.cpp), and a free [Groq](https://console.groq.com) API key.

```bash
# 1. Start the local model (leave running in its own terminal)
llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M --port 8080 --seed 42 \
  --ctx-size 4096 -ngl 0 -t 4 --jinja -np 1

# 2. Provide the Groq key (gitignored)
echo "GROQ_API_KEY=gsk_..." > .env

# 3. Day 0 feasibility gate
python scripts/day0_gate.py

# 4. Day 1 paired-item smoke test
python scripts/day1_smoke.py
```

## Layout

```
src/goodenough/     config (single source of truth), clients, scoring, store
scripts/            day0_gate, day1_smoke, probe
tests/              unit tests for the gate
PREREGISTRATION.md  the frozen experimental design
```

## Limitations

Two configurations cannot isolate the effect of model size; the comparison bundles quantization, execution location, and hardware. Both models may have seen these benchmarks in training, so this is a valid comparison on those items, not an uncontaminated generalization estimate. Results describe two pinned deployments on predeclared benchmark slices, not arbitrary production requests. Local cost is reported as zero incremental API spend, not as economically free.

## Prior art

Model routing and cascades are well studied: [FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/abs/2406.18665) (which trains partly on MMLU), and the [LLMRouterBench](https://arxiv.org/abs/2601.07206) benchmark. What that literature does not do is measure a quantized consumer-CPU deployment against a hosted reference with real wall-clock latency on the actual machine. That measurement is the contribution here.
