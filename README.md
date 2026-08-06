# GoodEnough

**Where is a 1.7B model running on a laptop CPU good enough to replace a hosted 70B model?**

Most teams pick one model and pay for it on every request. The obvious saving is to send easy requests to a small cheap model and hard ones to a large one, but almost nobody ships that, because you cannot easily prove the cheap path did not quietly get worse. GoodEnough measures the boundary directly: it runs a quantized 1.7B model on a consumer CPU and a fixed hosted 70B model over the same benchmark questions, and for each slice reports whether the local model is **non-inferior within a margin chosen before looking at the data**, below that margin, or inconclusive at the sample size available.

This is a reproducible case study of two pinned deployment configurations, not a new routing method. See [prior art](#prior-art). Unfamiliar terms (non-inferiority, discordant pairs, Clopper-Pearson) are defined plainly in [GLOSSARY.md](GLOSSARY.md).

## The claim, stated so it can be wrong

On at least one predeclared benchmark slice, the lower bound of a one-sided 95% confidence interval for `accuracy_local - accuracy_hosted` exceeds `-10 percentage points`, where the margin was fixed before any evaluation data was observed.

Non-inferior nowhere is a publishable result. Non-inferior everywhere means the benchmark is too easy and is reported as such. Mostly inconclusive is the honest expected outcome at small samples and is reported with a power analysis. No outcome leaves the project with nothing to show. The full design, frozen in advance, is in [PREREGISTRATION.md](PREREGISTRATION.md).

## Status

| Phase | State |
|---|---|
| Loader, scorer, resumable budget-aware runner, WAL store, analysis engine, map/GSM8K/router builders | done |
| Local evaluation (every split: MMLU map 800, MMLU dev 421, MMLU router 140, GSM8K map 150, GSM8K dev 50) | done |
| Hosted evaluation (Groq, rate- and budget-limited, resumes across days) | in progress |
| Map (per-slice non-inferiority, MMLU) | done |
| Map (GSM8K difficulty slope), router policy evaluation, cost accounting, full writeup | pending |

Exact hosted-evaluation counts move daily; query `data/results.sqlite` read-only if you need the current number rather than trusting this table.

## Results

Results are reported only once `reports/` contains them, generated directly from `data/results.sqlite` by the scripts below. Nothing here is asserted ahead of that.

- [reports/map.md](reports/map.md): MMLU per-slice non-inferiority map. **Available**, generated from the complete MMLU map hosted pass.
- `reports/gsm8k.md`: GSM8K difficulty-slope analysis. Pending hosted collection.
- `reports/router.md`: routing policy comparison against the held-out router split. Pending hosted collection.
- `reports/cost.md`: cash spent, hosted list-price-equivalent, local incremental spend, local machine occupancy. Pending.

## Pinned configuration

| | |
|---|---|
| Local | unsloth `Qwen3-1.7B-GGUF` Q4_K_M, llama.cpp, CPU only (`-ngl 0`), non-thinking mode |
| Hosted | Groq `llama-3.3-70b-versatile` (free plan) |
| Hardware | Intel Core Ultra 7 258V (Lunar Lake), 4P + 4LP-E, 32 GB |
| Benchmarks | MMLU (8 subjects) and GSM8K, public labels, deterministic scoring |

Exact hashes, sampling parameters, and seeds are frozen in [PREREGISTRATION.md](PREREGISTRATION.md) section 9.

## Environment constraint

The runner, scorer, and analysis code are pure Python standard library: no pandas, numpy, scipy, or `datasets`. The development machine blocks newly installed compiled libraries, so anything requiring a wheel with native code was not an option. `scripts/freeze_splits.py` pulls MMLU and GSM8K over Hugging Face's plain HTTP rows API for the same reason.

## Reproduce

Requires Python 3.11+, [llama.cpp](https://github.com/ggml-org/llama.cpp), and a free [Groq](https://console.groq.com) API key.

```bash
# 1. Start the local model (leave running in its own terminal)
llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M --port 8080 --seed 42 \
  --ctx-size 4096 -ngl 0 -t 4 --jinja -np 1

# 2. Provide the Groq key (gitignored)
echo "GROQ_API_KEY=gsk_..." > .env

# 3. Unit tests (no network calls)
python -m unittest discover -s tests -v

# 4. Splits are already frozen and committed under data/frozen/. Re-freezing is
#    only needed if you are starting the project over; it takes no arguments
#    beyond an optional dry run and re-fetches from Hugging Face.
python scripts/freeze_splits.py --dry-run

# 5. Evaluate a split against both models (resumable; stops cleanly at Groq's
#    daily token budget and picks up where it left off on the next run)
python scripts/run_eval.py --dataset mmlu --split map

# 6. Build the reports from whatever has been evaluated so far
python scripts/build_map.py
python scripts/build_gsm8k.py
python scripts/build_router.py
```

## Layout

```
src/goodenough/     config (single source of truth), clients, loader, scoring, store, analysis
scripts/            freeze_splits, run_eval, build_map, build_gsm8k, build_router, day0_gate, day1_smoke, probe
tests/              unit tests
data/frozen/        frozen benchmark splits and manifest (immutable once evaluation starts)
reports/            generated result tables, populated as hosted evaluation completes
PREREGISTRATION.md  the frozen experimental design
GLOSSARY.md         plain-language definitions of the statistical and ML terms used above
```

## Limitations

Two configurations cannot isolate the effect of model size; the comparison bundles quantization, execution location, and hardware. Both models may have seen these benchmarks in training, so this is a valid comparison on those items, not an uncontaminated generalization estimate. Results describe two pinned deployments on predeclared benchmark slices, not arbitrary production requests. Local cost is reported as zero incremental API spend, not as economically free. Full list in [PREREGISTRATION.md](PREREGISTRATION.md) section 14.

## License

MIT. See [LICENSE](LICENSE).

## Prior art

Model routing and cascades are well studied: [FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/abs/2406.18665) (which trains partly on MMLU), and the [LLMRouterBench](https://arxiv.org/abs/2601.07206) benchmark. What that literature does not do is measure a quantized consumer-CPU deployment against a hosted reference with real wall-clock latency on the actual machine. That measurement is the contribution here.
