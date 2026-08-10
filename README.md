# GoodEnough

**Where is a 1.7B model running on a laptop CPU good enough to replace a hosted 70B model?**

On none of the eight slices does the local model establish non-inferiority at the 10-point margin. Five fall below the margin. Three are inconclusive at n = 100, meaning the data cannot decide them in either direction. Local is `unsloth/Qwen3-1.7B-GGUF` Q4_K_M on a consumer CPU, hosted is Groq `llama-3.3-70b-versatile`, 100 paired items per slice, margin fixed before any data was collected.

| subject | | n | local | hosted | delta | one-sided 95% CI | verdict |
|---|---|---:|---:|---:|---:|:---:|---|
| high_school_geography | P | 100 | 0.73 | 0.95 | -0.220 | [-0.239, -0.152] | below_margin |
| formal_logic | P | 100 | 0.35 | 0.68 | -0.330 | [-0.396, -0.228] | below_margin |
| nutrition |  | 100 | 0.63 | 0.85 | -0.220 | [-0.239, -0.152] | below_margin |
| marketing |  | 100 | 0.75 | 0.91 | -0.160 | [-0.212, -0.076] | inconclusive |
| miscellaneous |  | 100 | 0.69 | 0.90 | -0.210 | [-0.262, -0.123] | below_margin |
| college_mathematics |  | 100 | 0.26 | 0.44 | -0.180 | [-0.261, -0.075] | inconclusive |
| professional_law |  | 100 | 0.35 | 0.64 | -0.290 | [-0.377, -0.175] | below_margin |
| high_school_psychology |  | 100 | 0.80 | 0.92 | -0.120 | [-0.179, -0.035] | inconclusive |

`delta` is local accuracy minus hosted accuracy. P marks the two primary slices named before the pilot. The pre-registered claim was that at least one primary slice would clear the margin. Both primary slices land below it, which is the outcome [PREREGISTRATION.md](PREREGISTRATION.md) section 15 named in advance as publishable. The three inconclusive slices are not counted against the model: the interval spans the margin there, so the study has no verdict on them either way.

```bash
python scripts/build_map.py && python scripts/build_gsm8k.py && python scripts/build_router.py && python scripts/build_cost.py && python scripts/build_figures.py
```

That rebuilds every table and figure in `reports/` from `data/results.sqlite`. The database itself is not committed (it holds every raw model response), so collecting it first is [Reproduce](#reproduce) below.

## Three numbers that carry the study

**5 of 140.** On the held-out router split, 5 items had the local model right and the hosted model wrong. 50 went the other way, 70 both models answered correctly, 15 neither did. Those 5 items are the whole accuracy budget any routing policy has to work with here. Full breakdown in [reports/router.md](reports/router.md).

**Oracle gap +0.0357, 95% CI [+0.0071, +0.0714].** A router with per-item knowledge of which model is right beats always-hosted by between 0.7 and 7.1 percentage points on this model pair at this sample size. That is a ceiling, not a policy: none of the four deployable policies evaluated here reaches it. Paired bootstrap, 10,000 resamples by item, seed 42.

**Total study cost: $0.2550 hosted list-price-equivalent and 10,362.9 seconds of local machine occupancy.** Those two are measured: 383,004 tokens passed through the hosted model, and $0.2550 is what they would have cost at Groq's published on-demand rate. No cash actually left an account, because collection ran under Groq's free tier, and local inference calls no metered API. Both of those zeros follow from how the study was billed rather than from anything counted, so [reports/cost.md](reports/cost.md) reports them separately from the measurements.

## What else is in reports/

- [reports/map.md](reports/map.md): the map above, plus verdicts at 5 and 15 point margins, unparseable rate per subject per model, and the correlation between slice difficulty and the size of the gap.
- [reports/gsm8k.md](reports/gsm8k.md): accuracy against gold reasoning-step count. The local slope is -0.512, 90% CI [-0.797, -0.298]. No hosted slope is reported: hosted made 2 errors in 150 items, too few to estimate one.
- [reports/router.md](reports/router.md): item-level breakdown, oracle gap, and five routing policies on the held-out split.
- [reports/cost.md](reports/cost.md): the four cost quantities, the latency inversion, and per-policy dollars and wall-clock seconds.
- [reports/figures/](reports/figures/): four paper figures, each emitted as a CSV and a vector PDF.

Every one of those files is generated from `data/results.sqlite` by the command above. Nothing in them is hand-entered.

## Why it is set up this way

Most teams pick one model and pay for it on every request. The obvious saving is to send easy requests to a small cheap model and hard ones to a large one, but almost nobody ships that, because you cannot easily prove the cheap path did not quietly get worse. GoodEnough measures the boundary directly and reports, per slice, whether the local model is **non-inferior within a margin chosen before looking at the data**, below that margin, or inconclusive at the sample size available.

This is a reproducible case study of two pinned deployment configurations, not a new routing method. See [prior art](#prior-art). Unfamiliar terms (non-inferiority, discordant pairs, Clopper-Pearson) are defined plainly in [GLOSSARY.md](GLOSSARY.md).

## The claim, stated so it can be wrong

On at least one predeclared benchmark slice, the lower bound of a one-sided 95% confidence interval for `accuracy_local - accuracy_hosted` exceeds `-10 percentage points`, where the margin was fixed before any evaluation data was observed.

Non-inferiority established nowhere is a publishable result. Non-inferior everywhere means the benchmark is too easy and is reported as such. Mostly inconclusive is the honest expected outcome at small samples and is reported with a power analysis. No outcome leaves the project with nothing to show. The full design, frozen in advance, is in [PREREGISTRATION.md](PREREGISTRATION.md).

## Status

Data collection is complete. Every split has been evaluated against both models and every report in `reports/` is generated from the finished database.

| Phase | State |
|---|---|
| Loader, scorer, resumable budget-aware runner, WAL store, analysis engine, report builders | done |
| Local evaluation (MMLU map 800, MMLU dev 421, MMLU router 140, GSM8K map 150, GSM8K dev 50) | done |
| Hosted evaluation (Groq, rate- and budget-limited, resumed across days) | done |
| Map, GSM8K difficulty, router policies, cost accounting, figures | done |
| Proxy and writeup | not built |

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

# 5. Evaluate each split against both models (resumable; stops cleanly at Groq's
#    daily token budget and picks up where it left off on the next run). The
#    full pass is ~287K hosted tokens against a 100K/day free-plan ceiling, so
#    it spans 3 to 4 days.
python scripts/run_eval.py --dataset mmlu  --split dev
python scripts/run_eval.py --dataset mmlu  --split map
python scripts/run_eval.py --dataset mmlu  --split router
python scripts/run_eval.py --dataset gsm8k --split dev
python scripts/run_eval.py --dataset gsm8k --split map

# 6. Build every report from whatever has been evaluated so far. This is the
#    one-command step at the top of this README.
python scripts/build_map.py && python scripts/build_gsm8k.py && python scripts/build_router.py && python scripts/build_cost.py && python scripts/build_figures.py
```

Each builder takes `--db` to point at a database other than `data/results.sqlite`, and each is read-only on it, so building reports while collection is still running is safe.

## Layout

```
src/goodenough/     config (single source of truth), clients, loader, scoring, store, analysis
scripts/            freeze_splits, run_eval, build_map, build_gsm8k, build_router, build_cost,
                    build_figures, day0_gate, day1_smoke, probe
tests/              unit tests
data/frozen/        frozen benchmark splits and manifest (immutable once evaluation starts)
reports/            generated result tables and figures, rebuilt from data/results.sqlite
PREREGISTRATION.md  the frozen experimental design
GLOSSARY.md         plain-language definitions of the statistical and ML terms used above
```

## Limitations

Two configurations cannot isolate the effect of model size; the comparison bundles quantization, execution location, and hardware. Both models may have seen these benchmarks in training, so this is a valid comparison on those items, not an uncontaminated generalization estimate. Results describe two pinned deployments on predeclared benchmark slices, not arbitrary production requests. Local cost is reported as zero incremental API spend, not as economically free. Full list in [PREREGISTRATION.md](PREREGISTRATION.md) section 14.

## License

MIT. See [LICENSE](LICENSE).

## Prior art

Model routing and cascades are well studied: [FrugalGPT](https://arxiv.org/abs/2305.05176), [RouteLLM](https://arxiv.org/abs/2406.18665) (which trains partly on MMLU), and the [LLMRouterBench](https://arxiv.org/abs/2601.07206) benchmark. What that literature does not do is measure a quantized consumer-CPU deployment against a hosted reference with real wall-clock latency on the actual machine. That measurement is the contribution here.
