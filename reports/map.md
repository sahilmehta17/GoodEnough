# GoodEnough map

Local: `unsloth/Qwen3-1.7B-GGUF:Q4_K_M`  Hosted: `llama-3.3-70b-versatile`  Margin: 0.10

Dev-set disagreement rate: 0.359 (151/421 items). This drives how many items are needed for a confident verdict.


## MMLU non-inferiority map

delta = local accuracy minus hosted accuracy. CI is the one-sided 95% bound. Verdict is at the 0.10 margin.


| subject | | n | local | hosted | delta | 95% CI | verdict |
|---|---|---:|---:|---:|---:|:---:|---|
| high_school_geography | P | 100 | 0.73 | 0.95 | -0.220 | [-0.239, -0.152] | **below_margin** |
| formal_logic | P | 100 | 0.35 | 0.68 | -0.330 | [-0.396, -0.228] | **below_margin** |
| nutrition |  | 100 | 0.63 | 0.85 | -0.220 | [-0.239, -0.152] | **below_margin** |
| marketing |  | 100 | 0.75 | 0.91 | -0.160 | [-0.212, -0.076] | **inconclusive** |
| miscellaneous |  | 100 | 0.69 | 0.90 | -0.210 | [-0.262, -0.123] | **below_margin** |
| college_mathematics |  | 100 | 0.26 | 0.44 | -0.180 | [-0.261, -0.075] | **inconclusive** |
| professional_law |  | 100 | 0.35 | 0.64 | -0.290 | [-0.377, -0.175] | **below_margin** |
| high_school_psychology |  | 100 | 0.80 | 0.92 | -0.120 | [-0.179, -0.035] | **inconclusive** |

P marks the two primary slices named before data collection.


Across the 8 slices at the 10 point margin: 0 establish non-inferiority, 5 fall below the margin, and 3 are inconclusive at n = 100.
Inconclusive means the interval spans the margin, so the data cannot decide those slices in either direction. It is not a finding that the local model is worse there (PREREGISTRATION.md section 7).


## Sensitivity to the margin

The primary margin is 0.10. PREREGISTRATION.md section 3 also asks for 5 and 15 points. Same interval per slice, reclassified; nothing is re-estimated.


| subject | | margin 5pp | margin 10pp | margin 15pp |
|---|---|---|---|---|
| high_school_geography | P | below_margin | below_margin | below_margin |
| formal_logic | P | below_margin | below_margin | below_margin |
| nutrition |  | below_margin | below_margin | below_margin |
| marketing |  | below_margin | inconclusive | inconclusive |
| miscellaneous |  | below_margin | below_margin | inconclusive |
| college_mathematics |  | below_margin | inconclusive | inconclusive |
| professional_law |  | below_margin | below_margin | below_margin |
| high_school_psychology |  | inconclusive | inconclusive | inconclusive |

3 of 8 slices change verdict across these margins: marketing, miscellaneous, college_mathematics.


## Unparseable responses (MMLU map)

An unparseable response scores incorrect and is never dropped (PREREGISTRATION.md section 12). The denominator is responses that returned without an API error; API failures are counted separately in the errors columns.


| subject | hosted unparseable | hosted rate | local unparseable | local rate | hosted errors | local errors |
|---|---:|---:|---:|---:|---:|---:|
| high_school_geography | 0/100 | 0.00 | 0/100 | 0.00 | 0 | 0 |
| formal_logic | 5/100 | 0.05 | 13/100 | 0.13 | 0 | 0 |
| nutrition | 0/100 | 0.00 | 0/100 | 0.00 | 0 | 0 |
| marketing | 0/100 | 0.00 | 0/100 | 0.00 | 0 | 0 |
| miscellaneous | 0/100 | 0.00 | 0/100 | 0.00 | 0 | 0 |
| college_mathematics | 20/100 | 0.20 | 25/100 | 0.25 | 0 | 0 |
| professional_law | 0/100 | 0.00 | 0/100 | 0.00 | 0 | 0 |
| high_school_psychology | 0/100 | 0.00 | 0/100 | 0.00 | 0 | 0 |

6 of the 8 subjects are clean on both models: zero unparseable responses from either.
The exceptions are formal_logic (hosted 5/100, local 13/100); college_mathematics (hosted 20/100, local 25/100).

college_mathematics is unparseable at 20% on the hosted 70B model and 25% on the local 1.7B model. A failure rate that high on both models is a harness limitation on that subject, in prompt format and answer extraction, not a capability difference between the models. All 45 of those responses scored incorrect, so the college_mathematics accuracies above are a floor for both models, not an estimate of what either model knows.

These rates are reported, not corrected. The parser was frozen on the dev split before any map response was seen; editing it against map data would invalidate the map (PREREGISTRATION.md section 12, CLAUDE.md).


## Difficulty and the size of the gap

Pearson correlation between hosted accuracy and delta across the 8 slices: r = +0.349, 95% bootstrap CI [-0.292, +0.946] (10,000 resamples, seed 42).


This is underpowered. One point per slice means n = 8, and the interval is correspondingly wide and contains zero, so the correlation is not distinguishable from zero at n = 8. The sign is the direction difficulty-based routing would predict: the local model falls further behind on the slices the hosted model also finds harder. That direction is worth recording and is not evidence for the mechanism at this sample size.


## Cost and latency (MMLU map)

| model | n | median latency (ms) | mean latency (ms) | total tokens |
|---|---:|---:|---:|---:|
| local | 800 | 2941 | 6786 | 177,249 |
| hosted | 800 | 283 | 544 | 184,529 |

Local token cost is zero incremental API spend; the tokens column for local reflects local compute only, not money.
