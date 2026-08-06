# GoodEnough map

Local: `unsloth/Qwen3-1.7B-GGUF:Q4_K_M`  Hosted: `llama-3.3-70b-versatile`  Margin: 0.10

Dev-set disagreement rate: 0.367 (110/300 items). This drives how many items are needed for a confident verdict.


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


## Cost and latency (MMLU map)

| model | n | median latency (ms) | mean latency (ms) | total tokens |
|---|---:|---:|---:|---:|
| local | 800 | 2941 | 6786 | 177,249 |
| hosted | 800 | 283 | 544 | 184,529 |

Local token cost is zero incremental API spend; the tokens column for local reflects local compute only, not money.
