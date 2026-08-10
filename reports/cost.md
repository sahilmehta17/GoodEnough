# Cost

## Project totals (every dataset and split evaluated so far)

| quantity | value |
|---|---:|
| actual cash spent | $0.00 |
| hosted list-price-equivalent | $0.2550 |
| local incremental API spend | $0.00 |
| local machine occupancy (seconds) | 10362.9 |

Hosted calls counted: 1,561 (237,655 input tokens, 145,349 output tokens). Actual cash spent is zero because collection runs on Groq's free plan; the list-price-equivalent is what those same tokens would cost at Groq's published on-demand rate. Local incremental API spend is zero by construction, not because local inference has no cost; it is not economically free, it draws no metered API dollars.


## Latency inversion

Pooled across every item evaluated so far, hosted responds faster than local by 8.5x (median end-to-end latency).


## Routing policy cost and latency (router split)

n = 140 paired router-split items.

| policy | accuracy | total $ | $ per 1,000 requests | total seconds | median seconds/request |
|---|---:|---:|---:|---:|---:|
| always_local | 0.536 | 0.0000 | 0.00 | 626.8 | 2.59 |
| always_hosted | 0.857 | 0.0182 | 0.13 | 71.0 | 0.37 |
| map_based | 0.857 | 0.0182 | 0.13 | 71.0 | 0.37 |
| cascade | 0.550 | 0.0020 | 0.01 | 634.1 | 2.59 |
| oracle | 0.893 | - | - | - | - |

Oracle has no cost or latency: it is a perfect-knowledge upper bound on accuracy, not a deployable policy.
