# Cost

## Measured totals (every dataset and split evaluated so far)

Everything in this table is counted from the results database.

| quantity | value | what it counts |
|---|---:|---|
| hosted tokens consumed | 383,004 | 237,655 input and 145,349 output, over 1,561 uncached hosted calls |
| hosted list-price-equivalent | $0.2550 | those tokens priced at Groq's published on-demand rate, $0.59 per 1M input and $0.79 per 1M output, read 2026-08-06 |
| local machine occupancy | 10,362.9 s | summed end-to-end wall clock of every uncached local call |
| local incremental API spend | $0.0000 | local token volume priced at the pinned local rate |

### Two quantities that are not measurements

**Cash outlay: none.** Collection ran under Groq's free tier, so no metered charge was incurred and no money left an account. That zero is a property of the billing plan, not something counted in this database, and it is stated here rather than placed in the table above. The measurement is the list-price-equivalent: 383,004 tokens really did pass through the hosted model, and $0.2550 is what they would have cost at the published rate had they been billed.


**Local incremental API spend: $0.0000.** The pinned local rate is $0.00 per 1M tokens because local inference calls no metered API, so this figure is zero by construction rather than by measurement. It is not a finding that local inference is free. The local cost this study did measure is the occupancy row above: 10,362.9 s of a laptop that could not be doing anything else. Electricity and hardware amortization are out of scope (PREREGISTRATION.md section 14).


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
