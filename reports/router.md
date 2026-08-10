# Router policies


## Item-level breakdown

Every one of the 140 paired items on the held-out router split, by which model got it right.


| outcome | items | share |
|---|---:|---:|
| both correct | 70 | 0.500 |
| hosted correct, local incorrect | 50 | 0.357 |
| **local correct, hosted incorrect** | **5** | **0.036** |
| neither correct | 15 | 0.107 |

**5 of 140.** That row is every item where the local 1.7B model was right and the hosted 70B model was wrong. It is the only row where routing can put accuracy above always-hosted, so those 5 items are the entire accuracy budget available to any policy in this study. The remaining 135 items score identically under every policy that routes them to the model that was already right: 70 both models get right, 50 only the hosted model gets right, and 15 neither gets right.


Routing still has a cost argument: sending an item to the local model costs no metered API dollars. What these counts bound is the accuracy argument. Against always-hosted, a policy gains on at most 5 items and loses on every one of the 50 hosted-correct, local-incorrect items it decides to keep local.


## Oracle gap over always-hosted

Oracle accuracy 0.893 minus always-hosted accuracy 0.857 = **+0.0357**, 95% paired bootstrap CI [+0.0071, +0.0714] (10,000 resamples by item, seed 42, n = 140).


The oracle needs per-item knowledge of which model is right, so it is a ceiling rather than a policy. On this model pair at this sample size, the ceiling on any routing policy is between 0.7 and 7.1 percentage points over always-hosted. The gap is reported only with that interval; the point estimate on its own is not a result.


## Policy comparison

Evaluated on the same held-out router split (140 items). hosted_calls is the number of paid calls (local is free). Oracle is a perfect-knowledge ceiling, not a deployable policy.

| policy | accuracy | hosted calls | notes |
|---|---:|---:|---|
| always_local | 0.536 | 0 | free, never calls hosted |
| always_hosted | 0.857 | 140 | most expensive, single-model ceiling |
| map_based | 0.857 | 140 | local where the map judged non-inferior |
| cascade | 0.550 | 4 | escalate only on local parse failure (escalated 3%) |
| oracle | 0.893 | - | upper bound, not real |