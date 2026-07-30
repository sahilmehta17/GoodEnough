# Pre-registration

**Committed before any evaluation data was observed.**

Commit this file, unmodified, before the first evaluation call. If it changes after data exists, the change must be recorded in the amendment log at the bottom with a date and reason, and the affected results must be relabelled exploratory.

---

## 1. Question

Where does a quantized 1.7B model running on a consumer laptop CPU meet a predeclared accuracy margin relative to a fixed hosted 70B model?

## 2. Primary hypothesis

On at least one predeclared benchmark slice, the lower bound of a one-sided 95% confidence interval for `accuracy_local - accuracy_hosted` exceeds `-delta`.

## 3. Margin

- **Primary: `delta = 10 percentage points`**
- Sensitivity also reported at 5 and 15 points
- Same margin applies to all slices
- This is a project definition of "good enough," not a claim about generally acceptable accuracy loss

## 4. Primary slices, named before the pilot

| Role | MMLU subject |
|---|---|
| Primary easy | `high_school_geography` |
| Primary hard | `formal_logic` |

These two carry the headline claim. Chosen so the contrast is reasoning demand rather than context length. `abstract_algebra` was rejected as the hard slice because small models sit near chance there and floor effects would make the result uninformative.

**The pilot sets sample size. It does not set which slices are primary.**

## 5. Exploratory slices

Six additional MMLU subjects, reported with unadjusted intervals and clearly labelled exploratory:

```
nutrition
marketing
miscellaneous
college_mathematics
professional_law
high_school_psychology
```

Headline results will not be phrased as "N of 8 passed."

## 6. Secondary analysis

GSM8K. Correctness regressed on gold reasoning-step count, slope reported with a confidence interval. Step count derived deterministically from annotated calculator operations in the GSM8K solution field.

No breakpoint search on evaluation data. If a threshold is estimated, it is estimated on dev and tested on map. The word "cliff" is not used unless earned that way.

## 7. Outcome classification

Every slice receives exactly one of:

- **Non-inferior** at `delta`
- **Below margin**
- **Inconclusive** at achieved n

Inconclusive is a reportable result, not a failure.

## 8. Statistical method

- **Primary:** exact matched-binary interval on discordant pairs, one-sided 95%
- **Secondary check:** paired bootstrap resampled by item, 10,000 resamples, one-sided 95%, fixed seed
- Slices where all paired differences are zero are handled by the exact method; the bootstrap's zero-width interval there is not reported as a result
- If the two methods disagree, both are reported

Planned n from `n >= (1.645^2 * d) / delta^2` where `d` is discordance estimated on the dev split. **This is a budget estimate, not a guarantee of conclusiveness.** Reported separately: planned n, observed discordance, achieved interval width.

## 9. Pinned configuration

**Local**
```
model:          Qwen/Qwen3-1.7B-GGUF
quantization:   Q8_0 (the only quantization the official repo publishes)
file hash:      <fill on Day 0>
runtime:        llama.cpp, commit <fill on Day 0>
mode:           non-thinking (/no_think)
sampling:       temperature 0.7, top_p 0.8, top_k 20, min_p 0, presence_penalty 1.5
seed:           <fixed, fill on Day 0>
max_tokens:     <fill on Day 0>
hardware:       Intel Core Ultra 7 <exact model>, <threads>, <RAM>
```

Greedy decoding is not used. Qwen's model card states "DO NOT use greedy decoding" and recommends `presence_penalty 1.5` specifically for quantized models.

**Hosted**
```
provider:       Groq (free plan)
model id:       llama-3.3-70b-versatile
seed support:   <confirm on Day 0>
```

Chosen over `openai/gpt-oss-120b` because reasoning tokens make that model's token budget unpredictable, and over `llama-3.1-8b-instant` because an 8B model is a poor reference for this claim.

**Single seed limitation.** One seed gives no estimate of sampling variance. Mitigated by running 3 seeds on the dev split only and reporting the spread as a variance floor. Not eliminated.

## 10. Prompting

Identical semantic prompt to both models. Chat templates differ by deployment and that difference is recorded in the manifest.

- Multiple choice: `Please show your choice in the answer field with only the choice letter, e.g., "answer": "C".`
- Math: `Please reason step by step, and put your final answer within \boxed{}.`

Zero-shot.

## 11. Splits, frozen

| Split | Size | Use |
|---|---|---|
| dev | ~150 items | Prompt selection, parser construction, discordance estimate, seed variance. Never reported. |
| map | ~100 per slice | The non-inferiority analysis. |
| router | ~150 items | Frozen routing policy evaluation only. |

Split by item id with a fixed seed. Split file committed with this document.

## 12. Scoring rules

- Unparseable responses score **incorrect**; unparseable rate reported per model as its own result
- Persistent API failures after retry score **incorrect**; reliability reported separately
- Items are never silently dropped
- Normalization rules frozen after dev-split work and versioned

## 13. Router policies

Evaluated on the frozen router split:

1. Always local
2. Always hosted
3. Map-based (requires caller-supplied `X-GoodEnough-Task` metadata)
4. Local-first cascade, escalate on parse failure (minimal operational baseline, not a production cascade)

Plus oracle routing from known per-item outcomes, reported only as an upper bound.

**Anticipated in advance:** the map-based router may route almost everything hosted. At 1.7B versus 70B with a 10-point margin, few or no slices may qualify. This is a legitimate result.

## 14. Declared limitations

- Two configurations cannot isolate the effect of model size. This comparison bundles model size, quantization, execution location, CPU versus accelerator, network latency, and provider queueing.
- Both models may have encountered MMLU and GSM8K during training. This is a valid comparison on those items, not an uncontaminated generalization estimate.
- Results describe two pinned deployment configurations on predeclared benchmark slices. They do not describe arbitrary production requests.
- Electricity and hardware amortization are out of scope. Local cost is reported as zero incremental API spend, not as economically free.

## 15. What would falsify the hypothesis

Non-inferiority not established on either primary slice at `delta = 10`. This is a publishable outcome and will be reported as such.

---

## Amendment log

| Date | Change | Reason | Results affected |
|---|---|---|---|
| | | | |
