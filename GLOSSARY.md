# Glossary

Plain-language definitions of every term used in this project, grouped by topic.
Where a word has a specific meaning *here*, that is noted.

---

## Models and running them

**LLM (large language model)**
A program that predicts text one token at a time. Both models in this project are LLMs.

**Parameters (the "1.7B" and "70B")**
The internal numbers a model learned during training, counted in billions. More parameters usually means a smarter, larger, slower, more expensive model. This project compares a small 1.7-billion-parameter model against a large 70-billion one.

**Token**
A chunk of text, roughly a word-piece. Models read and write in tokens, and costs and limits are measured in tokens. "The" is one token; "capital" might be one or two.

**Tokens per second (tok/s)**
How fast a model generates text. Your laptop produces about 40 tok/s; Groq's hardware is far faster.

**Inference**
Running a trained model to get an answer (as opposed to training it). Everything this project does is inference.

**Quantization / Q4_K_M / Q8_0**
Shrinking a model by storing its numbers at lower precision so it fits and runs on modest hardware. `Q4_K_M` is a common 4-bit setting; `Q8_0` is 8-bit (bigger, more precise). You use Q4_K_M because the official Q8_0 file produced garbage on your machine.

**GGUF**
The file format llama.cpp uses to store a quantized model. Your model file is a `.gguf`.

**llama.cpp / llama-server**
Free software that runs LLMs efficiently on ordinary CPUs. `llama-server` is its mode that exposes the model over a local web address so your scripts can call it.

**Local vs hosted**
"Local" means the small model running on your own laptop CPU. "Hosted" means the large model running on Groq's servers, reached over the internet.

**Groq**
A company that serves large models very fast on special hardware, with a free tier. It is your hosted reference model.

**Latency**
How long one request takes, end to end, in milliseconds. A core thing this project measures.

**TTFT (time to first token)**
How long until the model produces its first token. Only measurable with streaming, which this project does not use, so TTFT is deliberately out of scope.

---

## The experiment

**Benchmark**
A fixed set of questions with known correct answers, used to measure how good a model is.

**MMLU**
A well-known benchmark of multiple-choice questions across many subjects (geography, law, math, etc.). This project uses eight of its subjects.

**GSM8K**
A benchmark of grade-school math word problems, used here to study how the small model degrades as problems require more reasoning steps.

**MCQ (multiple-choice question)**
A question with options A-D and one correct letter. MMLU is all MCQs, which makes scoring a simple letter comparison.

**Slice / subject**
One MMLU subject, e.g. `formal_logic`. The project reports results per slice, so you can see where the small model is and is not good enough.

**Primary vs exploratory slices**
The two slices chosen in advance to carry the main conclusion (`high_school_geography`, `formal_logic`) are "primary". The other six are "exploratory", reported but not headline. Naming them in advance stops you from cherry-picking a flattering result later.

**Splits: dev, map, router**
Three separate pools of questions, kept apart on purpose:
- **dev**: a practice set for building and tuning the scorer. Never reported.
- **map**: the real measurement set that produces the results.
- **router**: a held-back set used only to test the routing step at the end.
Keeping them separate prevents you from accidentally tuning on the same data you report, which would inflate the result.

**Pre-registration**
Writing down the exact plan, including the pass/fail threshold, *before* looking at any results, so you cannot move the goalposts. `PREREGISTRATION.md` is that document, and it is frozen.

**Pinned config**
The exact, recorded settings (which model file, which sampling values, which seed) so anyone can reproduce the run. "Pinning" means fixing a value so it cannot silently drift.

**Seed**
A fixed starting number that makes a "random" process repeatable. With the same seed, the same items are selected and the model samples the same way every time. This project uses seed 42 everywhere.

**Deterministic**
Produces the same output every time. The item selection is deterministic because of the seed.

---

## The statistics

**Accuracy**
The fraction of questions a model gets right. The headline quality measure.

**Non-inferiority**
The specific claim being tested: not "is the small model better", but "is the small model *no worse than* the big model by more than an allowed margin". A deliberately humble, precise question.

**Margin (delta)**
The largest accuracy drop you are willing to call "good enough". Set to 10 percentage points here, chosen before seeing results. Also reported at 5 and 15 so readers can judge for themselves.

**Confidence interval (CI)**
A range that expresses uncertainty. Instead of "the gap is 4 points", you say "the gap is between 1 and 7 points with 95% confidence". Small samples give wide, less useful intervals.

**One-sided**
The interval only cares about one direction here: how far *below* the big model the small one might be. Being better is not the question.

**Discordance**
The fraction of questions where the two models disagree (one right, one wrong). It drives how many questions you need: the more they disagree, the more items required to reach a confident answer.

**Power / underpowered**
"Power" is the ability of a test to reach a clear conclusion. "Underpowered" means too few questions to decide, so the honest result is "inconclusive". Some small slices here are underpowered by necessity, and that is disclosed in advance.

**Inconclusive**
A legitimate third outcome, alongside "good enough" and "not good enough". It means the data cannot decide at this sample size. Reporting it honestly is a feature, not a failure.

**Bootstrap**
A method to estimate a confidence interval by resampling your data many times. Used here as a cross-check on the primary method.

**Matched-binary / paired**
Because both models answer the *same* questions, you compare them item by item (paired), which is more precise than comparing two separate averages.

**Sensitivity analysis**
Reporting the result at several margins (5, 10, 15) so the conclusion does not hinge on one arbitrary threshold.

---

## Routing (the final step)

**Router**
A dispatcher that decides, per request, whether to use the cheap local model or the expensive hosted one, based on the map.

**Policy**
A rule the router follows. The project compares several: always-local, always-hosted, map-based, and a cascade.

**Cascade**
Try the cheap model first, escalate to the expensive one only if the cheap answer looks bad. A simple baseline here, not a production-grade design.

**Oracle**
A cheat used only as a yardstick: a "router" that magically knows each answer in advance. It shows the best possible routing, so real policies can be measured against a ceiling.

**Metadata-aware**
The map is organized by subject, but a live request does not announce its subject. The map-based router therefore needs the caller to label the request. This is stated as a limitation.

---

## Infrastructure and tooling

**API**
A way for programs to talk to a service over the internet. Groq exposes its model through an API.

**Rate limit (RPM / RPD / TPD)**
Caps on how much you can use a free service: Requests Per Minute, Requests Per Day, Tokens Per Day. Groq's token-per-day cap is why the run spans several days.

**429 (Too Many Requests)**
The error a server returns when you exceed a rate limit. Both the freeze script and the runner handle it by waiting and retrying.

**Cache**
Saved past results so repeated work is instant and free. Essential here because cached tokens do not count against Groq's daily limit, and because a multi-day run must not redo finished work.

**Resumable**
A program that can stop partway (for example when the daily token cap is hit) and continue later without losing progress.

**SQLite**
A tiny self-contained database, stored as a single file, where every result row is saved.

**.env**
A file holding secrets like your Groq API key. It is listed in `.gitignore` so it is never committed to the public repo.

**.gitignore**
A list of files git should ignore, keeping secrets, large models, and databases out of the repository.

**Branch / commit / push**
Git terms. A **commit** is a saved snapshot of your work. A **branch** is a parallel line of commits. **Push** uploads your commits to GitHub. **main** is the primary branch others see first.

**Manifest**
A record file (`MANIFEST.json`) listing exactly what was frozen: sources, counts, seed, and checksums, so the frozen data is verifiable.

**Checksum (sha256)**
A short fingerprint of a file. If the file changes by even one character, the fingerprint changes, so it proves the frozen data has not been altered.
