# GSM8K difficulty

How each model's accuracy falls as a problem needs more reasoning steps.
Slope is the change in log-odds of a correct answer per extra step; a more negative slope means faster degradation.

| model | n | correct | incorrect | overall acc | slope per step | 90% CI |
|---|---:|---:|---:|---:|---:|:---:|
| local | 150 | 119 | 31 | 0.79 | -0.512 | [-0.797, -0.298] |
| hosted | 150 | 148 | 2 | 0.99 | not estimable | not estimable |

No slope is reported for hosted. It answered 148 of 150 items correctly (accuracy 0.99), leaving 2 errors across the whole split. That is too few errors to estimate how accuracy changes with step count: the logistic fit is near separable, so any slope it produces is set by those 2 responses and its interval runs to a boundary. The floor used here is 5 responses in the rarer outcome, applied to both models. The fitted value is kept in reports/gsm8k.json flagged `slope_interpretable: false`, and it should not be read as an estimate.


## Accuracy by reasoning steps

| steps | local n | local acc | hosted n | hosted acc |
|---:|---:|---:|---:|---:|
| 0 | 2 | 1.00 | 2 | 1.00 |
| 1 | 7 | 1.00 | 7 | 1.00 |
| 2 | 46 | 0.87 | 46 | 0.98 |
| 3 | 34 | 0.88 | 34 | 0.97 |
| 4 | 31 | 0.68 | 31 | 1.00 |
| 5 | 22 | 0.68 | 22 | 1.00 |
| 6 | 5 | 0.60 | 5 | 1.00 |
| 7 | 2 | 0.50 | 2 | 1.00 |
| 8 | 1 | 0.00 | 1 | 1.00 |