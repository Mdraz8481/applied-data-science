# Do Height and Weight Predict Grip Strength?

Grip strength is a common proxy for physical frailty, but it needs a dynamometer to
measure. Height and weight need a tape and a scale. If grip strength tracks closely with
either, the cheaper measurement could stand in for it.

**[→ Read the notebook](notebooks/health_metrics_correlation.ipynb)**

## Data

10 observations: height (inches), weight (pounds), age, grip strength, and a Y/N frailty
flag. Included in [`data/raw/`](data/raw/).

## Method

1. Encode the binary `Frailty` flag to 0/1.
2. For each predictor, compute Pearson *r*, its *p*-value, and a 95% confidence interval
   via the Fisher *z*-transform.
3. Plot each predictor against grip strength with a least-squares reference line.

The confidence intervals are the point of the analysis. A bare correlation coefficient
from 10 rows invites over-reading; the interval shows how little the number constrains.

## Findings

| Predictor | *r* | 95% CI | *p* | Verdict |
|---|---|---|---|---|
| Height (inches) | −0.17 | [−0.72, 0.52] | 0.64 | Inconclusive |
| Weight (pounds) | 0.03 | [−0.61, 0.65] | 0.93 | Inconclusive |
| Age | 0.13 | [−0.54, 0.70] | 0.71 | Inconclusive |

![Grip strength vs each predictor](results/grip_strength_scatter.png)

**Every interval spans zero, and every one is about 1.3 wide.** The data is equally
consistent with a strong positive relationship, a strong negative one, or none at all.

The honest conclusion is not "height and weight don't predict grip strength" — it's
**"10 observations cannot answer this question."** Detecting a moderate correlation
(*r* = 0.5) at 80% power needs roughly 29 observations.

## Limitations

- *n* = 10. This is the dominant constraint on everything above.
- Pearson correlation assumes a linear relationship; a non-linear one would be missed.
- No control for age or frailty when assessing height and weight.

## Next step

Collect more observations. Below roughly *n* = 30 this design cannot distinguish a real
moderate effect from noise.

## Outputs

- `data/processed/cleaned_yield_data.csv` — encoded, all columns retained
- `data/processed/cleaned_data.csv` — reduced to age, grip strength, frailty
- `results/grip_strength_scatter.png`
