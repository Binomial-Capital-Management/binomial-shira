# Bit-identical results

SHIRA has controls to produce results bit-identical to the stock library, and by default is much
faster within a very tight error bound.

## The default

For tree models SHIRA takes its fastest route by default. The model you get is a valid model of the
stock library, trained by a different route, so its numbers are not the same digits as the stock
library's.

## The settings

Set these environment variables before your program starts. SHIRA then gives exactly the stock
library's results, digit for digit.

| Models | Setting |
|---|---|
| XGBoost | `SHIRA_XGB_MODE=ordinary` |
| LightGBM | `SHIRA_LGB_MODE=ordinary` |
| CatBoost | `SHIRA_CAT_MODE=ordinary` |
| scikit-learn HistGradientBoosting | `SHIRA_HGB_MODE=ordinary` |
| scikit-learn RandomForest, ExtraTrees and single trees | `SHIRA_FOREST_MODE=ordinary` |
| scikit-learn GradientBoosting | `SHIRA_GB_MODE=ordinary` |

From a shell:

```
SHIRA_XGB_MODE=ordinary SHIRA_LGB_MODE=ordinary SHIRA_CAT_MODE=ordinary \
SHIRA_HGB_MODE=ordinary SHIRA_FOREST_MODE=ordinary SHIRA_GB_MODE=ordinary \
python your_program.py
```

In Python, set them before the first `import shira`. See
[`examples/09_bit_identical.py`](../examples/09_bit_identical.py).

In this mode a call that SHIRA cannot reproduce exactly runs on the stock library, so the result is
the stock library's either way.

## Linear and statistics models

scikit-learn's linear and statistics models (LinearRegression, Ridge, Lasso, ElasticNet,
LogisticRegression and the others) have no setting. SHIRA matches the stock library to rounding on
them.

## Comparing with stock yourself

scikit-learn's forests add their trees in the order their threads finish when they predict on several
threads, so a stock forest can differ from itself in the last digit from run to run. Predict with
`n_jobs=1` when you compare forests digit for digit.
