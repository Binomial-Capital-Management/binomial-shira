# Using SHIRA

## Change one import

| Before | After |
|---|---|
| `import xgboost as xgb` | `import shira.xgboost as xgb` |
| `import lightgbm as lgb` | `import shira.lightgbm as lgb` |
| `import catboost as cb` | `import shira.catboost as cb` |
| `from sklearn.ensemble import RandomForestRegressor` | `from shira.sklearn import RandomForestRegressor` |
| `from sklearn.linear_model import Ridge` | `from shira.sklearn import Ridge` |
| `from sklearn.model_selection import cross_val_score` | `from shira.sklearn import cross_val_score` |

Every scikit-learn class SHIRA covers is imported from `shira.sklearn`, whichever scikit-learn module
it lives in.

The rest of your code stays as it is: the same classes, functions, parameters and methods.

## Or switch a whole program

```python
import shira
shira.patch()          # once, before the program trains anything

import xgboost as xgb  # the program's own imports, unchanged
```

`shira.unpatch()` puts the stock libraries back.

## What you get back

The stock library's own object: an `xgboost.Booster`, a `lightgbm.Booster`, a CatBoost model, a
scikit-learn estimator. Saving, loading, predicting, pickling, pipelines and grid search work as
before, and a saved model opens in the stock library without SHIRA installed.

## Calls SHIRA does not cover

They run on the stock library, unchanged. SHIRA logs one line that says why, on the loggers
`shira.xgboost`, `shira.lightgbm`, `shira.catboost` and `shira.sklearn`. Each module's `last_backend`
tells you which one ran the last fit:

```python
import shira.lightgbm as lgb
lgb.LGBMRegressor().fit(X, y)
print(lgb.last_backend)        # "shira", or the stock library's name
```

`shira.sklearn.backend_history` keeps the list for every scikit-learn fit, in order.

## Threads

SHIRA respects the thread arguments you already pass (`n_jobs`, `nthread`, `num_threads`,
`thread_count`). `SHIRA_THREADS` or `OMP_NUM_THREADS` caps its threads when you pass none.

## Examples

One runnable script per use case is in [`examples/`](../examples).
