# SHIRA

Simons Hardware Integrated Runtime Architecture.


SHIRA makes scikit-learn, XGBoost, LightGBM and CatBoost faster. You keep the code you have and
change one import.

SHIRA has controls to produce results bit-identical to the stock library, and by default is much
faster within a very tight error bound.

The release note, with the full performance table in its appendix, is
[docs/SHIRA-0.1-release.pdf](docs/SHIRA-0.1-release.pdf).

## Install

SHIRA 0.1.0 runs on:

- macOS 15 or later on Apple Silicon
- Linux on AMD x86-64

It needs Python 3.13. Download the wheel for your platform from the
[Releases page](https://github.com/Binomial-Capital-Management/binomial-shira/releases) and install it:

```
pip install shira-0.1.0-py313-none-macosx_15_0_arm64.whl        # Mac
pip install shira-0.1.0-py313-none-manylinux_2_28_x86_64.whl    # Linux
```

On Linux the chip must support AVX2 (every AMD Zen chip, from 2017 on) and the system must have
glibc 2.27 or later.

pip also installs the versions of the stock libraries SHIRA is built against: scikit-learn 1.8.0,
XGBoost 3.2.0, LightGBM 4.6.0, CatBoost 1.2.10 and NumPy 2.4.2. Install into a virtual environment
with pip; other install layouts are not supported in this version.

## Use

Change the import. Everything after it stays the same.

| Before | After |
|---|---|
| `import xgboost as xgb` | `import shira.xgboost as xgb` |
| `import lightgbm as lgb` | `import shira.lightgbm as lgb` |
| `import catboost as cb` | `import shira.catboost as cb` |
| `from sklearn.ensemble import RandomForestRegressor` | `from shira.sklearn import RandomForestRegressor` |

```python
import shira.xgboost as xgb

model = xgb.XGBRegressor(n_estimators=200)
model.fit(X, y)
predictions = model.predict(X_test)
```

To switch a whole program without touching its imports, call `shira.patch()` once at the top:

```python
import shira
shira.patch()

import xgboost as xgb          # now runs on SHIRA
from sklearn.linear_model import Ridge
```

`shira.unpatch()` puts the stock libraries back.

What you get back is the stock library's own object (an `xgboost.Booster`, a `lightgbm.Booster`, a
CatBoost model, a scikit-learn estimator), so saving, loading, predicting and everything else you do
with it work as before.

A call SHIRA does not cover yet runs on the stock library, and SHIRA logs one line saying why. Each
module's `last_backend` tells you which one ran:

```python
import shira.lightgbm as lgb
lgb.LGBMRegressor().fit(X, y)
print(lgb.last_backend)        # "shira", or the stock library's name
```

## Bit-identical results

By default SHIRA takes its fastest route for tree models. Set these environment variables before
your program starts to get results bit-identical to the stock library instead:

| Models | Setting |
|---|---|
| XGBoost | `SHIRA_XGB_MODE=ordinary` |
| LightGBM | `SHIRA_LGB_MODE=ordinary` |
| CatBoost | `SHIRA_CAT_MODE=ordinary` |
| scikit-learn HistGradientBoosting | `SHIRA_HGB_MODE=ordinary` |
| scikit-learn RandomForest, ExtraTrees and single trees | `SHIRA_FOREST_MODE=ordinary` |
| scikit-learn GradientBoosting | `SHIRA_GB_MODE=ordinary` |

scikit-learn's linear and statistics models (LinearRegression, Ridge, Lasso, LogisticRegression, PCA
and the others) have no setting: SHIRA matches the stock library to rounding on them.

## What is covered today

<!-- coverage:start -->
**By default** SHIRA runs these calls itself:

| Library | Calls |
|---|---|
| XGBoost | default regression, pseudo-Huber, absolute error, quantile, Poisson, Tweedie, binary logistic, multiclass softprob, sample weights, missing values, early stopping, lossguide, subsample + colsample, monotone constraints, max_depth 12, max_bin 512, scale_pos_weight 3, num_parallel_tree 4, booster dart, tree_method approx, tree_method exact, ranking pairwise, ranking ndcg, categorical feature, custom objective, XGBRegressor.fit, XGBClassifier.fit, XGBClassifier multiclass, XGBRanker.fit, xgb.cv, QuantileDMatrix, pandas DataFrame |
| LightGBM | default regression, L1, Huber, quantile, Poisson, binary, multiclass, lambdarank, sample weights, missing values, early stopping, categorical feature, pre-built Dataset, bagging + feature_fraction, GOSS, DART, random forest mode, linear_tree, extra_trees, monotone constraints, max_depth 6, max_bin 511, LGBMRegressor.fit, LGBMClassifier.fit, LGBMRanker.fit, lgb.cv, pandas DataFrame |
| CatBoost | default regression (RMSE), MAE, Quantile, Poisson, binary (Logloss), MultiClass, sample weights, missing values, early stopping, depth 10, Ordered boosting, Bernoulli bootstrap, Bayesian bootstrap, MVS bootstrap, rsm 0.5, grow_policy Depthwise, grow_policy Lossguide, categorical feature (CTR), categorical one-hot, categorical + binary, categorical + MultiClass, categorical + weights + MultiClass, Pool input, pandas DataFrame |
| scikit-learn | `DecisionTreeRegressor`, `DecisionTreeClassifier`, `ExtraTreeRegressor`, `ExtraTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor`, `ExtraTreesClassifier`, `IsolationForest`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`, `LinearRegression`, `Ridge`, `LogisticRegression`, `Lasso`, `ElasticNet`, `BayesianRidge`, `ARDRegression`, `Lars`, `LassoLars`, `TruncatedSVD`, `FactorAnalysis`, `GaussianNB`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`, `PCA` |

**With the bit-identical settings** these calls give exactly the stock library's results:

| Library | Calls |
|---|---|
| XGBoost | default regression, pseudo-Huber, absolute error, quantile, Poisson, Tweedie, binary logistic, sample weights, missing values, early stopping, lossguide, subsample + colsample, monotone constraints, max_depth 12, max_bin 512, scale_pos_weight 3, ranking pairwise, ranking ndcg, custom objective, XGBRegressor.fit, XGBClassifier.fit, XGBRanker.fit, xgb.cv, QuantileDMatrix, pandas DataFrame |
| LightGBM | default regression, L1, Huber, quantile, Poisson, binary, multiclass, lambdarank, sample weights, missing values, early stopping, categorical feature, pre-built Dataset, bagging + feature_fraction, GOSS, DART, max_depth 6, max_bin 511, LGBMRegressor.fit, LGBMClassifier.fit, LGBMRanker.fit, lgb.cv, pandas DataFrame |
| CatBoost | default regression (RMSE), MAE, Quantile, sample weights, missing values, early stopping, depth 10, Bernoulli bootstrap, Bayesian bootstrap, MVS bootstrap, Pool input, pandas DataFrame |
| scikit-learn | `DecisionTreeRegressor`, `DecisionTreeClassifier`, `ExtraTreeRegressor`, `ExtraTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor`, `ExtraTreesClassifier`, `IsolationForest`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`, `LinearRegression`, `Ridge`, `LogisticRegression`, `Lasso`, `ElasticNet`, `BayesianRidge`, `ARDRegression`, `Lars`, `LassoLars`, `TruncatedSVD`, `FactorAnalysis`, `GaussianNB`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |

In that mode any other call runs on the stock library, so the result is the stock library's either way.
<!-- coverage:end -->

One difference on Linux: with the bit-identical settings, CatBoost fits with sample weights run on
the stock library.

## Next

Data loading, neural nets, LLM inference, GPUs, Intel, Windows.

## Licence

The Python files are open source under the Apache License, Version 2.0 (`LICENSE`). The compiled
library that ships inside the package is free to use and to redistribute with the package, under the
SHIRA Binary Licence (`LICENSE-BINARY`). `NOTICE` lists the libraries SHIRA works with and their
licences.

## Examples and docs

- [`examples/`](examples): one short runnable script per use case.
- [Install](docs/install.md), [using SHIRA](docs/usage.md), [bit-identical results](docs/bit-identical.md),
  [what is covered](docs/coverage.md), [troubleshooting](docs/troubleshooting.md).
