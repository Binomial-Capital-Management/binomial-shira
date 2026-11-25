"""Results bit-identical to the stock library.

By default SHIRA takes its fastest route for tree models. Set the mode variables BEFORE your program
starts (or, as here, before SHIRA is imported) and SHIRA gives exactly the stock library's results.
From a shell:

    SHIRA_XGB_MODE=ordinary SHIRA_LGB_MODE=ordinary SHIRA_CAT_MODE=ordinary \
    SHIRA_HGB_MODE=ordinary SHIRA_FOREST_MODE=ordinary SHIRA_GB_MODE=ordinary  python your_program.py
"""

import os

for name in ("SHIRA_XGB_MODE", "SHIRA_LGB_MODE", "SHIRA_CAT_MODE", "SHIRA_HGB_MODE", "SHIRA_FOREST_MODE", "SHIRA_GB_MODE"):
    os.environ[name] = "ordinary"

import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import xgboost  # noqa: E402

import shira.lightgbm  # noqa: E402
import shira.xgboost  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(20_000, 16))
y = 2.0 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.5, size=20_000)
X_test = rng.normal(size=(2_000, 16))

ours = shira.xgboost.XGBRegressor(n_estimators=100, random_state=0).fit(X, y).predict(X_test)
theirs = xgboost.XGBRegressor(n_estimators=100, random_state=0).fit(X, y).predict(X_test)
print(f"XGBoost   ran on: {shira.xgboost.last_backend}   identical to stock: {bool(np.array_equal(ours, theirs))}")

ours = shira.lightgbm.LGBMRegressor(n_estimators=100, random_state=0, verbose=-1).fit(X, y).predict(X_test)
theirs = lightgbm.LGBMRegressor(n_estimators=100, random_state=0, verbose=-1).fit(X, y).predict(X_test)
print(f"LightGBM  ran on: {shira.lightgbm.last_backend}   identical to stock: {bool(np.array_equal(ours, theirs))}")
