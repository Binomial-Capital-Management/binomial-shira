"""Switch a whole program to SHIRA without touching its imports.

Call shira.patch() once, before the program trains anything. The stock names then run on SHIRA.
shira.unpatch() puts the stock libraries back.
"""

import numpy as np

import shira

shira.patch()

# from here on these are the program's own, unchanged imports
import lightgbm as lgb  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(30_000, 16))
y = 2.0 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.5, size=30_000)

xgb.XGBRegressor(n_estimators=100).fit(X, y)
lgb.LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
RandomForestRegressor(n_estimators=20, n_jobs=4, random_state=0).fit(X, y)
Ridge(alpha=1.0).fit(X, y)

import shira.lightgbm  # noqa: E402
import shira.sklearn  # noqa: E402
import shira.xgboost  # noqa: E402

print("XGBRegressor          ran on:", shira.xgboost.last_backend)
print("LGBMRegressor         ran on:", shira.lightgbm.last_backend)
print("forest, then Ridge    ran on:", shira.sklearn.backend_history[-2:])

shira.unpatch()
print("unpatched: the stock libraries are back")
