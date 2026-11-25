"""See what SHIRA did with a call.

A call SHIRA covers runs on SHIRA. Any other call runs on the stock library, unchanged, and SHIRA logs
one line saying why. Each module's `last_backend` tells you which one ran; turn on logging to read the
reason.
"""

import logging

import numpy as np

import shira.xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

rng = np.random.default_rng(0)
X = rng.normal(size=(20_000, 8))
y = X[:, 0] + rng.normal(scale=0.5, size=20_000)

xgb.XGBRegressor(n_estimators=50).fit(X, y)
print("default call          ran on:", xgb.last_backend)

# the linear booster is not covered in this version: it runs on stock XGBoost, and the log line above says so
xgb.XGBRegressor(n_estimators=50, booster="gblinear").fit(X, y)
print("booster='gblinear'    ran on:", xgb.last_backend)
