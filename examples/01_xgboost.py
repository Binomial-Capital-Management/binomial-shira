"""XGBoost on SHIRA: change the import, keep the code.

    before:  import xgboost as xgb
    after:   import shira.xgboost as xgb
"""

import numpy as np

import shira.xgboost as xgb

rng = np.random.default_rng(0)
X = rng.normal(size=(50_000, 16))
y = 2.0 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.5, size=50_000)
X_train, X_test, y_train, y_test = X[:40_000], X[40_000:], y[:40_000], y[40_000:]

# the scikit-learn style
model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)
rmse = float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))
print(f"XGBRegressor   ran on: {xgb.last_backend}   test RMSE: {rmse:.4f}")

# the native style
booster = xgb.train({"objective": "reg:squarederror", "max_depth": 6, "eta": 0.1},
                    xgb.DMatrix(X_train, label=y_train), num_boost_round=200)
rmse = float(np.sqrt(np.mean((booster.predict(xgb.DMatrix(X_test)) - y_test) ** 2)))
print(f"xgb.train      ran on: {xgb.last_backend}   test RMSE: {rmse:.4f}")

# what you get back is a stock XGBoost model: save it, load it, use it anywhere
booster.save_model("model.json")
print("saved model.json, a standard XGBoost model file")
