"""LightGBM on SHIRA: change the import, keep the code.

    before:  import lightgbm as lgb
    after:   import shira.lightgbm as lgb
"""

import numpy as np

import shira.lightgbm as lgb

rng = np.random.default_rng(0)
X = rng.normal(size=(50_000, 16))
signal = 2.0 * X[:, 0] - X[:, 1] + np.sin(X[:, 2])
y = (signal + rng.normal(scale=1.0, size=50_000) > 0).astype(int)
X_train, X_test, y_train, y_test = X[:40_000], X[40_000:], y[:40_000], y[40_000:]

# the scikit-learn style
model = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.1, verbose=-1)
model.fit(X_train, y_train)
accuracy = float(np.mean(model.predict(X_test) == y_test))
print(f"LGBMClassifier ran on: {lgb.last_backend}   test accuracy: {accuracy:.4f}")

# the native style
booster = lgb.train({"objective": "binary", "num_leaves": 31, "learning_rate": 0.1, "verbose": -1},
                    lgb.Dataset(X_train, label=y_train), num_boost_round=200)
accuracy = float(np.mean((booster.predict(X_test) > 0.5) == y_test))
print(f"lgb.train      ran on: {lgb.last_backend}   test accuracy: {accuracy:.4f}")
