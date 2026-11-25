"""Cross-validation on SHIRA: xgb.cv, lgb.cv and scikit-learn's cross_val_score.

The calls are the stock ones; only the import changes.
"""

import numpy as np

import shira.lightgbm as lgb
import shira.xgboost as xgb
from shira.sklearn import Ridge, cross_val_score

rng = np.random.default_rng(0)
X = rng.normal(size=(30_000, 16))
y = 2.0 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.5, size=30_000)

result = xgb.cv({"objective": "reg:squarederror", "max_depth": 6, "eta": 0.1},
                xgb.DMatrix(X, label=y), num_boost_round=100, nfold=5, seed=0)
print(f"xgb.cv            final test RMSE: {float(result['test-rmse-mean'].iloc[-1]):.4f}")

result = lgb.cv({"objective": "regression", "num_leaves": 31, "learning_rate": 0.1, "verbose": -1},
                lgb.Dataset(X, label=y), num_boost_round=100, nfold=5, seed=0, stratified=False)
key = next(name for name in result if name.endswith("-mean"))
print(f"lgb.cv            final {key}: {float(result[key][-1]):.4f}")

scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=5)
print(f"cross_val_score   Ridge, mean R^2 over 5 folds: {float(np.mean(scores)):.4f}")
