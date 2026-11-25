"""scikit-learn linear and statistics models on SHIRA.

    before:  from sklearn.linear_model import Ridge
    after:   from shira.sklearn import Ridge

These models match the stock library to rounding; there is no mode to set.
"""

import numpy as np
from sklearn.linear_model import Ridge as StockRidge

import shira.sklearn as sk
from shira.sklearn import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge

rng = np.random.default_rng(0)
X = rng.normal(size=(200_000, 32))
coef = rng.normal(size=32)
y = X @ coef + rng.normal(scale=1.0, size=200_000)

for model in (LinearRegression(), Ridge(alpha=1.0), Lasso(alpha=0.01), ElasticNet(alpha=0.01, l1_ratio=0.5)):
    model.fit(X, y)
    error = float(np.max(np.abs(model.coef_ - coef)))
    print(f"{type(model).__name__:<18} ran on: {sk.last_backend:<8} largest coefficient error: {error:.4f}")

clf = LogisticRegression(max_iter=200).fit(X, (y > 0).astype(int))
print(f"{'LogisticRegression':<18} ran on: {sk.last_backend:<8} train accuracy: {clf.score(X, (y > 0).astype(int)):.4f}")

# the same answer as stock, to rounding
gap = float(np.max(np.abs(Ridge(alpha=1.0).fit(X, y).coef_ - StockRidge(alpha=1.0).fit(X, y).coef_)))
print(f"Ridge against stock scikit-learn, largest difference: {gap:.1e}")
