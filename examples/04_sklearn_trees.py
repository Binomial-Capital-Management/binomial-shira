"""scikit-learn trees, forests and boosting on SHIRA: import the same class from shira.sklearn.

    before:  from sklearn.ensemble import RandomForestRegressor
    after:   from shira.sklearn import RandomForestRegressor
"""

import numpy as np

import shira.sklearn as sk
from shira.sklearn import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(50_000, 16))
y = 2.0 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.5, size=50_000)
X_train, X_test, y_train, y_test = X[:40_000], X[40_000:], y[:40_000], y[40_000:]

for make in (lambda: RandomForestRegressor(n_estimators=50, n_jobs=4, random_state=0),
             lambda: ExtraTreesRegressor(n_estimators=50, n_jobs=4, random_state=0),
             lambda: HistGradientBoostingRegressor(max_iter=200, random_state=0),
             lambda: GradientBoostingRegressor(n_estimators=50, random_state=0)):
    model = make().fit(X_train, y_train)
    rmse = float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))
    print(f"{type(model).__name__:<30} ran on: {sk.last_backend:<8} test RMSE: {rmse:.4f}")

clf = RandomForestClassifier(n_estimators=50, n_jobs=4, random_state=0).fit(X_train, (y_train > 0).astype(int))
accuracy = float(np.mean(clf.predict(X_test) == (y_test > 0)))
print(f"{'RandomForestClassifier':<30} ran on: {sk.last_backend:<8} test accuracy: {accuracy:.4f}")

# every class is the scikit-learn class: pipelines, grid search and pickling work as before
print("is a scikit-learn estimator:", clf.__class__.__mro__[1].__module__.startswith("sklearn"))
