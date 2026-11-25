"""CatBoost on SHIRA: change the import, keep the code.

    before:  import catboost as cb
    after:   import shira.catboost as cb
"""

import numpy as np

import shira.catboost as cb

rng = np.random.default_rng(0)
X = rng.normal(size=(50_000, 16))
y = 2.0 * X[:, 0] - X[:, 1] + np.sin(X[:, 2]) + rng.normal(scale=0.5, size=50_000)
X_train, X_test, y_train, y_test = X[:40_000], X[40_000:], y[:40_000], y[40_000:]

model = cb.CatBoostRegressor(iterations=300, depth=6, verbose=False, allow_writing_files=False)
model.fit(X_train, y_train)
rmse = float(np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))
print(f"CatBoostRegressor  ran on: {cb.last_backend}   test RMSE: {rmse:.4f}")

# a classifier, the same way
labels = (y > 0).astype(int)
clf = cb.CatBoostClassifier(iterations=300, depth=6, verbose=False, allow_writing_files=False)
clf.fit(X_train, labels[:40_000])
accuracy = float(np.mean(clf.predict(X_test) == labels[40_000:]))
print(f"CatBoostClassifier ran on: {cb.last_backend}   test accuracy: {accuracy:.4f}")
