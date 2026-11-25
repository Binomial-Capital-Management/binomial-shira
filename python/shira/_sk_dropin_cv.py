from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn import model_selection as _ms
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import KFold, PredefinedSplit, TimeSeriesSplit, check_cv

from . import _sk_dropin_hgb as _hgb

STOCK = {"cross_val_score": _ms.cross_val_score, "cross_validate": _ms.cross_validate}
MAX_BLOCKS = 4096
LINEAR = ("Ridge", "LinearRegression")
HGB = ("HistGradientBoostingRegressor", "HistGradientBoostingClassifier")


def _runs(idx: np.ndarray) -> list[tuple[int, int]]:
    idx = np.asarray(idx, dtype=np.int64)
    assert idx.size > 0 and np.all(np.diff(idx) > 0)
    cut = np.flatnonzero(np.diff(idx) != 1)
    lo = np.concatenate([[idx[0]], idx[cut + 1]])
    hi = np.concatenate([idx[cut] + 1, [idx[-1] + 1]])
    return list(zip(lo.tolist(), hi.tolist()))


def _sorted(idx: Any) -> np.ndarray:
    idx = np.asarray(idx)
    if idx.size > 1 and bool((idx[1:] > idx[:-1]).all()):
        return idx
    return np.sort(idx)


def _deterministic(cv: Any) -> bool:
    if isinstance(cv, KFold):
        return not cv.shuffle
    return isinstance(cv, (TimeSeriesSplit, PredefinedSplit)) or type(cv).__name__ == "_CVIterableWrapper"


def _linear_ok(est: Any, name: str) -> bool:
    if getattr(est, "positive", False):
        return False
    if name == "Ridge":
        return bool(est.fit_intercept) and est.solver in ("auto", "cholesky") and np.ndim(est.alpha) == 0
    return True


def _plan(estimator: Any, X: Any, y: Any, groups: Any, cv: Any, params: Any) -> dict | None:
    from . import sklearn as sx

    name = sx.stock_class(estimator).__name__
    arrays = isinstance(X, np.ndarray) and X.ndim == 2 and isinstance(y, np.ndarray) and y.ndim == 1
    if not arrays or groups is not None or params or name not in LINEAR + HGB:
        return None
    cvo = check_cv(cv, y, classifier=is_classifier(estimator))
    if not _deterministic(cvo):
        return None
    splits = [(_sorted(tr), np.asarray(te)) for tr, te in cvo.split(X, y)]
    runs = [_runs(tr) for tr, _ in splits]
    if name in HGB:
        ok = all(len(r) == 1 for r in runs) and _hgb.why_not(estimator, X, y, {}) is None and _hgb.x86_guard(X) is None
        return {"kind": "hgb", "name": name, "splits": splits, "folds": [r[0] for r in runs]} if ok else None
    if not _linear_ok(estimator, name) or sum(len(r) for r in runs) > MAX_BLOCKS:
        return None
    return {"kind": "linear", "name": name, "splits": splits, "runs": runs}


def _fit_linear(estimator: Any, X: np.ndarray, y: np.ndarray, plan: dict) -> list:
    from . import linear as sl
    from . import linear_more as lm
    from . import windows as sw

    xf, yf = np.ascontiguousarray(X, dtype=np.float64), np.ascontiguousarray(y, dtype=np.float64)
    k0 = min(512, xf.shape[0])
    ref, ref_y = xf[:k0].mean(axis=0), float(yf[:k0].mean())
    edges = sorted({e for runs in plan["runs"] for r in runs for e in r})
    tots = sw.block_totals_many(xf, yf, np.ones(xf.shape[0]), np.asarray(edges, dtype=np.int64), ref, ref_y,
                                threads=_hgb.n_threads())
    assert tots.shape[0] == len(edges) - 1
    blocks = {(a, b): sl.unpack(tots[i], xf.shape[1]) for i, (a, b) in enumerate(zip(edges[:-1], edges[1:]))}
    out = []
    for runs in plan["runs"]:
        parts = [blocks[(a, b)] for lo, hi in runs for a, b in zip(edges[:-1], edges[1:]) if lo <= a and b <= hi]
        t = sl.sum_totals(parts)
        est = clone(estimator)
        est.n_features_in_ = xf.shape[1]
        if plan["name"] == "Ridge":
            est.coef_, est.intercept_ = sl.fit_ridge(t, float(est.alpha), ref, ref_y)
            est.n_iter_, est.solver_ = None, "cholesky"
        else:
            n_rows = int(sum(hi - lo for lo, hi in runs))
            r = lm.fit_linreg(t, ref, ref_y, n_rows, fit_intercept=bool(est.fit_intercept))
            est.coef_, est.rank_, est.singular_ = r["coef"], r["rank"], r["singular"]
            est.intercept_ = r["intercept"] if est.fit_intercept else 0.0
        out.append(est)
    return out


def _fit_all(estimator: Any, X: np.ndarray, y: np.ndarray, plan: dict) -> tuple:
    from . import sklearn as sx

    if plan["kind"] == "linear":
        try:
            return _fit_linear(estimator, X, y, plan), None
        except (RuntimeError, AssertionError) as exc:
            return None, f"engine: {exc}"
    base = sx.stock_class(estimator)

    def stock_fit(est: Any, x: Any, yy: Any, sw: Any) -> Any:
        return base.fit(est, x, yy, sample_weight=sw)

    return _hgb.fit_folds(lambda: clone(estimator), stock_fit, X, y, plan["folds"], plan["name"].endswith("Classifier"))


def _lockstep(estimator: Any, X: Any, y: Any, groups: Any, scoring: Any, cv: Any, params: Any) -> dict | None:
    from . import sklearn as sx

    plan = _plan(estimator, X, y, groups, cv, params)
    if plan is None or not (scoring is None or isinstance(scoring, str)):
        return None
    t0 = time.perf_counter()
    ests, why = _fit_all(estimator, X, y, plan)
    if ests is None:
        sx.LOG.warning("shira.sklearn: cross-validation on scikit-learn's own loop: %s", why)
        return None
    t1 = time.perf_counter()
    scorer = check_scoring(estimator, scoring=scoring)
    k = len(ests)
    if plan["kind"] == "hgb":
        scores, fast = _hgb.score_folds(ests, X, y, plan["splits"], scorer)
        sx._record("shira", f"{plan['name']} ({k} folds in one call, {fast}/{k} scored on SHIRA's forest)")
    else:
        scores = np.array([scorer(e, X[te], y[te]) for e, (_, te) in zip(ests, plan["splits"])])
        sx._record("shira", f"{plan['name']} ({k} folds in one call)")
    return {"test_score": scores, "fit_time": np.full(k, (t1 - t0) / k),
            "score_time": np.full(k, (time.perf_counter() - t1) / k)}


def cross_val_score(estimator: Any, X: Any, y: Any = None, *, groups: Any = None, scoring: Any = None,
                    cv: Any = None, **kwargs: Any) -> np.ndarray:
    res = _lockstep(estimator, X, y, groups, scoring, cv, kwargs.get("params"))
    if res is None:
        return STOCK["cross_val_score"](estimator, X, y, groups=groups, scoring=scoring, cv=cv, **kwargs)
    return res["test_score"]


def cross_validate(estimator: Any, X: Any, y: Any = None, *, groups: Any = None, scoring: Any = None,
                   cv: Any = None, **kwargs: Any) -> dict:
    fancy = any(kwargs.get(k) for k in ("return_train_score", "return_estimator", "return_indices", "params"))
    res = None if fancy else _lockstep(estimator, X, y, groups, scoring, cv, None)
    if res is None:
        return STOCK["cross_validate"](estimator, X, y, groups=groups, scoring=scoring, cv=cv, **kwargs)
    return res
