from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

LOG = logging.getLogger("shira.sklearn")
FAMILIES = ("RandomForestRegressor", "ExtraTreesRegressor", "RandomForestClassifier", "ExtraTreesClassifier",
            "DecisionTreeRegressor", "ExtraTreeRegressor", "DecisionTreeClassifier", "ExtraTreeClassifier")
CLF_CRITERIA = ("gini", "entropy", "log_loss")
REG_CRITERIA = ("squared_error", "friedman_mse", "poisson", "absolute_error")
PILOT_TREES = 6
PILOT_TOL = 0.02
PILOT_HOLD = 20_000
PILOT_VALUES = 200_000
log: list[dict] = []
last_check: dict = {}


def mode_on() -> bool:
    return os.environ.get("SHIRA_FOREST_MODE", "") != "ordinary"


def _switch(env: str, name: str, features: int) -> int:
    from . import _machine

    v = os.environ.get(env, "")
    out = int(v) if v else int(_machine.value(name, int(features)))
    assert out >= 0
    return out


def sample_rows(features: int) -> int:
    return _switch("SHIRA_FOREST_FIRST_READ_ROWS", "forest_first_read_rows", features)


def min_rows(features: int) -> int:
    return _switch("SHIRA_FOREST_FIRST_READ_MIN_ROWS", "forest_first_read_min_rows", features)


MIN_SAMPLE = 8


def shape_rows(n: int, features: int, env: str, name: str, floor_name: str = "sk_small_floor_rows", span: int = 2) -> int:
    from . import _machine

    m = _switch(env, name, features)
    assert n >= 0 and m >= 0 and span >= 2
    if m >= 1 and n >= span * m:
        return m
    floor = int(_machine.value(floor_name, int(features)))
    if m < 1 or os.environ.get(env, "").strip() or n <= floor:
        return 0
    s = max(floor, n * int(_machine.value("sk_small_sample_pct", int(features))) // 100)
    assert floor <= s < n
    return min(s, m) if s >= MIN_SAMPLE else 0


def check_on(features: int) -> bool:
    return _switch("SHIRA_FOREST_FIRST_READ_CHECK", "forest_first_read_check", features) == 1


SINGLE_EXTRA = ("ExtraTreeRegressor", "ExtraTreeClassifier")


def tree_check_on(est: Any, features: int) -> bool:
    assert features >= 1
    if type(est).__name__ == "ExtraTreesClassifier":
        return _switch("SHIRA_EXTRA_TREES_CLF_FIRST_READ_CHECK", "extra_trees_clf_first_read_check", features) == 1
    if type(est).__name__ not in SINGLE_EXTRA:
        return True
    return _switch("SHIRA_EXTRA_TREE_FIRST_READ_CHECK", "extra_tree_first_read_check", features) == 1


ETC_NAMES = ("ExtraTreesClassifier", "ExtraTreesRegressor")


def _partial_features(est: Any, features: int) -> bool:
    mf = getattr(est, "max_features", 1.0)
    if isinstance(mf, str):
        return mf in ("sqrt", "log2")
    if isinstance(mf, float):
        return mf < 1.0
    return isinstance(mf, int) and not isinstance(mf, bool) and mf < int(features)


def forest_rows(est: Any, n: int, features: int) -> int:
    floor = "extra_tree_small_floor_rows" if type(est).__name__ in SINGLE_EXTRA else "sk_small_floor_rows"
    span = 4 if type(est).__name__ in SINGLE_EXTRA else 2
    m = shape_rows(n, features, "SHIRA_FOREST_FIRST_READ_ROWS", "forest_first_read_rows", floor, span)
    assert n >= 0 and m >= 0
    if m < 1 or type(est).__name__ not in ETC_NAMES or not _partial_features(est, features):
        return m
    pct = _switch("SHIRA_FOREST_FIRST_READ_ETC_SAMPLE_PCT", "forest_first_read_etc_sample_pct", features)
    if pct < 1 or os.environ.get("SHIRA_FOREST_FIRST_READ_ROWS", "").strip():
        return m
    return max(m, min(n * pct // 100, n - 1))


def min_row_trees(features: int) -> int:
    return _switch("SHIRA_FOREST_FIRST_READ_MIN_ROW_TREES", "forest_first_read_min_row_trees", features)


def trees_of(est: Any) -> list:
    return [e.tree_ for e in est.estimators_] if hasattr(est, "estimators_") else [est.tree_]


def is_frame(X: Any) -> bool:
    dts = getattr(X, "dtypes", None)
    if dts is None or not hasattr(X, "iloc") or not hasattr(X, "to_numpy"):
        return False
    try:
        return len(dts) >= 1 and all(isinstance(d, np.dtype) and d.kind in "fiu" for d in dts)
    except TypeError:
        return False


def take(X: Any, rows: np.ndarray) -> Any:
    return X.iloc[rows] if is_frame(X) else np.ascontiguousarray(X[rows])


def values(X: Any) -> np.ndarray:
    return np.ascontiguousarray(X.to_numpy(dtype=np.float64) if is_frame(X) else X, dtype=np.float64)


def fit_args(args: tuple, kwargs: dict) -> tuple | None:
    if len(args) == 3 and not kwargs:
        X, y, w = args
    elif len(args) == 2 and set(kwargs) <= {"sample_weight"}:
        X, y, w = args[0], args[1], kwargs.get("sample_weight")
    else:
        return None
    array = isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype.kind in "fiu"
    if not (array or is_frame(X)) or np.ndim(y) != 1 or len(y) != X.shape[0]:
        return None
    if w is None:
        return X, y, None
    if not (isinstance(w, np.ndarray) and w.ndim == 1 and w.dtype.kind in "fiu" and w.shape[0] == X.shape[0]):
        return None
    w = np.ascontiguousarray(w, dtype=np.float64)
    return (X, y, w) if bool(np.isfinite(w).all()) and float(w.min()) >= 0.0 and float(w.sum()) > 0.0 else None


def why_not(est: Any, name: str, args: tuple, kwargs: dict) -> str | None:
    if not mode_on():
        return "SHIRA_FOREST_MODE=ordinary"
    if name not in FAMILIES:
        return f"{name}: no first read"
    xyw = fit_args(args, kwargs)
    if xyw is None:
        return "fit arguments, X, y or sample_weight outside the first read"
    X = xyw[0]
    clf = name.endswith("Classifier")
    if est.criterion not in (CLF_CRITERIA if clf else REG_CRITERIA):
        return f"criterion {est.criterion!r}"
    if est.criterion == "absolute_error" and (xyw[2] is not None or est.monotonic_cst is not None):
        return "criterion 'absolute_error' with sample weights or monotonic constraints (not yet)"
    held = len(getattr(est, "estimators_", ())) if getattr(est, "warm_start", False) else 0
    if held and int(est.n_estimators) <= held:
        return "warm_start with no new tree (stock warns)"
    m = forest_rows(est, X.shape[0], X.shape[1])
    lo = min_rows(X.shape[1])
    if m < 1 or X.shape[0] < lo:
        return f"{X.shape[0]} rows: under the first read's row bound (min rows {lo}, small-data floor or 2 x the sample)"
    if not hasattr(est, "n_estimators"):
        one = _switch("SHIRA_TREE_FIRST_READ_MIN_ROWS", "tree_first_read_min_rows", X.shape[1])
        return None if X.shape[0] >= one else f"{X.shape[0]} rows: under the single tree's first-read row bound ({one})"
    work = int(X.shape[0]) * int(est.n_estimators)
    if work < min_row_trees(X.shape[1]):
        return f"{work} rows x trees: under the first read's work bound ({min_row_trees(X.shape[1])})"
    return None


def stride(n: int, m: int) -> np.ndarray:
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    assert rows.size == m and int(rows[0]) == 0 and int(rows[-1]) == n - 1
    return rows


def class_rows(y: np.ndarray, rows: np.ndarray, pool: np.ndarray | None = None) -> np.ndarray:
    if y.ndim != 1:
        return rows
    src = np.arange(y.shape[0], dtype=np.int64) if pool is None else pool
    classes = np.unique(y[src])
    assert rows.size >= 1 and src.size >= rows.size
    if np.unique(y[rows]).size >= classes.size:
        return rows
    parts = []
    for c in classes:
        own = src[y[src] == c]
        k = min(own.size, max(1, int(round(rows.size * own.size / src.size))))
        parts.append(own[stride(own.size, k)] if k >= 2 else own[:1])
    out = np.sort(np.concatenate(parts))
    assert np.unique(y[out]).size == classes.size
    return out


def _class_sets(est: Any, y: np.ndarray, sets: list, vrows: np.ndarray, hold: np.ndarray, n: int) -> tuple[list, np.ndarray]:
    if not hasattr(est, "predict_proba"):
        return sets, vrows
    pool = np.setdiff1d(np.arange(n, dtype=np.int64), hold)
    out = [class_rows(y, r, pool) for r in sets]
    assert len(out) == len(sets) and not np.intersect1d(hold, np.concatenate(out)).size
    return out, np.union1d(vrows, np.concatenate(out))


def pilot_rows(n: int, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.int64)
    hold = idx[3::4]
    hold = hold[stride(hold.size, min(hold.size, PILOT_HOLD))]
    pool = idx[(idx & 3) != 3]
    sample = pool[stride(pool.size, min(m, pool.size))]
    values = np.union1d(sample, pool[stride(pool.size, min(pool.size, PILOT_VALUES))])
    assert hold.size >= 1 and sample.size >= 2 and not np.intersect1d(hold, values).size
    return sample[::2], sample, values, hold


def check_rule(features: int) -> tuple[int, float, int]:
    rule = _switch("SHIRA_FOREST_FIRST_READ_CHECK_RULE", "forest_first_read_check_rule", features)
    vals = _switch("SHIRA_FOREST_FIRST_READ_CHECK_VALUES", "forest_first_read_check_values", features)
    assert rule in (1, 2) and vals >= 1
    return rule, (0.005 if rule == 2 else PILOT_TOL), vals


def pilot_sets(n: int, m: int, rule: int, vals: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    idx = np.arange(n, dtype=np.int64)
    hold = idx[3::4]
    hold = hold[stride(hold.size, min(hold.size, PILOT_HOLD))]
    pool = idx[(idx & 3) != 3]
    top = pool[stride(pool.size, min((2 if rule == 2 else 1) * m, pool.size))]
    sets = [top[::4], top[::2], top] if rule == 2 else [top[::2], top]
    values = np.union1d(top, pool[stride(pool.size, min(pool.size, vals))])
    assert hold.size >= 1 and sets[0].size >= 2 and not np.intersect1d(hold, values).size
    return sets, values, hold


def _pilot_error(pilot: Any, xv: np.ndarray, yv: np.ndarray, xh: Any, yh: np.ndarray, wv: Any, wh: Any, threads: int) -> float:
    if not hasattr(pilot, "predict_proba"):
        resolve(pilot, xv, yv, threads, wv)
        return float(np.average((pilot.predict(xh) - yh) ** 2, weights=wh))
    resolve(pilot, xv, encode_classes(pilot, yv), threads, wv)
    onehot = np.eye(int(np.asarray(pilot.classes_).size))[encode_classes(pilot, yh)]
    return float(np.average(np.mean((pilot.predict_proba(xh) - onehot) ** 2, axis=1), weights=wh))


def _reuse_ok(est: Any, n: int, features: int, first: int) -> bool:
    if _switch("SHIRA_FOREST_FIRST_READ_CHECK_REUSE", "forest_first_read_check_reuse", features) != 1 or first:
        return False
    rs = est.random_state
    m = shape_rows(n, features, "SHIRA_FOREST_FIRST_READ_ROWS", "forest_first_read_rows")
    return (int(getattr(est, "n_estimators", 0)) >= PILOT_TREES and (rs is None or isinstance(rs, int))
            and m >= 2 and forest_rows(est, n, features) == m)


def _sub_forest(est: Any, k: int) -> Any:
    import copy

    pilot = copy.copy(est)
    pilot.estimators_ = [copy.deepcopy(t) for t in est.estimators_[:k]]
    pilot.n_estimators = len(pilot.estimators_)
    assert 1 <= len(pilot.estimators_) <= k
    return pilot


def learning_curve_flat(est: Any, X: Any, Xa: np.ndarray, y: np.ndarray, w: np.ndarray | None, fit_pilot: Any,
                        threads: int, wr: np.ndarray | None = None, m_model: Any = None) -> tuple[bool, float]:
    from sklearn.base import clone

    rule, bound, vals = check_rule(X.shape[1])
    if rule == 1 and m_model is None and vals == PILOT_VALUES:
        return _curve_rule1(est, X, Xa, y, w, fit_pilot, threads, wr)
    m = shape_rows(X.shape[0], X.shape[1], "SHIRA_FOREST_FIRST_READ_ROWS", "forest_first_read_rows")
    sets, vrows, hold = pilot_sets(X.shape[0], m, rule, vals)
    sets, vrows = _class_sets(est, y, sets, vrows, hold, X.shape[0])
    xv, xh = np.ascontiguousarray(Xa[vrows]), take(X, hold)
    wv, wh = (None, None) if w is None else (w[vrows], w[hold])
    wv = wv if wr is None else wr[vrows]
    if wh is not None and float(wh.sum()) <= 0.0:
        return False, float("nan")
    err = []
    for j, rows in enumerate(sets):
        if j == 1 and m_model is not None:
            pilot = _sub_forest(m_model, PILOT_TREES)
        else:
            pilot = clone(est).set_params(random_state=0)
            if hasattr(est, "n_estimators"):
                pilot.set_params(n_estimators=min(PILOT_TREES, int(est.n_estimators)))
            if not fit_pilot(pilot, take(X, rows), y[rows], None if w is None else w[rows]):
                return False, float("nan")
        try:
            err.append(_pilot_error(pilot, xv, y[vrows], xh, y[hold], wv, wh, threads))
        except ValueError:
            return False, float("nan")
    falls = [1.0 - err[i + 1] / err[i] if err[i] > 0.0 else 0.0 for i in range(len(err) - 1)]
    fall = max(falls)
    return fall <= bound, fall


def _curve_rule1(est: Any, X: Any, Xa: np.ndarray, y: np.ndarray, w: np.ndarray | None, fit_pilot: Any,
                 threads: int, wr: np.ndarray | None = None) -> tuple[bool, float]:
    from sklearn.base import clone

    half, sample, vrows, hold = pilot_rows(X.shape[0], shape_rows(X.shape[0], X.shape[1], "SHIRA_FOREST_FIRST_READ_ROWS",
                                                                  "forest_first_read_rows"))
    (half, sample), vrows = _class_sets(est, y, [half, sample], vrows, hold, X.shape[0])
    xv, xh = np.ascontiguousarray(Xa[vrows]), take(X, hold)
    clf = hasattr(est, "predict_proba")
    wv, wh = (None, None) if w is None else (w[vrows], w[hold])
    wv = wv if wr is None else wr[vrows]
    if wh is not None and float(wh.sum()) <= 0.0:
        return False, float("nan")
    err = []
    for rows in (half, sample):
        pilot = clone(est).set_params(random_state=0)
        if hasattr(est, "n_estimators"):
            pilot.set_params(n_estimators=min(PILOT_TREES, int(est.n_estimators)))
        if not fit_pilot(pilot, take(X, rows), y[rows], None if w is None else w[rows]):
            return False, float("nan")
        if not clf:
            resolve(pilot, xv, y[vrows], threads, wv)
            err.append(float(np.average((pilot.predict(xh) - y[hold]) ** 2, weights=wh)))
            continue
        try:
            resolve(pilot, xv, encode_classes(pilot, y[vrows]), threads, wv)
            yh = encode_classes(pilot, y[hold])
        except ValueError:
            return False, float("nan")
        onehot = np.eye(int(np.asarray(pilot.classes_).size))[yh]
        err.append(float(np.average(np.mean((pilot.predict_proba(xh) - onehot) ** 2, axis=1), weights=wh)))
    fall = 1.0 - err[1] / err[0] if err[0] > 0.0 else 0.0
    return fall <= PILOT_TOL, fall


def _impurity(frac: np.ndarray, criterion: str) -> np.ndarray:
    if criterion == "gini":
        return 1.0 - np.sum(frac * frac, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(frac > 0.0, frac * np.log2(frac), 0.0), axis=1)


def encode_classes(est: Any, y: np.ndarray) -> np.ndarray:
    cls = np.asarray(est.classes_)
    ix = np.clip(np.searchsorted(cls, y), 0, cls.size - 1)
    if not bool(np.all(cls[ix] == y)):
        raise ValueError("a label outside the sample's classes")
    return ix


def _row_counts(trees: list, x32: np.ndarray, threads: int) -> list[np.ndarray]:
    from shira_cpp import node_stats_cpp

    return node_stats_cpp.node_stats(trees, x32, np.zeros(x32.shape[0]), None, threads)[0]


def resolve_classes(est: Any, X: np.ndarray, yk: np.ndarray, threads: int, w: np.ndarray | None = None,
                    first: int = 0) -> None:
    from shira_cpp import node_stats_cpp

    trees = trees_of(est)[first:]
    K = int(np.asarray(est.classes_).size)
    assert trees and K >= 2 and all(t.n_outputs == 1 and t.value.shape[2] == K for t in trees)
    x32 = np.ascontiguousarray(X, dtype=np.float32)
    parts, total = [], None
    for k in range(K - 1):
        ow, owy, _ = node_stats_cpp.node_stats(trees, x32, (yk == k).astype(np.float64), w, threads)
        parts.append(owy)
        total = ow
    rows = total if w is None else _row_counts(trees, x32, threads)
    for t, tr in enumerate(trees):
        tot, cnt = total[t], rows[t]
        assert cnt.size == tr.node_count and float(cnt.min()) >= 1.0, "a node no row reaches (the sample is part of X)"
        has = tot > 0.0
        frac = np.array(tr.value[:, 0, :], dtype=np.float64)
        for k in range(K - 1):
            frac[has, k] = parts[k][t][has] / tot[has]
        frac[has, K - 1] = np.maximum(1.0 - frac[has, :K - 1].sum(axis=1), 0.0)
        tr.impurity[:] = np.maximum(_impurity(frac, est.criterion), 0.0)
        if est.monotonic_cst is not None:
            frac[:, 0] = _mono_values(tr, frac[:, 0], -np.asarray(est.monotonic_cst, dtype=np.int64))
            frac[:, 1] = 1.0 - frac[:, 0]
        tr.value[:, 0, :] = frac
        tr.n_node_samples[:] = cnt.astype(np.int64)
        tr.weighted_n_node_samples[:] = tot


def resolve(est: Any, X: np.ndarray, y: np.ndarray, threads: int, w: np.ndarray | None = None, first: int = 0) -> None:
    from shira_cpp import node_stats_cpp

    if hasattr(est, "classes_"):
        resolve_classes(est, X, y, threads, w, first)
        return
    trees = trees_of(est)[first:]
    assert trees and all(t.n_outputs == 1 for t in trees)
    x32 = np.ascontiguousarray(X, dtype=np.float32)
    if est.criterion == "absolute_error":
        _resolve_median(trees, x32, np.asarray(y, dtype=np.float64), threads)
        return
    ow, owy, owy2 = node_stats_cpp.node_stats(trees, x32, np.asarray(y, dtype=np.float64), w, threads)
    rows = ow if w is None else _row_counts(trees, x32, threads)
    ylogy = None
    if est.criterion == "poisson":
        yv = np.asarray(y, dtype=np.float64)
        ylogy = node_stats_cpp.node_stats(trees, x32, np.where(yv > 0.0, yv * np.log(np.where(yv > 0.0, yv, 1.0)), 0.0),
                                          w, threads)[1]
    for i, (t, tot, sy, sy2, cnt) in enumerate(zip(trees, ow, owy, owy2, rows)):
        assert cnt.size == t.node_count and float(cnt.min()) >= 1.0, "a node no row reaches (the sample is part of X)"
        has = tot > 0.0
        mean = sy[has] / tot[has]
        t.value[has, 0, 0] = mean
        if ylogy is None:
            t.impurity[has] = np.maximum(sy2[has] / tot[has] - mean * mean, 0.0)
        else:
            ok = has & (sy > 0.0)
            t.impurity[ok] = np.maximum((ylogy[i][ok] - np.log(sy[ok] / tot[ok]) * sy[ok]) / tot[ok], 0.0)
        t.n_node_samples[:] = cnt.astype(np.int64)
        t.weighted_n_node_samples[:] = tot
        if est.monotonic_cst is not None:
            t.value[:, 0, 0] = _mono_values(t, np.array(t.value[:, 0, 0], dtype=np.float64),
                                            np.asarray(est.monotonic_cst, dtype=np.int64))


def _resolve_median(trees: list, x32: np.ndarray, y: np.ndarray, threads: int) -> None:
    from shira_cpp import node_stats_cpp

    ids, nl = node_stats_cpp.leaf_ids(trees, x32, threads)
    counts = node_stats_cpp.node_stats(trees, x32, np.zeros(x32.shape[0]), None, threads)[0]
    for t, tr in enumerate(trees):
        med = node_stats_cpp.line_search_values(ids[t:t + 1], nl[t:t + 1], y, None, 0.0, 1.0, "absolute_error",
                                                nthread=max(1, min(64, threads)))
        leaves = np.flatnonzero(tr.children_left == -1)
        assert leaves.size == int(nl[t]) and float(counts[t].min()) >= 1.0
        got = counts[t][leaves] > 0
        tr.value[leaves[got], 0, 0] = med[0, :leaves.size][got]
        tr.n_node_samples[:] = counts[t].astype(np.int64)
        tr.weighted_n_node_samples[:] = counts[t]


def _mono_values(tr: Any, mean: np.ndarray, cst: np.ndarray) -> np.ndarray:
    n = int(tr.node_count)
    L, R, F = tr.children_left, tr.children_right, tr.feature
    assert mean.shape == (n,) and cst.ndim == 1
    lo, hi, out = np.full(n, -np.inf), np.full(n, np.inf), mean.copy()
    for v in range(n):
        out[v] = lo[v] if mean[v] < lo[v] else (hi[v] if mean[v] > hi[v] else mean[v])
        a, b = int(L[v]), int(R[v])
        if a < 0:
            continue
        assert a > v and b > v
        c = int(cst[F[v]])
        lo[a], hi[a], lo[b], hi[b] = lo[v], hi[v], lo[v], hi[v]
        if c != 0:
            mid = mean[a] / 2.0 + mean[b] / 2.0
            mid = min(max(mid, lo[v]), hi[v])
            if c > 0:
                hi[a], lo[b] = mid, mid
            else:
                lo[a], hi[b] = mid, mid
    return out


def _oob_full(est: Any, X: Any, rows: np.ndarray) -> tuple[str, np.ndarray]:
    clf = hasattr(est, "predict_proba")
    name = "oob_decision_function_" if clf else "oob_prediction_"
    samp = np.asarray(getattr(est, name))
    n = int(X.shape[0])
    assert samp.shape[0] == rows.size
    rest = np.setdiff1d(np.arange(n, dtype=np.int64), rows, assume_unique=True)
    full = np.empty((n,) + samp.shape[1:], dtype=np.float64)
    full[rows] = samp
    if rest.size:
        xr = take(X, rest)
        full[rest] = est.predict_proba(xr) if clf else np.asarray(est.predict(xr)).reshape((rest.size,) + samp.shape[1:])
    return name, full


def _class_weights(est: Any, y: np.ndarray, w: np.ndarray | None) -> np.ndarray | None:
    from sklearn.utils.class_weight import compute_sample_weight

    cw = getattr(est, "class_weight", None)
    if cw is None:
        return w
    ex = compute_sample_weight("balanced" if cw == "balanced_subsample" else cw, y)
    assert ex.shape == y.shape and np.isfinite(ex).all()
    return ex if w is None else ex * w


def fit(est: Any, name: str, X: Any, y: Any, w: np.ndarray | None, fit_any: Any, threads: int) -> bool:
    n = X.shape[0]
    rows = stride(n, forest_rows(est, n, X.shape[1]))
    clf = name.endswith("Classifier")
    y1 = np.asarray(y) if clf else np.asarray(y, dtype=np.float64)
    rows = class_rows(y1, rows) if clf else rows
    live = np.random.get_state() if est.random_state is None else None
    wr = _class_weights(est, y1, w) if clf else w
    first = len(getattr(est, "estimators_", ())) if getattr(est, "warm_start", False) else 0
    tp = time.perf_counter()
    Xa = values(X)
    checked = check_on(X.shape[1]) and tree_check_on(est, X.shape[1])
    reuse = checked and _reuse_ok(est, n, X.shape[1], first)
    if reuse:
        rows = pilot_sets(n, shape_rows(n, X.shape[1], "SHIRA_FOREST_FIRST_READ_ROWS", "forest_first_read_rows"),
                          *check_rule(X.shape[1])[::2])[0][1]
        if not fit_any(est, take(X, rows), y1[rows], None if w is None else w[rows]):
            return False
    flat, fall = learning_curve_flat(est, X, Xa, y1, w, fit_any, threads, wr, est if reuse else None) if checked \
        else (True, float("nan"))
    t0 = time.perf_counter()
    if checked:
        last_check.update(n=n, flat=flat, fall=fall, seconds=t0 - tp)
    if not flat and live is not None and reuse:
        np.random.set_state(live)
    if not flat or (not reuse and not fit_any(est, take(X, rows), y1[rows], None if w is None else w[rows])):
        return False
    t1 = time.perf_counter()
    oob = _oob_full(est, X, rows) if getattr(est, "oob_score", False) else None
    try:
        resolve(est, Xa, encode_classes(est, y1) if clf else y1, threads, wr, first)
    except ValueError as e:
        LOG.warning("shira.sklearn: %s: the first read did not apply (ValueError: %s); today's route", name, e)
        if live is not None:
            np.random.set_state(live)
        return False
    t2 = time.perf_counter()
    if oob is not None:
        setattr(est, oob[0], oob[1])
    est._shira_fr_rows = rows
    count = len(trees_of(est))
    check = (f"of which {t0 - tp:.2f} s the learning-curve check: the held-out error fell {100 * fall:.1f} % when the "
             f"pilot sample doubled (the larger fall, rule {check_rule(X.shape[1])[0]}), bound "
             f"{100 * check_rule(X.shape[1])[1]:.1f} %" if checked else "no learning-curve check")
    line = (f"fitted the first-read way (SHIRA's default): {count} trees grown by SHIRA's engine on a stride "
            f"sample of {rows.size} of {n} rows (not stock's trees; accuracy depends on the data), every node's value, impurity and "
            f"row counts taken from all {n} rows ({t2 - tp:.2f} s, {check}); "
            f"SHIRA_FOREST_MODE=ordinary turns it off")
    log.append({"name": name, "n": n, "m": int(rows.size), "trees": count, "threads": threads,
                "t_check": t0 - tp, "t_shapes": t1 - t0, "t_values": t2 - t1, "fall": fall, "line": line})
    del log[:-64]
    return True


GB_NAME = "GradientBoostingRegressor"
GB_NAMES = (GB_NAME, "GradientBoostingClassifier")


def gb_why_not(est: Any, name: str, args: tuple, kwargs: dict) -> str | None:
    if os.environ.get("SHIRA_GB_MODE", "") == "ordinary":
        return "SHIRA_GB_MODE=ordinary"
    xyw = fit_args(args, kwargs)
    if name not in GB_NAMES or xyw is None:
        return "not a GradientBoosting fit(X, y[, sample_weight]) the first read takes"
    X = xyw[0]
    why = _gb_loss_why(est, name)
    if why:
        return why
    done = len(getattr(est, "estimators_", ())) if est.warm_start else 0
    if est.warm_start and done and int(est.n_estimators) <= done:
        return "warm_start with no new stage (stock warns or raises)"
    m = shape_rows(X.shape[0], X.shape[1], "SHIRA_GB_FIRST_READ_ROWS", "gb_first_read_rows")
    lo = _switch("SHIRA_GB_FIRST_READ_MIN_ROWS", "gb_first_read_min_rows", X.shape[1])
    if m < 1 or X.shape[0] < lo:
        return f"{X.shape[0]} rows: under the first read's row bound (min rows {lo}, small-data floor or 2 x the sample)"
    return None


def _gb_loss_why(est: Any, name: str) -> str | None:
    plain = est.init is None and not (est.warm_start and len(getattr(est, "estimators_", ())))
    if name == GB_NAME and est.loss == "squared_error":
        return None
    if name == GB_NAME and est.loss in ("absolute_error", "quantile", "huber"):
        return None if plain else f"loss {est.loss!r} with an init estimator or a warm start (not yet)"
    if name != GB_NAME and est.loss == "log_loss":
        return None
    return f"loss {est.loss!r} (not yet)"


def _gb_start(est: Any, x32: np.ndarray, first: int) -> np.ndarray:
    from sklearn.ensemble._gradient_boosting import predict_stages

    raw = np.ascontiguousarray(est._raw_predict_init(x32), dtype=np.float64)
    if first > 0:
        predict_stages(est.estimators_[:first], x32, float(est.learning_rate), raw)
    assert raw.shape == (x32.shape[0], 1) and np.isfinite(raw).all()
    return raw[:, 0]


def gb_resolve(est: Any, X: np.ndarray, y: np.ndarray, threads: int, w: np.ndarray | None = None,
               first: int = 0) -> None:
    from shira_cpp import node_stats_cpp, product_cpp

    if est.loss != "squared_error":
        _gb_resolve_other(est, X, y, threads, w, first)
        return
    trees = [e[0].tree_ for e in est.estimators_[first:]]
    assert trees and est.estimators_.shape[1] == 1 and first >= 0
    x32 = np.ascontiguousarray(X, dtype=np.float32)
    ids, nl = node_stats_cpp.leaf_ids(trees, x32, threads)
    lr = float(est.learning_rate)
    if est.init is not None or first > 0:
        val, _ = product_cpp.product(ids, nl, np.ascontiguousarray(y, dtype=np.float32), 0.0, "reg:squarederror", eta=lr,
                                     reg_lambda=0.0, min_child_weight=0.0, max_delta_step=0.0, nthread=threads,
                                     weight=None if w is None else np.ascontiguousarray(w, dtype=np.float32),
                                     pred=_gb_start(est, x32, first).astype(np.float32))
        _gb_write(trees, val, nl, lr)
        return
    base = float(np.float32(np.average(y, weights=w)))
    val = product_cpp.product_window(ids, 0, X.shape[0], nl, np.ascontiguousarray(y, dtype=np.float32), base, "reg:squarederror",
                                     eta=lr, reg_lambda=0.0, min_child_weight=0.0, max_delta_step=0.0,
                                     weight=None if w is None else np.ascontiguousarray(w, dtype=np.float32), nthread=threads)
    _gb_write(trees, val, nl, lr)
    est.init_.constant_ = np.array([[base]], dtype=np.float64)


def _gb_write(trees: list, val: np.ndarray, nl: np.ndarray, lr: float) -> None:
    assert val.shape[0] == len(trees) and np.isfinite(val).all()
    for t, tr in enumerate(trees):
        leaves = np.flatnonzero(tr.children_left == -1)
        assert leaves.size == int(nl[t])
        tr.value[leaves, 0, 0] = val[t, :leaves.size].astype(np.float64) / lr


def _gb_resolve_other(est: Any, X: np.ndarray, y: np.ndarray, threads: int, w: np.ndarray | None, first: int) -> None:
    from shira_cpp import node_stats_cpp

    x32 = np.ascontiguousarray(X, dtype=np.float32)
    wv = np.ones(X.shape[0]) if w is None else np.asarray(w, dtype=np.float64)
    if est.init is None and first == 0:
        est.init_.fit(x32, y, sample_weight=w)
    trees = [e.tree_ for stage in est.estimators_[first:] for e in stage]
    ids, nl = node_stats_cpp.leaf_ids(trees, x32, threads)
    assert ids.shape == (len(trees), X.shape[0]) and first >= 0
    lr = float(est.learning_rate)
    if hasattr(est, "classes_"):
        val = _gb_classes(est, x32, y, wv, ids, nl, first, lr, threads)
    elif est.loss in ("absolute_error", "quantile"):
        base = float(np.ravel(est._raw_predict_init(x32[:1]))[0])
        val = node_stats_cpp.line_search_values(ids, nl, np.asarray(y, dtype=np.float64), wv, base, lr, est.loss,
                                                float(est.alpha), nthread=max(1, min(64, threads)))
    else:
        base = float(np.ravel(est._raw_predict_init(x32[:1]))[0])
        val = node_stats_cpp.huber_values(ids, nl, np.asarray(y, dtype=np.float64), wv, base, lr, float(est.alpha),
                                          nthread=max(1, min(64, threads)))
        _gb_write(trees, val, nl, 1.0)
        return
    _gb_write(trees, val, nl, lr)


def _gb_classes(est: Any, x32: np.ndarray, y: np.ndarray, wv: np.ndarray, ids: np.ndarray, nl: np.ndarray, first: int,
                lr: float, threads: int) -> np.ndarray:
    from shira_cpp import node_stats_cpp, product_cpp

    cls = np.asarray(est.classes_)
    yk = np.clip(np.searchsorted(cls, y), 0, cls.size - 1)
    if not bool(np.all(cls[yk] == y)):
        raise ValueError("a label outside the sample's classes")
    K = int(est.estimators_.shape[1])
    if K == 1:
        start = _gb_start(est, x32, first).astype(np.float32)
        val, _ = product_cpp.product(ids, nl, yk.astype(np.float32), 0.0, "binary:logistic", eta=lr, reg_lambda=0.0,
                                     min_child_weight=0.0, max_delta_step=0.0, weight=wv.astype(np.float32),
                                     nthread=threads, pred=start)
        return val
    if est.init is not None or first > 0:
        raise ValueError("multiclass with an init estimator or a warm start (not yet)")
    base = np.ravel(est._raw_predict_init(x32[:1])).astype(np.float64)
    assert base.shape == (K,)
    return node_stats_cpp.softmax_values(ids, nl, yk, wv, base, lr * (K - 1) / K, 0.0, nthread=max(1, min(64, threads)))


def _gb_huber(est: Any, x32: np.ndarray, y: np.ndarray, wv: np.ndarray, ids: np.ndarray, nl: np.ndarray,
              lr: float) -> np.ndarray:
    from sklearn.utils.stats import _weighted_percentile

    raw = np.ravel(est._raw_predict_init(x32)).astype(np.float64).copy()
    out = np.zeros((ids.shape[0], int(nl.max())))
    for t in range(ids.shape[0]):
        diff = y - raw
        est._loss.closs.delta = float(_weighted_percentile(np.abs(diff), wv, 100 * est.alpha))
        order = np.argsort(ids[t], kind="stable")
        cut = np.concatenate([[0], np.cumsum(np.bincount(ids[t], minlength=int(nl[t])))])
        d, wo = diff[order], wv[order]
        for j in range(int(nl[t])):
            a, b = int(cut[j]), int(cut[j + 1])
            if b > a:
                out[t, j] = lr * float(np.ravel(est._loss.fit_intercept_only(y_true=d[a:b], sample_weight=wo[a:b]))[0])
        raw += out[t, :int(nl[t])][ids[t]]
    assert np.isfinite(out).all()
    return out


def gb_fit(est: Any, X: Any, y: Any, w: np.ndarray | None, fit_any: Any, threads: int) -> bool:
    n = X.shape[0]
    rows = stride(n, shape_rows(n, X.shape[1], "SHIRA_GB_FIRST_READ_ROWS", "gb_first_read_rows"))
    y1 = np.asarray(y) if hasattr(est, "predict_proba") else np.asarray(y, dtype=np.float64)
    first = len(getattr(est, "estimators_", ())) if est.warm_start else 0
    t0 = time.perf_counter()
    engine = fit_any(est, take(X, rows), y1[rows], None if w is None else w[rows])
    if not engine and len(getattr(est, "estimators_", ())) <= first:
        return False
    t1 = time.perf_counter()
    try:
        gb_resolve(est, values(X), y1, threads, w, first)
    except ValueError as e:
        LOG.warning("shira.sklearn: %s: the first read did not apply (ValueError: %s); today's route", type(est).__name__, e)
        return False
    t2 = time.perf_counter()
    grower = "SHIRA's engine" if engine else "stock scikit-learn (the engine refused a setting)"
    line = (f"fitted the first-read way (SHIRA's default): {len(est.estimators_) - first} stage shapes grown by {grower} on "
            f"a stride sample of {rows.size} of {n} rows (not stock's trees; accuracy depends on the data; train_score_, "
            f"oob_improvement_, the stage count early stopping chose and the node counts are the sample fit's), every "
            f"leaf value solved on all {n} rows ({t2 - t0:.2f} s); "
            f"SHIRA_GB_MODE=ordinary turns it off")
    log.append({"name": type(est).__name__, "n": n, "m": int(rows.size), "trees": len(est.estimators_), "threads": threads,
                "t_check": 0.0, "t_shapes": t1 - t0, "t_values": t2 - t1, "fall": float("nan"), "line": line})
    del log[:-64]
    return True
