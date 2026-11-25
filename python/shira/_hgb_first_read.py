from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np

LOG = logging.getLogger("shira.sklearn")
SOLVE_ERRORS = (AssertionError, ValueError, RuntimeError, AttributeError, TypeError, ArithmeticError)
log: list[dict] = []


def mode_on() -> bool:
    return os.environ.get("SHIRA_HGB_MODE", "") != "ordinary"


def sample_rows(features: int, n: int | None = None) -> int:
    from . import _machine

    env = os.environ.get("SHIRA_HGB_FIRST_READ_ROWS", "")
    if env:
        return int(env)
    m = int(_machine.value("hgb_first_read_rows", int(features)))
    small = int(_machine.value("hgb_first_read_rows_small", int(features)))
    if n is not None and small > 0 and n < int(_machine.value("hgb_first_read_small_below_rows", int(features))):
        m = small
    floor = int(_machine.value("sk_small_floor_rows", int(features)))
    if n is not None and floor < n <= m:
        m = max(floor, n * int(_machine.value("sk_small_sample_pct", int(features))) // 100)
        assert floor <= m < n
    assert m >= 0
    return m


def min_rows(features: int) -> int:
    from . import _machine

    env = os.environ.get("SHIRA_HGB_FIRST_READ_MIN_ROWS", "")
    return int(env) if env else int(_machine.value("hgb_first_read_min_rows", int(features)))


def pass_threads(features: int, engine: int) -> int:
    from . import _machine, _threads

    env = os.environ.get("SHIRA_HGB_FIRST_READ_THREADS", "")
    th = int(env) if env else int(_machine.value("hgb_first_read_threads", int(features)))
    assert th >= 0 and engine >= 1
    return engine if th == 0 else max(1, min(th, _threads.env_budget()))


def why_not(est: Any, X: np.ndarray, classifier: bool) -> str | None:
    if not mode_on():
        return "SHIRA_HGB_MODE=ordinary"
    if classifier and est.loss != "log_loss":
        return "classifier with a loss other than log_loss"
    if not classifier and est.loss not in _REG_OBJ and est.loss not in _LINE_SEARCH:
        return f"loss {est.loss!r}"
    m, lo = sample_rows(X.shape[1], X.shape[0]), min_rows(X.shape[1])
    if m < 1 or X.shape[0] <= m or X.shape[0] < lo:
        return f"{X.shape[0]} rows: under the first read's row bound ({max(lo, m + 1)})"
    return None


def _values(cst: Any) -> Any:
    return cst.values() if isinstance(cst, dict) else cst


def _stride(n: int, m: int) -> np.ndarray:
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    assert rows.size == m and int(rows[0]) == 0 and int(rows[-1]) == n - 1
    return rows


def _leaf_lists(est: Any) -> list[np.ndarray]:
    out = [np.flatnonzero(pr[0].nodes["is_leaf"].astype(bool)) for pr in est._predictors]
    assert out and all(v.size >= 1 for v in out)
    return out


def _ids(est: Any, X: np.ndarray, th: int) -> tuple[np.ndarray, np.ndarray]:
    from shira_cpp import binning_cpp, product_cpp

    if getattr(est, "_preprocessor", None) is not None:
        return _ids_cat(est, X, th)
    if not isinstance(X, np.ndarray):
        X = np.ascontiguousarray(est._preprocess_X(X, reset=False), dtype=np.float64)
    bm = est._bin_mapper
    preds = [pr[0] for pr in est._predictors]
    assert len(preds) >= 1 and all(len(pr) == 1 for pr in est._predictors), "one tree per iteration (regression)"
    assert not any(bool(pr.nodes["is_categorical"].any()) for pr in preds), "categorical splits"
    count = np.array([pr.nodes.size for pr in preds], dtype=np.int32)
    nodes = np.zeros((len(preds), int(count.max()), 5), dtype=np.int32)
    for t, pr in enumerate(preds):
        nd, c = pr.nodes, int(count[t])
        nodes[t, :c, 0] = np.where(nd["is_leaf"].astype(bool), -1, nd["feature_idx"])
        nodes[t, :c, 1] = nd["bin_threshold"].astype(np.int32) + 1
        nodes[t, :c, 2] = nd["missing_go_to_left"]
        nodes[t, :c, 3] = nd["left"]
        nodes[t, :c, 4] = nd["right"]
    binned = binning_cpp.hgb_bin(X, [np.asarray(v, dtype=np.float64) for v in bm.bin_thresholds_], int(bm.n_bins), nthread=th)
    codes = np.empty(binned.shape, dtype=np.uint16, order="C")
    np.add(binned, 1, out=codes, dtype=np.uint16, casting="unsafe")
    codes[binned == bm.missing_values_bin_idx_] = 0
    return product_cpp.leaf_ids(codes, nodes, count, nthread=th)


def _encoded(est: Any, y: np.ndarray, classifier: bool) -> np.ndarray:
    if not classifier:
        return np.ascontiguousarray(y, dtype=np.float64)
    cls = np.asarray(est.classes_)
    if cls.size < 2 or est.n_trees_per_iteration_ != (1 if cls.size == 2 else cls.size):
        raise ValueError(f"{cls.size} classes and {est.n_trees_per_iteration_} trees per iteration")
    ix = np.clip(np.searchsorted(cls, y), 0, cls.size - 1)
    if not bool(np.all(cls[ix] == y)):
        raise ValueError("a label outside the sample's classes")
    return ix.astype(np.float64)


def _resolve(est: Any, X: np.ndarray, y: np.ndarray, th: int, w: np.ndarray | None = None,
             classifier: bool = False) -> tuple[float, float]:
    from shira_cpp import product_cpp

    if classifier and int(est.n_trees_per_iteration_) > 1:
        return _resolve_multi(est, X, y, th, w)
    if not classifier and est.loss in _LINE_SEARCH:
        return _resolve_line_search(est, X, y, th, w)
    t0 = time.perf_counter()
    ids, nl = _ids(est, X, th)
    leaves = _leaf_lists(est)
    assert ids.shape == (len(leaves), X.shape[0]) and all(int(nl[t]) == leaves[t].size for t in range(len(leaves)))
    t1 = time.perf_counter()
    if classifier and getattr(est, "class_weight", None) is not None:
        w = est._finalize_sample_weight(w, y)
        assert w is not None and w.shape == y.shape
    y32 = np.ascontiguousarray(y, dtype=np.float32)
    base = float(np.float32(np.ravel(est._loss.fit_intercept_only(y_true=y, sample_weight=w))[0]))
    obj = "binary:logistic" if classifier else _REG_OBJ[est.loss]
    val = product_cpp.product_window(ids, 0, X.shape[0], nl, y32, base, obj, eta=float(est.learning_rate),
                                     reg_lambda=float(est.l2_regularization), min_child_weight=0.0, max_delta_step=0.0,
                                     obj_mds=0.0,
                                     weight=None if w is None else np.ascontiguousarray(w, dtype=np.float32), nthread=th)
    assert val.shape[0] == len(leaves) and np.isfinite(val).all()
    for t, lf in enumerate(leaves):
        est._predictors[t][0].nodes["value"][lf] = val[t, :lf.size].astype(np.float64)
    est._baseline_prediction = np.array([[base]], dtype=np.float64)
    return t1 - t0, time.perf_counter() - t1


_REG_OBJ = {"squared_error": "reg:squarederror", "poisson": "count:poisson", "gamma": "reg:gamma"}
_LINE_SEARCH = ("absolute_error", "quantile")


def _resolve_line_search(est: Any, X: np.ndarray, y: np.ndarray, th: int, w: np.ndarray | None) -> tuple[float, float]:
    t0 = time.perf_counter()
    ids, nl = _ids(est, X, th)
    leaves = _leaf_lists(est)
    assert ids.shape == (len(leaves), X.shape[0]) and all(int(nl[t]) == leaves[t].size for t in range(len(leaves)))
    t1 = time.perf_counter()
    yv = np.ascontiguousarray(y, dtype=np.float64)
    base = float(np.ravel(est._loss.fit_intercept_only(y_true=yv, sample_weight=w))[0])
    from shira_cpp import node_stats_cpp

    nl = np.array([lf.size for lf in leaves], dtype=np.int32)
    val = node_stats_cpp.line_search_values(ids, nl, yv, w, base, float(est.learning_rate), str(est.loss),
                                            float(est.quantile or 0.5), nthread=max(1, min(64, int(th))))
    assert val.shape == (len(leaves), int(nl.max())) and np.isfinite(val).all()
    for t, lf in enumerate(leaves):
        est._predictors[t][0].nodes["value"][lf] = val[t, :lf.size]
    est._baseline_prediction = np.array([[base]], dtype=np.float64)
    return t1 - t0, time.perf_counter() - t1


def _solver(est: Any, classifier: bool) -> Any:
    if not classifier and est.loss in _LINE_SEARCH:
        return _resolve
    return _resolve_mono if _has_mono(est) else _resolve


def _resolve_multi(est: Any, X: np.ndarray, y: np.ndarray, th: int, w: np.ndarray | None) -> tuple[float, float]:
    import types

    t0 = time.perf_counter()
    K = int(est.n_trees_per_iteration_)
    view = types.SimpleNamespace(_bin_mapper=est._bin_mapper, _predictors=[[pr] for it in est._predictors for pr in it],
                                 _preprocessor=getattr(est, "_preprocessor", None), _preprocess_X=est._preprocess_X)
    ids, nl = _ids(view, X, th)
    leaves = _leaf_lists(view)
    assert len(leaves) == K * len(est._predictors) and ids.shape == (len(leaves), X.shape[0])
    assert all(int(nl[t]) == leaves[t].size for t in range(len(leaves)))
    t1 = time.perf_counter()
    if getattr(est, "class_weight", None) is not None:
        w = est._finalize_sample_weight(w, y)
    base = np.ravel(est._loss.fit_intercept_only(y_true=y, sample_weight=w)).astype(np.float64)
    assert base.shape == (K,) and np.isfinite(base).all()
    vals = _softmax_values(ids, leaves, y.astype(np.int64), w, base, float(est.learning_rate),
                           float(est.l2_regularization), th)
    for t, lf in enumerate(leaves):
        view._predictors[t][0].nodes["value"][lf] = vals[t]
    est._baseline_prediction = base.reshape(1, K)
    return t1 - t0, time.perf_counter() - t1


def _softmax_values(ids: np.ndarray, leaves: list, yk: np.ndarray, w: np.ndarray | None, base: np.ndarray, eta: float,
                    l2: float, th: int = 1) -> list[np.ndarray]:
    from shira_cpp import node_stats_cpp

    T = len(leaves)
    assert T % base.size == 0 and ids.shape == (T, yk.size)
    nl = np.array([lf.size for lf in leaves], dtype=np.int32)
    val = node_stats_cpp.softmax_values(ids, nl, yk, w, base, eta, l2, nthread=max(1, min(64, int(th))))
    assert val.shape == (T, int(nl.max())) and np.isfinite(val).all()
    return [val[t, :int(nl[t])] for t in range(T)]


def fit(est: Any, stock_fit: Any, X: np.ndarray, y: Any, sample_weight: Any, classifier: bool,
        fit_ordinary: Any) -> tuple[str | None, str]:
    n = X.shape[0]
    rows = _stride(n, sample_rows(X.shape[1], n))
    y1 = np.asarray(y)
    w1 = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    from . import _sk_dropin_hgb as dh

    live, before, _ = dh.seed_plan(est)
    t0 = time.perf_counter()
    why = fit_ordinary(est, stock_fit, np.ascontiguousarray(X[rows]), y1[rows], None if w1 is None else w1[rows], classifier)
    if why:
        return why, ""
    m2 = _richer_sample(est, X, y1, w1, rows, classifier)
    if m2 > rows.size:
        rows = _stride(n, m2)
        if live is not None:
            live.set_state(before)
        why = fit_ordinary(est, stock_fit, np.ascontiguousarray(X[rows]), y1[rows], None if w1 is None else w1[rows], classifier)
        if why:
            return why, ""
    t1 = time.perf_counter()
    th = pass_threads(X.shape[1], dh.train_threads(n))
    try:
        solve = _solver(est, classifier)
        t_ids, t_val = solve(est, np.ascontiguousarray(X, dtype=np.float64), _encoded(est, y1, classifier), th, w1,
                             classifier)
    except SOLVE_ERRORS as e:
        LOG.warning("shira.sklearn: %s: the first read did not apply (%s: %s); the ordinary engine fits every row",
                    type(est).__name__, type(e).__name__, e)
        if live is not None:
            live.set_state(before)
        return fit_ordinary(est, stock_fit, X, y, sample_weight, classifier), ""
    line = (f"fitted the first-read way (SHIRA's default): {est.n_iter_} tree shapes grown by SHIRA's engine on a stride "
            f"sample of {rows.size} of {n} rows (not stock's trees; accuracy depends on the data; the tree count, the scores, the bins "
            f"and the node counts are the sample fit's), every leaf value solved on all {n} rows "
            f"({time.perf_counter() - t0:.2f} s); SHIRA_HGB_MODE=ordinary turns it off")
    log.append({"n": n, "m": int(rows.size), "trees": int(est.n_iter_), "threads": th, "t_shapes": t1 - t0, "t_ids": t_ids,
                "t_values": t_val, "line": line})
    del log[:-64]
    return None, line


def _has_mono(est: Any) -> bool:
    cst = est.monotonic_cst
    return cst is not None and any(int(v) != 0 for v in np.ravel(list(_values(cst))))


def _mono_trees(est: Any, leaves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.utils.validation import _check_monotonic_cst

    sign = np.asarray(_check_monotonic_cst(est, est.monotonic_cst), dtype=np.int32)
    cat = getattr(est, "is_categorical_", None)
    if cat is not None:
        sign = np.concatenate([sign[cat], sign[~cat]])
    preds = [pr[0] for pr in est._predictors]
    count = np.array([pr.nodes.size for pr in preds], dtype=np.int32)
    tree = np.zeros((len(preds), int(count.max()), 4), dtype=np.int32)
    for t, pr in enumerate(preds):
        nd, c = pr.nodes, int(count[t])
        leaf = nd["is_leaf"].astype(bool)
        assert np.array_equal(np.flatnonzero(leaf), leaves[t]) and bool((nd["left"][~leaf] > np.flatnonzero(~leaf)).all())
        tree[t, :c, 0] = -1
        tree[t, leaves[t], 0] = np.arange(leaves[t].size)
        tree[t, :c, 1] = np.where(leaf, 0, nd["left"])
        tree[t, :c, 2] = np.where(leaf, 0, nd["right"])
        tree[t, :c, 3] = np.where(leaf, 0, sign[nd["feature_idx"].astype(np.int64)])
    return tree, count


def _resolve_mono(est: Any, X: np.ndarray, y: np.ndarray, th: int, w: np.ndarray | None = None,
                  classifier: bool = False) -> tuple[float, float]:
    from shira_cpp import product_cpp

    t0 = time.perf_counter()
    ids, nl = _ids(est, X, th)
    leaves = _leaf_lists(est)
    assert ids.shape == (len(leaves), X.shape[0]) and all(int(nl[t]) == leaves[t].size for t in range(len(leaves)))
    tree, count = _mono_trees(est, leaves)
    t1 = time.perf_counter()
    if classifier and getattr(est, "class_weight", None) is not None:
        w = est._finalize_sample_weight(w, y)
        assert w is not None and w.shape == y.shape
    base = float(np.float32(np.ravel(est._loss.fit_intercept_only(y_true=y, sample_weight=w))[0]))
    val, _ = product_cpp.product_bounded(ids, nl, tree, count, np.ascontiguousarray(y, dtype=np.float32), base,
                                         "binary:logistic" if classifier else _REG_OBJ[est.loss],
                                         eta=float(est.learning_rate), reg_lambda=float(est.l2_regularization),
                                         obj_mds=0.0,
                                         min_child_weight=0.0, weight=None if w is None else np.ascontiguousarray(w, dtype=np.float32),
                                         nthread=th)
    assert val.shape[0] == len(leaves) and np.isfinite(val).all()
    for t, lf in enumerate(leaves):
        est._predictors[t][0].nodes["value"][lf] = val[t, :lf.size].astype(np.float64)
    est._baseline_prediction = np.array([[base]], dtype=np.float64)
    return t1 - t0, time.perf_counter() - t1


RICH_LEVELS = ((0.05, 200_000), (0.35, 400_000))
RICH_HOLD = 20_000
RICH_MAX_FEATURES = 16


def explained(est: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, rows: np.ndarray, classifier: bool) -> float:
    assert rows.size >= 2 and X.shape[0] == y.shape[0]
    half = int(rows[1] - rows[0]) // 2
    if half < 1:
        return 0.0
    hold = (rows[:-1] + half)[:: max(1, (rows.size - 1) // RICH_HOLD)]
    try:
        yh = _encoded(est, y[hold], classifier)
    except ValueError:
        return 0.0
    wh = None if w is None else np.ascontiguousarray(w[hold], dtype=np.float64)
    raw = np.ascontiguousarray(est._raw_predict(_take(X, hold)), dtype=np.float64)
    base = np.ascontiguousarray(np.repeat(np.asarray(est._baseline_prediction, dtype=np.float64), hold.size, axis=0))
    model = float(est._loss(y_true=yh, raw_prediction=raw, sample_weight=wh))
    floor = float(est._loss(y_true=yh, raw_prediction=base, sample_weight=wh))
    return 0.0 if not (np.isfinite(model) and np.isfinite(floor) and abs(floor) > 0.0) else 1.0 - model / floor


def _richer_sample(est: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, rows: np.ndarray, classifier: bool) -> int:
    n, m = X.shape[0], int(rows.size)
    if os.environ.get("SHIRA_HGB_RICH", "") != "1" or os.environ.get("SHIRA_HGB_FIRST_READ_ROWS", ""):
        return m
    if X.shape[1] > RICH_MAX_FEATURES:
        return m
    share = explained(est, X, y, w, rows, classifier)
    want = max([m] + [lv for cut, lv in RICH_LEVELS if share >= cut])
    log_rich.append({"n": n, "m": m, "explained": share, "want": min(want, n // 2)})
    del log_rich[:-64]
    return max(m, min(want, n // 2))


log_rich: list[dict] = []


STOCK_GROWN = ("categorical_features", "fit arguments", "X is not a numeric numpy array", "verbose output", "warm_start")
STOCK_GROWN_ARGS = frozenset({"X_val", "y_val", "sample_weight_val"})


def _take(X: Any, rows: np.ndarray) -> Any:
    return X.iloc[rows] if hasattr(X, "iloc") else np.ascontiguousarray(X[rows])


CAT_WORDS = 9


def _ids_cat(est: Any, X: Any, th: int) -> tuple[np.ndarray, np.ndarray]:
    from shira_cpp import product_cpp

    bm = est._bin_mapper
    binned = bm.transform(est._preprocess_X(X, reset=False))
    miss = int(bm.missing_values_bin_idx_)
    codes = np.empty(binned.shape, dtype=np.uint16, order="C")
    np.add(binned, 1, out=codes, dtype=np.uint16, casting="unsafe")
    codes[binned == miss] = 0
    preds = [pr[0] for pr in est._predictors]
    assert len(preds) >= 1 and all(len(pr) == 1 for pr in est._predictors), "one tree per iteration"
    count = np.array([pr.nodes.size for pr in preds], dtype=np.int32)
    nodes = np.zeros((len(preds), int(count.max()), 5), dtype=np.int32)
    words: list[np.ndarray] = []
    for t, pr in enumerate(preds):
        nd, c = pr.nodes, int(count[t])
        leaf = nd["is_leaf"].astype(bool)
        nodes[t, :c, 0] = np.where(leaf, -1, nd["feature_idx"])
        nodes[t, :c, 1] = nd["bin_threshold"].astype(np.int32) + 1
        nodes[t, :c, 2] = nd["missing_go_to_left"]
        nodes[t, :c, 3] = nd["left"]
        nodes[t, :c, 4] = nd["right"]
        sets = np.asarray(pr.binned_left_cat_bitsets, dtype="<u4")
        for i in np.flatnonzero(nd["is_categorical"].astype(bool) & ~leaf).tolist():
            bits = int.from_bytes(np.ascontiguousarray(sets[int(nd["bitset_idx"][i])]).tobytes(), "little")
            moved = ((bits & ~(1 << miss)) << 1) | ((bits >> miss) & 1)
            nodes[t, i, 1], nodes[t, i, 2] = CAT_WORDS * len(words), 2
            words.append(np.frombuffer(moved.to_bytes(4 * CAT_WORDS, "little"), dtype="<u4"))
    if not words:
        return product_cpp.leaf_ids(codes, nodes, count, nthread=th)
    return product_cpp.leaf_ids_cat(codes, nodes, count, np.concatenate(words), nthread=th)


def _ids_stock(est: Any, X: Any, th: int) -> tuple[np.ndarray, np.ndarray]:
    Xp = est._preprocess_X(X, reset=False)
    known, fmap = est._bin_mapper.make_known_categories_bitsets()
    preds = [pr[0] for pr in est._predictors]
    assert len(preds) >= 1 and all(len(pr) == 1 for pr in est._predictors), "one tree per iteration"
    nl = np.array([int(pr.nodes["is_leaf"].sum()) for pr in preds], dtype=np.int32)
    ids = np.empty((len(preds), Xp.shape[0]), dtype=np.uint8 if int(nl.max()) <= 256 else np.uint16)
    for t, pr in enumerate(preds):
        nd = pr.nodes
        leaf = np.flatnonzero(nd["is_leaf"].astype(bool))
        keep = nd["value"].copy()
        try:
            nd["value"][leaf] = np.arange(leaf.size)
            ids[t] = pr.predict(Xp, known, fmap, max(1, int(th)))
        finally:
            nd["value"][:] = keep
    return ids, nl


def why_stock_grown(est: Any, X: Any, y: Any, why: str, classifier: bool, fit_kwargs: dict) -> str | None:
    from . import _sk_dropin_hgb as dh

    if not why.startswith(STOCK_GROWN):
        return why
    frame = hasattr(X, "iloc") and hasattr(X, "shape") and len(X.shape) == 2
    if not frame and not (isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype.kind in "fiu"):
        return "X is neither a numeric numpy array nor a DataFrame"
    if set(fit_kwargs) - STOCK_GROWN_ARGS:
        return "a fit argument outside the validation set"
    if est.warm_start and est._is_fitted() and getattr(est, "_shira_fr_rows", None) != (X.shape[0], sample_rows(X.shape[1], X.shape[0])):
        return "warm_start continuing a model that was not grown on this sample"
    if not (dh._seeded(est) or est.random_state is None or isinstance(est.random_state, np.random.RandomState)):
        return "random_state is not an int, None or a RandomState"
    if np.ndim(y) != 1 or len(y) != X.shape[0]:
        return "y is not one column"
    return why_not(est, X, classifier)


def fit_stock_grown(est: Any, base_fit: Any, X: Any, y: Any, sample_weight: Any, classifier: bool,
                    fit_kwargs: dict) -> tuple[str | None, str]:
    from threadpoolctl import threadpool_limits

    from . import _sk_dropin_hgb as dh

    n = X.shape[0]
    y1 = np.asarray(y)
    w1 = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    live, before, _ = dh.seed_plan(est)
    t0 = time.perf_counter()

    def grow(rows: np.ndarray) -> None:
        if live is not None:
            live.set_state(before)
        with threadpool_limits(limits=dh.train_threads(int(rows.size)), user_api="openmp"):
            base_fit(est, _take(X, rows), y1[rows], sample_weight=None if w1 is None else w1[rows], **fit_kwargs)

    rows = _stride(n, sample_rows(X.shape[1], n))
    grow(rows)
    t1 = time.perf_counter()
    th = pass_threads(X.shape[1], dh.train_threads(n))
    try:
        solve = _solver(est, classifier)
        t_ids, t_val = solve(est, X, _encoded(est, y1, classifier), th, w1, classifier)
    except SOLVE_ERRORS as e:
        LOG.warning("shira.sklearn: %s: the first read with stock-grown shapes did not apply (%s: %s); stock fits",
                    type(est).__name__, type(e).__name__, e)
        if live is not None:
            live.set_state(before)
        return f"the first read with stock-grown shapes did not apply ({e})", ""
    line = (f"fitted the first-read way (SHIRA's default): {est.n_iter_} tree shapes grown by scikit-learn on a stride "
            f"sample of {rows.size} of {n} rows (not stock's trees; accuracy depends on the data; the tree count, the scores, the bins "
            f"and the node counts are the sample fit's), every leaf value solved on all {n} rows "
            f"({time.perf_counter() - t0:.2f} s); SHIRA_HGB_MODE=ordinary turns it off")
    est._shira_fr_rows = (n, int(rows.size))
    log.append({"n": n, "m": int(rows.size), "trees": int(est.n_iter_), "threads": th, "t_shapes": t1 - t0, "t_ids": t_ids,
                "t_values": t_val, "line": line, "grower": "stock"})
    del log[:-64]
    return None, line


def numeric_frame(est: Any, X: Any) -> np.ndarray | None:
    if not (hasattr(X, "iloc") and hasattr(X, "dtypes") and hasattr(X, "to_numpy")) or isinstance(est.monotonic_cst, dict):
        return None
    try:
        if not all(isinstance(t, np.dtype) and t.kind in "fiub" for t in X.dtypes):
            return None
        out = np.ascontiguousarray(X.to_numpy(dtype=np.float64))
    except (TypeError, ValueError):
        return None
    return out if out.ndim == 2 and out.shape == X.shape else None


def set_frame_names(est: Any, X: Any) -> None:
    from sklearn.utils.validation import _check_feature_names

    _check_feature_names(est, X, reset=True)
