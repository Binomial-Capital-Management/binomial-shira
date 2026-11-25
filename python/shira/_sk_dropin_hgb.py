from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE, X_BITSET_INNER_DTYPE
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

MAX_THREADS = 64
X86_STOCK_MIN_ROWS = 450_000


def why_not(est: Any, X: Any, y: Any, fit_kwargs: dict) -> str | None:
    if fit_kwargs:
        return f"fit arguments {sorted(fit_kwargs)}"
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype.kind not in "fiu":
        return "X is not a numeric numpy array"
    if est.categorical_features not in (None, "from_dtype"):
        return "categorical_features"
    if est.verbose:
        return "verbose output"
    if est.warm_start:
        return "warm_start"
    if not (_seeded(est) or est.random_state is None or isinstance(est.random_state, np.random.RandomState)):
        return "random_state is not an int, None or a RandomState"
    return None if np.ndim(y) == 1 and len(y) == X.shape[0] else "y is not one column"


def x86_guard(X: Any) -> str | None:
    from . import _machine

    lo = int(_machine.value("hgb_stock_min_rows", int(X.shape[1])))
    assert lo >= 0 and X.shape[0] >= 0
    if X.shape[0] >= lo:
        return f"{X.shape[0]} rows: stock is faster from {lo} rows on this machine (switch hgb_stock_min_rows)"
    return None


def _preorder(nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order, depth, stack = [], [], [(0, 0)]
    for _ in range(2 * nodes.shape[0] + 2):
        if not stack:
            break
        v, d = stack.pop()
        order.append(v)
        depth.append(d)
        if nodes[v, 0] >= 0:
            stack += [(int(nodes[v, 3]), d + 1), (int(nodes[v, 2]), d + 1)]
    assert not stack and len(order) == nodes.shape[0]
    return np.asarray(order), np.asarray(depth)


def _records(nodes: np.ndarray, value: np.ndarray, gain: np.ndarray, thresholds: list,
             count: np.ndarray | None = None) -> np.ndarray:
    order, depth = _preorder(nodes)
    new = np.empty_like(order)
    new[order] = np.arange(order.size)
    nd = nodes[order]
    leaf = nd[:, 0] < 0
    rec = np.zeros(order.size, dtype=PREDICTOR_RECORD_DTYPE)
    rec["value"], rec["gain"], rec["depth"], rec["is_leaf"] = value[order], gain[order], depth, leaf
    if count is not None:
        assert count.shape[0] == nodes.shape[0]
        rec["count"] = count[order]
    it = np.flatnonzero(~leaf)
    f, b = nd[it, 0], nd[it, 1]
    rec["feature_idx"][it], rec["bin_threshold"][it] = f, b
    rec["missing_go_to_left"][it] = nd[it, 4] & 1
    rec["left"][it], rec["right"][it] = new[nd[it, 2]], new[nd[it, 3]]
    rec["num_threshold"][it] = [float(thresholds[fi][bi]) if bi < len(thresholds[fi]) else np.inf
                                for fi, bi in zip(f.tolist(), b.tolist())]
    return rec


def _fill_counts(recs: list[np.ndarray], x_train: np.ndarray, nf: int) -> None:
    from shira_cpp import outputs_cpp as oc

    trees = []
    for r in recs:
        feat = np.where(r["is_leaf"].astype(bool), -1, r["feature_idx"]).astype(np.int32)
        tr = oc.RawTree(feat, np.where(feat < 0, 0.0, r["num_threshold"]),
                        r["missing_go_to_left"].astype(np.int32) * (feat >= 0),
                        r["left"].astype(np.int32), r["right"].astype(np.int32), r["value"].copy(),
                        None, None, np.arange(feat.size, dtype=np.int32))
        trees.append(tr)
    fr = oc.build(trees, nf, 1, 1, 1, base=0.0)
    w, _, _ = fr.node_sums(np.ascontiguousarray(x_train, dtype=np.float64))
    assert w.size == sum(r.size for r in recs)
    pos = 0
    for r in recs:
        r["count"] = np.rint(w[pos:pos + r.size]).astype(np.uint32)
        pos += r.size


def _seeded(est: Any) -> bool:
    rs = est.random_state
    return isinstance(rs, (int, np.integer)) and not isinstance(rs, bool)


def seed_plan(est: Any) -> tuple:
    import copy

    from sklearn.utils import check_random_state

    if _seeded(est):
        return None, None, None
    live = check_random_state(est.random_state)
    twin = copy.deepcopy(live)
    assert twin is not live and isinstance(twin, np.random.RandomState)
    top = np.iinfo(np.uint32).max
    pair = int(twin.randint(top, dtype="u8")), int(twin.randint(top, dtype="u8"))
    assert all(0 <= s < top for s in pair)
    return live, live.get_state(legacy=False), pair


def _settle(est: Any, why: str | None, live: Any, before: Any, pair: Any) -> str | None:
    if live is None:
        return why
    assert pair is not None and before is not None
    drawn = getattr(est, "_random_seed", None)
    if drawn is None or int(drawn) != pair[0]:
        return "the random generator was moved by other code during the fit"
    if why is not None:
        live.set_state(before)
    return why


def _train(est: Any, X: np.ndarray, y_enc: np.ndarray, sw: Any, classifier: bool, threads: int,
           folds: list, pair: Any = None) -> tuple:
    from shira_cpp import hgb_cpp

    params = {k: v for k, v in est.get_params(deep=False).items() if k in hgb_cpp.DEFAULTS}
    extra = sorted(set(est.get_params(deep=False)) - set(hgb_cpp.DEFAULTS))
    if extra:
        raise ValueError(f"parameters the engine does not know: {extra}")
    prep = hgb_cpp.prepare(X, y_enc, folds, params, "classifier" if classifier else "regression",
                           sample_weight=sw, threads=threads, seed_pair=pair)
    return prep, hgb_cpp.train(prep, threads=threads, trees_out=True)


def _predictors(res: dict, k: int, thresholds: list, x_train: np.ndarray | None) -> list:
    n_iter, multi = int(np.asarray(res["n_iter"])[k]), res["nnodes"].ndim == 3
    n_class = res["nnodes"].shape[2] if multi else 1
    counts = res.get("count")
    assert counts is not None or x_train is not None
    recs = []
    for r in range(n_iter):
        for t in range(n_class):
            ix = (k, r, t) if multi else (k, r)
            nn = int(res["nnodes"][ix])
            nodes = res["nodes"][ix][:nn]
            if (nodes[:, 4] & 2).any():
                raise ValueError("categorical split")
            cnt = None if counts is None else counts[ix][:nn]
            recs.append(_records(nodes, res["value"][ix][:nn], res["gain"][ix][:nn], thresholds, cnt))
    if counts is None:
        _fill_counts(recs, x_train, x_train.shape[1])
    empty = np.zeros((0, 8), dtype=X_BITSET_INNER_DTYPE)
    preds = [TreePredictor(r, empty.copy(), empty.copy()) for r in recs]
    return [preds[i * n_class:(i + 1) * n_class] for i in range(n_iter)]


def _fast_ok(est: Any, X: Any, sw: Any, prep: dict, res: dict) -> bool:
    plain = isinstance(X, np.ndarray) and sw is None and prep["is_cat"] is None and isinstance(est.loss, str)
    no_rng = float(est.max_features) == 1.0
    no_cw = getattr(est, "class_weight", None) is None
    return bool(plain and no_rng and no_cw and est.scoring == "loss" and res.get("count") is not None)


def _fast_state(est: Any, X: np.ndarray, y: Any, prep: dict, res: dict, k: int) -> None:
    from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
    from sklearn.utils import check_random_state
    from sklearn.utils.validation import _check_y

    Xv, _ = est._preprocess_X(X, reset=True)
    y_enc = est._encode_y(_check_y(y, estimator=est))
    rng = check_random_state(est.random_state)
    est._random_seed = rng.randint(np.iinfo(np.uint32).max, dtype="u8")
    est._feature_subsample_rng = np.random.default_rng(rng.randint(np.iinfo(np.uint32).max, dtype="u8"))
    n_samples, est._n_features = Xv.shape
    est._loss = est._get_loss(sample_weight=None)
    est.do_early_stopping_ = n_samples > 10_000 if est.early_stopping == "auto" else est.early_stopping
    est._use_validation_data = est.validation_fraction is not None
    bm = _BinMapper(n_bins=est.max_bins + 1, is_categorical=None, known_categories=None,
                    random_state=est._random_seed, n_threads=_openmp_effective_n_threads())
    thr = [np.asarray(t, dtype=np.float64).copy() for t in prep["thresholds"][k]]
    bm.is_categorical_ = np.zeros(est._n_features, dtype=np.uint8)
    bm.missing_values_bin_idx_ = bm.n_bins - 1
    bm.bin_thresholds_ = thr
    bm.n_bins_non_missing_ = np.array([t.size + 1 for t in thr], dtype=np.uint32)
    est._bin_mapper = bm
    tr = prep["train_rows"][k]
    assert tr.size >= 1 and len(thr) == est._n_features
    est._baseline_prediction = est._loss.fit_intercept_only(y_true=y_enc[tr], sample_weight=None).reshape((1, -1))
    est._scorer = None


def _inject_fast(est: Any, X: Any, y: Any, prep: dict, res: dict, k: int) -> str | None:
    _fast_state(est, X, y, prep, res, k)
    try:
        preds = _predictors(res, k, est._bin_mapper.bin_thresholds_, None)
    except (ValueError, RuntimeError, AssertionError) as exc:
        return f"tree conversion: {exc}"
    est._predictors = preds
    if est.do_early_stopping_:
        est.train_score_ = np.asarray(res["train_score"][k], dtype=np.float64).copy()
        est.validation_score_ = np.asarray(res["validation_score"][k], dtype=np.float64).copy()
    else:
        est.train_score_, est.validation_score_ = np.asarray([]), np.asarray([])
    assert est.n_iter_ == len(preds)
    return None


def _inject(est: Any, stock_fit: Any, X: Any, y: Any, sw: Any, prep: dict, res: dict, k: int) -> str | None:
    if _fast_ok(est, X, sw, prep, res):
        return _inject_fast(est, X, y, prep, res, k)
    saved = est.max_iter
    est.max_iter = 1
    try:
        stock_fit(est, X, y, sw)
    finally:
        est.max_iter = saved
    mine, theirs = prep["thresholds"][k], est._bin_mapper.bin_thresholds_
    if len(mine) != len(theirs) or not all(np.array_equal(a, b) for a, b in zip(mine, theirs)):
        return "bin thresholds differ from sklearn's bin mapper"
    xf = np.ascontiguousarray(X, dtype=np.float64)
    try:
        preds = _predictors(res, k, theirs, xf[prep["train_rows"][k]])
    except (ValueError, RuntimeError, AssertionError) as exc:
        return f"tree conversion: {exc}"
    est._predictors = preds
    if est.do_early_stopping_:
        est.train_score_ = np.asarray(res["train_score"][k], dtype=np.float64).copy()
        est.validation_score_ = np.asarray(res["validation_score"][k], dtype=np.float64).copy()
    assert est.n_iter_ == len(preds)
    return None


def _encode(y: Any, classifier: bool) -> np.ndarray:
    y1 = np.asarray(y)
    return np.unique(y1, return_inverse=True)[1].astype(np.float64) if classifier else y1.astype(np.float64)


def n_threads() -> int:
    return max(1, min(int(_openmp_effective_n_threads()), MAX_THREADS))


def train_threads(rows: int) -> int:
    from . import _threads

    return _threads.default_threads("hgb", int(rows), n_threads())


def fit_on_shira(est: Any, stock_fit: Any, X: Any, y: Any, sample_weight: Any, classifier: bool) -> str | None:
    xf = np.ascontiguousarray(X, dtype=np.float64)
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    live, before, pair = seed_plan(est)
    try:
        prep, res = _train(est, xf, _encode(y, classifier), sw, classifier, train_threads(xf.shape[0]),
                           [(0, xf.shape[0])], pair)
    except (ValueError, RuntimeError, AssertionError) as exc:
        return f"engine: {exc}"
    return _settle(est, _inject(est, stock_fit, X, y, sample_weight, prep, res, 0), live, before, pair)


def fit_folds(make: Any, stock_fit: Any, X: np.ndarray, y: Any, folds: list, classifier: bool) -> tuple:
    assert folds and all(0 <= a < b <= X.shape[0] for a, b in folds)
    xf = np.ascontiguousarray(X, dtype=np.float64)
    y1 = np.asarray(y)
    y_enc = _encode(y1, classifier)
    proto = make()
    if not _seeded(proto):
        return None, "random_state is not an int (one fit per fold, each on SHIRA)"
    split = proto.early_stopping == "auto"
    groups: dict = {}
    for k, (a, b) in enumerate(folds):
        groups.setdefault((b - a > 10_000) if split else True, []).append(k)
    out: list = [None] * len(folds)
    for ks in groups.values():
        try:
            prep, res = _train(proto, xf, y_enc, None, classifier, train_threads(max(folds[k][1] - folds[k][0] for k in ks)),
                               [folds[k] for k in ks])
        except (ValueError, RuntimeError, AssertionError) as exc:
            return None, f"engine: {exc}"
        for j, k in enumerate(ks):
            a, b = folds[k]
            est = make()
            why = _inject(est, stock_fit, X[a:b], y1[a:b], None, prep, res, j)
            if why:
                return None, why
            out[k] = est
    assert all(e is not None for e in out)
    return out, None


CERT_ROWS = 256


def _forest_raw(est: Any, xt: np.ndarray) -> Any:
    from shira_cpp import outputs_cpp as oc

    stock = type(est)._raw_predict
    ok = (est.n_trees_per_iteration_ == 1 and getattr(est, "_preprocessor", None) is None
          and isinstance(xt, np.ndarray) and xt.dtype == np.float64 and xt.ndim == 2 and xt.shape[0] > 0)
    if not ok:
        return None
    try:
        forest = oc.from_hgb(est, est._n_features)
        m = min(CERT_ROWS, xt.shape[0])
        got = forest.predict(xt[:m], threads=n_threads()).reshape(-1, 1)
    except (AssertionError, ValueError, RuntimeError, OSError):
        return None
    if not np.array_equal(got, stock(est, xt[:m])):
        return None
    threads = n_threads()

    def raw(X: Any, n_threads: Any = None) -> np.ndarray:
        if X is not xt:
            return stock(est, X, n_threads)
        out = forest.predict(xt, threads=threads).reshape(-1, 1)
        assert out.shape == (xt.shape[0], 1) and out.dtype == np.float64
        return out

    return raw


def score_folds(ests: list, X: np.ndarray, y: Any, splits: list, scorer: Any) -> tuple:
    assert len(ests) == len(splits) and ests
    y1 = np.asarray(y)
    scores, fast = [], 0
    for est, (_, te) in zip(ests, splits):
        xt = X[te]
        raw = _forest_raw(est, xt)
        if raw is not None:
            est._raw_predict = raw
            fast += 1
        try:
            scores.append(scorer(est, xt, y1[te]))
        finally:
            est.__dict__.pop("_raw_predict", None)
    assert all("_raw_predict" not in vars(e) for e in ests)
    return np.array(scores), fast
