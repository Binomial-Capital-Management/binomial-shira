from __future__ import annotations

import ctypes
import logging
from numbers import Integral

import numpy as np

from . import _lib

_P = ctypes.c_void_p
LOG = logging.getLogger("shira.skgb")
_FAST_WARNED = [False]
LOSS = {"squared_error": 0, "absolute_error": 1, "huber": 2, "quantile": 3, "binomial": 4, "multinomial": 5,
        "tree": 6}
ERRORS = {-1: "bad arguments", -2: "a code 0 (bin 0 must be empty: codes >= 1)",
          -3: "min_weight_fraction_leaf needs weights exact in the per-tree integer scale",
          -4: "a positive weight underflows the per-tree integer scale", -5: "a tree with no in-bag row",
          -6: "the exact (sort-based) builder failed", -9: "an internal check failed (caught in the worker thread)"}
_GB_KEYS = {"loss", "learning_rate", "n_estimators", "subsample", "criterion", "min_samples_split",
            "min_samples_leaf", "min_weight_fraction_leaf", "max_depth", "min_impurity_decrease", "init",
            "random_state", "max_features", "alpha", "max_leaf_nodes", "warm_start", "validation_fraction",
            "n_iter_no_change", "tol", "ccp_alpha", "verbose"}


def _lib_skgb() -> ctypes.CDLL:
    lib = _lib()
    if not hasattr(lib, "_skgb_ready"):
        lib.shira_skgb_batch.restype = ctypes.c_int
        lib.shira_skgb_batch.argtypes = [_P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, _P, _P, _P, _P,
                                         ctypes.c_int32, _P, _P, _P, ctypes.c_int32, ctypes.c_int32,
                                         _P, _P, _P, _P, _P, _P]
        i32, d = ctypes.c_int32, ctypes.c_double
        lib.shira_skgb_labels.restype = ctypes.c_int
        lib.shira_skgb_labels.argtypes = [_P, ctypes.c_int64, _P, _P, i32, _P, _P, _P]
        lib.shira_fold_offsets.restype = ctypes.c_int64
        lib.shira_fold_offsets.argtypes = [_P, _P, i32, _P]
        lib.shira_skgb_predict_raw.restype = ctypes.c_int
        lib.shira_skgb_predict_raw.argtypes = [_P, ctypes.c_int64, i32, _P, _P, _P, i32, i32, i32, d, _P, i32, _P]
        lib._skgb_ready = True
    return lib


def _refuse(p: dict, kind: str) -> None:
    bad = set(p) - _GB_KEYS - {"splitter"}
    if bad:
        raise ValueError(f"unknown / unsupported parameters: {sorted(bad)}")
    checks = [("max_features", None), ("min_impurity_decrease", 0.0), ("ccp_alpha", 0.0), ("warm_start", False),
              ("n_iter_no_change", None), ("splitter", "best")]
    for key, ok in checks:
        if key in p and p[key] != ok:
            raise ValueError(f"{key}={p[key]!r} is not supported in sklearn mode (only {ok!r})")
    rs = p.get("random_state")
    if not isinstance(rs, Integral) or not 0 <= int(rs) < 2 ** 31:
        raise ValueError("random_state must be an int in [0, 2**31): sklearn's draws are reproduced from it")
    for key in ("min_samples_split", "min_samples_leaf"):
        if key in p and not isinstance(p[key], Integral):
            raise ValueError(f"{key} must be an int (fractions are not supported)")
    crit = p.get("criterion", "squared_error" if kind == "dtr" else "friedman_mse")
    if (kind == "dtr") != (crit == "squared_error") or crit not in ("friedman_mse", "squared_error"):
        raise ValueError(f"criterion={crit!r} not supported for {kind} (GB: friedman_mse, tree: squared_error)")


def _cfg(kind: str, p: dict, ncls_trees: int, sw_none: bool, n_classes: int) -> tuple[np.ndarray, np.ndarray, int]:
    if kind == "dtr":
        loss = "tree"
    elif kind == "gbc":
        if p.get("loss", "log_loss") != "log_loss":
            raise ValueError("GradientBoostingClassifier: only loss='log_loss' (exponential is not supported)")
        loss = "binomial" if n_classes == 2 else "multinomial"
    else:
        loss = p.get("loss", "squared_error")
        if loss not in ("squared_error", "absolute_error", "huber", "quantile"):
            raise ValueError(f"loss={loss!r} not supported")
    depth = p.get("max_depth", None if kind == "dtr" else 3)
    mln = p.get("max_leaf_nodes")
    if mln is None and depth is None:
        raise ValueError("max_depth=None needs max_leaf_nodes (unbounded depth-first trees are refused)")
    init = p.get("init")
    init_mode = 0 if init is None else (1 if isinstance(init, str) and init == "zero" else 2)
    rounds = 1 if kind == "dtr" else int(p.get("n_estimators", 100))
    opt_i = np.array([LOSS[loss], ncls_trees, 0 if depth is None else int(depth), 0 if mln is None else int(mln),
                      int(p.get("min_samples_split", 2)), int(p.get("min_samples_leaf", 1)),
                      1 if kind == "dtr" else 0, init_mode, int(sw_none), rounds, int(p["random_state"]), 0],
                     dtype=np.int32)
    lr = 1.0 if kind == "dtr" else float(p.get("learning_rate", 0.1))
    sub = 1.0 if kind == "dtr" else float(p.get("subsample", 1.0))
    opt_d = np.array([lr, sub, float(p.get("alpha", 0.9)), float(p.get("min_weight_fraction_leaf", 0.0))])
    maxn = (1 << (int(depth) + 1)) + 2 if depth is not None else 0
    if mln is not None:
        maxn = 2 * int(mln) + 2 if maxn == 0 else min(maxn, 2 * int(mln) + 2)
    if not 3 <= maxn <= (1 << 21):
        raise ValueError(f"tree too large for sklearn mode (max nodes {maxn})")
    return opt_d, opt_i, maxn


def fit(kind: str, params: dict, codes: np.ndarray, n_bins: int, y: np.ndarray, folds: list[tuple[int, int]],
        sample_weight: np.ndarray | None = None, threads: int = 1, exact: bool = True) -> list[dict]:
    assert kind in ("gbr", "gbc", "dtr") and folds and threads >= 1
    p = dict(params)
    _refuse(p, kind)
    codes = np.ascontiguousarray(codes.view(np.uint16) if codes.dtype == np.int16 else codes.astype(np.uint16))
    n, nf = codes.shape
    classes = None
    yd = np.ascontiguousarray(y, dtype=np.float64)
    n_classes = 0
    if kind == "gbc":
        classes, yd = _labels(y, folds)
        n_classes = classes.size
    ncls = n_classes if n_classes > 2 else 1
    sw_none = sample_weight is None
    sw = np.ones(n) if sw_none else np.ascontiguousarray(sample_weight, dtype=np.float64)
    if (sw < 0).any() or not np.isfinite(sw).all():
        raise ValueError("negative or non-finite sample weights are not supported")
    opt_d, opt_i, maxn = _cfg(kind, p, ncls, sw_none, n_classes)
    if not exact:
        opt_i[11] = 1
        if not _FAST_WARNED[0]:
            LOG.warning("skgb fast mode (exact=False): not bit-identical to scikit-learn")
            _FAST_WARNED[0] = True
    init = p.get("init")
    init_raw = None
    if opt_i[7] == 2:
        if not isinstance(init, np.ndarray):
            raise ValueError("init: None, 'zero' or an init raw-prediction array (n,) / (n, ncls); "
                             "estimator objects are not supported")
        init_raw = np.ascontiguousarray(init.reshape(n, ncls), dtype=np.float64)
    return _run(codes, n, nf, n_bins, yd, sw, folds, init_raw, opt_d, opt_i, maxn, ncls, threads, classes)


def _labels(y, folds) -> tuple[np.ndarray, np.ndarray]:
    ya = np.asarray(y)
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    if ya.dtype.kind not in "biuf":
        classes = np.unique(ya)
        for a, b in folds:
            if np.unique(ya[a:b]).size != classes.size:
                raise ValueError("every fold must contain every class (sklearn encodes labels per fit)")
        return classes, np.ascontiguousarray(np.searchsorted(classes, ya), dtype=np.float64)
    yv = np.ascontiguousarray(ya, dtype=np.float64)
    cls, ncl, codes = np.empty(yv.size), np.zeros(1, dtype=np.int64), np.empty(yv.size)
    rc = _lib_skgb().shira_skgb_labels(yv.ctypes.data, yv.size, lo.ctypes.data, hi.ctypes.data, len(folds),
                                       cls.ctypes.data, ncl.ctypes.data, codes.ctypes.data)
    if rc == -2:
        raise ValueError("every fold must contain every class (sklearn encodes labels per fit)")
    if rc != 0:
        raise ValueError(f"shira_skgb_labels returned {rc} (NaN label or bad folds)")
    return cls[:int(ncl[0])].astype(ya.dtype), codes


def _run(codes, n, nf, n_bins, yd, sw, folds, init_raw, opt_d, opt_i, maxn, ncls, threads, classes) -> list[dict]:
    k, rounds = len(folds), int(opt_i[9])
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    nodes = np.zeros((k, rounds, ncls, maxn, 5), dtype=np.int32)
    leaf = np.zeros((k, rounds, ncls, maxn), dtype=np.float64)
    count = np.zeros((k, rounds, ncls), dtype=np.int32)
    pred = np.zeros((int((hi - lo).sum()), ncls), dtype=np.float64)
    init = np.zeros((k, ncls), dtype=np.float64)
    exact = np.zeros((k, rounds, ncls), dtype=np.uint8)
    rc = _lib_skgb().shira_skgb_batch(codes.ctypes.data, n, nf, int(n_bins), yd.ctypes.data, sw.ctypes.data,
                                      lo.ctypes.data, hi.ctypes.data, k,
                                      None if init_raw is None else init_raw.ctypes.data, opt_d.ctypes.data,
                                      opt_i.ctypes.data, maxn, threads, nodes.ctypes.data, leaf.ctypes.data,
                                      count.ctypes.data, pred.ctypes.data, init.ctypes.data,
                                      exact.ctypes.data)
    if rc != 0:
        raise ValueError(f"shira_skgb_batch rc={rc}: {ERRORS.get(rc, 'unknown')}")
    offs = np.zeros(k + 1, dtype=np.int64)
    assert _lib_skgb().shira_fold_offsets(lo.ctypes.data, hi.ctypes.data, k, offs.ctypes.data) == pred.shape[0]
    out = []
    for j in range(k):
        trees = [[(nodes[j, r, c, :count[j, r, c]].copy(), leaf[j, r, c, :count[j, r, c]].copy())
                  for c in range(ncls)] for r in range(rounds)]
        out.append({"init": init[j], "raw": pred[offs[j]:offs[j + 1]], "trees": trees, "lr": float(opt_d[0]),
                    "classes": classes, "exact_trees": int(exact[j].sum()),
                    "_packed": (nodes[j], leaf[j], count[j], maxn)})
    return out


def apply(tree_nodes: np.ndarray, codes: np.ndarray) -> np.ndarray:
    leaf = np.zeros(codes.shape[0], dtype=np.int64)
    for _ in range(tree_nodes.shape[0]):
        f = tree_nodes[leaf, 0]
        live = f >= 0
        if not live.any():
            break
        r = np.nonzero(live)[0]
        go_left = codes[r, f[r]] <= tree_nodes[leaf[r], 1]
        leaf[r] = np.where(go_left, tree_nodes[leaf[r], 3], tree_nodes[leaf[r], 4])
    return leaf


def _pack_model(model: dict) -> tuple:
    if "_packed" in model:
        return model["_packed"]
    trees = model["trees"]
    rounds, ncls = len(trees), len(trees[0])
    maxn = max(nd.shape[0] for stage in trees for nd, _ in stage)
    nodes = np.zeros((rounds, ncls, maxn, 5), dtype=np.int32)
    leaf = np.zeros((rounds, ncls, maxn), dtype=np.float64)
    count = np.zeros((rounds, ncls), dtype=np.int32)
    for r, stage in enumerate(trees):
        for c, (nd, val) in enumerate(stage):
            nodes[r, c, :nd.shape[0]], leaf[r, c, :nd.shape[0]], count[r, c] = nd, val, nd.shape[0]
    return nodes, leaf, count, maxn


def predict_raw(model: dict, codes: np.ndarray) -> np.ndarray:
    ncls = model["init"].shape[0]
    cd = np.ascontiguousarray(codes.view(np.uint16) if codes.dtype == np.int16 else codes.astype(np.uint16))
    per_row = "init_rows" in model
    init = np.ascontiguousarray(np.asarray(model["init_rows"] if per_row else model["init"], dtype=np.float64))
    if per_row:
        init = np.ascontiguousarray(init.reshape(-1, ncls))
    nodes, leaf, count, maxn = _pack_model(model)
    nodes, leaf = np.ascontiguousarray(nodes, dtype=np.int32), np.ascontiguousarray(leaf, dtype=np.float64)
    count = np.ascontiguousarray(count, dtype=np.int32)
    out = np.empty((cd.shape[0], ncls), dtype=np.float64)
    rc = _lib_skgb().shira_skgb_predict_raw(cd.ctypes.data, cd.shape[0], cd.shape[1], nodes.ctypes.data,
                                            leaf.ctypes.data, count.ctypes.data, len(model["trees"]), ncls,
                                            int(maxn), float(model["lr"]), init.ctypes.data, int(per_row),
                                            out.ctypes.data)
    if rc != 0:
        raise RuntimeError(f"shira_skgb_predict_raw returned {rc}")
    return out
