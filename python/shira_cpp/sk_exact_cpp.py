from __future__ import annotations

import ctypes
import functools
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
LIB_PATH = (Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB")
            else SHIRA / "lib" / ("libshira.dylib" if sys.platform == "darwin" else "libshira.so"))
_P = ctypes.c_void_p
_I64 = ctypes.c_int64
_LIB: ctypes.CDLL | None = None
MAX_THREADS = 64


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        assert LIB_PATH.exists(), f"build first: {SHIRA / 'build.sh'}"
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.shira_sk_build_batch.restype = ctypes.c_int
        lib.shira_sk_build_batch.argtypes = [_P, _I64, _I64, _P, _P, _I64, _P, _P, ctypes.c_int32,
                                             _P, ctypes.c_int32, _I64, _P, _P, _P, _P, _P, _P, _P, _P]
        lib.shira_sk_apply.restype = ctypes.c_int
        lib.shira_sk_apply.argtypes = [_P, _I64, _I64, _P, _P, _P, _P, _I64, _P]
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray) -> int:
    return a.ctypes.data


def _cap(w2: np.ndarray, max_depth: int) -> int:
    w = np.ascontiguousarray(w2, dtype=np.float64)
    cap = int(_lib_w().shira_skx_cap(_ptr(w), w.shape[0], w.shape[1], int(max_depth), -1))
    assert cap >= 1, "shira_skx_cap: bad arguments"
    return cap


def build_batch(X, y, yc, k, W, crit, splitter, max_depth, min_leaf, max_features, seeds,
                threads: int = 6, legacy: bool = False) -> list[tuple]:
    x = np.ascontiguousarray(X, dtype=np.float32)
    assert x.ndim == 2 and 1 <= threads <= MAX_THREADS
    n, nf = x.shape
    w2 = np.ascontiguousarray(np.atleast_2d(W), dtype=np.float64)
    sd = np.ascontiguousarray(seeds, dtype=np.uint32)
    t_n = sd.size
    assert w2.shape == (t_n, n) and t_n >= 1, "one weight row per seed"
    yv = np.ascontiguousarray(y if y is not None else np.zeros(n), dtype=np.float64)
    ycv = np.ascontiguousarray(yc if yc is not None else np.zeros(n, np.int64), dtype=np.int64)
    assert yv.size == n and ycv.size == n
    kk, cap = max(int(k), 1), _cap(w2, int(max_depth))
    rec_i = np.array([crit, splitter, max_depth, min_leaf, max_features, int(legacy)], dtype=np.int64)
    feat, lch, rch = (np.empty((t_n, cap), np.int64) for _ in range(3))
    nnode, depth = np.empty((t_n, cap), np.int64), np.empty((t_n, cap), np.int64)
    thr, val = np.empty((t_n, cap)), np.empty((t_n, cap, kk))
    count = np.zeros(t_n, np.int64)
    rc = _lib().shira_sk_build_batch(_ptr(x), n, nf, _ptr(yv), _ptr(ycv), int(k), _ptr(w2), _ptr(sd),
                                     t_n, _ptr(rec_i), int(threads), cap, _ptr(feat), _ptr(thr),
                                     _ptr(lch), _ptr(rch), _ptr(val), _ptr(nnode), _ptr(depth),
                                     _ptr(count))
    assert rc == 0, f"shira_sk_build_batch returned {rc}"
    return [(feat[t, :c].copy(), thr[t, :c].copy(), lch[t, :c].copy(), rch[t, :c].copy(),
             val[t, :c].copy(), nnode[t, :c].copy(), depth[t, :c].copy())
            for t, c in enumerate(count.tolist())]


def build_generic(X, y, yc, k, w, crit, splitter, max_depth, min_leaf, max_features, seed):
    return build_batch(X, y, yc, k, np.asarray(w)[None, :], crit, splitter, max_depth, min_leaf,
                       max_features, [seed], threads=1)[0]


def build(X, y, w, max_depth, min_leaf, max_features, seed):
    arrs = build_batch(X, y, None, 0, np.asarray(w)[None, :], 0, 0, max_depth, min_leaf,
                       max_features, [seed], threads=1, legacy=True)[0]
    return arrs[0], arrs[1], arrs[2], arrs[3], arrs[4][:, 0].copy()


def apply(X, feature, threshold, left, right) -> np.ndarray:
    x = np.ascontiguousarray(X, dtype=np.float32)
    f, t = np.ascontiguousarray(feature, np.int64), np.ascontiguousarray(threshold, np.float64)
    lc, rc_ = np.ascontiguousarray(left, np.int64), np.ascontiguousarray(right, np.int64)
    out = np.empty(x.shape[0], np.int64)
    rc = _lib().shira_sk_apply(_ptr(x), x.shape[0], x.shape[1], _ptr(f), _ptr(t), _ptr(lc), _ptr(rc_),
                               f.size, _ptr(out))
    assert rc == 0, f"shira_sk_apply returned {rc}"
    return out


def predict(X, feature, threshold, left, right, value) -> np.ndarray:
    return np.asarray(value)[apply(X, feature, threshold, left, right)]


RAND_R_MAX = 0x7FFFFFFF
INT32_MAX = np.iinfo(np.int32).max
CRIT_CLF = {"gini": 1, "entropy": 2, "log_loss": 2}
CRIT_REG = {"squared_error": 0, "friedman_mse": 3, "absolute_error": 4, "poisson": 5}
NAN_REG = ("squared_error", "friedman_mse", "poisson")
ESTIMATORS = {
    "DecisionTreeRegressor": (False, "best", False, False, None),
    "DecisionTreeClassifier": (True, "best", False, False, None),
    "ExtraTreeRegressor": (False, "random", False, False, 1.0),
    "ExtraTreeClassifier": (True, "random", False, False, "sqrt"),
    "RandomForestRegressor": (False, "best", True, True, 1.0),
    "RandomForestClassifier": (True, "best", True, True, "sqrt"),
    "ExtraTreesRegressor": (False, "random", True, False, 1.0),
    "ExtraTreesClassifier": (True, "random", True, False, "sqrt"),
}
TREE_KEYS = {"criterion", "splitter", "max_depth", "min_samples_split", "min_samples_leaf",
             "min_weight_fraction_leaf", "max_features", "random_state", "max_leaf_nodes",
             "min_impurity_decrease", "class_weight", "ccp_alpha", "monotonic_cst"}
FOREST_KEYS = (TREE_KEYS - {"splitter"}) | {"n_estimators", "bootstrap", "oob_score", "n_jobs",
                                             "verbose", "warm_start", "max_samples"}


def _lib_x() -> ctypes.CDLL:
    lib = _lib()
    if not hasattr(lib, "_skx_ready"):
        lib.shira_skx_build_batch.restype = ctypes.c_int
        lib.shira_skx_build_batch.argtypes = ([_P, _I64, _I64, _P, _I64, _P, _P, _P, ctypes.c_int32,
                                               _P, _P, _P, _P, ctypes.c_int32, _I64] + [_P] * 12)
        lib.shira_skx_apply.restype = ctypes.c_int
        lib.shira_skx_apply.argtypes = [_P, _I64, _I64, _P, _P, _P, _P, _P, _I64, _P]
        lib.shira_skx_build_batch_pre.restype = ctypes.c_int
        lib.shira_skx_build_batch_pre.argtypes = [_P] + lib.shira_skx_build_batch.argtypes
        lib.shira_skx_pre_new.restype = ctypes.c_int
        lib.shira_skx_pre_new.argtypes = [ctypes.POINTER(_P), _P, _I64, _I64, ctypes.c_int32]
        lib.shira_skx_pre_free.restype, lib.shira_skx_pre_free.argtypes = None, [_P]
        lib._skx_ready = True
    return lib


def _nptr(a) -> int | None:
    return None if a is None else a.ctypes.data


def _lib_w() -> ctypes.CDLL:
    lib = _lib()
    if not hasattr(lib, "_skw_ready"):
        i32, d = ctypes.c_int32, ctypes.c_double
        lib.shira_skx_forest_weights.restype = ctypes.c_int
        lib.shira_skx_forest_weights.argtypes = [_I64, i32, _I64, i32, _I64, _P, d, _P, _I64, _P, i32, i32,
                                                 _P, _P, _P, _P]
        lib.shira_class_sample_weight.restype = ctypes.c_int
        lib.shira_class_sample_weight.argtypes = [_P, _I64, _I64, _P, _P, i32, _P]
        lib.shira_class_sample_weight_sub.restype = ctypes.c_int
        lib.shira_class_sample_weight_sub.argtypes = [_P, _I64, _I64, _P, _P, _P]
        lib.shira_index_counts.restype = ctypes.c_int
        lib.shira_index_counts.argtypes = [_P, _I64, _I64, _P]
        lib.shira_skx_weight_mul.restype = ctypes.c_int
        lib.shira_skx_weight_mul.argtypes = [_P, _P, _I64, d, _P, _P]
        lib.shira_skx_param_counts.restype = ctypes.c_int
        lib.shira_skx_param_counts.argtypes = [_I64, _I64, i32, _I64, i32, d, i32, d, i32, d, i32, _I64, _P]
        lib.shira_skx_cap.restype = _I64
        lib.shira_skx_cap.argtypes = [_P, i32, _I64, _I64, _I64]
        lib.shira_skx_predict_forest.restype = ctypes.c_int
        lib.shira_skx_predict_forest.argtypes = [_P, _I64, _I64, i32, _P, _P, _P, _P, _P, _P, _P, _I64, i32, i32, _P]
        lib._skw_ready = True
    return lib


def _encode_labels(y2: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    classes, enc = [], np.zeros(y2.shape, dtype=np.float64)
    for k in range(y2.shape[1]):
        ck, enc[:, k] = np.unique(y2[:, k], return_inverse=True)
        classes.append(ck)
    return np.ascontiguousarray(enc), np.array([c.shape[0] for c in classes], dtype=np.int64), classes


def build_ex(X, Y, ncls, W, seeds, crit: int, splitter: int, max_depth: int, min_split: int,
             min_leaf: int, max_features: int, max_leaf_nodes: int, min_wl, min_imp_dec: float,
             ccp_alpha: float, miss=None, mono=None, threads: int = 2, path_mode: int = 0, pre=None) -> list[dict]:
    path_mode = int(os.environ.get("SHIRA_SKX_PATH_MODE", path_mode))
    x = np.ascontiguousarray(X, dtype=np.float32)
    y2 = np.ascontiguousarray(np.asarray(Y, np.float64).reshape(x.shape[0], -1))
    w2 = np.ascontiguousarray(np.atleast_2d(W), dtype=np.float64)
    sd = np.ascontiguousarray(seeds, dtype=np.uint32)
    n, nf = x.shape
    t_n, n_out = sd.size, y2.shape[1]
    assert w2.shape == (t_n, n) and 1 <= threads <= MAX_THREADS
    nc = None if ncls is None else np.ascontiguousarray(ncls, dtype=np.int64)
    mnc = 1 if nc is None else int(nc.max())
    stride = n_out * mnc
    cap = int(_lib_w().shira_skx_cap(_ptr(w2), t_n, n, int(max_depth), int(max_leaf_nodes)))
    assert cap >= 1, "shira_skx_cap: bad arguments"
    rec_i = np.array([crit, splitter, max_depth, min_split, min_leaf, max_features, max_leaf_nodes,
                      path_mode], dtype=np.int64)
    wl = np.broadcast_to(np.asarray(min_wl, dtype=np.float64), (t_n,))
    rec_d = np.concatenate([[min_imp_dec, ccp_alpha], wl]).astype(np.float64)
    ms = None if miss is None else np.ascontiguousarray(miss, dtype=np.uint8)
    mo = None if mono is None else np.ascontiguousarray(mono, dtype=np.int8)
    i64 = [np.empty((t_n, cap), np.int64) for _ in range(5)]
    f64 = [np.empty((t_n, cap)) for _ in range(3)]
    val, mgl, count = np.empty((t_n, cap, stride)), np.empty((t_n, cap), np.uint8), np.zeros(t_n, np.int64)
    path = np.zeros(t_n, np.uint8)
    lib = _lib_x()
    assert pre is None or x is X, "a shared presort needs the X buffer it was built on"
    fn = lib.shira_skx_build_batch if pre is None else functools.partial(lib.shira_skx_build_batch_pre, pre)
    rc = fn(
        _ptr(x), n, nf, _ptr(y2), n_out, _nptr(nc), _ptr(w2), _ptr(sd), t_n, _ptr(rec_i), _ptr(rec_d),
        _nptr(ms), _nptr(mo), int(threads), cap, _ptr(i64[0]), _ptr(f64[0]), _ptr(i64[1]), _ptr(i64[2]),
        _ptr(val), _ptr(i64[3]), _ptr(f64[1]), _ptr(f64[2]), _ptr(mgl), _ptr(i64[4]), _ptr(count), _ptr(path))
    if rc != 0:
        raise RuntimeError(f"shira_skx_build_batch returned {rc}")
    out = []
    for t, c in enumerate(count.tolist()):
        out.append({"feature": i64[0][t, :c].copy(), "threshold": f64[0][t, :c].copy(),
                    "children_left": i64[1][t, :c].copy(), "children_right": i64[2][t, :c].copy(),
                    "value": val[t, :c].reshape(c, n_out, mnc).copy(),
                    "n_node_samples": i64[3][t, :c].copy(), "weighted_n_node_samples": f64[1][t, :c].copy(),
                    "impurity": f64[2][t, :c].copy(), "missing_go_to_left": mgl[t, :c].copy(),
                    "depth": i64[4][t, :c].copy(), "fast_path": bool(path[t] == 1),
                    "path": int(path[t])})
    return out


def apply_ex(X, tree: dict) -> np.ndarray:
    x = np.ascontiguousarray(X, dtype=np.float32)
    a = [np.ascontiguousarray(tree[k], dt) for k, dt in (
        ("feature", np.int64), ("threshold", np.float64), ("children_left", np.int64),
        ("children_right", np.int64), ("missing_go_to_left", np.uint8))]
    out = np.empty(x.shape[0], np.int64)
    rc = _lib_x().shira_skx_apply(_ptr(x), x.shape[0], x.shape[1], *[_ptr(v) for v in a],
                                  a[0].size, _ptr(out))
    assert rc == 0, f"shira_skx_apply returned {rc}"
    return out


def seed_state(random_state: int) -> int:
    return int(np.random.RandomState(random_state).randint(0, RAND_R_MAX))


def _class_weight_vec(cw, classes, y) -> np.ndarray:
    if cw is None or len(cw) == 0:
        return np.ones(classes.shape[0], dtype=np.float64)
    if isinstance(cw, str):
        assert cw == "balanced", cw
        le_classes, y_ind = np.unique(y, return_inverse=True)
        enc = np.ascontiguousarray(y_ind, dtype=np.float64)
        k = np.array([le_classes.shape[0]], dtype=np.int64)
        per_row = np.empty(enc.shape[0], dtype=np.float64)
        rc = _lib_w().shira_class_sample_weight(_ptr(enc), enc.shape[0], 1, _ptr(k), None, 1, _ptr(per_row))
        assert rc == 0, f"shira_class_sample_weight returned {rc}"
        recip = np.empty(le_classes.shape[0], dtype=np.float64)
        recip[y_ind] = per_row
        return recip[np.searchsorted(le_classes, classes)]
    weight = np.ones(classes.shape[0], dtype=np.float64)
    missing = [c for c in classes.tolist() if c not in cw]
    for i, c in enumerate(classes.tolist()):
        if c in cw:
            weight[i] = cw[c]
    if missing and len(classes) - len(missing) != len(cw):
        raise ValueError(f"The classes, {missing}, are not in class_weight")
    return weight


def compute_sample_weight(cw, y, indices=None) -> np.ndarray:
    y2 = np.asarray(y).reshape(np.asarray(y).shape[0], -1)
    if indices is not None and cw != "balanced":
        raise ValueError("The only valid class_weight for subsampling is 'balanced'.")
    enc, ncls, classes = _encode_labels(y2)
    n, n_out = enc.shape
    out = np.empty(n, dtype=np.float64)
    if indices is not None:
        ind = np.ascontiguousarray(np.asarray(indices), dtype=np.int64)
        cnt = np.empty(n, dtype=np.int64)
        if _lib_w().shira_index_counts(_ptr(ind), ind.size, n, _ptr(cnt)) != 0:
            raise ValueError("indices must lie in [0, n)")
        rc = _lib_w().shira_class_sample_weight_sub(_ptr(enc), n, n_out, _ptr(ncls), _ptr(cnt), _ptr(out))
        assert rc == 0, f"shira_class_sample_weight_sub returned {rc}"
        return out
    if cw == "balanced":
        rc = _lib_w().shira_class_sample_weight(_ptr(enc), n, n_out, _ptr(ncls), None, 1, _ptr(out))
    else:
        vec = np.ascontiguousarray(np.concatenate(
            [_class_weight_vec(cw if n_out == 1 else cw[k], classes[k], y2[:, k]) for k in range(n_out)]),
            dtype=np.float64)
        rc = _lib_w().shira_class_sample_weight(_ptr(enc), n, n_out, _ptr(ncls), _ptr(vec), 0, _ptr(out))
    assert rc == 0, f"shira_class_sample_weight returned {rc}"
    return out


def _refuse(msg: str):
    raise NotImplementedError(f"SHIRA sklearn-exact: {msg}")


def _params(name: str, given: dict) -> dict:
    clf, splitter, forest, boot, mf = ESTIMATORS[name]
    allowed = FOREST_KEYS if forest else TREE_KEYS
    bad = set(given) - allowed
    if bad:
        _refuse(f"{name} parameters not supported: {sorted(bad)}")
    p = {"criterion": "gini" if clf else "squared_error", "splitter": splitter, "max_depth": None,
         "min_samples_split": 2, "min_samples_leaf": 1, "min_weight_fraction_leaf": 0.0,
         "max_features": mf, "random_state": None, "max_leaf_nodes": None,
         "min_impurity_decrease": 0.0, "class_weight": None, "ccp_alpha": 0.0, "monotonic_cst": None,
         "n_estimators": 100, "bootstrap": boot, "oob_score": False, "warm_start": False,
         "max_samples": None}
    p.update(given)
    if not isinstance(p["random_state"], (int, np.integer)) or isinstance(p["random_state"], bool):
        _refuse("random_state must be an int (sklearn's own draws are replayed from it)")
    if p["oob_score"] or p["warm_start"]:
        _refuse("oob_score and warm_start are not covered")
    if p["splitter"] not in ("best", "random") or p["criterion"] not in (CRIT_CLF if clf else CRIT_REG):
        _refuse(f"splitter {p['splitter']!r} / criterion {p['criterion']!r}")
    return p


def _counts(p: dict, n: int, nf: int) -> tuple[int, int, int, int, int]:
    msl, mss, mf = p["min_samples_leaf"], p["min_samples_split"], p["max_features"]
    msl_int, mss_int = isinstance(msl, (int, np.integer)), isinstance(mss, (int, np.integer))
    if isinstance(mf, str):
        mfk, mfv = (1 if mf == "sqrt" else 2), 0.0
    elif mf is None:
        mfk, mfv = 0, 0.0
    else:
        mfk, mfv = (3 if isinstance(mf, (int, np.integer)) else 4), float(mf)
    out = np.zeros(5, dtype=np.int64)
    rc = _lib_w().shira_skx_param_counts(
        n, nf, int(p["max_depth"] is None), 0 if p["max_depth"] is None else int(p["max_depth"]),
        0 if msl_int else 1, float(msl), 0 if mss_int else 1, float(mss), mfk, mfv,
        int(p["max_leaf_nodes"] is None), 0 if p["max_leaf_nodes"] is None else int(p["max_leaf_nodes"]), _ptr(out))
    assert rc == 0, f"shira_skx_param_counts returned {rc}"
    return int(out[0]), int(out[1]), int(out[2]), int(out[3]), int(out[4])


def _encode_y(y, clf: bool):
    y2 = np.atleast_1d(np.asarray(y))
    y2 = y2.reshape(-1, 1) if y2.ndim == 1 else y2
    if not clf:
        return np.ascontiguousarray(y2, dtype=np.float64), None, None, y2
    classes, enc = [], np.zeros(y2.shape, dtype=np.int64)
    for k in range(y2.shape[1]):
        ck, enc[:, k] = np.unique(y2[:, k], return_inverse=True)
        classes.append(ck)
    ncls = np.array([c.shape[0] for c in classes], dtype=np.int64)
    return np.ascontiguousarray(enc, dtype=np.float64), classes, ncls, y2


def _missing_mask(x32, p: dict, clf: bool, forest: bool):
    if np.isinf(x32).any():
        raise ValueError("Input X contains infinity or a value too large for dtype('float32').")
    if not np.isnan(x32).any():
        return None
    ok = p["criterion"] in (CRIT_CLF if clf else NAN_REG)
    if not ok or (p["monotonic_cst"] is not None and not forest):
        raise ValueError("Input X contains NaN.")
    nan = np.isnan(x32)
    sk = _sk_nan_mask(nan)
    if not np.array_equal(sk, nan.any(axis=0)):
        _refuse("a NaN column that sklearn 1.8's _any_isnan_axis0 leaves unmarked (its per-row break); "
                "sklearn then sorts that NaN as a value, which this engine does not replay")
    return sk.astype(np.uint8)


def _sk_nan_mask(nan) -> np.ndarray:
    n_rows, nf = nan.shape
    assert nan.dtype == np.bool_ and nf >= 1, "boolean NaN matrix expected"
    marked = np.zeros(nf, dtype=bool)
    i = 0
    for _ in range(nf):
        rows = nan[i:, ~marked].any(axis=1) if i < n_rows else np.zeros(0, dtype=bool)
        if not rows.any():
            break
        r = i + int(np.argmax(rows))
        marked[int(np.flatnonzero(nan[r] & ~marked)[0])] = True
        i = r + 1
    assert marked.sum() <= nf
    return marked


def _monotonic(p: dict, nf: int, n_out: int, ncls):
    cst = p["monotonic_cst"]
    if cst is None:
        return None
    if n_out > 1:
        raise ValueError("Monotonicity constraints are not supported with multiple outputs.")
    cst = np.asarray(cst)
    if cst.shape[0] != nf or not np.all(np.isin(cst, (-1, 0, 1))):
        raise ValueError("monotonic_cst must be an array-like of -1, 0 or 1, one per feature")
    cst = cst.astype(np.int8)
    if ncls is not None:
        if ncls[0] > 2:
            raise ValueError("Monotonicity constraints are not supported with multiclass classification")
        cst = cst * np.int8(-1)
    return cst


class Model:

    def __init__(self, name, trees, classes, ncls, n_outputs, forest, monotonic):
        self.name, self.trees_, self.classes_ = name, trees, classes
        self.n_classes_, self.n_outputs_, self.forest = ncls, n_outputs, forest
        self.monotonic = monotonic

    def predict(self, X) -> np.ndarray:
        return predict_model(self, X)

    def predict_proba(self, X):
        return predict_proba_model(self, X)


def _n_bootstrap(n: int, max_samples) -> int:
    if max_samples is None:
        return n
    if isinstance(max_samples, (int, np.integer)):
        if max_samples > n:
            raise ValueError(f"`max_samples` must be <= n_samples={n} but got value {max_samples}")
        return int(max_samples)
    return max(round(n * max_samples), 1)


def _forest_weights(p: dict, n: int, sw, yenc, y_orig, clf: bool, threads: int = 1):
    cw = p["class_weight"] if clf else None
    if cw is not None:
        if isinstance(cw, str) and cw not in ("balanced", "balanced_subsample"):
            raise ValueError(f'Valid presets for class_weight include "balanced" and "balanced_subsample". Given "{cw}".')
        if cw != "balanced_subsample" or not p["bootstrap"]:
            exp = compute_sample_weight("balanced" if cw == "balanced_subsample" else cw, y_orig)
            sw, _ = _weight_mul(sw, exp, n, None)
    if not p["bootstrap"] and p["max_samples"] is not None:
        raise ValueError("`max_sample` cannot be set if `bootstrap=False`.")
    t_n = int(p["n_estimators"])
    n_boot = _n_bootstrap(n, p["max_samples"]) if p["bootstrap"] else 0
    sub = cw == "balanced_subsample" and bool(p["bootstrap"])
    swc = None if sw is None else np.ascontiguousarray(sw, dtype=np.float64)
    ye = np.ascontiguousarray(yenc, dtype=np.float64) if sub else None
    nc = np.ascontiguousarray([np.unique(yenc[:, k]).shape[0] for k in range(yenc.shape[1])], dtype=np.int64) \
        if sub else None
    w2 = np.empty((t_n, n), dtype=np.float64)
    seeds, tree_rs, wl = np.empty(t_n, np.uint32), np.empty(t_n, np.int64), np.empty(t_n, np.float64)
    rc = _lib_w().shira_skx_forest_weights(
        n, t_n, int(p["random_state"]), int(bool(p["bootstrap"])), int(n_boot), _nptr(swc),
        float(p["min_weight_fraction_leaf"]), _nptr(ye), 0 if ye is None else ye.shape[1], _nptr(nc), int(sub),
        int(max(1, threads)), _ptr(w2), _ptr(seeds), _ptr(tree_rs), _ptr(wl))
    if rc != 0:
        raise ValueError(f"shira_skx_forest_weights returned {rc}")
    return w2, [int(v) for v in seeds], [float(v) for v in wl], [int(v) for v in tree_rs]


def _tree_weights(p: dict, n: int, sw, y_orig, clf: bool):
    cw = p["class_weight"] if clf else None
    if cw is not None:
        exp = compute_sample_weight(cw, y_orig)
        sw, _ = _weight_mul(sw, exp, n, None)
    w, wl = _weight_mul(sw, None, n, p["min_weight_fraction_leaf"])
    return w[None, :], [seed_state(p["random_state"])], [wl], [p["random_state"]]


def _weight_mul(sw, ex, n: int, frac):
    a = None if sw is None else np.ascontiguousarray(sw, dtype=np.float64)
    b = None if ex is None else np.ascontiguousarray(ex, dtype=np.float64)
    out, wl = np.empty(n, dtype=np.float64), ctypes.c_double(0.0)
    rc = _lib_w().shira_skx_weight_mul(_nptr(a), _nptr(b), n, float(0.0 if frac is None else frac), _ptr(out),
                                       None if frac is None else ctypes.byref(wl))
    assert rc == 0, f"shira_skx_weight_mul returned {rc}"
    return out, (None if frac is None else wl.value)


MODES = {"exact": 0, "fast": 2}
MODE_ENV = "SHIRA_SK_MODE"


def resolve_mode(mode=None) -> int:
    m = mode if mode is not None else os.environ.get(MODE_ENV, "exact")
    if m not in MODES:
        raise ValueError(f"shira_mode must be one of {sorted(MODES)}, got {m!r}")
    return MODES[m]


def fit(name: str, X, y, sample_weight=None, threads: int = 2, path_mode=None, mode=None, **params) -> Model:
    path_mode = resolve_mode(mode) if path_mode is None else int(path_mode)
    if name not in ESTIMATORS:
        _refuse(f"estimator {name!r}")
    if hasattr(X, "tocsc"):
        _refuse("sparse X")
    clf, _, forest, _, _ = ESTIMATORS[name]
    p = _params(name, params)
    x32 = np.ascontiguousarray(X, dtype=np.float32)
    n, nf = x32.shape
    miss = _missing_mask(x32, p, clf, forest)
    yenc, classes, ncls, y_orig = _encode_y(y, clf)
    assert yenc.shape[0] == n, "one label row per sample"
    if p["criterion"] == "poisson" and (np.any(yenc < 0) or np.sum(yenc) <= 0):
        raise ValueError("Poisson regression needs y >= 0 with a positive sum.")
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).reshape(n)
    mono = _monotonic(p, nf, yenc.shape[1], ncls)
    crit = (CRIT_CLF if clf else CRIT_REG)[p["criterion"]]
    md, mss, msl, mf, mln = _counts(p, n, nf)
    if forest:
        w2, seeds, wl, _ = _forest_weights(p, n, sw, yenc, y_orig, clf, threads)
    else:
        w2, seeds, wl, _ = _tree_weights(p, n, sw, y_orig, clf)
    trees = build_ex(x32, yenc, ncls, w2, seeds, crit, 0 if p["splitter"] == "best" else 1, md, mss, msl,
                     mf, mln, wl, float(p["min_impurity_decrease"]), float(p["ccp_alpha"]), miss, mono,
                     threads, path_mode)
    return Model(name, trees, classes, ncls, yenc.shape[1], forest, mono is not None)


def _pack_trees(model: Model) -> dict:
    pk = getattr(model, "_packed", None)
    if pk is not None:
        return pk
    trees = model.trees_
    offs = np.zeros(len(trees) + 1, dtype=np.int64)
    offs[1:] = np.cumsum([t["feature"].shape[0] for t in trees])
    cat = lambda key, dt: np.ascontiguousarray(np.concatenate([t[key] for t in trees]), dtype=dt)
    stride = int(np.prod(trees[0]["value"].shape[1:]))
    pk = {"offs": offs, "feature": cat("feature", np.int64), "threshold": cat("threshold", np.float64),
          "left": cat("children_left", np.int64), "right": cat("children_right", np.int64),
          "mgl": cat("missing_go_to_left", np.uint8),
          "value": np.ascontiguousarray(np.concatenate([t["value"].reshape(-1, stride) for t in trees])),
          "stride": stride, "shape": trees[0]["value"].shape[1:]}
    model._packed = pk
    return pk


def _predict_all(model: Model, X) -> np.ndarray:
    x32 = np.ascontiguousarray(X, dtype=np.float32)
    if model.monotonic and np.isnan(x32).any():
        raise ValueError("Input X contains NaN (monotonic_cst models do not accept NaN, as sklearn).")
    pk = _pack_trees(model)
    out = np.empty((x32.shape[0], pk["stride"]), dtype=np.float64)
    rc = _lib_w().shira_skx_predict_forest(
        _ptr(x32), x32.shape[0], x32.shape[1], len(model.trees_), _ptr(pk["offs"]), _ptr(pk["feature"]),
        _ptr(pk["threshold"]), _ptr(pk["left"]), _ptr(pk["right"]), _ptr(pk["mgl"]), _ptr(pk["value"]),
        pk["stride"], int(bool(model.forest)), 1, _ptr(out))
    if rc != 0:
        raise RuntimeError(f"shira_skx_predict_forest returned {rc}")
    return out.reshape((x32.shape[0],) + tuple(pk["shape"]))


def predict_model(model: Model, X) -> np.ndarray:
    assert model.classes_ is None, "use predict_proba for classifiers"
    pr = _predict_all(model, X)[:, :, 0]
    return pr.reshape(pr.shape[0]) if model.n_outputs_ == 1 else pr


def predict_proba_model(model: Model, X):
    assert model.classes_ is not None, "use predict for regressors"
    pr = _predict_all(model, X)
    out = [np.ascontiguousarray(pr[:, k, :c]) for k, c in enumerate(model.n_classes_.tolist())]
    return out[0] if model.n_outputs_ == 1 else out
