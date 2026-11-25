from __future__ import annotations

import ctypes
import functools
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
LIB_PATH = (Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB")
            else SHIRA / "lib" / ("libshira.dylib" if sys.platform == "darwin" else "libshira.so"))
_P = ctypes.c_void_p
_I64 = ctypes.c_int64
_I32 = ctypes.c_int32
_LIB: ctypes.CDLL | None = None
MAX_THREADS = 64
ALMOST_INF = 1e300
SUBSAMPLE = int(2e5)
N_IN, N_OUT = 23, 11
LOSS_IDS = {"squared_error": 0, "binary": 1, "multiclass": 2, "absolute_error": 3, "quantile": 4,
            "poisson": 5, "gamma": 6}
REG_LOSSES = ("squared_error", "absolute_error", "quantile", "poisson", "gamma")

DEFAULTS = {"loss": None, "learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 31, "max_depth": None,
            "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0, "max_bins": 255,
            "categorical_features": "from_dtype", "monotonic_cst": None, "interaction_cst": None,
            "warm_start": False, "early_stopping": "auto", "scoring": "loss", "validation_fraction": 0.1,
            "n_iter_no_change": 10, "tol": 1e-7, "verbose": 0, "random_state": None, "class_weight": None,
            "quantile": None}


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        if not LIB_PATH.exists():
            raise FileNotFoundError(f"build first: {SHIRA / 'build.sh'}")
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.shira_train_hgb.restype = ctypes.c_int
        lib.shira_train_hgb.argtypes = [_P, _I64, _I32, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P,
                                        _I32, _I32, _P, _P, _P, _P, _P]
        lib.shira_train_hgb_ex.restype = ctypes.c_int
        lib.shira_train_hgb_ex.argtypes = [_P, _P, _P, _P]
        lib.shira_hgb_value_codes.restype = ctypes.c_int
        lib.shira_hgb_value_codes.argtypes = [_P, _I64, _I32, _I32, _P, _P, _I64, _P]
        lib.shira_hgb_pairwise_sum.restype = ctypes.c_double
        lib.shira_hgb_pairwise_sum.argtypes = [_P, _I64]
        lib.shira_hgb_weighted_percentile.restype = ctypes.c_int
        lib.shira_hgb_weighted_percentile.argtypes = [_P, _P, _I64, ctypes.c_double, _P]
        lib.shira_hgb_seeds.restype = ctypes.c_int
        lib.shira_hgb_seeds.argtypes = [_I64, _P]
        lib.shira_hgb_class_weights.restype = ctypes.c_int
        lib.shira_hgb_class_weights.argtypes = [_P, _I64, _I32, _P, _P, _I32, _P, _P]
        lib.shira_hgb_val_split.restype = ctypes.c_int
        lib.shira_hgb_val_split.argtypes = [_P, _I64, _I64, _I64, _I64, _I32, _P, _P]
        lib.shira_hgb_fold_lut.restype = ctypes.c_int
        lib.shira_hgb_fold_lut.argtypes = [_P, _P, _P, _P, _P, _P, _I32, _I64, _P]
        lib.shira_hgb_baseline.restype = ctypes.c_int
        lib.shira_hgb_baseline.argtypes = [_P, _P, _I64, _I32, _I32, ctypes.c_double, ctypes.c_double, ctypes.c_double, _P]
        lib.shira_hgb_interaction_states.restype = ctypes.c_int
        lib.shira_hgb_interaction_states.argtypes = [_P] * 3 + [_I32] * 3 + [_P] * 4
        lib.shira_hgb_label_counts.restype = ctypes.c_int
        lib.shira_hgb_label_counts.argtypes = [_P, _I64, _P, _P, _I32, _I32]
        lib.shira_hgb_prepare_fold.restype = ctypes.c_int
        lib.shira_hgb_prepare_fold.argtypes = [_P, _I64, _I32, _I64, _I64, _I32, _I64, ctypes.c_uint32, _I64, _P,
                                               _I64, _I32, _P, _P, _P, _P, _P, _P]
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray | None) -> int | None:
    return None if a is None else a.ctypes.data


def _refuse(cond: bool, msg: str) -> None:
    if cond:
        raise ValueError(f"hgb_cpp: {msg}")


def _check_loss(p: dict, objective: str) -> str:
    _refuse(objective not in ("regression", "binary", "multiclass", "classifier"), f"objective {objective}")
    if objective == "regression":
        loss = "squared_error" if p["loss"] is None else p["loss"]
        _refuse(loss not in REG_LOSSES, f"loss {loss} (regression: {REG_LOSSES})")
        if loss == "quantile":
            q = p["quantile"]
            _refuse(q is None or not 0.0 < float(q) < 1.0, "quantile loss needs quantile in (0, 1)")
        _refuse(p["class_weight"] is not None, "class_weight is a classifier option")
        return loss
    _refuse(p["loss"] not in (None, "log_loss"), f"loss {p['loss']} (classifier: log_loss)")
    cw = p["class_weight"]
    _refuse(cw is not None and cw != "balanced" and not isinstance(cw, dict), "class_weight: None, 'balanced' or dict")
    return "log_loss"


def _check_es(p: dict) -> None:
    es = p["early_stopping"]
    _refuse(es not in (True, False, "auto"), "early_stopping must be True, False or 'auto'")
    if es is not False:
        _refuse(p["scoring"] != "loss", "early stopping is supported with scoring='loss' only")
    vf = p["validation_fraction"]
    ok = vf is None or (isinstance(vf, (float, np.floating)) and 0 < vf < 1) or (
        isinstance(vf, (int, np.integer)) and not isinstance(vf, bool) and vf >= 1)
    _refuse(not ok, "validation_fraction: None, float in (0, 1) or int >= 1")
    _refuse(int(p["n_iter_no_change"]) < 1, "n_iter_no_change >= 1")


def check_params(params: dict, objective: str) -> dict:
    bad = sorted(set(params) - set(DEFAULTS))
    _refuse(bool(bad), f"unknown / unsupported parameters {bad}")
    p = {**DEFAULTS, **params}
    loss = _check_loss(p, objective)
    _check_es(p)
    _refuse(p["warm_start"] is not False, "warm_start not supported")
    _refuse(not 0.0 < float(p["max_features"]) <= 1.0, "max_features in (0, 1]")
    _refuse(isinstance(p["monotonic_cst"], dict), "monotonic_cst as a dict (feature names) not supported")
    cf = p["categorical_features"]
    _refuse(isinstance(cf, str) and cf != "from_dtype", "categorical features by name not supported")
    _refuse(not isinstance(p["random_state"], (int, np.integer)), "random_state must be an int")
    _refuse(p["max_leaf_nodes"] is None or int(p["max_leaf_nodes"]) < 2, "max_leaf_nodes must be an int >= 2")
    _refuse(p["max_depth"] is not None and int(p["max_depth"]) < 1, "max_depth must be None or >= 1")
    _refuse(not 2 <= int(p["max_bins"]) <= 255, "max_bins in [2, 255]")
    _refuse(not float(p["learning_rate"]) > 0 or float(p["l2_regularization"]) < 0, "bad learning_rate / l2")
    q = float(p["quantile"]) if loss == "quantile" else 0.5
    return {"objective": objective, "loss": loss, "lr": float(p["learning_rate"]), "rounds": int(p["max_iter"]),
            "max_leaves": int(p["max_leaf_nodes"]), "max_depth": 0 if p["max_depth"] is None else int(p["max_depth"]),
            "min_leaf": int(p["min_samples_leaf"]), "l2": float(p["l2_regularization"]),
            "max_bins": int(p["max_bins"]), "random_state": int(p["random_state"]), "min_hess": 1e-3,
            "min_gain": 0.0, "quantile": q, "qq": float(np.true_divide(100 * q, 100)),
            "wq": (100 * q) / 100 if loss == "quantile" else 50 / 100, "max_features": float(p["max_features"]),
            "early_stopping": p["early_stopping"], "validation_fraction": p["validation_fraction"],
            "n_iter_no_change": int(p["n_iter_no_change"]), "tol": float(p["tol"]),
            "monotonic_cst": p["monotonic_cst"], "interaction_cst": p["interaction_cst"],
            "categorical_features": cf, "class_weight": p["class_weight"]}


@functools.lru_cache(maxsize=64)
def seeds(random_state: int) -> tuple[np.uint64, np.uint64]:
    rs_i = int(random_state)
    if 0 <= rs_i <= 0xFFFFFFFF:
        out = np.zeros(2, dtype=np.uint64)
        _refuse(_lib().shira_hgb_seeds(rs_i, _ptr(out)) != 0, "shira_hgb_seeds failed")
        return np.uint64(out[0]), np.uint64(out[1])
    return _ref_seeds(random_state)


def _ref_seeds(random_state: int) -> tuple[np.uint64, np.uint64]:
    rs = np.random.RandomState(random_state)
    a = rs.randint(np.iinfo(np.uint32).max, dtype="u8")
    b = rs.randint(np.iinfo(np.uint32).max, dtype="u8")
    return a, b


def binning_seed(random_state: int) -> np.uint64:
    return seeds(random_state)[0]


def cfg_seeds(cfg: dict) -> tuple[np.uint64, np.uint64]:
    pair = cfg.get("seed_pair")
    if pair is None:
        return seeds(cfg["random_state"])
    assert len(pair) == 2 and cfg["random_state"] is None
    return pair


def check_seed_pair(seed_pair) -> tuple[np.uint64, np.uint64] | None:
    if seed_pair is None:
        return None
    top = int(np.iinfo(np.uint32).max)
    ok = len(seed_pair) == 2 and all(isinstance(s, (int, np.integer)) and 0 <= int(s) < top for s in seed_pair)
    _refuse(not ok, "seed_pair must be two ints in [0, 2^32 - 2]")
    return np.uint64(int(seed_pair[0])), np.uint64(int(seed_pair[1]))


def rng_state(random_state: int) -> np.ndarray:
    return pcg_words(seeds(random_state)[1])


def pcg_words(feature_seed) -> np.ndarray:
    st = np.random.default_rng(feature_seed).bit_generator.state
    s, inc = int(st["state"]["state"]), int(st["state"]["inc"])
    m64 = (1 << 64) - 1
    return np.array([s >> 64, s & m64, inc >> 64, inc & m64, st["has_uint32"], st["uinteger"]], dtype=np.uint64)


def categorical_mask(cf, nf: int) -> np.ndarray | None:
    if cf is None or (isinstance(cf, str) and cf == "from_dtype"):
        return None
    arr = np.asarray(cf)
    if arr.size == 0:
        return None
    _refuse(arr.dtype.kind not in ("i", "b"), "categorical_features must be a bool mask or int indices")
    if arr.dtype.kind == "i":
        _refuse(arr.max() >= nf or arr.min() < 0, "categorical_features indices in [0, n_features - 1]")
        mask = np.zeros(nf, dtype=bool)
        mask[arr] = True
    else:
        _refuse(arr.shape[0] != nf, "categorical_features mask must have shape (n_features,)")
        mask = arr.astype(bool)
    return mask if mask.any() else None


def monotonic(cst, nf: int, objective: str) -> np.ndarray | None:
    if cst is None:
        return None
    arr = np.asarray(cst, dtype=np.int8)
    _refuse(arr.shape != (nf,), f"monotonic_cst must have shape ({nf},)")
    _refuse(not np.isin(arr, (-1, 0, 1)).all(), "monotonic_cst values in {-1, 0, 1}")
    _refuse(objective == "multiclass" and bool((arr != 0).any()), "monotonic constraints with multiclass")
    return arr if bool((arr != 0).any()) else None


def interaction_groups(cst, nf: int) -> list | None:
    if cst is None:
        return None
    if isinstance(cst, str) and cst == "no_interactions":
        groups = [[i] for i in range(nf)]
    elif isinstance(cst, str) and cst == "pairwise":
        groups = [(i, j) for i in range(nf) for j in range(i + 1, nf)]
    else:
        _refuse(isinstance(cst, str), f"interaction_cst {cst!r}")
        groups = cst
    sets = [set(g) for g in groups]
    for g in sets:
        _refuse(any(not isinstance(x, (int, np.integer)) or not 0 <= x < nf for x in g), f"bad group {g}")
    rest = set(range(nf)) - set().union(*sets)
    if len(rest) > 0:
        sets.append(rest)
    return sets


def interaction_states_cpp(cst, nf: int, sets: list):
    if isinstance(cst, str):
        groups = [[i] for i in range(nf)] if cst == "no_interactions" else \
            [(i, j) for i in range(nf) for j in range(i + 1, nf)]
    else:
        groups = cst
    base = set().__sizeof__()
    off, feat, mask = [0], [], []
    for g in groups:
        if type(g) in (set, frozenset):
            sz = g.__sizeof__()
            mask.append(7 if sz == base else (sz - base) // 16 - 1)
        elif isinstance(g, (set, frozenset)):
            return interaction_states(sets, nf)
        else:
            mask.append(0)
        feat.extend(int(x) for x in g)
        off.append(len(feat))
    go = np.ascontiguousarray(off, dtype=np.int32)
    gf = np.ascontiguousarray(feat if feat else [0], dtype=np.int32)
    gm = np.ascontiguousarray(mask if mask else [0], dtype=np.int64)
    rc, n_st = -2, ctypes.c_int32(0)
    for cap in (4096, 100_000):
        st_off = np.zeros(cap + 1, dtype=np.int32)
        st_feat = np.zeros(cap * max(nf, 1), dtype=np.int32)
        st_trans = np.zeros(cap * max(nf, 1), dtype=np.int32)
        rc = _lib().shira_hgb_interaction_states(_ptr(go), _ptr(gf), _ptr(gm), len(groups), nf, cap, ctypes.byref(n_st),
                                                 _ptr(st_off), _ptr(st_feat), _ptr(st_trans))
        if rc != -2:
            break
    _refuse(rc == -2, "too many interaction states")
    if rc == -3:
        return interaction_states(sets, nf)
    _refuse(rc != 0, f"shira_hgb_interaction_states failed ({rc})")
    S = n_st.value
    return st_off[: S + 1].copy(), st_feat[: st_off[S]].copy(), st_trans[: S * nf].reshape(S, nf).copy()


def interaction_states(sets: list, nf: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = set().union(*sets)
    feats = [list(np.fromiter(root, dtype=np.uint32, count=len(root)))]
    idx_of = [list(range(len(sets)))]
    key_of: dict[tuple, int] = {}
    trans: list[list[int]] = [[-1] * nf]
    q = 0
    while q < len(feats):
        _refuse(len(feats) > 100_000, "too many interaction states")
        for f in feats[q]:
            kids, allowed = [], set()
            for i in idx_of[q]:
                if f in sets[i]:
                    kids.append(i)
                    allowed.update(sets[i])
            key = tuple(kids)
            if key not in key_of:
                key_of[key] = len(feats)
                feats.append(list(np.fromiter(allowed, dtype=np.uint32, count=len(allowed))))
                idx_of.append(kids)
                trans.append([-1] * nf)
            trans[q][int(f)] = key_of[key]
        q += 1
    off = np.concatenate([[0], np.cumsum([len(f) for f in feats])]).astype(np.int32)
    flat = np.array([int(v) for f in feats for v in f], dtype=np.int32)
    return off, flat, np.array(trans, dtype=np.int32)


def ordinal_categories(col: np.ndarray, max_bins: int) -> np.ndarray:
    cats = np.unique(col[~np.isnan(col)])
    _refuse(cats.size > max_bins, f"categorical feature cardinality {cats.size} > max_bins {max_bins}")
    _refuse(cats.size == 0, "categorical feature without any non-missing value")
    return cats


def value_codes(x: np.ndarray, threads: int = 1) -> tuple[np.ndarray, list]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    n, nf = x.shape
    cap = min(n, 65535)
    codes = np.zeros((n, nf), dtype=np.uint16)
    uniq = np.zeros((nf, cap), dtype=np.float64)
    nu = np.zeros(nf, dtype=np.int64)
    rc = _lib().shira_hgb_value_codes(_ptr(x), n, nf, int(threads), _ptr(codes), _ptr(uniq), cap, _ptr(nu))
    _refuse(rc == -1, "too many distinct values for 16-bit codes (or bad arguments)")
    _refuse(rc != 0, f"shira_hgb_value_codes failed ({rc})")
    return codes, [uniq[j, : nu[j]].copy() for j in range(nf)]


def fold_thresholds(x: np.ndarray, rows: np.ndarray, cfg: dict, is_cat, threads: int = 1) -> list:
    n = rows.size
    if n > SUBSAMPLE:
        rows = rows[_bc().hgb_subsample_rows(n, SUBSAMPLE, int(cfg_seeds(cfg)[0]))]
    thr, _ = _bc().hgb_thresholds(x, cfg["max_bins"] + 1, rows=rows, nthread=threads)
    return [None if is_cat is not None and is_cat[j] else thr[j] for j in range(x.shape[1])]


def _bc():
    try:
        from . import binning_cpp as bc
    except ImportError:
        import binning_cpp as bc
    return bc


def _ref_weighted_percentile(a: np.ndarray, w: np.ndarray, rank: float) -> float:
    idx = np.argsort(a)
    cdf = np.cumsum(w[idx])
    adj = np.asarray(np.float64(rank) / 100 * cdf[-1])
    if adj == 0:
        adj = np.nextafter(adj, adj + 1)
    j = int(np.clip(np.searchsorted(cdf, adj), 0, a.size - 1))
    return float(a[idx[j]])


def weighted_percentile(a: np.ndarray, w: np.ndarray, rank: float) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    w = np.ascontiguousarray(w, dtype=np.float64)
    _refuse(a.ndim != 1 or w.shape != a.shape or a.size < 1, "weighted_percentile needs 1-d a, w of one size")
    out = np.zeros(1)
    rc = _lib().shira_hgb_weighted_percentile(_ptr(a), _ptr(w), a.size, float(np.float64(rank) / 100), _ptr(out))
    _refuse(rc != 0, f"shira_hgb_weighted_percentile failed ({rc})")
    return float(out[0])


def baseline(y: np.ndarray, sw: np.ndarray | None, cfg: dict, n_classes: int) -> np.ndarray:
    yc = np.ascontiguousarray(y, dtype=np.float64)
    w = None if sw is None else np.ascontiguousarray(sw, dtype=np.float64)
    _refuse(yc.ndim != 1 or yc.size < 1 or (w is not None and w.shape != yc.shape), "baseline needs 1-d y (and sw)")
    out = np.zeros(n_classes if n_classes > 2 else 1)
    rc = _lib().shira_hgb_baseline(_ptr(yc), _ptr(w), yc.size, int(cfg["loss_id"]), int(n_classes),
                                   float(cfg["quantile"]), float(cfg["qq"]), float(cfg["wq"]), _ptr(out))
    if rc == -3:
        raise ZeroDivisionError("Weights sum to zero, can't be normalized")
    if rc != 0:
        return _ref_baseline(y, sw, cfg, n_classes)
    return out


def _ref_baseline(y: np.ndarray, sw: np.ndarray | None, cfg: dict, n_classes: int) -> np.ndarray:
    loss = cfg["loss"]
    if loss == "absolute_error":
        return np.array([np.median(y) if sw is None else _ref_weighted_percentile(y, sw, 50)])
    if loss == "quantile":
        q = cfg["quantile"]
        return np.array([np.percentile(y, 100 * q) if sw is None else _ref_weighted_percentile(y, sw, 100 * q)])
    if n_classes > 2:
        from scipy.stats import gmean
        out = np.zeros(n_classes, dtype=y.dtype)
        eps = np.finfo(y.dtype).eps
        for k in range(n_classes):
            out[k] = np.average(y == k, weights=sw, axis=0)
            out[k] = np.clip(out[k], eps, 1 - eps)
        yp = out[None, :]
        return np.log(yp / gmean(yp, axis=1)[:, np.newaxis]).reshape(-1)
    y_pred = np.average(y, weights=sw, axis=0)
    eps = 10 * np.finfo(y_pred.dtype).eps
    if loss == "squared_error":
        return np.array([y_pred])
    if loss in ("poisson", "gamma"):
        return np.array([np.log(np.clip(y_pred, 0 + eps, None))])
    from scipy.special import logit
    return np.array([logit(np.clip(y_pred, 0 + eps, 1 - eps))])


def _ref_approximate_mode(class_counts: np.ndarray, n_draws: int, rng) -> np.ndarray:
    continuous = class_counts / class_counts.sum() * n_draws
    floored = np.floor(continuous)
    need_to_add = int(n_draws - floored.sum())
    if need_to_add > 0:
        remainder = continuous - floored
        values = np.sort(np.unique(remainder))[::-1]
        for value in values:
            (inds,) = np.where(remainder == value)
            add_now = min(len(inds), need_to_add)
            inds = rng.choice(inds, size=add_now, replace=False)
            floored[inds] += 1
            need_to_add -= add_now
            if need_to_add == 0:
                break
    return floored.astype(int)


def split_sizes(n: int, vf) -> tuple[int, int]:
    n_test = math.ceil(vf * n) if isinstance(vf, (float, np.floating)) else int(vf)
    _refuse(not 0 < n_test < n, f"validation_fraction {vf} leaves no training rows")
    return n - n_test, n_test


def train_val_split(y: np.ndarray, vf, seed, stratify: bool) -> tuple[np.ndarray, np.ndarray]:
    n = y.shape[0]
    n_train, n_test = split_sizes(n, vf)
    yc = np.ascontiguousarray(y, dtype=np.float64) if stratify else None
    if stratify and not bool(np.isfinite(yc).all()):
        return _ref_train_val_split(y, vf, seed, stratify)
    tr = np.empty(n_train, dtype=np.int64)
    te = np.empty(n_test, dtype=np.int64)
    rc = _lib().shira_hgb_val_split(_ptr(yc), n, n_train, n_test, int(seed), int(bool(stratify)), _ptr(tr), _ptr(te))
    _refuse(rc == -3, "stratified split impossible")
    _refuse(rc != 0, f"shira_hgb_val_split failed ({rc})")
    return tr, te


def _ref_train_val_split(y: np.ndarray, vf, seed, stratify: bool) -> tuple[np.ndarray, np.ndarray]:
    n = y.shape[0]
    n_train, n_test = split_sizes(n, vf)
    rng = np.random.RandomState(seed)
    if not stratify:
        perm = rng.permutation(n)
        return perm[n_test:n_test + n_train], perm[:n_test]
    classes, y_idx, counts = np.unique(y, return_inverse=True, return_counts=True)
    _refuse(counts.min() < 2 or n_train < classes.size or n_test < classes.size, "stratified split impossible")
    class_indices = np.split(np.argsort(y_idx, kind="mergesort"), np.cumsum(counts)[:-1])
    n_i = _ref_approximate_mode(counts, n_train, rng)
    t_i = _ref_approximate_mode(counts - n_i, n_test, rng)
    train, test = [], []
    for i in range(classes.size):
        perm = rng.permutation(counts[i])
        pi = class_indices[i].take(perm, mode="clip")
        train.extend(pi[: n_i[i]])
        test.extend(pi[n_i[i]: n_i[i] + t_i[i]])
    return rng.permutation(train), rng.permutation(test)


def class_weights(y: np.ndarray, cw) -> np.ndarray:
    yc = np.ascontiguousarray(y, dtype=np.float64)
    bal = isinstance(cw, str) and cw == "balanced"
    try:
        keys = None if bal else np.ascontiguousarray([float(c) for c in cw.keys()], dtype=np.float64)
        vals = None if bal else np.ascontiguousarray([float(v) for v in cw.values()], dtype=np.float64)
    except (TypeError, ValueError, AttributeError):
        return _ref_class_weights(y, cw)
    if yc.ndim != 1 or yc.size < 1 or not bool(np.isfinite(yc).all()):
        return _ref_class_weights(y, cw)
    out = np.empty(yc.size, dtype=np.float64)
    bad = ctypes.c_int32(-1)
    rc = _lib().shira_hgb_class_weights(_ptr(yc), yc.size, int(bal), _ptr(keys), _ptr(vals),
                                        0 if bal else keys.size, _ptr(out), ctypes.byref(bad))
    if rc == -3:
        _refuse(True, f"class_weight label {list(cw.keys())[bad.value]} not in y")
    _refuse(rc != 0, f"shira_hgb_class_weights failed ({rc})")
    return out


def _ref_class_weights(y: np.ndarray, cw) -> np.ndarray:
    classes = np.unique(y)
    if cw == "balanced":
        y_ind = np.searchsorted(classes, y)
        recip = len(y) / (len(classes) * np.bincount(y_ind).astype(np.float64))
        weight = recip[np.searchsorted(classes, classes)]
    else:
        weight = np.ones(classes.shape[0], dtype=np.float64)
        for c, wv in cw.items():
            i = np.searchsorted(classes, c)
            _refuse(i >= classes.size or classes[i] != c, f"class_weight label {c} not in y")
            weight[i] = wv
    expanded = weight[np.searchsorted(classes, y)]
    return np.prod(expanded[None, :], axis=0, dtype=np.float64)


def _classes(y: np.ndarray, folds, objective: str) -> tuple[str, int]:
    if objective == "regression":
        return "regression", 1
    yc = np.ascontiguousarray(y, dtype=np.float64)
    lo = np.ascontiguousarray([f[0] for f in folds], dtype=np.int64)
    hi = np.ascontiguousarray([f[1] for f in folds], dtype=np.int64)
    k = _lib().shira_hgb_label_counts(_ptr(yc), yc.size, _ptr(lo), _ptr(hi), lo.size, int(min(yc.size, 2**31 - 1)))
    _refuse(k == -1, "bad folds")
    _refuse(k in (-3, -5), "labels must be 0 .. K-1")
    _refuse(k == -4, "every fold must hold every class")
    return _objective_of(objective, k)


def _objective_of(objective: str, k: int) -> tuple[str, int]:
    _refuse(k < 2, "need two or more classes")
    if objective == "classifier":
        objective = "binary" if k == 2 else "multiclass"
    _refuse((objective == "binary") != (k == 2), f"{objective} with {k} classes")
    return objective, (1 if k == 2 else k)


def _fold_weights(y, sw, folds, cfg, objective) -> np.ndarray | None:
    if sw is None and cfg["class_weight"] is None:
        return None
    parts = []
    for lo, hi in folds:
        w = None if sw is None else sw[lo:hi]
        if cfg["class_weight"] is not None and objective != "regression":
            cwv = class_weights(y[lo:hi], cfg["class_weight"])
            w = cwv if w is None else w * cwv
        parts.append(np.asarray(w, dtype=np.float64))
    return np.ascontiguousarray(np.concatenate(parts))


def _fold_split(y_fold, cfg, seed, stratify) -> tuple[np.ndarray, np.ndarray | None]:
    n = y_fold.shape[0]
    es = cfg["early_stopping"]
    do_es = (n > 10_000) if es == "auto" else bool(es)
    if not do_es or cfg["validation_fraction"] is None:
        return np.arange(n), None
    return train_val_split(y_fold, cfg["validation_fraction"], seed, stratify)


def _fold_tables(x, codes, uniq, x_fold_cats, tr_rows, cfg, is_cat, threads, thr=None):
    nf, c_max = codes.shape[1], max(u.size for u in uniq)
    lut = np.zeros((nf, c_max), dtype=np.uint8)
    nbnm = np.zeros(nf, dtype=np.int32)
    if thr is None:
        thr = fold_thresholds(x, tr_rows, cfg, is_cat, threads)
    assert len(thr) == nf
    tabs = []
    for j in range(nf):
        if is_cat is not None and is_cat[j]:
            cats = x_fold_cats[j]
            thr[j] = np.arange(cats.size, dtype=np.float64)
            nbnm[j] = cats.size
            tabs.append(np.asarray(cats, dtype=np.float64))
        else:
            nbnm[j] = thr[j].shape[0] + 1
            tabs.append(np.asarray(thr[j], dtype=np.float64))
    u_off = np.zeros(nf + 1, dtype=np.int64)
    u_off[1:] = np.cumsum([u.size for u in uniq])
    t_off = np.zeros(nf + 1, dtype=np.int64)
    t_off[1:] = np.cumsum([t.size for t in tabs])
    uf = np.ascontiguousarray(np.concatenate(list(uniq) + [np.zeros(1)]), dtype=np.float64)
    tf = np.ascontiguousarray(np.concatenate(tabs + [np.zeros(1)]), dtype=np.float64)
    ic = None if is_cat is None else np.ascontiguousarray(is_cat, dtype=np.uint8)
    rc = _lib().shira_hgb_fold_lut(_ptr(uf), _ptr(u_off), _ptr(tf), _ptr(t_off), _ptr(ic), _ptr(nbnm), nf, c_max,
                                   _ptr(lut))
    _refuse(rc != 0, f"shira_hgb_fold_lut failed ({rc})")
    return lut, nbnm, thr


def prepare(x: np.ndarray, y: np.ndarray, folds, params: dict, objective: str,
            sample_weight: np.ndarray | None = None, threads: int = 1, direct: bool = True,
            seed_pair=None) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _refuse(x.ndim != 2 or y.shape != (x.shape[0],), "x must be (n, F) and y (n,)")
    _refuse(sample_weight is not None and np.shape(sample_weight) != y.shape, "sample_weight must be (n,)")
    sw = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    pair = check_seed_pair(seed_pair)
    cfg = check_params(params if pair is None else {**params, "random_state": 0}, objective)
    if pair is not None:
        cfg["random_state"], cfg["seed_pair"] = None, pair
    objective, ntree = _classes(y, folds, objective)
    cfg["objective"], cfg["ntree"] = objective, ntree
    cfg["loss_id"] = LOSS_IDS[cfg["loss"] if objective == "regression" else objective]
    _refuse(cfg["loss"] == "poisson" and bool((y < 0).any()), "poisson needs y >= 0")
    _refuse(cfg["loss"] == "gamma" and bool((y <= 0).any()), "gamma needs y > 0")
    nf = x.shape[1]
    cat = categorical_mask(cfg["categorical_features"], nf)
    perm = np.arange(nf) if cat is None else np.concatenate([np.flatnonzero(cat), np.flatnonzero(~cat)])
    x = np.ascontiguousarray(x) if cat is None else np.ascontiguousarray(x[:, perm])
    is_cat = None if cat is None else cat[perm].astype(np.uint8)
    mono = monotonic(cfg["monotonic_cst"], nf, objective)
    mono = None if mono is None else np.ascontiguousarray(mono[perm])
    _refuse(mono is not None and is_cat is not None and bool(((mono != 0) & (is_cat != 0)).any()),
            "categorical features cannot have monotonic constraints")
    cst = cfg["interaction_cst"]
    if cst is not None and not isinstance(cst, str):
        cst = [g if type(g) in (list, tuple, set, frozenset, range) else list(g) for g in cst]
    groups = interaction_groups(cst, nf)
    wts = _fold_weights(y, sw, folds, cfg, objective)
    seed = cfg_seeds(cfg)[0]
    states = None if groups is None else interaction_states_cpp(cst, nf, groups)
    if direct and is_cat is None:
        per = _prepare_folds_direct(x, y, wts, folds, cfg, seed, threads)
        return {"cfg": cfg, "codes": None, "uniq": None, "y": y, "sw": wts, "folds": [(int(a), int(b)) for a, b in folds],
                "perm": perm, "is_cat": is_cat, "mono": mono, "groups": groups, "states": states, **per}
    thr_cache = None
    try:
        codes, uniq = value_codes(x, threads)
    except ValueError as err:
        _refuse("failed (-2)" not in str(err) or is_cat is not None, str(err).removeprefix("hgb_cpp: "))
        codes, uniq, thr_cache = wide_codes(x, y, folds, cfg, threads, keep_thresholds=True)
    per = _prepare_folds(x, y, wts, folds, cfg, codes, uniq, is_cat, seed, threads, thr_cache)
    return {"cfg": cfg, "codes": codes, "uniq": uniq, "y": y, "sw": wts, "folds": [(int(a), int(b)) for a, b in folds],
            "perm": perm, "is_cat": is_cat, "mono": mono, "groups": groups, "states": states, "fold_bins": None,
            **per}


def _union_codes(x: np.ndarray, unions: list, threads: int) -> np.ndarray:
    from shira_cpp import outputs_cpp as oc

    n, nf = x.shape
    if not all(bool(np.isfinite(u).all()) for u in unions):
        out = np.zeros((n, nf), dtype=np.uint16)
        for j, u in enumerate(unions):
            col = x[:, j]
            nan = np.isnan(col)
            c = np.searchsorted(u, np.where(nan, 0.0, col), side="left") + 1
            out[:, j] = np.where(nan, 0, c).astype(np.uint16)
        return out
    off = np.zeros(nf + 1, dtype=np.int32)
    off[1:] = np.cumsum([u.size for u in unions])
    thr = np.ascontiguousarray(np.concatenate(unions) if off[-1] else np.zeros(1), dtype=np.float64)
    xc = np.ascontiguousarray(x, dtype=np.float64)
    out = np.empty((n, nf), dtype=np.uint16)
    assert xc.shape == (n, nf) and int(off[-1]) <= thr.size
    oc._check(oc._lib().shira_bin_raw_f64(oc._ptr(xc), n, nf, oc._ptr(off), oc._ptr(thr), 1,
                                          max(1, min(int(threads), oc.MAX_THREADS)), oc._ptr(out)),
              "shira_bin_raw_f64")
    return out


def wide_codes(x: np.ndarray, y: np.ndarray, folds, cfg: dict, threads: int, keep_thresholds: bool = False):
    n, nf = x.shape
    seed = cfg_seeds(cfg)[0]
    thr_folds = []
    for lo, hi in folds:
        tr, _ = _fold_split(y[lo:hi], cfg, seed, cfg["objective"] != "regression")
        thr_folds.append(fold_thresholds(x, lo + tr, cfg, None, threads))
    unions, uniq = [], []
    for j in range(nf):
        u = np.unique(np.concatenate([np.asarray(t[j], dtype=np.float64) for t in thr_folds]))
        _refuse(u.size + 2 > 65535, f"feature {j}: union of fold thresholds too large")
        unions.append(u)
    codes = _union_codes(x, unions, threads)
    for j, u in enumerate(unions):
        top = np.nextafter(u[-1], np.inf) if u.size else 0.0
        reps = np.concatenate([u, [top]])
        nan = codes[:, j] == 0
        if nan.any():
            reps = np.concatenate([reps, [np.nan]])
        codes[:, j] = np.where(nan, np.uint16(u.size + 1), codes[:, j] - np.uint16(1))
        assert int(codes[:, j].max()) < reps.size
        uniq.append(reps)
    assert len(uniq) == nf and codes.shape == (n, nf)
    return (codes, uniq, thr_folds) if keep_thresholds else (codes, uniq)


def _has_missing(codes, uniq, lo: int, hi: int, tr: np.ndarray) -> np.ndarray:
    plain = tr.size == hi - lo and bool((tr == np.arange(tr.size)).all())
    out = np.zeros(codes.shape[1], dtype=np.uint8)
    for j, u in enumerate(uniq):
        if u.size and np.isnan(u[-1]):
            col = codes[lo:hi, j] if plain else codes[lo + tr, j]
            out[j] = bool((col == u.size - 1).any())
    return out


def _prepare_folds(x, y, wts, folds, cfg, codes, uniq, is_cat, seed, threads=1, thr_cache=None) -> dict:
    nf, k_n = x.shape[1], len(folds)
    c_max = max(u.size for u in uniq)
    lut = np.zeros((k_n, nf, c_max), dtype=np.uint8)
    nbnm = np.zeros((k_n, nf), dtype=np.int32)
    miss = np.zeros((k_n, nf), dtype=np.uint8)
    bases = np.zeros((k_n, cfg["ntree"]))
    thr_all, tr_all, va_all = [], [], []
    w_off = 0
    for k, (lo, hi) in enumerate(folds):
        yk = y[lo:hi]
        wk = None if wts is None else wts[w_off: w_off + hi - lo]
        w_off += hi - lo
        tr, va = _fold_split(yk, cfg, seed, cfg["objective"] != "regression")
        cats = None
        if is_cat is not None:
            cats = [ordinal_categories(x[lo:hi, j], cfg["max_bins"]) if is_cat[j] else None for j in range(nf)]
        thr_k = None if thr_cache is None else list(thr_cache[k])
        lut[k], nbnm[k], thr = _fold_tables(x, codes, uniq, cats, lo + tr, cfg, is_cat, threads, thr_k)
        miss[k] = _has_missing(codes, uniq, lo, hi, tr)
        bases[k] = baseline(yk[tr], None if wk is None else wk[tr], cfg, cfg["ntree"] if cfg["ntree"] > 1 else 2)
        thr_all.append(thr)
        tr_all.append(tr.astype(np.int64))
        va_all.append(None if va is None else va.astype(np.int64))
    return {"lut": lut, "nbnm": nbnm, "miss": miss, "thresholds": thr_all, "base": bases, "train_rows": tr_all,
            "val_rows": va_all}


def _fold_rows_cpp(x, lo: int, hi: int, cfg, seed, tr_given, n_test: int, threads: int):
    n, nf = x.shape
    span, nb = hi - lo, cfg["max_bins"] + 1
    tr_out = np.empty(span - n_test, dtype=np.int64) if n_test > 0 else None
    va_out = np.empty(n_test, dtype=np.int64) if n_test > 0 else None
    tg = None if tr_given is None else np.ascontiguousarray(tr_given, dtype=np.int64)
    ptrs = np.zeros(nf + 1, dtype=np.int32)
    flat = np.zeros(max(1, nf * (nb - 2)), dtype=np.float64)
    nbm = np.zeros(nf, dtype=np.uint32)
    miss = np.zeros(nf, dtype=np.uint8)
    rc = _lib().shira_hgb_prepare_fold(_ptr(x), n, nf, lo, hi, nb, SUBSAMPLE, int(seed), n_test, _ptr(tg),
                                       0 if tg is None else tg.size, max(1, min(int(threads), MAX_THREADS)),
                                       _ptr(tr_out), _ptr(va_out), _ptr(ptrs), _ptr(flat), _ptr(nbm), _ptr(miss))
    _refuse(rc != 0, f"shira_hgb_prepare_fold failed ({rc})")
    thr = [flat[ptrs[j]: ptrs[j + 1]].copy() for j in range(nf)]
    if n_test > 0:
        tr, va = tr_out, va_out
    else:
        tr, va = (np.arange(span, dtype=np.int64) if tg is None else tg), None
    return tr, va, thr, nbm.astype(np.int32), miss


def _prepare_folds_direct(x, y, wts, folds, cfg, seed, threads=1) -> dict:
    nf, k_n = x.shape[1], len(folds)
    nbnm = np.zeros((k_n, nf), dtype=np.int32)
    miss = np.zeros((k_n, nf), dtype=np.uint8)
    bases = np.zeros((k_n, cfg["ntree"]))
    thr_all, tr_all, va_all = [], [], []
    stratify = cfg["objective"] != "regression"
    es = cfg["early_stopping"]
    w_off = 0
    plain = []
    for k, (lo, hi) in enumerate(folds):
        yk = y[lo:hi]
        wk = None if wts is None else wts[w_off: w_off + hi - lo]
        w_off += hi - lo
        do_es = (hi - lo > 10_000) if es == "auto" else bool(es)
        split = do_es and cfg["validation_fraction"] is not None
        tr_given, va_given, n_test = None, None, 0
        if split and stratify:
            tr_given, va_given = _fold_split(yk, cfg, seed, True)
        elif split:
            n_test = split_sizes(hi - lo, cfg["validation_fraction"])[1]
        tr, va, thr, nbnm[k], miss[k] = _fold_rows_cpp(x, lo, hi, cfg, seed, tr_given, n_test, threads)
        plain.append(tr_given is None and n_test == 0)
        if va_given is not None:
            va = va_given.astype(np.int64)
        bases[k] = baseline(yk[tr], None if wk is None else wk[tr], cfg, cfg["ntree"] if cfg["ntree"] > 1 else 2)
        thr_all.append(thr)
        tr_all.append(tr)
        va_all.append(va)
    off = np.zeros(k_n * nf + 1, dtype=np.int64)
    off[1:] = np.cumsum([t.size for th in thr_all for t in th])
    flat = np.ascontiguousarray(np.concatenate([t for th in thr_all for t in th] + [np.zeros(1)]), dtype=np.float64)
    assert off[-1] == int((nbnm - 1).sum())
    return {"lut": None, "nbnm": nbnm, "miss": miss, "thresholds": thr_all, "base": bases, "train_rows": tr_all,
            "val_rows": va_all, "fold_bins": None, "x_raw": x, "thr_flat": flat, "thr_off": off, "plain_rows": plain}


def _row_tables(prep: dict):
    tr, va, folds = prep["train_rows"], prep["val_rows"], prep["folds"]
    plain = all(v is None for v in va) and all(t.size == hi - lo and (t == np.arange(t.size)).all()
                                               for t, (lo, hi) in zip(tr, folds))
    if plain:
        return None, None, None, None
    tr_off = np.concatenate([[0], np.cumsum([t.size for t in tr])]).astype(np.int64)
    va_l = [np.zeros(0, np.int64) if v is None else v for v in va]
    va_off = np.concatenate([[0], np.cumsum([v.size for v in va_l])]).astype(np.int64)
    return tr_off, np.ascontiguousarray(np.concatenate(tr)), va_off, np.ascontiguousarray(np.concatenate(va_l))


def _rec(prep: dict, threads: int, n_states: int, use_val: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = prep["cfg"]
    n, nf = prep["y"].shape[0], len(prep["thresholds"][0])
    es = cfg["early_stopping"]
    early = int(es is True or (es == "auto" and any(hi - lo > 10_000 for lo, hi in prep["folds"])))
    if es == "auto":
        _refuse(len({hi - lo > 10_000 for lo, hi in prep["folds"]}) > 1,
                "early_stopping='auto' with folds on both sides of 10,000 rows: run them in separate calls")
    c_max = 1 if prep.get("lut") is None else prep["lut"].shape[2]
    rec_i = np.array([cfg["max_leaves"], cfg["max_depth"], cfg["min_leaf"], cfg["loss_id"], cfg["max_bins"] + 1,
                      cfg["ntree"], cfg["rounds"], cfg["n_iter_no_change"], early, use_val, n, nf,
                      c_max, len(prep["folds"]), threads, n_states], dtype=np.int64)
    rec_d = np.array([cfg["l2"], cfg["lr"], cfg["min_hess"], cfg["min_gain"], cfg["quantile"], cfg["qq"], cfg["wq"],
                      cfg["tol"], cfg["max_features"]], dtype=np.float64)
    return rec_i, rec_d


OUT_KEYS = ("tr", "va", "niter", "tsc", "vsc", "nodes", "value", "gain", "nnodes", "bits", "count")


def pack(prep: dict, threads: int, trees_out: bool) -> dict:
    cfg, folds = prep["cfg"], prep["folds"]
    _refuse(not 1 <= threads <= MAX_THREADS, "threads in [1, 64]")
    lo = np.ascontiguousarray([f[0] for f in folds], dtype=np.int64)
    hi = np.ascontiguousarray([f[1] for f in folds], dtype=np.int64)
    tr_off, tr_idx, va_off, va_idx = _row_tables(prep)
    use_val = int(va_off is not None and va_off[-1] > 0)
    st = prep["states"]
    rec_i, rec_d = _rec(prep, threads, 0 if st is None else st[0].size - 1, use_val)
    k_n, rounds, nn, T = len(folds), cfg["rounds"], 2 * cfg["max_leaves"] - 1, cfg["ntree"]
    m_tot = int((hi - lo).sum()) if tr_off is None else int(tr_off[-1])
    mv_tot = 0 if va_off is None else int(va_off[-1])
    o = {"tr": np.zeros(m_tot * T), "va": np.zeros(max(mv_tot, 1) * T), "niter": np.zeros(k_n, np.int32),
         "tsc": np.zeros((k_n, rounds + 1)), "vsc": np.zeros((k_n, rounds + 1)),
         "nodes": np.zeros((k_n, rounds, T, nn, 5), np.int32) if trees_out else None,
         "value": np.zeros((k_n, rounds, T, nn)) if trees_out else None,
         "gain": np.zeros((k_n, rounds, T, nn)) if trees_out else None,
         "nnodes": np.zeros((k_n, rounds, T), np.int32),
         "bits": np.zeros((k_n, rounds, T, nn, 8), np.uint32) if trees_out and prep["is_cat"] is not None else None,
         "count": np.zeros((k_n, rounds, T, nn), np.uint32) if trees_out else None}
    rng = pcg_words(cfg_seeds(cfg)[1]) if cfg["max_features"] < 1.0 else None
    fused = prep.get("thr_flat") is not None
    direct = fused or prep.get("fold_bins") is not None
    keep = [None if direct else np.ascontiguousarray(prep["codes"]), prep["y"], prep["sw"], lo, hi, tr_off, tr_idx,
            va_off, va_idx, None if direct else np.ascontiguousarray(prep["lut"]), np.ascontiguousarray(prep["nbnm"]),
            np.ascontiguousarray(prep["miss"]), np.ascontiguousarray(prep["base"], dtype=np.float64), prep["is_cat"],
            prep["mono"], None if st is None else st[0], None if st is None else st[1],
            None if st is None else np.ascontiguousarray(st[2]), rng, prep.get("fold_bins"),
            prep["x_raw"] if fused else None, prep["thr_flat"] if fused else None, prep["thr_off"] if fused else None]
    assert len(keep) == N_IN
    return {"keep": keep, "o": o, "rec_i": rec_i, "rec_d": rec_d, "tr_off": tr_off, "va_off": va_off,
            "use_val": use_val, "T": T}


def train(prep: dict, threads: int = 6, trees_out: bool = False) -> dict:
    pk = pack(prep, threads, trees_out)
    o, T, k_n = pk["o"], pk["T"], len(prep["folds"])
    ins = (ctypes.c_void_p * N_IN)(*[_ptr(a) for a in pk["keep"]])
    outs = (ctypes.c_void_p * N_OUT)(*[_ptr(o[k]) for k in OUT_KEYS])
    t0 = time.perf_counter()
    rc = _lib().shira_train_hgb_ex(ins, outs, _ptr(pk["rec_i"]), _ptr(pk["rec_d"]))
    secs = time.perf_counter() - t0
    if rc != 0:
        why = {-1: "rejected its arguments", -2: "a worker failed an internal check (message on stderr)",
               -3: "failed outside the workers (allocation?)"}.get(rc, "failed")
        raise ValueError(f"shira_train_hgb_ex {why} ({rc})")
    trees = {k: (None if o[k] is None else (o[k][:, :, 0] if T == 1 else o[k]))
             for k in ("nodes", "value", "gain", "nnodes", "bits", "count")}
    use_val = pk["use_val"]
    return {"preds": _assemble(prep, o, pk["tr_off"], pk["va_off"]), "seconds": secs, "n_iter": o["niter"],
            "train_score": [o["tsc"][k, : o["niter"][k] + 1] for k in range(k_n)],
            "validation_score": [o["vsc"][k, : o["niter"][k] + 1] if use_val else np.zeros(0) for k in range(k_n)],
            **trees}


def _assemble(prep: dict, o: dict, tr_off, va_off) -> list:
    T, out = prep["cfg"]["ntree"], []
    pos_t = pos_v = 0
    plain = prep.get("plain_rows")
    for k, (lo, hi) in enumerate(prep["folds"]):
        tr = prep["train_rows"][k]
        if plain is not None and plain[k]:
            seg = o["tr"][pos_t * T: (pos_t + tr.size) * T]
            pos_t += tr.size
            out.append(seg.copy() if T == 1 else seg.reshape(-1, T).copy())
            continue
        full = np.zeros((hi - lo, T))
        full[tr] = o["tr"][pos_t * T: (pos_t + tr.size) * T].reshape(-1, T)
        pos_t += tr.size
        va = prep["val_rows"][k]
        if va is not None:
            full[va] = o["va"][pos_v * T: (pos_v + va.size) * T].reshape(-1, T)
            pos_v += va.size
        out.append(full[:, 0].copy() if T == 1 else full)
    return out


def pairwise_sum(a: np.ndarray) -> float:
    a = np.ascontiguousarray(a, dtype=np.float64)
    return float(_lib().shira_hgb_pairwise_sum(_ptr(a), a.size))


def fold_bins(prep: dict, k: int) -> np.ndarray:
    lo, hi = prep["folds"][k]
    if prep.get("thr_flat") is not None:
        b = _bc().hgb_bin(prep["x_raw"][lo:hi], prep["thresholds"][k], prep["cfg"]["max_bins"] + 1).astype(np.int64)
        return np.where(b == prep["cfg"]["max_bins"], prep["nbnm"][k][None, :].astype(np.int64), b)
    if prep.get("fold_bins") is not None:
        nf = prep["nbnm"].shape[1]
        off = sum((b - a) for a, b in prep["folds"][:k]) * nf
        b = prep["fold_bins"][off: off + (hi - lo) * nf].reshape(nf, hi - lo).T.astype(np.int64)
        return np.where(b == prep["cfg"]["max_bins"], prep["nbnm"][k][None, :].astype(np.int64), b)
    c = prep["codes"][lo:hi].astype(np.int64)
    return np.stack([prep["lut"][k, j][c[:, j]] for j in range(c.shape[1])], axis=1)


def tree_output(res: dict, k: int, r: int, bins: np.ndarray, t: int = 0, nbnm: np.ndarray | None = None) -> np.ndarray:
    ix = (k, r) if res["nnodes"].ndim == 2 else (k, r, t)
    nodes, value = res["nodes"][ix], res["value"][ix]
    bits = None if res.get("bits") is None else res["bits"][ix]
    cur = np.zeros(bins.shape[0], dtype=np.int64)
    rows = np.arange(bins.shape[0])
    for _ in range(int(res["nnodes"][ix])):
        act = nodes[cur, 0] >= 0
        if not act.any():
            break
        nd = cur[act]
        f = nodes[nd, 0]
        b = bins[rows[act], f]
        flags = nodes[nd, 4]
        go = b <= nodes[nd, 1]
        if bits is not None:
            catl = ((bits[nd, b >> 5] >> (b & 31).astype(np.uint32)) & 1).astype(bool)
            go = np.where((flags & 2) != 0, catl, go)
        if nbnm is not None:
            go = np.where(b == nbnm[f], (flags & 1) != 0, go)
        cur[act] = np.where(go, nodes[nd, 2], nodes[nd, 3])
    assert (nodes[cur, 0] < 0).all()
    return value[cur]
