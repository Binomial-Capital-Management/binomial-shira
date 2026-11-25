from __future__ import annotations

import ctypes

import numpy as np

from .binning_cpp import _check, _lib as _blib, _ptr

_P, _I32, _I64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64
_REG: list[int] = []


def _lib() -> ctypes.CDLL:
    lib = _blib()
    if not _REG:
        lib.shira_tree_node_stats.argtypes = [_P, _I64, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _I32, _P, _P, _P]
        lib.shira_tree_node_stats.restype = ctypes.c_int
        lib.shira_sk_leaf_ids.argtypes = [_P, _I64, _I32, _P, _P, _P, _P, _P, _P, _I32, _I32, _I32, _P, _P]
        lib.shira_sk_leaf_ids.restype = ctypes.c_int
        _REG.append(1)
    return lib


def _arrays(trees: list) -> tuple:
    off = np.concatenate([[0], np.cumsum([t.node_count for t in trees])]).astype(np.int64)
    feat = np.ascontiguousarray(np.concatenate([t.feature for t in trees]), dtype=np.int32)
    thr = np.ascontiguousarray(np.concatenate([t.threshold for t in trees]), dtype=np.float64)
    left = np.ascontiguousarray(np.concatenate([t.children_left for t in trees]), dtype=np.int32)
    right = np.ascontiguousarray(np.concatenate([t.children_right for t in trees]), dtype=np.int32)
    mgl = np.ascontiguousarray(np.concatenate([t.missing_go_to_left for t in trees]), dtype=np.uint8)
    return off, feat, thr, left, right, mgl


def leaf_ids(trees: list, x: np.ndarray, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert trees and x.ndim == 2 and x.dtype == np.float32 and x.flags.c_contiguous and 1 <= nthread <= 64
    off, feat, thr, left, right, mgl = _arrays(trees)
    most = max(int(t.n_leaves) for t in trees)
    assert most <= 65536, f"{most} leaves in one tree: over the 2-byte leaf id"
    dt = np.uint8 if most <= 256 else np.uint16
    out = np.empty((len(trees), x.shape[0]), dtype=dt)
    nl = np.zeros(len(trees), dtype=np.int32)
    _check(_lib().shira_sk_leaf_ids(_ptr(x), x.shape[0], x.shape[1], _ptr(feat), _ptr(thr), _ptr(left), _ptr(right), _ptr(mgl),
                                    _ptr(off), len(trees), int(nthread), np.dtype(dt).itemsize, _ptr(out), _ptr(nl)),
           "shira_sk_leaf_ids")
    return out, nl


def node_stats(trees: list, x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None,
               nthread: int = 1) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    assert trees and x.ndim == 2 and x.dtype == np.float32 and x.flags.c_contiguous and 1 <= nthread <= 64
    n = x.shape[0]
    yv = np.ascontiguousarray(y, dtype=np.float64)
    wv = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    assert yv.shape == (n,) and (wv is None or wv.shape == (n,))
    off, feat, thr, left, right, mgl = _arrays(trees)
    total = int(off[-1])
    ow, owy, owy2 = np.empty(total), np.empty(total), np.empty(total)
    _check(_lib().shira_tree_node_stats(_ptr(x), n, x.shape[1], _ptr(yv), _ptr(wv), _ptr(feat), _ptr(thr), _ptr(left),
                                        _ptr(right), _ptr(mgl), _ptr(off), len(trees), int(nthread), _ptr(ow), _ptr(owy),
                                        _ptr(owy2)), "shira_tree_node_stats")
    cut = off[1:-1]
    return np.split(ow, cut), np.split(owy, cut), np.split(owy2, cut)


_REG_SM: list[int] = []


def softmax_values(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, w: np.ndarray | None, base: np.ndarray, eta: float,
                   l2: float, nthread: int = 1) -> np.ndarray:
    lib = _blib()
    if not _REG_SM:
        lib.shira_hgb_softmax_values.argtypes = [_P, _I32, _I64, _I32, _I32, _P, _I32, _P, _P, _P, ctypes.c_double,
                                                 ctypes.c_double, _I32, _P]
        lib.shira_hgb_softmax_values.restype = ctypes.c_int
        _REG_SM.append(1)
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16) and 1 <= nthread <= 64
    T, n = ids.shape
    bv = np.ascontiguousarray(base, dtype=np.float64)
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    yv = np.ascontiguousarray(y, dtype=np.int32)
    wv = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    assert bv.ndim == 1 and nl.shape == (T,) and yv.shape == (n,) and (wv is None or wv.shape == (n,))
    lmax = int(nl.max())
    out = np.zeros((T, lmax), dtype=np.float64)
    _check(lib.shira_hgb_softmax_values(_ptr(ids), ids.itemsize, n, T, bv.size, _ptr(nl), lmax, _ptr(yv), _ptr(wv),
                                        _ptr(bv), float(eta), float(l2), int(nthread), _ptr(out)),
           "shira_hgb_softmax_values")
    return out


_REG_LS: list[int] = []


def line_search_values(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, w: np.ndarray | None, base: float, lr: float,
                       loss: str, quantile: float = 0.5, nthread: int = 1) -> np.ndarray:
    lib = _blib()
    if not _REG_LS:
        lib.shira_hgb_line_search_values.argtypes = [_P, _I32, _I64, _I32, _P, _I32, _P, _P, ctypes.c_double, ctypes.c_double,
                                                     _I32, ctypes.c_double, ctypes.c_double, _I32, _P]
        lib.shira_hgb_line_search_values.restype = ctypes.c_int
        _REG_LS.append(1)
    assert loss in ("absolute_error", "quantile") and 1 <= nthread <= 64
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    yv = np.ascontiguousarray(y, dtype=np.float64)
    wv = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    assert nl.shape == (T,) and yv.shape == (n,) and (wv is None or wv.shape == (n,))
    q = float(quantile) if loss == "quantile" else 0.5
    qq = float(np.true_divide(100 * q, 100)) if loss == "quantile" else 0.5
    wq = (100 * q) / 100 if loss == "quantile" else 50 / 100
    out = np.zeros((T, int(nl.max())), dtype=np.float64)
    rc = lib.shira_hgb_line_search_values(_ptr(ids), ids.itemsize, n, T, _ptr(nl), int(nl.max()), _ptr(yv), _ptr(wv),
                                          float(base), float(lr), 3 if loss == "absolute_error" else 4, qq, wq,
                                          int(nthread), _ptr(out))
    if rc != 0:
        raise RuntimeError(f"shira_hgb_line_search_values returned {rc}")
    return out


_REG_HU: list[int] = []


def huber_values(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, w: np.ndarray, base: float, lr: float, alpha: float,
                 nthread: int = 1) -> np.ndarray:
    lib = _blib()
    if not _REG_HU:
        lib.shira_gb_huber_values.argtypes = [_P, _I32, _I64, _I32, _P, _I32, _P, _P, ctypes.c_double, ctypes.c_double,
                                              ctypes.c_double, _I32, _P]
        lib.shira_gb_huber_values.restype = ctypes.c_int
        _REG_HU.append(1)
    assert 1 <= nthread <= 64 and 0.0 < alpha < 1.0
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    yv = np.ascontiguousarray(y, dtype=np.float64)
    wv = np.ascontiguousarray(w, dtype=np.float64)
    assert nl.shape == (T,) and yv.shape == (n,) and wv.shape == (n,)
    out = np.zeros((T, int(nl.max())), dtype=np.float64)
    rc = lib.shira_gb_huber_values(_ptr(ids), ids.itemsize, n, T, _ptr(nl), int(nl.max()), _ptr(yv), _ptr(wv), float(base),
                                   float(lr), float(alpha), int(nthread), _ptr(out))
    if rc != 0:
        raise RuntimeError(f"shira_gb_huber_values returned {rc}")
    return out


_REG_CAT: list[int] = []


def cat_oblivious_ids(x32: np.ndarray, doc: dict, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    lib = _blib()
    if not _REG_CAT:
        lib.shira_cat_oblivious_ids.argtypes = [_P, _I64, _I32, _P, _P, _P, _I32, _I32, _P, _P, _P, _I32, _P]
        lib.shira_cat_oblivious_ids.restype = ctypes.c_int
        _REG_CAT.append(1)
    x32 = np.ascontiguousarray(x32, dtype=np.float32)
    n, F = x32.shape
    ff = doc["features_info"]["float_features"]
    assert len(ff) == F and all(int(f["flat_feature_index"]) == i for i, f in enumerate(ff))
    bl = [np.asarray(f.get("borders") or [], dtype=np.float32) for f in ff]
    assert all(b.size <= 255 and (b.size < 2 or bool(np.all(np.diff(b) > 0))) for b in bl)
    boff = np.concatenate([[0], np.cumsum([b.size for b in bl])]).astype(np.int64)
    borders = np.concatenate(bl + [np.zeros(1, np.float32)]).astype(np.float32)
    nan_true = np.array([1 if f.get("nan_value_treatment") == "AsTrue" else 0 for f in ff], dtype=np.uint8)
    trees = doc["oblivious_trees"]
    T, maxd = len(trees), max(1, max(len(t["splits"] or []) for t in trees))
    depth = np.array([len(t["splits"] or []) for t in trees], dtype=np.int32)
    sfeat = np.zeros((T, maxd), dtype=np.int32)
    sbin = np.zeros((T, maxd), dtype=np.int32)
    for t, tree in enumerate(trees):
        for i, s in enumerate(tree["splits"] or []):
            assert s["split_type"] == "FloatFeature"
            f = int(s["float_feature_index"])
            k = int(np.searchsorted(bl[f], np.float32(s["border"])))
            assert k < bl[f].size and bl[f][k] == np.float32(s["border"])
            sfeat[t, i], sbin[t, i] = f, k
    out = np.empty((T, n), dtype=np.uint8)
    _check(lib.shira_cat_oblivious_ids(_ptr(x32), n, F, _ptr(borders), _ptr(boff), _ptr(nan_true), T, maxd, _ptr(depth),
                                       _ptr(sfeat), _ptr(sbin), int(nthread), _ptr(out)), "shira_cat_oblivious_ids")
    return out, (1 << depth).astype(np.int32)


_REG_NT: list[int] = []


def cat_logloss_newton(ids8: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, bias: float, eta: float,
                       lam: float, max_steps: int, nthread: int = 1, F0: np.ndarray | None = None,
                       force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lib = _blib()
    if not _REG_NT:
        for name in ("shira_cat_logloss_newton", "shira_cat_logloss_newton16"):
            fn = getattr(lib, name)
            fn.argtypes = [_P, _I64, _I32, _P, _P, _P, _P, ctypes.c_double, ctypes.c_double, ctypes.c_double, _I32, _I32, _I32,
                           _P, _P]
            fn.restype = ctypes.c_int
        _REG_NT.append(1)
    wide = np.asarray(ids8).dtype == np.uint16
    ids8 = np.ascontiguousarray(ids8, dtype=np.uint16 if wide else np.uint8)
    T, n = ids8.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float64)
    ww = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    f0 = None if F0 is None else np.ascontiguousarray(F0, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and (ww is None or ww.shape == (n,)) and (f0 is None or f0.shape == (n,))
    out = np.empty(int(nl.sum()), dtype=np.float64)
    steps = np.empty(T, dtype=np.int32)
    fn = lib.shira_cat_logloss_newton16 if wide else lib.shira_cat_logloss_newton
    _check(fn(_ptr(ids8), n, T, _ptr(nl), _ptr(yy), _ptr(ww), _ptr(f0), float(bias), float(eta), float(lam), int(max_steps),
              int(bool(force)), int(nthread), _ptr(out), _ptr(steps)), "shira_cat_logloss_newton" + ("16" if wide else ""))
    return out, steps


_REG_MC: list[int] = []


def cat_multiclass_values(ids8: np.ndarray, nl: np.ndarray, yk: np.ndarray, w: np.ndarray | None, K: int, eta: float,
                          lam: float, nthread: int = 1, F0: np.ndarray | None = None) -> np.ndarray:
    lib = _blib()
    if not _REG_MC:
        lib.shira_cat_multiclass_values.argtypes = [_P, _I64, _I32, _P, _P, _P, _I32, _P, ctypes.c_double, ctypes.c_double,
                                                    _I32, _P]
        lib.shira_cat_multiclass_values.restype = ctypes.c_int
        _REG_MC.append(1)
    ids8 = np.ascontiguousarray(ids8, dtype=np.uint8)
    T, n = ids8.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(yk, dtype=np.int32)
    ww = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    f0 = None if F0 is None else np.ascontiguousarray(F0, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and (ww is None or ww.shape == (n,)) and (f0 is None or f0.shape == (n, K))
    out = np.empty(int(nl.sum()) * int(K), dtype=np.float64)
    _check(lib.shira_cat_multiclass_values(_ptr(ids8), n, T, _ptr(nl), _ptr(yy), _ptr(ww), int(K), _ptr(f0), float(eta),
                                           float(lam), int(nthread), _ptr(out)), "shira_cat_multiclass_values")
    return out


_REG_Q: list[int] = []


def cat_quantile_values(ids8: np.ndarray, nl: np.ndarray, y: np.ndarray, bias: float, alpha: float, delta: float, eta: float,
                        nthread: int = 1) -> np.ndarray:
    lib = _blib()
    if not _REG_Q:
        lib.shira_cat_quantile_values.argtypes = [_P, _I64, _I32, _P, _P, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                                  ctypes.c_double, _I32, _P]
        lib.shira_cat_quantile_values.restype = ctypes.c_int
        _REG_Q.append(1)
    ids8 = np.ascontiguousarray(ids8, dtype=np.uint8)
    T, n = ids8.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and 0.0 < alpha < 1.0
    out = np.empty(int(nl.sum()), dtype=np.float64)
    _check(lib.shira_cat_quantile_values(_ptr(ids8), n, T, _ptr(nl), _ptr(yy), float(bias), float(alpha), float(delta), float(eta),
                                         int(nthread), _ptr(out)), "shira_cat_quantile_values")
    return out


_REG_TREE: list[int] = []


def _cat_tables(doc: dict, F: int) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    ff = doc["features_info"]["float_features"]
    assert len(ff) == F and all(int(f["flat_feature_index"]) == i for i, f in enumerate(ff))
    bl = [np.asarray(f.get("borders") or [], dtype=np.float32) for f in ff]
    assert all(b.size <= 255 and (b.size < 2 or bool(np.all(np.diff(b) > 0))) for b in bl)
    boff = np.concatenate([[0], np.cumsum([b.size for b in bl])]).astype(np.int64)
    borders = np.concatenate(bl + [np.zeros(1, np.float32)]).astype(np.float32)
    nan_true = np.array([1 if f.get("nan_value_treatment") == "AsTrue" else 0 for f in ff], dtype=np.uint8)
    return bl, boff, borders, nan_true


def _cat_split(bl: list, s: dict) -> tuple[int, int]:
    assert s["split_type"] == "FloatFeature"
    f = int(s["float_feature_index"])
    k = int(np.searchsorted(bl[f], np.float32(s["border"])))
    assert k < bl[f].size and bl[f][k] == np.float32(s["border"])
    return f, k


def _cat_lib16() -> ctypes.CDLL:
    lib = _blib()
    if not _REG_TREE:
        lib.shira_cat_oblivious_ids16.argtypes = [_P, _I64, _I32, _P, _P, _P, _I32, _I32, _P, _P, _P, _I32, _P]
        lib.shira_cat_oblivious_ids16.restype = ctypes.c_int
        lib.shira_cat_tree_ids.argtypes = [_P, _I64, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P, _P, _I32, _P]
        lib.shira_cat_tree_ids.restype = ctypes.c_int
        _REG_TREE.append(1)
    return lib


def cat_oblivious_ids16(x32: np.ndarray, doc: dict, nthread: int = 1, t0: int = 0, t1: int | None = None) -> tuple:
    x32 = np.ascontiguousarray(x32, dtype=np.float32)
    n, F = x32.shape
    bl, boff, borders, nan_true = _cat_tables(doc, F)
    trees = doc["oblivious_trees"][t0:t1]
    T, maxd = len(trees), max(1, max(len(t["splits"] or []) for t in trees))
    depth = np.array([len(t["splits"] or []) for t in trees], dtype=np.int32)
    sfeat = np.zeros((T, maxd), dtype=np.int32)
    sbin = np.zeros((T, maxd), dtype=np.int32)
    for t, tree in enumerate(trees):
        for i, s in enumerate(tree["splits"] or []):
            sfeat[t, i], sbin[t, i] = _cat_split(bl, s)
    out = np.empty((T, n), dtype=np.uint16)
    _check(_cat_lib16().shira_cat_oblivious_ids16(_ptr(x32), n, F, _ptr(borders), _ptr(boff), _ptr(nan_true), T, maxd,
                                                  _ptr(depth), _ptr(sfeat), _ptr(sbin), int(nthread), _ptr(out)),
           "shira_cat_oblivious_ids16")
    return out, (1 << depth).astype(np.int32)


def _cat_flatten(tree: dict, steps: list, to_leaf: list, bl: list, rows: list) -> int:
    nleaf = int(sum(1 for v in to_leaf if 0 <= int(v) < (1 << 31)))
    if "split" not in tree:
        assert len(steps) == 1 and int(to_leaf[0]) == 0
        return 1
    base = len(rows)
    stack = [(tree, 0, -1, 0)]
    for _ in range(1 << 18):
        if not stack:
            break
        node, i, parent, side = stack.pop()
        if "split" in node:
            f, k = _cat_split(bl, node["split"])
            rows.append([f, k, -1, -1])
            ref = len(rows) - 1 - base
            (dl, dr) = (int(steps[i][0]), int(steps[i][1]))
            for child, d, col in ((node["right"], dr, 3), (node["left"], dl, 2)):
                if d == 0:
                    assert "split" not in child
                    rows[base + ref][col] = -(int(to_leaf[i]) + 1)
                else:
                    stack.append((child, i + d, ref, col))
        else:
            assert tuple(int(v) for v in steps[i]) == (0, 0)
            ref = -(int(to_leaf[i]) + 1)
        if parent >= 0:
            rows[base + parent][side] = ref
    assert not stack and nleaf >= 2
    return nleaf


def cat_tree_ids(x32: np.ndarray, doc: dict, model: object, nthread: int = 1, t0: int = 0, t1: int | None = None) -> tuple:
    x32 = np.ascontiguousarray(x32, dtype=np.float32)
    n, F = x32.shape
    bl, boff, borders, nan_true = _cat_tables(doc, F)
    trees = doc["trees"][t0:t1]
    rows: list = []
    node_off, nl = [0], []
    for t, tree in enumerate(trees):
        ti = (t0 or 0) + t
        nl.append(_cat_flatten(tree, model._get_tree_step_nodes(ti), model._get_tree_node_to_leaf(ti), bl, rows))
        node_off.append(len(rows))
    tab = np.asarray(rows, dtype=np.int32).reshape(-1, 4)
    cols = [np.ascontiguousarray(tab[:, c]) for c in range(4)]
    T = len(trees)
    nlv, offs = np.asarray(nl, dtype=np.int32), np.asarray(node_off, dtype=np.int32)
    out = np.empty((T, n), dtype=np.uint16)
    _check(_cat_lib16().shira_cat_tree_ids(_ptr(x32), n, F, _ptr(borders), _ptr(boff), _ptr(nan_true), T, _ptr(offs),
                                           _ptr(cols[0]), _ptr(cols[1]), _ptr(cols[2]), _ptr(cols[3]), _ptr(nlv), int(nthread),
                                           _ptr(out)), "shira_cat_tree_ids")
    return out, nlv


_REG_OS: list[int] = []


def cat_logloss_onestep(ids8: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, bias: float, eta: float,
                        lam: float, nthread: int = 1, F0: np.ndarray | None = None) -> np.ndarray:
    lib = _blib()
    if not _REG_OS:
        for name in ("shira_cat_logloss_onestep", "shira_cat_logloss_onestep16"):
            fn = getattr(lib, name)
            fn.argtypes = [_P, _I64, _I32, _P, _P, _P, _P, ctypes.c_double, ctypes.c_double, ctypes.c_double, _I32, _P]
            fn.restype = ctypes.c_int
        _REG_OS.append(1)
    wide = np.asarray(ids8).dtype == np.uint16
    ids8 = np.ascontiguousarray(ids8, dtype=np.uint16 if wide else np.uint8)
    name = "shira_cat_logloss_onestep16" if wide else "shira_cat_logloss_onestep"
    T, n = ids8.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float64)
    ww = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    f0 = None if F0 is None else np.ascontiguousarray(F0, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and (ww is None or ww.shape == (n,)) and (f0 is None or f0.shape == (n,))
    out = np.empty(int(nl.sum()), dtype=np.float64)
    _check(getattr(lib, name)(_ptr(ids8), n, T, _ptr(nl), _ptr(yy), _ptr(ww), _ptr(f0), float(bias), float(eta), float(lam),
                              int(nthread), _ptr(out)), name)
    return out


REG_LOSS = {"Poisson": 1, "Tweedie": 2, "Huber": 3}
REG_LOSS.update({"Lq": 4, "Expectile": 5})
_REG_RO: list[int] = []


def cat_reg_onestep(loss: str, param: float, ids8: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None,
                    bias: float, eta: float, lam: float, nthread: int = 1, F0: np.ndarray | None = None) -> np.ndarray:
    lib = _blib()
    if not _REG_RO:
        for name in ("shira_cat_reg_onestep", "shira_cat_reg_onestep16"):
            fn = getattr(lib, name)
            fn.argtypes = [_I32, ctypes.c_double, _P, _I64, _I32, _P, _P, _P, _P, ctypes.c_double, ctypes.c_double, ctypes.c_double,
                           _I32, _P]
            fn.restype = ctypes.c_int
        _REG_RO.append(1)
    assert loss in REG_LOSS and np.isfinite(param)
    wide = np.asarray(ids8).dtype == np.uint16
    ids8 = np.ascontiguousarray(ids8, dtype=np.uint16 if wide else np.uint8)
    fname = "shira_cat_reg_onestep16" if wide else "shira_cat_reg_onestep"
    T, n = ids8.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float64)
    ww = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    f0 = None if F0 is None else np.ascontiguousarray(F0, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and (ww is None or ww.shape == (n,)) and (f0 is None or f0.shape == (n,))
    out = np.empty(int(nl.sum()), dtype=np.float64)
    _check(getattr(lib, fname)(REG_LOSS[loss], float(param), _ptr(ids8), n, T, _ptr(nl), _ptr(yy), _ptr(ww), _ptr(f0),
                               float(bias), float(eta), float(lam), int(nthread), _ptr(out)), fname)
    return out


_REG_PW: list[int] = []


def cat_poisson_walk(ids8: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, bias: float, eta: float,
                     lam: float, max_steps: int, nthread: int = 1, F0: np.ndarray | None = None,
                     force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lib = _blib()
    if not _REG_PW:
        for name in ("shira_cat_poisson_walk", "shira_cat_poisson_walk16"):
            fn = getattr(lib, name)
            fn.argtypes = [_P, _I64, _I32, _P, _P, _P, _P, ctypes.c_double, ctypes.c_double, ctypes.c_double, _I32, _I32, _I32,
                           _P, _P]
            fn.restype = ctypes.c_int
        _REG_PW.append(1)
    assert 1 <= int(max_steps) <= 64
    wide = np.asarray(ids8).dtype == np.uint16
    ids8 = np.ascontiguousarray(ids8, dtype=np.uint16 if wide else np.uint8)
    fname = "shira_cat_poisson_walk16" if wide else "shira_cat_poisson_walk"
    T, n = ids8.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float64)
    ww = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    f0 = None if F0 is None else np.ascontiguousarray(F0, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and (ww is None or ww.shape == (n,)) and (f0 is None or f0.shape == (n,))
    out = np.empty(int(nl.sum()), dtype=np.float64)
    taken = np.zeros(T, dtype=np.int32)
    _check(getattr(lib, fname)(_ptr(ids8), n, T, _ptr(nl), _ptr(yy), _ptr(ww), _ptr(f0), float(bias), float(eta), float(lam),
                               int(max_steps), int(bool(force)), int(nthread), _ptr(out), _ptr(taken)), fname)
    return out, taken


_REG_WK: list[int] = []
WALK_LOSS = {"Expectile": 5, "LogCosh": 6}


def cat_reg_walk(loss: str, param: float, ids8: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, bias: float,
                 eta: float, lam: float, max_steps: int, nthread: int = 1, F0: np.ndarray | None = None,
                 force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    lib = _blib()
    if not _REG_WK:
        for name in ("shira_cat_reg_walk", "shira_cat_reg_walk16"):
            fn = getattr(lib, name)
            fn.argtypes = [_I32, ctypes.c_double, _P, _I64, _I32, _P, _P, _P, _P, ctypes.c_double, ctypes.c_double,
                           ctypes.c_double, _I32, _I32, _I32, _P, _P]
            fn.restype = ctypes.c_int
        _REG_WK.append(1)
    assert loss in WALK_LOSS and 1 <= int(max_steps) <= 64 and np.isfinite(param)
    wide = np.asarray(ids8).dtype == np.uint16
    ids8 = np.ascontiguousarray(ids8, dtype=np.uint16 if wide else np.uint8)
    fname = "shira_cat_reg_walk16" if wide else "shira_cat_reg_walk"
    T, n = ids8.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float64)
    ww = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
    f0 = None if F0 is None else np.ascontiguousarray(F0, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and (ww is None or ww.shape == (n,)) and (f0 is None or f0.shape == (n,))
    out = np.empty(int(nl.sum()), dtype=np.float64)
    taken = np.zeros(T, dtype=np.int32)
    _check(getattr(lib, fname)(WALK_LOSS[loss], float(param), _ptr(ids8), n, T, _ptr(nl), _ptr(yy), _ptr(ww), _ptr(f0),
                               float(bias), float(eta), float(lam), int(max_steps), int(bool(force)), int(nthread), _ptr(out),
                               _ptr(taken)), fname)
    return out, taken


_REG_QX: list[int] = []


def cat_quantile_values_x(ids: np.ndarray, nl: np.ndarray, y: np.ndarray, bias: float, alpha: float, delta: float, eta: float,
                          nthread: int = 1, F0: np.ndarray | None = None, f32: bool = False) -> np.ndarray:
    lib = _blib()
    if not _REG_QX:
        for name in ("shira_cat_quantile_values_x", "shira_cat_quantile_values_x16"):
            fn = getattr(lib, name)
            fn.argtypes = [_P, _I64, _I32, _P, _P, _P, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _I32,
                           _I32, _P]
            fn.restype = ctypes.c_int
        _REG_QX.append(1)
    wide = np.asarray(ids).dtype == np.uint16
    ids = np.ascontiguousarray(ids, dtype=np.uint16 if wide else np.uint8)
    fname = "shira_cat_quantile_values_x16" if wide else "shira_cat_quantile_values_x"
    T, n = ids.shape
    nl = np.ascontiguousarray(nl, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float64)
    f0 = None if F0 is None else np.ascontiguousarray(F0, dtype=np.float64)
    assert nl.shape == (T,) and yy.shape == (n,) and (f0 is None or f0.shape == (n,)) and 0.0 < alpha < 1.0
    out = np.empty(int(nl.sum()), dtype=np.float64)
    _check(getattr(lib, fname)(_ptr(ids), n, T, _ptr(nl), _ptr(yy), _ptr(f0), float(bias), float(alpha), float(delta), float(eta),
                               int(bool(f32)), int(nthread), _ptr(out)), fname)
    return out
