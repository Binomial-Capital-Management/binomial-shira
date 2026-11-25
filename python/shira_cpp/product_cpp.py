from __future__ import annotations

import ctypes

import numpy as np
from typing import Any

from .binning_cpp import _check, _lib as _blib, _ptr

_P = ctypes.c_void_p
_I32, _I64, _F, _D = ctypes.c_int32, ctypes.c_int64, ctypes.c_float, ctypes.c_double
OBJ = {"reg:pseudohubererror": 0, "binary:logistic": 1, "reg:squarederror": 2, "reg:logistic": 3, "binary:logitraw": 4,
       "count:poisson": 5, "reg:tweedie": 6, "reg:gamma": 7}
_REG: list[int] = []


def _lib() -> ctypes.CDLL:
    lib = _blib()
    if not _REG:
        lib.shira_tree_leaf_ids.argtypes = [_P, _I64, _I32, _P, _P, _I32, _I32, _I32, _I32, _P, _P]
        lib.shira_tree_leaf_ids.restype = ctypes.c_int
        lib.shira_lgb_node_counts.argtypes = [_P, _I32, _I64, _I32, _P, _I32, _P, _I32, _I32, _P, _I32, _P, _P, _P]
        lib.shira_lgb_node_counts.restype = ctypes.c_int
        lib.shira_tree_leaf_ids_c8.argtypes = [_P, _I64, _I32, _P, _P, _I32, _I32, _I32, _I32, _P, _P]
        lib.shira_tree_leaf_ids_c8.restype = ctypes.c_int
        lib.shira_tree_leaf_ids_cat.argtypes = [_P, _I64, _I32, _P, _P, _I32, _I32, _I32, _I32, _P, _P, _P, _I64]
        lib.shira_tree_leaf_ids_cat.restype = ctypes.c_int
        lib.shira_tree_product.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _F, _F, _D, _D, _D, _D, _I32, _P, _P]
        lib.shira_tree_product.restype = ctypes.c_int
        lib.shira_tree_product_ld.argtypes = [_P, _I32, _I64, _I64, _P, _I32, _I32, _P, _P, _I32, _F, _F, _D, _D, _D, _D, _I32,
                                              _P, _P]
        lib.shira_tree_product_ld.restype = ctypes.c_int
        lib.shira_tree_pair_update.argtypes = [_P, _I32, _I64, _I64, _P, _I32, _I32, _P, _P, _D, _I32, _P, _P, _P]
        lib.shira_tree_pair_update.restype = ctypes.c_int
        lib.shira_tree_pair_solve.argtypes = [_P, _P, _P, _P, _I32, _I32, _D, _D, _D, _D, _D, _P]
        lib.shira_tree_product_bounded.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _P, _P, _I32, _F, _F, _D,
                                                   _D, _D, _I32, _P, _P]
        lib.shira_tree_product_bounded.restype = ctypes.c_int
        lib.shira_tree_linear_values.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _I32, _P, _P, _I32, _P, _P, _I32, _D, _D,
                                                 _D, _D, _I32, _P, _P, _P, _P]
        lib.shira_tree_linear_values.restype = ctypes.c_int
        lib.shira_tree_pair_solve.restype = ctypes.c_int
        lib.shira_tree_leaf_sums.argtypes = [_P, _I32, _I64, _I64, _P, _I32, _I32, _P, _P, _D, _I32, _P, _P]
        lib.shira_tree_leaf_sums.restype = ctypes.c_int
        lib.shira_tree_corr_sweep.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _P, _I32, _D, _D, _D, _I32, _I32, _D,
                                              _I32, _I32, _P, _P, _P]
        lib.shira_tree_corr_sweep.restype = ctypes.c_int
        lib.shira_tree_product_l1.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _F, _F, _D, _D, _D, _D, _D, _I32,
                                              _P, _P]
        lib.shira_tree_product_l1.restype = ctypes.c_int
        lib.shira_tree_product_softmax.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _I32, _P, _P, _D, _D, _D, _D, _I32, _P,
                                                   _P]
        lib.shira_tree_product_softmax.restype = ctypes.c_int
        lib.shira_tree_product_bounded_reg.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _P, _P, _I32, _F, _F,
                                                       _D, _D, _D, _D, _D, _I32, _P, _P]
        lib.shira_tree_product_bounded_reg.restype = ctypes.c_int
        lib.shira_tree_product_smooth.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _P, _P, _I32, _F, _F, _D,
                                                  _D, _D, _D, _D, _I32, _P, _P]
        lib.shira_tree_product_smooth.restype = ctypes.c_int
        lib.shira_tree_product_dart.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _D, _P, _P, _P, _D, _D, _I32,
                                                _P, _P]
        lib.shira_tree_product_dart.restype = ctypes.c_int
        lib.shira_tree_product_softmax_reg.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _I32, _P, _P, _I32, _P, _P, _D, _D,
                                                       _D, _D, _D, _I32, _P, _P]
        lib.shira_tree_product_softmax_reg.restype = ctypes.c_int
        lib.shira_tree_product_softmax_lgb.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _I32, _P, _P, _D, _D, _D, _D, _I32,
                                                       _P, _P]
        lib.shira_tree_product_softmax_lgb.restype = ctypes.c_int
        lib.shira_tree_product_lgb.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _D, _D, _D, _D, _D, _I32, _P,
                                               _P]
        lib.shira_tree_product_lgb.restype = ctypes.c_int
        lib.shira_xgb_rank_ndcg_values.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I64, _P, _I32, _I32, _I32, _I32,
                                                   _D, _D, _D, _D, _D, _I32, _P, _P]
        lib.shira_xgb_rank_ndcg_values.restype = ctypes.c_int
        lib.shira_xgb_mot_values.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _I32, _P, _P, _I32, _D, _D, _D, _D, _D, _I32, _P,
                                             _P, _D]
        lib.shira_xgb_mot_values.restype = ctypes.c_int
        lib.shira_xgb_given_values.argtypes = [_P, _I32, _I64, _I32, _P, _P, _D, _D, _D, _D, _D, _I32, _P, _P]
        lib.shira_xgb_given_values.restype = ctypes.c_int
        lib.shira_lgb_xendcg_values.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I64, ctypes.c_uint32, _D, _D, _D, _D,
                                                _I32, _P, _P]
        lib.shira_lgb_xendcg_values.restype = ctypes.c_int
        lib.shira_tree_product_parallel.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _I32, _P, _P, _I32, _F, _F, _D, _D, _D,
                                                    _D, _I32, _P, _P]
        lib.shira_tree_product_parallel.restype = ctypes.c_int
        lib.shira_tree_product_softmax_lgb_mono.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _I32, _P, _P, _I32, _P, _P, _D,
                                                            _D, _D, _D, _I32, _P, _P]
        lib.shira_tree_product_softmax_lgb_mono.restype = ctypes.c_int
        lib.shira_tree_product_lgb_mono.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _P, _P, _I32, _D, _D, _D,
                                                    _D, _D, _I32, _P, _P]
        lib.shira_tree_product_lgb_mono.restype = ctypes.c_int
        lib.shira_tree_product_renew.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _D, _D, _I32, _P, _P]
        lib.shira_tree_product_renew.restype = ctypes.c_int
        lib.shira_tree_product_renew_w.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _D, _D, _I32, _P, _P]
        lib.shira_tree_product_renew_w.restype = ctypes.c_int
        lib.shira_tree_product_renew_xgb.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _P, _D, _F, _I32, _P, _P]
        lib.shira_tree_product_renew_xgb.restype = ctypes.c_int
        lib.shira_tree_product_obj2.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I32, _D, _D, _D, _D, _I32, _P,
                                                _P]
        lib.shira_tree_product_obj2.restype = ctypes.c_int
        lib.shira_leaf_ids_transpose.argtypes = [_P, _I64, _I32, _I32, _I32, _P]
        lib.shira_leaf_ids_transpose.restype = ctypes.c_int
        lib.shira_tree_leaf_ids_lanes.argtypes = [_P, _I32, _I64, _I32, _P, _P, _I32, _I32, _I32, _I32, _P, _P, _P,
                                                  _I64]
        lib.shira_tree_leaf_ids_lanes.restype = ctypes.c_int
        _REG.append(1)
    return lib


def node_counts(ids: np.ndarray, nl: np.ndarray, nodes: np.ndarray, leaf: np.ndarray,
                nthread: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idc = np.ascontiguousarray(ids)
    assert idc.ndim == 2 and idc.dtype in (np.uint8, np.uint16), idc.dtype
    T, n = idc.shape
    nlc = np.ascontiguousarray(nl, dtype=np.int32)
    nd = np.ascontiguousarray(nodes[:T], dtype=np.int32)
    lf = np.ascontiguousarray(leaf[:T], dtype=np.float64)
    assert nlc.shape == (T,) and nd.ndim == 3 and nd.shape[0] == T and lf.shape[0] == T
    L, Li = int(lf.shape[1]), int(nd.shape[1])
    cnt = np.empty((T, L), dtype=np.float64)
    icnt = np.empty((T, Li), dtype=np.float64)
    isum = np.empty((T, Li), dtype=np.float64)
    _check(_lib().shira_lgb_node_counts(_ptr(idc), idc.dtype.itemsize, n, T, _ptr(nlc), L, _ptr(nd), Li, int(nd.shape[2]),
                                        _ptr(lf), max(1, int(nthread)), _ptr(cnt), _ptr(icnt), _ptr(isum)),
           "shira_lgb_node_counts")
    return cnt, icnt, isum


def leaf_ids_lanes(codes: np.ndarray, nodes: np.ndarray, count: np.ndarray, nthread: int = 1,
                   cat: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    c = np.ascontiguousarray(codes)
    assert c.ndim == 2 and c.dtype in (np.uint8, np.uint16)
    nd = np.ascontiguousarray(nodes, dtype=np.int32)
    ct = np.ascontiguousarray(count, dtype=np.int32)
    tab = None if cat is None else np.ascontiguousarray(cat, dtype=np.uint32).reshape(-1)
    assert nd.ndim == 3 and nd.shape[2] == 5 and ct.shape == (nd.shape[0],)
    T, maxn = nd.shape[0], nd.shape[1]
    leaves_max = int(((nd[:, :, 0] < 0) & (np.arange(maxn)[None, :] < ct[:, None])).sum(axis=1).max())
    dt = np.uint8 if leaves_max <= 256 else np.uint16
    out = np.empty((T, c.shape[0]), dtype=dt)
    nl = np.zeros(T, dtype=np.int32)
    _check(_lib().shira_tree_leaf_ids_lanes(_ptr(c), c.dtype.itemsize, c.shape[0], c.shape[1], _ptr(nd), _ptr(ct), T, maxn,
                                            int(nthread), np.dtype(dt).itemsize, _ptr(out), _ptr(nl), _ptr(tab),
                                            0 if tab is None else tab.size), "shira_tree_leaf_ids_lanes")
    return out, nl


def leaf_ids_cat(codes: np.ndarray, nodes: np.ndarray, count: np.ndarray, cat: np.ndarray,
                 nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    c = np.ascontiguousarray(codes, dtype=np.uint16)
    nd = np.ascontiguousarray(nodes, dtype=np.int32)
    ct = np.ascontiguousarray(count, dtype=np.int32)
    tab = np.ascontiguousarray(cat, dtype=np.uint32).reshape(-1)
    assert c.ndim == 2 and nd.ndim == 3 and nd.shape[2] == 5 and ct.shape == (nd.shape[0],) and tab.size >= 9
    T, maxn = nd.shape[0], nd.shape[1]
    leaves_max = int(((nd[:, :, 0] < 0) & (np.arange(maxn)[None, :] < ct[:, None])).sum(axis=1).max())
    dt = np.uint8 if leaves_max <= 256 else np.uint16
    out = np.empty((T, c.shape[0]), dtype=dt)
    nl = np.zeros(T, dtype=np.int32)
    _check(_lib().shira_tree_leaf_ids_cat(_ptr(c), c.shape[0], c.shape[1], _ptr(nd), _ptr(ct), T, maxn, int(nthread),
                                           np.dtype(dt).itemsize, _ptr(out), _ptr(nl), _ptr(tab), tab.size),
           "shira_tree_leaf_ids_cat")
    return out, nl


def leaf_ids(codes: np.ndarray, nodes: np.ndarray, count: np.ndarray, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    c8 = codes.dtype == np.uint8
    c = np.ascontiguousarray(codes) if c8 else np.ascontiguousarray(codes, dtype=np.uint16)
    nd = np.ascontiguousarray(nodes, dtype=np.int32)
    ct = np.ascontiguousarray(count, dtype=np.int32)
    assert c.ndim == 2 and nd.ndim == 3 and nd.shape[2] == 5 and ct.shape == (nd.shape[0],)
    T, maxn = nd.shape[0], nd.shape[1]
    leaves_max = int(((nd[:, :, 0] < 0) & (np.arange(maxn)[None, :] < ct[:, None])).sum(axis=1).max())
    dt = np.uint8 if leaves_max <= 256 else np.uint16
    out = np.empty((T, c.shape[0]), dtype=dt)
    nl = np.zeros(T, dtype=np.int32)
    fn = _lib().shira_tree_leaf_ids_c8 if c8 else _lib().shira_tree_leaf_ids
    _check(fn(_ptr(c), c.shape[0], c.shape[1], _ptr(nd), _ptr(ct), T, maxn, int(nthread), np.dtype(dt).itemsize, _ptr(out),
              _ptr(nl)), "shira_tree_leaf_ids")
    return out, nl


def product(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, base_margin: float, objective: str = "reg:squarederror",
            eta: float = 0.3, reg_lambda: float = 1.0, min_child_weight: float = 1.0, max_delta_step: float = 0.0,
            weight: np.ndarray | None = None, obj_a: float = 1.0, obj_mds: float = 0.7, nthread: int = 1,
            pred: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    assert objective in OBJ, f"objective {objective!r}"
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    pr = np.full(n, np.float32(base_margin), dtype=np.float32) if pred is None else np.ascontiguousarray(pred, dtype=np.float32).copy()
    assert yy.shape == (n,) and pr.shape == (n,) and (w is None or w.shape == (n,))
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w), OBJ[objective],
                                     float(obj_a), float(obj_mds), float(eta), float(reg_lambda), float(min_child_weight),
                                     float(max_delta_step), int(nthread), _ptr(pr), _ptr(leaf)), "shira_tree_product")
    return leaf, pr


def product_parallel(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, base_margin: float, objective: str, K: int,
                     eta: float = 0.3, reg_lambda: float = 1.0, min_child_weight: float = 1.0, max_delta_step: float = 0.0,
                     weight: np.ndarray | None = None, obj_a: float = 1.0, obj_mds: float = 0.7,
                     nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert objective in OBJ, f"objective {objective!r}"
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16) and K >= 1
    T, n = ids.shape
    assert T % K == 0, "whole rounds"
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product_parallel(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T // K, int(K), lmax, _ptr(yy),
                                              _ptr(w), OBJ[objective], float(obj_a), float(obj_mds), float(eta),
                                              float(reg_lambda), float(min_child_weight), float(max_delta_step),
                                              int(nthread), _ptr(pr), _ptr(leaf)), "shira_tree_product_parallel")
    return leaf, pr


def product_l1(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, base_margin: float, objective: str = "reg:squarederror",
               eta: float = 0.3, reg_lambda: float = 1.0, reg_alpha: float = 0.0, min_child_weight: float = 1.0,
               max_delta_step: float = 0.0, weight: np.ndarray | None = None, obj_a: float = 1.0, obj_mds: float = 0.7,
               nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert objective in OBJ, f"objective {objective!r}"
    assert reg_alpha >= 0.0
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product_l1(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w),
                                        OBJ[objective], float(obj_a), float(obj_mds), float(eta), float(reg_lambda),
                                        float(reg_alpha), float(min_child_weight), float(max_delta_step), int(nthread),
                                        _ptr(pr), _ptr(leaf)), "shira_tree_product_l1")
    return leaf, pr


def product_softmax(ids: np.ndarray, nleaves: np.ndarray, num_class: int, y: np.ndarray, base_margin: Any,
                    eta: float = 0.3, reg_lambda: float = 1.0, min_child_weight: float = 1.0, max_delta_step: float = 0.0,
                    weight: np.ndarray | None = None, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16) and num_class >= 2
    T, n = ids.shape
    assert T % num_class == 0, "whole rounds"
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    bm = np.asarray(base_margin, dtype=np.float32)
    assert bm.ndim == 0 or bm.shape == (num_class,)
    pr = np.ascontiguousarray(np.broadcast_to(bm.reshape(-1, 1) if bm.ndim else bm, (num_class, n)), dtype=np.float32)
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product_softmax(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T // num_class, int(num_class),
                                             lmax, _ptr(yy), _ptr(w), float(eta), float(reg_lambda),
                                             float(min_child_weight), float(max_delta_step), int(nthread), _ptr(pr),
                                             _ptr(leaf)), "shira_tree_product_softmax")
    return leaf, pr


def product_softmax_lgb(ids: np.ndarray, nleaves: np.ndarray, num_class: int, y: np.ndarray, init: np.ndarray,
                        eta: float = 0.1, reg_lambda: float = 0.0, reg_alpha: float = 0.0, min_child_weight: float = 1e-3,
                        weight: np.ndarray | None = None, nthread: int = 1, tree: np.ndarray | None = None,
                        count: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16) and num_class >= 2
    T, n = ids.shape
    assert T % num_class == 0, "whole rounds"
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    ini = np.asarray(init, dtype=np.float64)
    assert yy.shape == (n,) and (w is None or w.shape == (n,)) and ini.shape == (num_class,)
    pr = np.ascontiguousarray(np.repeat(ini[:, None], n, axis=1))
    leaf = np.zeros((T, lmax), dtype=np.float64)
    if tree is not None:
        tr, ct = _mono_table(tree, count, T)
        _check(_lib().shira_tree_product_softmax_lgb_mono(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T // num_class,
                                                          int(num_class), lmax, _ptr(tr), _ptr(ct), tr.shape[1], _ptr(yy),
                                                          _ptr(w), float(eta), float(reg_lambda), float(reg_alpha),
                                                          float(min_child_weight), int(nthread), _ptr(pr), _ptr(leaf)),
               "shira_tree_product_softmax_lgb_mono")
        return leaf, pr
    _check(_lib().shira_tree_product_softmax_lgb(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T // num_class,
                                                 int(num_class), lmax, _ptr(yy), _ptr(w), float(eta), float(reg_lambda),
                                                 float(reg_alpha), float(min_child_weight), int(nthread), _ptr(pr),
                                                 _ptr(leaf)), "shira_tree_product_softmax_lgb")
    return leaf, pr


def lgb_xendcg(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, group: np.ndarray, seed: int, eta: float,
               lambda_l1: float = 0.0, lambda_l2: float = 0.0, max_delta_step: float = 0.0,
               nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    gp = np.ascontiguousarray(np.concatenate([[0], np.cumsum(np.asarray(group, dtype=np.int64))]), dtype=np.int64)
    Q = int(gp.size) - 1
    assert yy.shape == (n,) and int(gp[-1]) == n and Q >= 1
    sc = np.zeros(n, dtype=np.float64)
    leaf = np.zeros((T, lmax), dtype=np.float64)
    _check(_lib().shira_lgb_xendcg_values(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(gp), Q,
                                          ctypes.c_uint32(int(seed) & 0xFFFFFFFF), float(eta), float(lambda_l1),
                                          float(lambda_l2), float(max_delta_step), int(nthread), _ptr(sc), _ptr(leaf)),
           "shira_lgb_xendcg_values")
    return leaf, sc


def xgb_mot(ids: np.ndarray, nleaves: np.ndarray, Y: np.ndarray, base_margin: np.ndarray, logistic: bool,
            weight: np.ndarray | None = None, eta: float = 0.3, reg_lambda: float = 1.0, reg_alpha: float = 0.0,
            min_child_weight: float = 1.0, max_delta_step: float = 0.0, nthread: int = 1,
            scale_pos_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(Y, dtype=np.float32)
    assert yy.ndim == 2 and yy.shape[0] == n
    K = int(yy.shape[1])
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert w is None or w.shape == (n,)
    pr = np.ascontiguousarray(np.tile(np.asarray(base_margin, dtype=np.float32).reshape(1, K), (n, 1)))
    leaf = np.zeros((T, lmax, K), dtype=np.float32)
    _check(_lib().shira_xgb_mot_values(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, K, _ptr(yy), _ptr(w),
                                       1 if logistic else 0, float(eta), float(reg_lambda), float(reg_alpha),
                                       float(min_child_weight), float(max_delta_step), int(nthread), _ptr(pr), _ptr(leaf),
                                       float(scale_pos_weight)), "shira_xgb_mot_values")
    return leaf, pr


def xgb_given(ids_row: np.ndarray, nleaves: int, g: np.ndarray, h: np.ndarray, pred: np.ndarray, eta: float = 0.3,
              reg_lambda: float = 1.0, reg_alpha: float = 0.0, min_child_weight: float = 1.0, max_delta_step: float = 0.0,
              nthread: int = 1) -> np.ndarray:
    ids = np.ascontiguousarray(ids_row)
    assert ids.ndim == 1 and ids.dtype in (np.uint8, np.uint16)
    n = int(ids.size)
    gg, hh = np.ascontiguousarray(g, dtype=np.float32).reshape(-1), np.ascontiguousarray(h, dtype=np.float32).reshape(-1)
    assert gg.size == n and hh.size == n and pred.dtype == np.float32 and pred.shape == (n,) and pred.flags.c_contiguous
    leaf = np.zeros(int(nleaves), dtype=np.float32)
    _check(_lib().shira_xgb_given_values(_ptr(ids), ids.dtype.itemsize, n, int(nleaves), _ptr(gg), _ptr(hh), float(eta),
                                         float(reg_lambda), float(reg_alpha), float(min_child_weight), float(max_delta_step),
                                         int(nthread), _ptr(pred), _ptr(leaf)), "shira_xgb_given_values")
    return leaf


def xgb_rank_ndcg(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, group_ptr: np.ndarray, base_margin: float,
                  group_weight: np.ndarray | None = None, topk: int = 32, exp_gain: bool = True, norm_by_diff: bool = True,
                  normalization: bool = True, eta: float = 0.3, reg_lambda: float = 1.0, reg_alpha: float = 0.0,
                  min_child_weight: float = 1.0, max_delta_step: float = 0.0, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    gp = np.ascontiguousarray(group_ptr, dtype=np.int64)
    Q = int(gp.size) - 1
    assert yy.shape == (n,) and Q >= 1 and int(gp[0]) == 0 and int(gp[-1]) == n
    gw = None if group_weight is None else np.ascontiguousarray(group_weight, dtype=np.float32)
    assert gw is None or gw.shape == (Q,)
    pr = (np.array(base_margin, dtype=np.float32) if np.ndim(base_margin) == 1
          else np.full(n, base_margin, dtype=np.float32))
    assert pr.shape == (n,)
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_xgb_rank_ndcg_values(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(gp), Q,
                                             _ptr(gw), int(topk), int(exp_gain), int(norm_by_diff), int(normalization),
                                             float(eta), float(reg_lambda), float(reg_alpha), float(min_child_weight),
                                             float(max_delta_step), int(nthread), _ptr(pr), _ptr(leaf)),
           "shira_xgb_rank_ndcg_values")
    return leaf, pr


LGB_KIND = {"huber": 0, "binary": 1, "fair": 2}


def _mono_table(tree: np.ndarray, count: np.ndarray | None, T: int) -> tuple[np.ndarray, np.ndarray]:
    tr = np.ascontiguousarray(tree, dtype=np.int32)
    assert count is not None
    ct = np.ascontiguousarray(count, dtype=np.int32)
    assert tr.ndim == 3 and tr.shape[0] == T and tr.shape[2] == 4 and ct.shape == (T,)
    return tr, ct


def product_lgb(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, init: float, kind: str, a: float, eta: float = 0.1,
                reg_lambda: float = 0.0, reg_alpha: float = 0.0, min_child_weight: float = 1e-3,
                weight: np.ndarray | None = None, nthread: int = 1, tree: np.ndarray | None = None,
                count: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    assert kind in LGB_KIND and a > 0.0
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    pr = np.full(n, float(init), dtype=np.float64)
    leaf = np.zeros((T, lmax), dtype=np.float64)
    if tree is not None:
        tr, ct = _mono_table(tree, count, T)
        _check(_lib().shira_tree_product_lgb_mono(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(tr), _ptr(ct),
                                                  tr.shape[1], _ptr(yy), _ptr(w), LGB_KIND[kind], float(a), float(eta),
                                                  float(reg_lambda), float(reg_alpha), float(min_child_weight),
                                                  int(nthread), _ptr(pr), _ptr(leaf)), "shira_tree_product_lgb_mono")
        return leaf, pr
    _check(_lib().shira_tree_product_lgb(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w),
                                         LGB_KIND[kind], float(a), float(eta), float(reg_lambda), float(reg_alpha),
                                         float(min_child_weight), int(nthread), _ptr(pr), _ptr(leaf)),
           "shira_tree_product_lgb")
    return leaf, pr


def product_renew(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, init: float, alpha: float, eta: float = 0.1,
                  nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert 0.0 < alpha < 1.0
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    assert yy.shape == (n,)
    pr = np.full(n, float(init), dtype=np.float64)
    leaf = np.zeros((T, lmax), dtype=np.float64)
    _check(_lib().shira_tree_product_renew(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), float(alpha),
                                           float(eta), int(nthread), _ptr(pr), _ptr(leaf)), "shira_tree_product_renew")
    return leaf, pr


def product_renew_xgb(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, base_margin: float, alpha: float,
                      eta: float = 0.3, weight: np.ndarray | None = None, fallback: np.ndarray | None = None,
                      nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert 0.0 <= alpha <= 1.0 and eta > 0.0
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    fb = None if fallback is None else np.ascontiguousarray(fallback, dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,)) and (fb is None or fb.shape == (T, lmax))
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product_renew_xgb(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w),
                                               _ptr(fb), float(alpha), float(eta), int(nthread), _ptr(pr), _ptr(leaf)),
           "shira_tree_product_renew_xgb")
    return leaf, pr


OBJ2 = {"reg:squaredlogerror": 0, "binary:hinge": 1}


def leaf_transpose(leaf_nT: np.ndarray, wide: bool, nthread: int = 1) -> np.ndarray:
    a = np.ascontiguousarray(leaf_nT, dtype=np.int32)
    assert a.ndim == 2
    n, T = a.shape
    out = np.empty((T, n), dtype=np.uint16 if wide else np.uint8)
    _check(_lib().shira_leaf_ids_transpose(_ptr(a), n, T, 2 if wide else 1, int(nthread), _ptr(out)),
           "shira_leaf_ids_transpose")
    return out


def product_obj2(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, base_margin: float, objective: str,
                 eta: float = 0.3, reg_lambda: float = 1.0, min_child_weight: float = 1.0, max_delta_step: float = 0.0,
                 weight: np.ndarray | None = None, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert objective in OBJ2, objective
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product_obj2(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w),
                                          OBJ2[objective], float(eta), float(reg_lambda), float(min_child_weight),
                                          float(max_delta_step), int(nthread), _ptr(pr), _ptr(leaf)),
           "shira_tree_product_obj2")
    return leaf, pr


def product_renew_w(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, weight: np.ndarray, init: float, alpha: float,
                    eta: float = 0.1, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert 0.0 < alpha < 1.0
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (n,) and w.shape == (n,)
    pr = np.full(n, float(init), dtype=np.float64)
    leaf = np.zeros((T, lmax), dtype=np.float64)
    _check(_lib().shira_tree_product_renew_w(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w),
                                             float(alpha), float(eta), int(nthread), _ptr(pr), _ptr(leaf)),
           "shira_tree_product_renew_w")
    return leaf, pr


def product_smooth(ids: np.ndarray, nleaves: np.ndarray, tree: np.ndarray, count: np.ndarray, y: np.ndarray,
                   base_margin: float, objective: str, path_smooth: float, eta: float = 0.1, reg_lambda: float = 0.0,
                   reg_alpha: float = 0.0, max_delta_step: float = 0.0, weight: np.ndarray | None = None,
                   nthread: int = 1, obj_a: float = 1.0, obj_mds: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    assert objective in OBJ, f"objective {objective!r}"
    assert path_smooth > 0.0
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    tr = np.ascontiguousarray(tree, dtype=np.int32)
    ct = np.ascontiguousarray(count, dtype=np.int32)
    assert tr.ndim == 3 and tr.shape[0] == T and tr.shape[2] == 4 and ct.shape == (T,) and nl.shape == (T,)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product_smooth(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(tr), _ptr(ct),
                                            tr.shape[1], _ptr(yy), _ptr(w), OBJ[objective], float(obj_a), float(obj_mds),
                                            float(eta), float(reg_lambda), float(reg_alpha), float(max_delta_step),
                                            float(path_smooth), int(nthread), _ptr(pr), _ptr(leaf)),
           "shira_tree_product_smooth")
    return leaf, pr


def product_dart(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, init: float, obj: int, drops: list,
                 lambda_l1: float = 0.0, lambda_l2: float = 0.0, weight: np.ndarray | None = None,
                 nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    assert obj in (0, 1)
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    assert len(drops) == T
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    ptr = np.ascontiguousarray(np.concatenate([[0], np.cumsum([len(d[0]) for d in drops])]), dtype=np.int32)
    idx = np.ascontiguousarray(np.concatenate([np.asarray(d[0], dtype=np.int32) for d in drops] + [np.zeros(1, np.int32)]),
                               dtype=np.int32)
    rate = np.ascontiguousarray([float(d[2]) for d in drops], dtype=np.float64)
    pr = np.zeros(n, dtype=np.float64)
    leaf = np.zeros((T, lmax), dtype=np.float64)
    _check(_lib().shira_tree_product_dart(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w),
                                          int(obj), float(init), _ptr(ptr), _ptr(idx), _ptr(rate), float(lambda_l1),
                                          float(lambda_l2), int(nthread), _ptr(pr), _ptr(leaf)), "shira_tree_product_dart")
    return leaf, pr


def product_softmax_reg(ids: np.ndarray, nleaves: np.ndarray, num_class: int, y: np.ndarray, base_margin: Any,
                        tree: np.ndarray | None = None, count: np.ndarray | None = None, eta: float = 0.3,
                        reg_lambda: float = 1.0, reg_alpha: float = 0.0, min_child_weight: float = 1.0,
                        max_delta_step: float = 0.0, weight: np.ndarray | None = None,
                        nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16) and num_class >= 2
    T, n = ids.shape
    assert T % num_class == 0, "whole rounds"
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,)) and (tree is None) == (count is None)
    tr = None if tree is None else np.ascontiguousarray(tree, dtype=np.int32)
    ct = None if count is None else np.ascontiguousarray(count, dtype=np.int32)
    assert tr is None or (tr.ndim == 3 and tr.shape[0] == T and tr.shape[2] == 4 and ct.shape == (T,))
    bm = np.asarray(base_margin, dtype=np.float32)
    assert bm.ndim == 0 or bm.shape == (num_class,)
    pr = np.ascontiguousarray(np.broadcast_to(bm.reshape(-1, 1) if bm.ndim else bm, (num_class, n)), dtype=np.float32)
    leaf = np.zeros((T, lmax), dtype=np.float32)
    _check(_lib().shira_tree_product_softmax_reg(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T // num_class,
                                                 int(num_class), lmax, _ptr(tr), _ptr(ct),
                                                 0 if tr is None else tr.shape[1], _ptr(yy), _ptr(w), float(eta),
                                                 float(reg_lambda), float(reg_alpha), float(min_child_weight),
                                                 float(max_delta_step), int(nthread), _ptr(pr), _ptr(leaf)),
           "shira_tree_product_softmax_reg")
    return leaf, pr


def product_window(ids_all: np.ndarray, lo: int, hi: int, nleaves: np.ndarray, y: np.ndarray, base_margin: float,
                   objective: str = "reg:squarederror", eta: float = 0.3, reg_lambda: float = 1.0,
                   min_child_weight: float = 1.0, max_delta_step: float = 0.0, weight: np.ndarray | None = None,
                   obj_a: float = 1.0, obj_mds: float = 0.7, nthread: int = 1) -> np.ndarray:
    assert objective in OBJ, f"objective {objective!r}"
    assert ids_all.ndim == 2 and ids_all.flags.c_contiguous and ids_all.dtype in (np.uint8, np.uint16)
    T, N = ids_all.shape
    assert 0 <= lo < hi <= N
    n = hi - lo
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    leaf = np.zeros((T, lmax), dtype=np.float32)
    first = ctypes.c_void_p(ids_all.ctypes.data + lo * ids_all.itemsize)
    _check(_lib().shira_tree_product_ld(first, ids_all.itemsize, N, n, _ptr(nl), T, lmax, _ptr(yy), _ptr(w), OBJ[objective],
                                        float(obj_a), float(obj_mds), float(eta), float(reg_lambda), float(min_child_weight),
                                        float(max_delta_step), int(nthread), _ptr(pr), _ptr(leaf)), "shira_tree_product_ld")
    return leaf


def pair_tables(nleaves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = int(np.asarray(nleaves).sum())
    assert L >= 1 and L <= 8192, f"{L} leaves: the pair table would be {8 * L * L / 2**20:.0f} MiB"
    return np.zeros(L), np.zeros(L), np.zeros((L, L))


def pair_update(ids_all: np.ndarray, lo: int, hi: int, nleaves: np.ndarray, y: np.ndarray, tables: tuple,
                sign: float = 1.0, weight: np.ndarray | None = None, nthread: int = 1) -> None:
    assert ids_all.ndim == 2 and ids_all.flags.c_contiguous and ids_all.dtype in (np.uint8, np.uint16)
    T, N_all = ids_all.shape
    assert 0 <= lo <= hi <= N_all and sign in (1.0, -1.0)
    if hi == lo:
        return
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (hi - lo,) and (w is None or w.shape == yy.shape)
    Nt, St, Ct = tables
    assert Nt.size == int(nl.sum()) and Ct.shape == (Nt.size, Nt.size) and Ct.flags.c_contiguous
    first = ctypes.c_void_p(ids_all.ctypes.data + lo * ids_all.itemsize)
    _check(_lib().shira_tree_pair_update(first, ids_all.itemsize, N_all, hi - lo, _ptr(nl), T, int(nl.max()), _ptr(yy),
                                         _ptr(w), float(sign), int(nthread), _ptr(Nt), _ptr(St), _ptr(Ct)),
           "shira_tree_pair_update")


def product_bounded(ids: np.ndarray, nleaves: np.ndarray, tree: np.ndarray, count: np.ndarray, y: np.ndarray,
                    base_margin: float, objective: str = "reg:squarederror", eta: float = 0.3, reg_lambda: float = 1.0,
                    min_child_weight: float = 1.0, weight: np.ndarray | None = None, nthread: int = 1,
                    obj_mds: float = 0.7, reg_alpha: float = 0.0, max_delta_step: float = 0.0,
                    obj_a: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    assert reg_alpha >= 0.0 and max_delta_step >= 0.0
    assert objective in OBJ, f"objective {objective!r}"
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    tr = np.ascontiguousarray(tree, dtype=np.int32)
    ct = np.ascontiguousarray(count, dtype=np.int32)
    assert tr.ndim == 3 and tr.shape[0] == T and tr.shape[2] == 4 and ct.shape == (T,) and nl.shape == (T,)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    assert yy.shape == (n,) and (w is None or w.shape == (n,))
    leaf = np.zeros((T, lmax), dtype=np.float32)
    if reg_alpha != 0.0 or max_delta_step != 0.0 or obj_a != 1.0:
        _check(_lib().shira_tree_product_bounded_reg(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(tr), _ptr(ct),
                                                     tr.shape[1], _ptr(yy), _ptr(w), OBJ[objective], float(obj_a),
                                                     float(obj_mds), float(eta), float(reg_lambda), float(reg_alpha),
                                                     float(min_child_weight), float(max_delta_step), int(nthread), _ptr(pr),
                                                     _ptr(leaf)), "shira_tree_product_bounded_reg")
        return leaf, pr
    _check(_lib().shira_tree_product_bounded(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(tr), _ptr(ct),
                                             tr.shape[1], _ptr(yy), _ptr(w), OBJ[objective], 1.0, float(obj_mds), float(eta),
                                             float(reg_lambda), float(min_child_weight), int(nthread), _ptr(pr),
                                             _ptr(leaf)), "shira_tree_product_bounded")
    return leaf, pr


def linear_values(ids: np.ndarray, nleaves: np.ndarray, x: np.ndarray, fptr: np.ndarray, feats: np.ndarray, y: np.ndarray,
                  base_margin: float, objective: str = "reg:squarederror", eta: float = 0.1, reg_lambda: float = 0.0,
                  min_child_weight: float = 1e-3, linear_lambda: float = 0.0, weight: np.ndarray | None = None,
                  nthread: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert objective in OBJ, f"objective {objective!r}"
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    xx = np.ascontiguousarray(x, dtype=np.float64)
    fp = np.ascontiguousarray(fptr, dtype=np.int32)
    ft = np.ascontiguousarray(feats if len(feats) else [0], dtype=np.int32)
    assert xx.shape[0] == n and fp.shape == (T, lmax + 1)
    kmax = int(np.diff(fp, axis=1).max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    pr = np.full(n, np.float32(base_margin), dtype=np.float32)
    leaf, const, coeff = np.zeros((T, lmax), dtype=np.float32), np.zeros((T, lmax)), np.zeros(ft.size)
    _check(_lib().shira_tree_linear_values(_ptr(ids), ids.dtype.itemsize, n, _ptr(nl), T, lmax, _ptr(xx), xx.shape[1],
                                           _ptr(fp), _ptr(ft), kmax, _ptr(yy), _ptr(w), OBJ[objective], float(eta),
                                           float(reg_lambda), float(min_child_weight), float(linear_lambda), int(nthread),
                                           _ptr(pr), _ptr(leaf), _ptr(const), _ptr(coeff)), "shira_tree_linear_values")
    return leaf, const, coeff, pr


def pair_solve(tables: tuple, nleaves: np.ndarray, base_margin: float, eta: float = 0.3, reg_lambda: float = 1.0,
               min_child_weight: float = 1.0, max_delta_step: float = 0.0) -> np.ndarray:
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    Nt, St, Ct = tables
    assert Nt.size == int(nl.sum()) and Ct.shape == (Nt.size, Nt.size)
    leaf = np.zeros((nl.size, int(nl.max())), dtype=np.float32)
    _check(_lib().shira_tree_pair_solve(_ptr(Nt), _ptr(St), _ptr(Ct), _ptr(nl), nl.size, int(nl.max()), float(base_margin),
                                        float(eta), float(reg_lambda), float(min_child_weight), float(max_delta_step),
                                        _ptr(leaf)), "shira_tree_pair_solve")
    return leaf


def leaf_sums(ids_all: np.ndarray, lo: int, hi: int, nleaves: np.ndarray, y: np.ndarray, sums: tuple,
              sign: float = 1.0, weight: np.ndarray | None = None, nthread: int = 1) -> None:
    assert ids_all.ndim == 2 and ids_all.flags.c_contiguous and ids_all.dtype in (np.uint8, np.uint16)
    T, N_all = ids_all.shape
    assert 0 <= lo <= hi <= N_all and sign in (1.0, -1.0)
    if hi == lo:
        return
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    assert yy.shape == (hi - lo,) and (w is None or w.shape == yy.shape)
    Nt, St = sums
    assert Nt.shape == St.shape == (int(nl.sum()),) and Nt.dtype == St.dtype == np.float64
    first = ctypes.c_void_p(ids_all.ctypes.data + lo * ids_all.itemsize)
    _check(_lib().shira_tree_leaf_sums(first, ids_all.itemsize, N_all, hi - lo, _ptr(nl), T, int(nl.max()), _ptr(yy),
                                       _ptr(w), float(sign), int(nthread), _ptr(Nt), _ptr(St)), "shira_tree_leaf_sums")


CORR_OBJ = ("binary:logistic", "reg:squarederror", "reg:logistic", "binary:logitraw")


def corr_sweep(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, weight: np.ndarray | None, hold: np.ndarray,
               objective: str, reg_lambda: float, min_child_weight: float, margin: np.ndarray, leaf: np.ndarray,
               base_margin: float = 0.0, init: bool = False, nthread: int = 1, damp: float = 1.0,
               trees: tuple[int, int] | None = None) -> float:
    assert 0.0 < damp <= 1.0
    assert objective in CORR_OBJ, f"objective {objective!r}"
    assert ids.ndim == 2 and ids.flags.c_contiguous and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    yy = np.ascontiguousarray(y, dtype=np.float32)
    w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
    hd = np.ascontiguousarray(hold, dtype=np.uint8)
    assert yy.shape == (n,) and hd.shape == (n,) and (w is None or w.shape == (n,)) and nl.shape == (T,)
    assert margin.shape == (n,) and margin.dtype == np.float64 and margin.flags.c_contiguous
    assert leaf.ndim == 2 and leaf.shape[0] == T and leaf.dtype == np.float32 and leaf.flags.c_contiguous
    t0, t1 = (0, T) if trees is None else (int(trees[0]), int(trees[1]))
    assert 0 <= t0 < t1 <= T
    out = np.zeros(1, dtype=np.float64)
    _check(_lib().shira_tree_corr_sweep(_ptr(ids), ids.itemsize, n, _ptr(nl), T, int(leaf.shape[1]), _ptr(yy), _ptr(w),
                                        _ptr(hd), OBJ[objective], float(reg_lambda), float(min_child_weight), float(damp),
                                        int(t0), int(t1), float(base_margin), int(bool(init)), int(nthread), _ptr(margin), _ptr(leaf),
                                        _ptr(out)), "shira_tree_corr_sweep")
    return float(out[0])


CORR_PROBE_SHARE = 16
CORR_PROBE_GAIN = 5e-4


def _hold_loss(objective: str, margin: np.ndarray, y: np.ndarray, weight: np.ndarray | None, hold: np.ndarray) -> float:
    h = np.flatnonzero(hold)
    m, yy = margin[h], np.asarray(y, dtype=np.float32)[h].astype(np.float64)
    if objective == "reg:squarederror":
        e = (m - yy) ** 2
    else:
        e = np.where(m > 0.0, m + np.log1p(np.exp(-np.abs(m))), np.log1p(np.exp(np.minimum(m, 0.0)))) - yy * m
    w = None if weight is None else np.asarray(weight, dtype=np.float32)[h].astype(np.float64)
    out = float(np.sum(e) / h.size) if w is None else float(np.sum(w * e) / np.sum(w))
    assert np.isfinite(out) and h.size > 0
    return out
CORR_DAMPS = (1.0, 0.5, 0.25, 0.125)


def corr_drive(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, weight: np.ndarray | None, objective: str,
               reg_lambda: float, min_child_weight: float, base_margin: float, leaf: np.ndarray, shape_rows: np.ndarray,
               sweeps: int, hold_every: int, nthread: int = 1, gain: float = 1e-4,
               margin0: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    assert sweeps >= 1 and hold_every >= 2 and leaf.ndim == 2
    n = int(ids.shape[1])
    outside = np.ones(n, dtype=bool)
    outside[shape_rows] = False
    cand = np.flatnonzero(outside)
    if cand.size == 0:
        return None, None, {"sweeps": 0, "reason": "every row is in the shape sample"}
    step = max(1, cand.size // max(1, n // hold_every))
    hold = np.zeros(n, dtype=np.uint8)
    hold[cand[step - 1:: step]] = 1
    lv = np.array(leaf, dtype=np.float32, order="C")
    if margin0 is None:
        margin = np.empty(n, dtype=np.float64)
        best = corr_sweep(ids, nleaves, y, weight, hold, objective, reg_lambda, min_child_weight, margin, lv,
                          base_margin=float(base_margin), init=True, nthread=nthread)
    else:
        margin = np.array(margin0, dtype=np.float64, order="C")
        assert margin.shape == (n,)
        best = _hold_loss(objective, margin, y, weight, hold)
    args = (nleaves, y, weight, hold, objective, reg_lambda, min_child_weight, margin)
    sched = (1.0,) if objective == "reg:squarederror" else CORR_DAMPS
    T = int(ids.shape[0])
    pv, pm = lv.copy(), margin.copy()
    probe = corr_sweep(ids, nleaves, y, weight, hold, objective, reg_lambda, min_child_weight, pm, pv, nthread=nthread,
                       damp=sched[-1], trees=(0, max(1, -(-T // CORR_PROBE_SHARE))))
    if not probe < best * (1.0 - CORR_PROBE_GAIN):
        return None, None, {"sweeps": 0, "probe": probe, "losses": [best], "held_back": int(hold.sum())}
    del pv, pm
    losses, kept, keep_m, di, damps = [best], 0, None, 0, []
    for _ in range(sweeps + len(sched)):
        if kept >= sweeps:
            break
        prev, prev_m = lv.copy(), margin.copy()
        loss = corr_sweep(ids, *args, lv, nthread=nthread, damp=sched[di])
        losses.append(loss)
        if loss < best * (1.0 - gain):
            best, kept, keep_m = loss, kept + 1, margin
            damps.append(sched[di])
            continue
        lv, margin, keep_m = prev, prev_m, prev_m
        args = (nleaves, y, weight, hold, objective, reg_lambda, min_child_weight, margin)
        di += 1
        if di >= len(sched):
            break
    assert len(losses) >= 2 and kept <= sweeps
    log = {"sweeps": kept, "damps": damps, "losses": losses, "probe": probe, "held_back": int(hold.sum())}
    return (lv, keep_m, log) if kept > 0 else (None, None, log)
