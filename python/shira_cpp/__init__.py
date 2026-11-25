from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
from shira import model as trees

_LIB_NAME = "libshira.dylib" if sys.platform == "darwin" else "libshira.so"
LIB_PATH = Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB") else SHIRA / "lib" / _LIB_NAME
_P = ctypes.c_void_p
_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        assert LIB_PATH.exists(), f"build first: {SHIRA / 'build.sh'}"
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.shira_train.restype = ctypes.c_int
        lib.shira_train.argtypes = [_P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, _P, _P, _P,
                                    ctypes.c_int32, _P, _P, ctypes.c_int32, ctypes.c_int32,
                                    _P, _P, _P, _P]
        lib.shira_train_xgb.restype = ctypes.c_int
        lib.shira_train_xgb.argtypes = [_P, ctypes.c_int64, ctypes.c_int32, _P, _P, _P, _P,
                                        ctypes.c_int32, _P, _P, ctypes.c_int32, ctypes.c_int32,
                                        _P, _P, _P, _P]
        _LIB = lib
    return _LIB


def _recipe_arrays(rec: trees.Recipe) -> tuple[np.ndarray, np.ndarray]:
    lam_q = int(round(rec.lam * (1 << trees.SCALE)))
    mcw_q = int(round(rec.min_hess * (1 << trees.SCALE)))
    assert lam_q > 0, "lambda must be > 0 (chip format divides by H + lambda)"
    obj = {"pseudohuber": 0, "logistic": 1}[rec.objective]
    rec_d = np.array([rec.lam, rec.min_hess, rec.min_gain, rec.eta, rec.base_score], dtype=np.float64)
    rec_i = np.array([rec.max_depth, obj, lam_q, mcw_q], dtype=np.int64)
    return rec_d, rec_i


def _boosters(rec, nodes, leaf, count, rounds: int, k_models: int) -> list[trees.Booster]:
    out = []
    for k in range(k_models):
        b = trees.Booster(rec)
        for r in range(rounds):
            tree = []
            for v in range(int(count[k, r])):
                f, cut, d, lft, rgt = (int(x) for x in nodes[k, r, v])
                if f < 0:
                    tree.append(trees.Node(leaf=float(leaf[k, r, v])))
                else:
                    tree.append(trees.Node(feature=f, cut=cut, dir=d, left=lft, right=rgt))
            b.trees.append(tree)
        out.append(b)
    return out


def train(params: dict, dtrain: trees.DMatrix, num_boost_round: int,
          folds: list[tuple[int, int]], threads: int = 6) -> tuple[list[trees.Booster], list[np.ndarray]]:
    rec = trees.recipe_xgboost(params)
    assert 1 <= rec.max_depth <= 10 and 0 < num_boost_round <= 100000 and folds
    codes = np.ascontiguousarray(dtrain.codes.astype(np.uint16))
    y = np.ascontiguousarray(dtrain.y, dtype=np.float32)
    n, nf = codes.shape
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    k_models, maxn = len(folds), 1 << (rec.max_depth + 1)
    rec_d, rec_i = _recipe_arrays(rec)
    nodes = np.zeros((k_models, num_boost_round, maxn, 5), dtype=np.int32)
    leaf = np.zeros((k_models, num_boost_round, maxn), dtype=np.float32)
    count = np.zeros((k_models, num_boost_round), dtype=np.int32)
    pred = np.zeros((n, k_models), dtype=np.float32)
    rc = _lib().shira_train(codes.ctypes.data, n, nf, dtrain.n_bins, y.ctypes.data, lo.ctypes.data,
                            hi.ctypes.data, k_models, rec_d.ctypes.data, rec_i.ctypes.data,
                            num_boost_round, threads, nodes.ctypes.data, leaf.ctypes.data,
                            count.ctypes.data, pred.ctypes.data)
    assert rc == 0, f"shira_train failed rc={rc}"
    preds = [pred[a:b, k].copy() for k, (a, b) in enumerate(folds)]
    return _boosters(rec, nodes, leaf, count, num_boost_round, k_models), preds


def train_xgb(params: dict, dtrain: trees.DMatrix, num_boost_round: int,
              folds: list[tuple[int, int]], threads: int = 6,
              boosters: bool = True) -> tuple[list[trees.Booster] | None, list[np.ndarray]]:
    samp_keys = ("subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode", "seed", "nthread",
                 "shira_lowcost")
    samp = {k: params[k] for k in samp_keys if k in params}
    rec = trees.recipe_xgboost({k: v for k, v in params.items() if k not in samp_keys})
    assert 1 <= rec.max_depth <= 10 and 0 < num_boost_round <= 100000 and folds
    codes = dtrain.codes.view(np.uint16) if dtrain.codes.dtype == np.int16 else dtrain.codes.astype(np.uint16)
    codes = np.ascontiguousarray(codes)
    y = np.ascontiguousarray(dtrain.y, dtype=np.float32)
    n, nf = codes.shape
    nbf = np.ascontiguousarray(np.diff(dtrain.ip) - 1, dtype=np.int32)
    assert nbf.size == nf
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    k_models, maxn = len(folds), 1 << (rec.max_depth + 1)
    obj = {"pseudohuber": 0, "logistic": 1}[rec.objective]
    rec_d = np.array([rec.lam, rec.min_hess, float(params.get("gamma", 0.0)), rec.eta, rec.base_score,
                      float(samp.get("subsample", 1.0)), float(samp.get("colsample_bytree", 1.0)),
                      float(samp.get("colsample_bylevel", 1.0)), float(samp.get("colsample_bynode", 1.0))],
                     dtype=np.float64)
    rec_i = np.array([rec.max_depth, obj, int(samp.get("seed", 0)) & 0xFFFFFFFF,
                      int(samp.get("nthread", threads)), int(samp.get("shira_lowcost", 0))], dtype=np.int64)
    nodes = np.zeros((k_models, num_boost_round, maxn, 5), dtype=np.int32)
    leaf = np.zeros((k_models, num_boost_round, maxn), dtype=np.float32)
    count = np.zeros((k_models, num_boost_round), dtype=np.int32)
    pred = np.empty((k_models, n), dtype=np.float32)
    rc = _lib().shira_train_xgb(codes.ctypes.data, n, nf, nbf.ctypes.data, y.ctypes.data,
                                lo.ctypes.data, hi.ctypes.data, k_models, rec_d.ctypes.data,
                                rec_i.ctypes.data, num_boost_round, threads, nodes.ctypes.data,
                                leaf.ctypes.data, count.ctypes.data, pred.ctypes.data)
    assert rc == 0, f"shira_train_xgb failed rc={rc}"
    preds = [pred[k, a:b] for k, (a, b) in enumerate(folds)]
    out = _boosters(rec, nodes, leaf, count, num_boost_round, k_models) if boosters else None
    return out, preds


_GROWTH = {"depthwise": 0, "lossguide": 1, "symmetric": 2}


def _core_lib() -> ctypes.CDLL:
    lib = _lib()
    if not hasattr(lib, "_core_ready"):
        lib.shira_grow_batch.restype = ctypes.c_int
        lib.shira_grow_batch.argtypes = [_P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, _P, _P, _P, _P,
                                         _P, ctypes.c_int32, _P, _P, ctypes.c_int32, ctypes.c_int32,
                                         _P, _P, _P, _P]
        lib._core_ready = True
    return lib


def grow_batch(codes: np.ndarray, n_bins: int, rec, jobs: list[tuple[int, int, np.ndarray, np.ndarray]],
               threads: int = 6, opts: dict | None = None) -> list[tuple[list, np.ndarray]]:
    assert rec.growth in _GROWTH and jobs
    codes = np.ascontiguousarray(codes.view(np.uint16) if codes.dtype == np.int16 else codes.astype(np.uint16))
    n, nf = codes.shape
    if rec.growth == "lossguide":
        assert rec.max_leaves > 0
        maxn = 2 * rec.max_leaves + 2
    else:
        assert 1 <= rec.max_depth <= 16
        maxn = (1 << (rec.max_depth + 1)) + 2
    sizes = np.array([hi - lo for lo, hi, _, _ in jobs], dtype=np.int64)
    goff = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    g = np.ascontiguousarray(np.concatenate([np.asarray(j[2], dtype=np.float64) for j in jobs]))
    h = np.ascontiguousarray(np.concatenate([np.asarray(j[3], dtype=np.float64) for j in jobs]))
    lo = np.array([j[0] for j in jobs], dtype=np.int64)
    hi = np.array([j[1] for j in jobs], dtype=np.int64)
    rec_d, rec_i = _core_rec(rec)
    k = len(jobs)
    nodes = np.zeros((k, maxn, 5), dtype=np.int32)
    leaf = np.zeros((k, maxn), dtype=np.float64)
    count = np.zeros(k, dtype=np.int32)
    row_leaf = np.zeros(int(sizes.sum()), dtype=np.int32)
    if opts:
        oi = np.array([int(opts.get("max_features", 0)), int(opts.get("min_samples_split", 0))], dtype=np.int32)
        sd = np.ascontiguousarray(np.broadcast_to(np.asarray(opts.get("seeds", 1), dtype=np.uint32), (k,)))
        lib = _core_lib()
        lib.shira_grow_batch_opts.restype = ctypes.c_int
        lib.shira_grow_batch_opts.argtypes = [_P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32] + [_P] * 5 + [
            ctypes.c_int32, _P, _P, _P, _P, ctypes.c_int32, ctypes.c_int32, _P, _P, _P, _P]
        rc = lib.shira_grow_batch_opts(codes.ctypes.data, n, nf, int(n_bins), g.ctypes.data, h.ctypes.data,
                                       lo.ctypes.data, hi.ctypes.data, goff.ctypes.data, k, rec_d.ctypes.data,
                                       rec_i.ctypes.data, oi.ctypes.data, sd.ctypes.data, maxn, threads,
                                       nodes.ctypes.data, leaf.ctypes.data, count.ctypes.data, row_leaf.ctypes.data)
    else:
        rc = _core_lib().shira_grow_batch(codes.ctypes.data, n, nf, int(n_bins), g.ctypes.data, h.ctypes.data,
                                          lo.ctypes.data, hi.ctypes.data, goff.ctypes.data, k, rec_d.ctypes.data,
                                          rec_i.ctypes.data, maxn, threads, nodes.ctypes.data, leaf.ctypes.data,
                                          count.ctypes.data, row_leaf.ctypes.data)
    assert rc == 0, f"shira_grow_batch failed rc={rc}"
    out = []
    for j in range(k):
        tree = []
        for v in range(int(count[j])):
            f, cut, d, lft, rgt = (int(x) for x in nodes[j, v])
            tree.append(trees.Node(leaf=float(leaf[j, v])) if f < 0 else
                        trees.Node(feature=f, cut=cut, dir=d, left=lft, right=rgt))
        out.append((tree, row_leaf[goff[j]:goff[j] + sizes[j]]))
    return out


def _core_rec(rec) -> tuple[np.ndarray, np.ndarray]:
    rec_d = np.array([rec.lam, rec.min_hess, rec.min_gain, rec.eta, rec.leaf_eps], dtype=np.float64)
    rec_i = np.array([_GROWTH[rec.growth], rec.max_depth, rec.max_leaves, int(rec.zero_below_mcw),
                      int(rec.midpoint), int(rec.min_count), {"true": 0, "lgb": 1}[rec.count_rule]],
                     dtype=np.int32)
    return rec_d, rec_i


def _maxn(rec) -> int:
    if rec.growth == "lossguide":
        assert rec.max_leaves > 0
        return 2 * rec.max_leaves + 2
    assert 1 <= rec.max_depth <= 16
    return (1 << (rec.max_depth + 1)) + 2


LOSSES = {"squared": 0, "squared_f32": 1, "pseudohuber": 2, "lgb_binary": 3, "hgb_binary": 4,
          "gbc_logloss": 5, "cat_logloss": 6}


def boost_batch(codes: np.ndarray, n_bins: int, rec, y: np.ndarray, folds: list[tuple[int, int]],
                loss: str, rounds: int, base=0.0, threads: int = 6, trees_out: bool = True,
                leaf_eta: float = 1.0):
    assert folds and rounds > 0 and loss in LOSSES
    lib = _core_lib()
    if not hasattr(lib, "_boost_ready"):
        lib.shira_boost_batch.restype = ctypes.c_int
        lib.shira_boost_batch.argtypes = [_P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, _P, _P, _P, _P,
                                          ctypes.c_int32, _P, _P, ctypes.c_int32, ctypes.c_double, ctypes.c_int32,
                                          ctypes.c_int32, ctypes.c_int32, _P, _P, _P, _P]
        lib._boost_ready = True
    codes = np.ascontiguousarray(codes.view(np.uint16) if codes.dtype == np.int16 else codes.astype(np.uint16))
    n, nf = codes.shape
    yd = np.ascontiguousarray(y, dtype=np.float64)
    maxn, k = _maxn(rec), len(folds)
    rec_d, rec_i = _core_rec(rec)
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    bases = np.ascontiguousarray(np.broadcast_to(np.asarray(base, dtype=np.float64), (k,)))
    nodes = np.zeros((k, rounds, maxn, 5), dtype=np.int32)
    leaf = np.zeros((k, rounds, maxn), dtype=np.float64)
    count = np.zeros((k, rounds), dtype=np.int32)
    pred = np.zeros(int((hi - lo).sum()), dtype=np.float64)
    rc = lib.shira_boost_batch(codes.ctypes.data, n, nf, int(n_bins), yd.ctypes.data, lo.ctypes.data,
                               hi.ctypes.data, bases.ctypes.data, k, rec_d.ctypes.data, rec_i.ctypes.data,
                               LOSSES[loss], float(leaf_eta), rounds, maxn, threads, nodes.ctypes.data,
                               leaf.ctypes.data,
                               count.ctypes.data, pred.ctypes.data)
    assert rc == 0, f"shira_boost_batch failed rc={rc}"
    offs = np.concatenate([[0], np.cumsum(hi - lo)])
    preds = [pred[offs[j]:offs[j + 1]] for j in range(k)]
    if not trees_out:
        return None, preds
    out = []
    for j in range(k):
        ts = []
        for r in range(rounds):
            tree = []
            for v in range(int(count[j, r])):
                f, cut, d, lft, rgt = (int(x) for x in nodes[j, r, v])
                tree.append(trees.Node(leaf=float(leaf[j, r, v])) if f < 0 else
                            trees.Node(feature=f, cut=cut, dir=d, left=lft, right=rgt))
            ts.append(tree)
        out.append(ts)
    return out, preds
