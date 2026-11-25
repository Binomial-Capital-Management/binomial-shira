from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from ._lib import check as _check
from ._lib import lib as _lib
from .export import LazyTrees

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shira_cpp import binning_cpp, xgb_cpp

F32 = np.float32
MEAN_RULE = frozenset({"reg:squarederror", "reg:logistic", "binary:logistic", "binary:logitraw",
                       "count:poisson", "reg:tweedie", "reg:gamma"})
NEWTON_RULE = frozenset({"reg:pseudohubererror"})
K_RT_EPS = 1e-6


def _blocks(n: int, threads: int) -> list[tuple[int, int]]:
    assert n >= 0 and threads >= 1
    c, extra = divmod(n, threads)
    out, s = [], 0
    for t in range(threads):
        m = c + (1 if t < extra else 0)
        out.append((s, s + m))
        s += m
    assert s == n
    return out


def _seq_sum(a: np.ndarray) -> float:
    return float(np.cumsum(a, dtype=np.float64)[-1]) if a.size else 0.0


def _mean_rule(y: np.ndarray, w: np.ndarray | None, threads: int) -> F32:
    y64 = y.astype(np.float64)
    if w is None:
        terms = y64 / float(y.size)
    else:
        sum_w = _seq_sum(w.astype(np.float64))
        terms = (y64 / sum_w) * w.astype(np.float64)
    total = 0.0
    for lo, hi in _blocks(y.size, threads):
        total += _seq_sum(terms[lo:hi])
    return F32(total)


def _newton_huber(y: np.ndarray, w: np.ndarray | None, threads: int, slope: float) -> F32:
    s = F32(slope)
    z = (F32(0) - y).astype(F32)
    root = np.sqrt(F32(1) + (z * z) / (s * s)).astype(F32)
    grad = (z / root).astype(F32)
    hess = ((s * s) / ((s * s + z * z).astype(F32) * root)).astype(F32)
    if w is not None:
        grad, hess = (grad * w).astype(F32), (hess * w).astype(F32)
    g_tot = h_tot = 0.0
    for lo, hi in _blocks(y.size, threads):
        g_tot += _seq_sum(grad[lo:hi])
        h_tot += _seq_sum(hess[lo:hi])
    return F32(-g_tot / max(h_tot, K_RT_EPS))


def intercept(objective: str, y: np.ndarray, w: np.ndarray | None, threads: int | None,
              huber_slope: float = 1.0) -> F32 | None:
    assert y.dtype == F32 and y.ndim == 1 and y.size > 0
    assert w is None or (w.dtype == F32 and w.shape == y.shape)
    if threads is None or threads < 1:
        return None
    if objective not in MEAN_RULE and objective not in NEWTON_RULE:
        return None
    out = np.zeros(1, dtype=F32)
    ya = np.ascontiguousarray(y, dtype=F32)
    wa = None if w is None else np.ascontiguousarray(w, dtype=F32)
    _check(_lib().shira_xgb_base_score(0 if objective in MEAN_RULE else 1, ya.ctypes.data,
                                       None if wa is None else wa.ctypes.data, ya.size, int(threads),
                                       float(F32(huber_slope)), out.ctypes.data), "shira_xgb_base_score")
    return F32(out[0])


def cuts(x: np.ndarray, max_bin: int, threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    ptrs, vals, mins = binning_cpp.xgb_cuts(x, int(max_bin), max(1, int(threads)))
    ip, cv = binning_cpp.quantile_cut_layout(ptrs, vals, mins)
    assert ip.size == x.shape[1] + 1 and cv.dtype == F32
    return np.asarray(ip, dtype=np.int64), cv


def codes(x: np.ndarray, ip: np.ndarray, cv: np.ndarray, threads: int = 1) -> np.ndarray:
    nf = ip.size - 1
    ptrs = (ip - np.arange(nf + 1)).astype(np.uint32)
    keep = np.ones(cv.size, dtype=bool)
    keep[ip[:-1]] = False
    out = binning_cpp.xgb_codes(x, ptrs, cv[keep], max(1, int(threads)))
    assert out.shape == x.shape and out.dtype == np.uint16
    return out


def _trees(out: dict) -> LazyTrees:
    count = out["count"][0]
    trained = int(out["rounds"][0]) * int(out["num_class"]) if "rounds" in out else count.size
    assert 0 < trained <= count.size and bool(np.all(count[:trained] > 0))
    lt = LazyTrees(out["nodes"][0, :trained], out["leaf"][0, :trained], count[:trained])
    if "cover" in out:
        lt.cover, lt.gain = out["cover"][0, :trained], out["gain"][0, :trained]
    if "bw" in out:
        lt.bw = out["bw"][0, :trained]
    return lt


def engine_es(params: dict, cd: np.ndarray, ip: np.ndarray, y: np.ndarray, rounds: int,
              weight: np.ndarray | None, evals: tuple | None, esr: int,
              grad_fn=None, group_ptr: np.ndarray | None = None) -> tuple[LazyTrees, int, np.ndarray | None]:
    nbf = np.diff(ip) - 1
    assert nbf.size == cd.shape[1] and rounds >= 1
    threads = max(1, min(64, int(params.get("nthread", os.cpu_count() or 1))))
    if "nthread" not in params:
        from . import _threads

        nthread = _threads.xgb_default_nthread()
        params = {**params, "nthread": nthread}
        threads = _threads.default_threads("xgboost", int(cd.shape[0]), max(1, min(64, nthread)))
    else:
        from . import _threads

        threads = _threads.explicit_threads(threads)
    out = xgb_cpp.train(params, cd, nbf, y, [(0, cd.shape[0])], int(rounds), threads=threads, weight=weight,
                        evals=evals, early_stopping_rounds=int(esr), grad_fn=grad_fn,
                        group_ptr=group_ptr)
    trees = _trees(out)
    n_rounds = len(trees) // int(out["num_class"])
    assert n_rounds * int(out["num_class"]) == len(trees)
    hist = np.array(out["eval"][0, :n_rounds]) if evals is not None else None
    return trees, n_rounds, hist


def engine(params: dict, cd: np.ndarray, ip: np.ndarray, y: np.ndarray, rounds: int,
           weight: np.ndarray | None, evals: tuple | None, esr: int) -> tuple[LazyTrees, int]:
    trees, n_rounds, _ = engine_es(params, cd, ip, y, rounds, weight, evals, esr)
    return trees, n_rounds


ENGINE_KEYS = xgb_cpp.KEYS
engine_base_margin = xgb_cpp.base_margin
eval_codes = xgb_cpp.bin_codes
