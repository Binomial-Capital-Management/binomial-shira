from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

import numpy as np

from ._lib import check, f64, lib, ptr


@dataclass(frozen=True)
class Totals:
    w: float
    sx: np.ndarray
    sy: float
    xx: np.ndarray
    xy: np.ndarray
    yy: float
    buf: np.ndarray | None = field(default=None, repr=False, compare=False)


def pack(t: Totals) -> np.ndarray:
    p = t.sx.size
    if t.buf is not None and t.buf.size == 3 + 2 * p + p * p:
        return t.buf
    out = np.concatenate([[t.w, t.sy, t.yy], t.sx, t.xy, np.asarray(t.xx).reshape(-1)])
    assert out.size == lib().shira_totals_len(p)
    return f64(out)


def unpack(a: np.ndarray, p: int) -> Totals:
    assert a.size == 3 + 2 * p + p * p and a.flags.c_contiguous and a.dtype == np.float64
    return Totals(float(a[0]), a[3:3 + p], float(a[1]), a[3 + 2 * p:].reshape(p, p),
                  a[3 + p:3 + 2 * p], float(a[2]), a)


def block_totals(x: np.ndarray, y: np.ndarray, w: np.ndarray, ref: np.ndarray, ref_y: float) -> Totals:
    x = f64(x)
    n, p = x.shape
    y, w, ref = f64(y, (n,)), f64(w, (n,)), f64(ref, (p,))
    out = np.empty(lib().shira_totals_len(p))
    check(lib().shira_block_totals(ptr(x), ptr(y), ptr(w), n, p, ptr(ref), float(ref_y), ptr(out)),
          "shira_block_totals")
    return unpack(out, p)


def block_totals_mt(x: np.ndarray, y: np.ndarray, w: np.ndarray, ref: np.ndarray, ref_y: float,
                    threads: int | None = None) -> Totals:
    x = f64(x)
    n, p = x.shape
    y, w, ref = f64(y, (n,)), f64(w, (n,)), f64(ref, (p,))
    t = def_threads() if threads is None else int(threads)
    assert t >= 1
    out = np.empty(lib().shira_totals_len(p))
    check(lib().shira_block_totals_mt(ptr(x), ptr(y), ptr(w), n, p, ptr(ref), float(ref_y), ptr(out), t),
          "shira_block_totals_mt")
    return unpack(out, p)


def def_threads() -> int:
    from . import _threads

    return _threads.env_budget()


def block_totals_acc(x: np.ndarray, y: np.ndarray, w: np.ndarray, ref: np.ndarray, ref_y: float,
                     threads: int | None = None) -> Totals:
    x = f64(x)
    n, p = x.shape
    y, w, ref = f64(y, (n,)), f64(w, (n,)), f64(ref, (p,))
    t = def_threads() if threads is None else int(threads)
    assert t >= 1
    out = np.empty(lib().shira_totals_len(p))
    check(lib().shira_block_totals_acc_mt(ptr(x), ptr(y), ptr(w), n, p, ptr(ref), float(ref_y), ptr(out), t),
          "shira_block_totals_acc_mt")
    return unpack(out, p)


def diag_totals(x: np.ndarray, w: np.ndarray, ref: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    x = f64(x)
    n, p = x.shape
    w, ref = f64(w, (n,)), f64(ref, (p,))
    sw, s1, s2 = np.empty(1), np.empty(p), np.empty(p)
    check(lib().shira_diag_totals(ptr(x), ptr(w), n, p, ptr(ref), ptr(sw), ptr(s1), ptr(s2)), "shira_diag_totals")
    return float(sw[0]), s1, s2


def sum_totals(blocks: list[Totals]) -> Totals:
    assert blocks, "empty window"
    p = blocks[0].sx.size
    bufs = [pack(b) for b in blocks]
    ptrs = (ctypes.c_void_p * len(bufs))(*[b.ctypes.data for b in bufs])
    out = np.empty(3 + 2 * p + p * p)
    check(lib().shira_sum_totals_ptrs(ctypes.addressof(ptrs), len(bufs), p, ptr(out)), "shira_sum_totals")
    return unpack(out, p)


def fit_ridge(t: Totals, alpha: float, ref: np.ndarray, ref_y: float) -> tuple[np.ndarray, float]:
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    coef, icpt = np.empty(p), np.empty(1)
    check(lib().shira_fit_ridge(ptr(tot), p, float(alpha), ptr(refa), float(ref_y),
                                ptr(coef), ptr(icpt)), "shira_fit_ridge")
    return coef, float(icpt[0])


def fit_ridge_capped(t: Totals, alpha: float, ref: np.ndarray, ref_y: float,
                     limit: float) -> tuple[np.ndarray, float, float]:
    p = t.sx.size
    assert alpha > 0.0 and limit > 0.0
    tot, refa = pack(t), f64(ref, (p,))
    coef, icpt, cond = np.empty(p), np.empty(1), np.empty(1)
    check(lib().shira_fit_ridge_capped(ptr(tot), p, float(alpha), ptr(refa), float(ref_y), float(limit),
                                       ptr(coef), ptr(icpt), ptr(cond)), "shira_fit_ridge_capped")
    return coef, float(icpt[0]), float(cond[0])


def fit_enet(t: Totals, alpha: float, l1_ratio: float, ref: np.ndarray, ref_y: float,
             tol: float = 1e-4, b0: np.ndarray | None = None) -> tuple[np.ndarray, float, int]:
    p = t.sx.size
    coef, icpt, it = np.empty(p), np.empty(1), np.zeros(1, dtype=np.int32)
    b0a = None if b0 is None else f64(b0, (p,))
    tot, refa = pack(t), f64(ref, (p,))
    check(lib().shira_fit_enet(ptr(tot), p, float(alpha), float(l1_ratio), ptr(refa),
                               float(ref_y), float(tol), ptr(b0a), ptr(coef), ptr(icpt), ptr(it)),
          "shira_fit_enet")
    return coef, float(icpt[0]), int(it[0])


def fit_enet_sk(t: Totals, alpha: float, l1_ratio: float, ref: np.ndarray, ref_y: float, tol: float,
                max_iter: int) -> dict:
    h = lib()
    if not hasattr(h, "_enet_sk_ready"):
        h.shira_fit_enet_sk.restype = ctypes.c_int
        h.shira_fit_enet_sk.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_double, ctypes.c_double,
                                        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int32,
                                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_void_p]
        h._enet_sk_ready = True
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    coef, icpt, gap = np.empty(p), np.empty(1), np.empty(1)
    it, conv = np.zeros(1, np.int32), np.zeros(1, np.int32)
    check(h.shira_fit_enet_sk(ptr(tot), p, float(alpha), float(l1_ratio), ptr(refa), float(ref_y), float(tol),
                              int(max_iter), ptr(coef), ptr(icpt), ptr(it), ptr(gap), ptr(conv)), "shira_fit_enet_sk")
    return {"coef": coef, "intercept": float(icpt[0]), "n_iter": int(it[0]), "gap": float(gap[0]),
            "converged": bool(conv[0])}
