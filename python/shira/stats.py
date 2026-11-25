from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from ._lib import check, f64, lib, ptr
from .linear import Totals, pack, unpack


class WindowTree:

    def __init__(self, blocks: list[Totals], s: int = 16):
        assert blocks and s >= 2, "need blocks and super-block size >= 2"
        self.p, self.s, self.n = blocks[0].sx.size, s, len(blocks)
        self.blocks = list(blocks)
        self._packed = f64(np.stack([pack(b) for b in blocks]))
        self.supers = np.empty((-(-self.n // s), self._packed.shape[1]))
        check(lib().shira_wtree_build(ptr(self._packed), self.n, self.p, s, ptr(self.supers)),
              "shira_wtree_build")

    def _call(self, lo: int, hi: int) -> tuple[np.ndarray, int]:
        out, used = np.empty(self._packed.shape[1]), np.zeros(1, dtype=np.int32)
        check(lib().shira_wtree_window(ptr(self._packed), ptr(self.supers), self.n, self.p, self.s,
                                       lo, hi, ptr(out), ptr(used)), "shira_wtree_window")
        return out, int(used[0])

    def window(self, lo: int, hi: int) -> Totals:
        return unpack(self._call(lo, hi)[0], self.p)

    def touched(self, lo: int, hi: int) -> int:
        return self._call(lo, hi)[1]


def covariance(t: Totals, ddof: int = 0) -> tuple[np.ndarray, np.ndarray]:
    p = t.sx.size
    tot, mean, cov = pack(t), np.empty(p), np.empty((p, p))
    check(lib().shira_covariance(ptr(tot), p, int(ddof), ptr(mean), ptr(cov)), "shira_covariance")
    return mean, cov


def pca(t: Totals, k: int, ref: np.ndarray) -> dict:
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    mean, comp, ev, evr, noise = np.empty(p), np.empty((k, p)), np.empty(k), np.empty(k), np.empty(1)
    check(lib().shira_pca(ptr(tot), p, int(k), ptr(refa), ptr(mean), ptr(comp), ptr(ev),
                          ptr(evr), ptr(noise)), "shira_pca")
    return {"mean": mean, "components": comp, "explained_variance": ev,
            "explained_variance_ratio": evr, "noise_variance": float(noise[0])}


@dataclass(frozen=True)
class DiagTotals:
    n: np.ndarray
    s1: np.ndarray
    s2: np.ndarray


def class_diag_totals(x: np.ndarray, y: np.ndarray, classes: np.ndarray, ref: np.ndarray) -> DiagTotals:
    x = f64(x)
    n, p = x.shape
    yi = np.ascontiguousarray(y, dtype=np.int64)
    ci = np.ascontiguousarray(classes, dtype=np.int64)
    k = ci.size
    cnt, s1, s2, refa = np.empty(k), np.empty((k, p)), np.empty((k, p)), f64(ref, (p,))
    check(lib().shira_class_diag_totals(ptr(x), ptr(yi), n, p, ptr(ci), k, ptr(refa),
                                        ptr(cnt), ptr(s1), ptr(s2)), "shira_class_diag_totals")
    return DiagTotals(cnt, s1, s2)


def _sum_arrays(arrs: list[np.ndarray]) -> np.ndarray:
    bufs = [f64(a) for a in arrs]
    assert all(b.shape == bufs[0].shape for b in bufs), "parts differ in shape"
    h = lib()
    fn = h.shira_sum_arrays_ptrs
    fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p], ctypes.c_int
    ptrs = (ctypes.c_void_p * len(bufs))(*[b.ctypes.data for b in bufs])
    out = np.empty(bufs[0].shape)
    check(fn(ctypes.addressof(ptrs), len(bufs), bufs[0].size, ptr(out)), "shira_sum_arrays_ptrs")
    return out


def sum_diag(parts: list[DiagTotals]) -> DiagTotals:
    assert parts
    return DiagTotals(_sum_arrays([d.n for d in parts]), _sum_arrays([d.s1 for d in parts]),
                      _sum_arrays([d.s2 for d in parts]))


def gaussian_nb(d: DiagTotals, ref: np.ndarray, var_smoothing: float = 1e-9) -> dict:
    k, p = d.s1.shape
    theta, var, prior, eps = np.empty((k, p)), np.empty((k, p)), np.empty(k), np.empty(1)
    cn, c1, c2, refa = f64(d.n), f64(d.s1), f64(d.s2), f64(ref, (p,))
    check(lib().shira_gaussian_nb(ptr(cn), ptr(c1), ptr(c2), k, p,
                                  ptr(refa), float(var_smoothing), ptr(theta), ptr(var),
                                  ptr(prior), ptr(eps)), "shira_gaussian_nb")
    return {"theta": theta, "var": var, "class_prior": prior, "epsilon": float(eps[0])}


def gaussian_nb_np(x: np.ndarray, yi: np.ndarray, k: int, var_smoothing: float = 1e-9) -> dict:
    x = f64(x)
    n, p = x.shape
    yv = np.ascontiguousarray(yi, dtype=np.int64)
    assert yv.shape == (n,) and k >= 1
    theta, var, cnt, prior, eps = np.empty((k, p)), np.empty((k, p)), np.empty(k), np.empty(k), np.empty(1)
    check(lib().shira_gaussian_nb_np(ptr(x), ptr(yv), n, p, k, float(var_smoothing), ptr(theta), ptr(var),
                                     ptr(cnt), ptr(prior), ptr(eps)), "shira_gaussian_nb_np")
    return {"theta": theta, "var": var, "class_count": cnt, "class_prior": prior, "epsilon": float(eps[0])}


def pca_np(x: np.ndarray, k: int) -> dict:
    x = f64(x)
    n, p = x.shape
    assert 1 <= k <= p and n > 1
    mean, comp, ev, evr, sv, noise = np.empty(p), np.empty((k, p)), np.empty(k), np.empty(k), np.empty(k), np.empty(1)
    check(lib().shira_pca_np(ptr(x), n, p, k, ptr(mean), ptr(comp), ptr(ev), ptr(evr), ptr(sv), ptr(noise)),
          "shira_pca_np")
    return {"mean": mean, "components": comp, "explained_variance": ev, "explained_variance_ratio": evr,
            "singular_values": sv, "noise_variance": float(noise[0])}


def gram_np(x: np.ndarray, raw: bool = True, centred: bool = True) -> dict:
    x = f64(x)
    n, p = x.shape
    mean, var = np.empty(p), np.empty(p)
    g = np.empty((p, p)) if raw else None
    gc = np.empty((p, p)) if centred else None
    check(lib().shira_gram_np(ptr(x), n, p, ptr(mean), ptr(var), ptr(g), ptr(gc)), "shira_gram_np")
    return {"mean": mean, "var": var, "g": g, "gc": gc, "n": n}


def lda_lsqr(class_totals: list[Totals], ref: np.ndarray) -> dict:
    k, p = len(class_totals), class_totals[0].sx.size
    assert k >= 2
    tots = f64(np.stack([pack(t) for t in class_totals]))
    rows = 1 if k == 2 else k
    coef, icpt = np.empty((rows, p)), np.empty(rows)
    means, cov, priors = np.empty((k, p)), np.empty((p, p)), np.empty(k)
    refa = f64(ref, (p,))
    check(lib().shira_lda_lsqr(ptr(tots), k, p, ptr(refa), ptr(coef), ptr(icpt), ptr(means),
                               ptr(cov), ptr(priors)), "shira_lda_lsqr")
    return {"coef": coef, "intercept": icpt, "means": means, "covariance": cov, "priors": priors}
