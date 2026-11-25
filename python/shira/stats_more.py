from __future__ import annotations

import ctypes

import numpy as np

from ._lib import check, f64, lib, ptr
from .linear import Totals, pack
from .stats import pca as _pca

_P = ctypes.c_void_p
_I32, _D, _INT = ctypes.c_int32, ctypes.c_double, ctypes.c_int
_SIGS = {
    "shira_truncated_svd": ([_P, _I32, _I32, _P, _P, _P, _P, _P], _INT),
    "shira_factor_analysis": ([_P, _I32, _I32, _P, _P, _P, _P, _P, _P, _P, _P], _INT),
    "shira_qda": ([_P, _I32, _I32, _P, _I32, _D, _D, _D, _P, _P, _P, _P], _INT),
    "shira_lda_svd": ([_P, _I32, _I32, _P, _D, _P, _P, _P, _P, _P, _P, _P, _P], _INT),
    "shira_lda_eigen": ([_P, _I32, _I32, _P, _D, _P, _P, _P, _P, _P, _P], _INT),
}
_DONE: list[bool] = []


def _l() -> ctypes.CDLL:
    h = lib()
    if not _DONE:
        for name, (args, res) in _SIGS.items():
            fn = getattr(h, name)
            fn.argtypes, fn.restype = args, res
        _DONE.append(True)
    return h


def _stack(class_totals: list[Totals]) -> np.ndarray:
    return f64(np.stack([pack(t) for t in class_totals]))


def pca_full(t: Totals, k: int, ref: np.ndarray) -> dict:
    return _pca(t, k, ref)


def truncated_svd(t: Totals, k: int, ref: np.ndarray) -> dict:
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    comp, sv, ev, evr = np.empty((k, p)), np.empty(k), np.empty(k), np.empty(k)
    check(_l().shira_truncated_svd(ptr(tot), p, int(k), ptr(refa), ptr(comp), ptr(sv), ptr(ev), ptr(evr)),
          "shira_truncated_svd")
    return {"components": comp, "singular_values": sv, "explained_variance": ev,
            "explained_variance_ratio": evr}


def factor_analysis(t: Totals, k: int, ref: np.ndarray, max_iter: int = 1000, tol: float = 1e-2,
                    psi0: np.ndarray | None = None) -> dict:
    p = t.sx.size
    tot, refa, prm = pack(t), f64(ref, (p,)), f64([max_iter, tol])
    psia = None if psi0 is None else f64(psi0, (p,))
    mean, comp, noise, ll = np.empty(p), np.empty((k, p)), np.empty(p), np.empty(max_iter)
    it = np.zeros(1, dtype=np.int32)
    check(_l().shira_factor_analysis(ptr(tot), p, int(k), ptr(refa), ptr(prm), ptr(psia), ptr(mean),
                                     ptr(comp), ptr(noise), ptr(ll), ptr(it)), "shira_factor_analysis")
    n = int(it[0])
    return {"mean": mean, "components": comp, "noise_variance": noise, "loglike": ll[:n].copy(),
            "n_iter": n}


def qda(class_totals: list[Totals], ref: np.ndarray, solver: str = "svd", reg_param: float = 0.0,
        shrinkage: float | None = None, tol: float = 1e-4) -> dict:
    assert solver in ("svd", "eigen")
    k, p = len(class_totals), class_totals[0].sx.size
    tots, refa = _stack(class_totals), f64(ref, (p,))
    means, scal, rot, pri = np.empty((k, p)), np.empty((k, p)), np.empty((k, p, p)), np.empty(k)
    shr = float("nan") if shrinkage is None else float(shrinkage)
    check(_l().shira_qda(ptr(tots), k, p, ptr(refa), 0 if solver == "svd" else 1, float(reg_param), shr,
                         float(tol), ptr(means), ptr(scal), ptr(rot), ptr(pri)), "shira_qda")
    return {"means": means, "scalings": list(scal), "rotations": list(rot), "priors": pri}


def lda_svd(class_totals: list[Totals], ref: np.ndarray, tol: float = 1e-4) -> dict:
    k, p = len(class_totals), class_totals[0].sx.size
    tots, refa = _stack(class_totals), f64(ref, (p,))
    rows, mc, slots = (1 if k == 2 else k), min(k - 1, p), min(k, p)
    coef, icpt, xbar = np.empty((rows, p)), np.empty(rows), np.empty(p)
    means, pri, evr, scal = np.empty((k, p)), np.empty(k), np.empty(mc), np.empty(p * slots)
    ns = np.zeros(1, dtype=np.int32)
    check(_l().shira_lda_svd(ptr(tots), k, p, ptr(refa), float(tol), ptr(coef), ptr(icpt), ptr(xbar),
                             ptr(means), ptr(pri), ptr(evr), ptr(scal), ptr(ns)), "shira_lda_svd")
    r = int(ns[0])
    return {"coef": coef, "intercept": icpt, "xbar": xbar, "means": means, "priors": pri,
            "explained_variance_ratio": evr, "scalings": scal[:p * r].reshape(p, r).copy()}


def lda_eigen(class_totals: list[Totals], ref: np.ndarray, shrinkage: float | None = None) -> dict:
    k, p = len(class_totals), class_totals[0].sx.size
    tots, refa = _stack(class_totals), f64(ref, (p,))
    rows, mc = (1 if k == 2 else k), min(k - 1, p)
    coef, icpt, means = np.empty((rows, p)), np.empty(rows), np.empty((k, p))
    cov, pri, evr = np.empty((p, p)), np.empty(k), np.empty(mc)
    shr = float("nan") if shrinkage is None else float(shrinkage)
    check(_l().shira_lda_eigen(ptr(tots), k, p, ptr(refa), shr, ptr(coef), ptr(icpt), ptr(means), ptr(cov),
                               ptr(pri), ptr(evr)), "shira_lda_eigen")
    return {"coef": coef, "intercept": icpt, "means": means, "covariance": cov, "priors": pri,
            "explained_variance_ratio": evr}
