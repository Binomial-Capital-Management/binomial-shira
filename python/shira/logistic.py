from __future__ import annotations

import ctypes

import numpy as np

from ._lib import check, f64, lib, ptr

_P = ctypes.c_void_p
_SIG = ([_P, _P, _P, ctypes.c_int64, ctypes.c_int32, ctypes.c_double, ctypes.c_int32, ctypes.c_double,
         _P, _P, _P], ctypes.c_int)
_DONE: list[bool] = []


def _l() -> ctypes.CDLL:
    h = lib()
    if not _DONE:
        h.shira_logreg_l2.argtypes, h.shira_logreg_l2.restype = _SIG
        _DONE.append(True)
    return h


def fit_logistic(x: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0, fit_intercept: bool = True,
                 tol: float = 1e-12) -> dict:
    x = f64(x)
    n, p = x.shape
    y, w = f64(y, (n,)), f64(w, (n,))
    coef, icpt, hyp = np.empty(p), np.empty(1), np.empty(2)
    check(_l().shira_logreg_l2(ptr(x), ptr(y), ptr(w), n, p, float(C), int(bool(fit_intercept)), float(tol),
                               ptr(coef), ptr(icpt), ptr(hyp)), "shira_logreg_l2")
    return {"coef": coef, "intercept": float(icpt[0]), "n_iter": int(hyp[0]), "objective": float(hyp[1])}


_SIG_LB = ([_P, _P, _P, ctypes.c_int64, ctypes.c_int32, ctypes.c_double, ctypes.c_int32, ctypes.c_double,
            ctypes.c_int32, _P, _P, _P], ctypes.c_int)


def fit_logistic_lbfgs(x: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0, fit_intercept: bool = True,
                       tol: float = 1e-4, max_iter: int = 100) -> dict:
    h = _l()
    h.shira_logreg_lbfgs.argtypes, h.shira_logreg_lbfgs.restype = _SIG_LB
    x = f64(x)
    n, p = x.shape
    y, w = f64(y, (n,)), f64(w, (n,))
    coef, icpt, hyp = np.empty(p), np.empty(1), np.empty(3)
    check(h.shira_logreg_lbfgs(ptr(x), ptr(y), ptr(w), n, p, float(C), int(bool(fit_intercept)), float(tol),
                               int(max_iter), ptr(coef), ptr(icpt), ptr(hyp)), "shira_logreg_lbfgs")
    return {"coef": coef, "intercept": float(icpt[0]), "n_iter": int(hyp[0]), "objective": float(hyp[1]),
            "n_fev": int(hyp[2])}


_SIG_FUSED = ([_P, _P, _P, ctypes.c_int64, ctypes.c_int32, ctypes.c_double, ctypes.c_int32, ctypes.c_double,
               ctypes.c_int32, ctypes.c_int32, _P, _P, _P, _P], ctypes.c_int)


def fit_logistic_lbfgs_fused(x: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0, fit_intercept: bool = True,
                             tol: float = 1e-4, max_iter: int = 100, threads: int = 1,
                             start: np.ndarray | None = None) -> dict:
    h = _l()
    h.shira_logreg_lbfgs_fused.argtypes, h.shira_logreg_lbfgs_fused.restype = _SIG_FUSED
    x = f64(x)
    n, p = x.shape
    y, w = f64(y, (n,)), f64(w, (n,))
    x0 = None if start is None else f64(start, (p + (1 if fit_intercept else 0),))
    coef, icpt, hyp = np.empty(p), np.empty(1), np.empty(3)
    check(h.shira_logreg_lbfgs_fused(ptr(x), ptr(y), ptr(w), n, p, float(C), int(bool(fit_intercept)), float(tol),
                                     int(max_iter), int(threads), None if x0 is None else ptr(x0), ptr(coef), ptr(icpt),
                                     ptr(hyp)), "shira_logreg_lbfgs_fused")
    return {"coef": coef, "intercept": float(icpt[0]), "n_iter": int(hyp[0]), "objective": float(hyp[1]),
            "n_fev": int(hyp[2])}


_SIG_MULTI = ([_P, _P, _P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_int32,
               ctypes.c_double, ctypes.c_int32, _P, _P, _P], ctypes.c_int)


def fit_logistic_multi_lbfgs(x: np.ndarray, y: np.ndarray, w: np.ndarray, n_classes: int, C: float = 1.0,
                             fit_intercept: bool = True, tol: float = 1e-4, max_iter: int = 100) -> dict:
    h = _l()
    h.shira_logreg_multi_lbfgs.argtypes, h.shira_logreg_multi_lbfgs.restype = _SIG_MULTI
    x = f64(x)
    n, p = x.shape
    y, w = f64(y, (n,)), f64(w, (n,))
    k = int(n_classes)
    coef, icpt, hyp = np.empty((k, p)), np.empty(k), np.empty(3)
    check(h.shira_logreg_multi_lbfgs(ptr(x), ptr(y), ptr(w), n, p, k, float(C), int(bool(fit_intercept)), float(tol),
                                     int(max_iter), ptr(coef), ptr(icpt), ptr(hyp)), "shira_logreg_multi_lbfgs")
    return {"coef": coef, "intercept": icpt, "n_iter": int(hyp[0]), "objective": float(hyp[1]), "n_fev": int(hyp[2])}


_SIG_MULTI_FUSED = ([_P, _P, _P, ctypes.c_int64, ctypes.c_int32, ctypes.c_int32, ctypes.c_double, ctypes.c_int32,
                     ctypes.c_double, ctypes.c_int32, ctypes.c_int32, _P, _P, _P, _P], ctypes.c_int)


def fit_logistic_multi_lbfgs_fused(x: np.ndarray, y: np.ndarray, w: np.ndarray, n_classes: int, C: float = 1.0,
                                   fit_intercept: bool = True, tol: float = 1e-4, max_iter: int = 100, threads: int = 1,
                                   start: np.ndarray | None = None) -> dict:
    h = _l()
    h.shira_logreg_multi_lbfgs_fused.argtypes, h.shira_logreg_multi_lbfgs_fused.restype = _SIG_MULTI_FUSED
    x = f64(x)
    n, p = x.shape
    y, w = f64(y, (n,)), f64(w, (n,))
    k = int(n_classes)
    x0 = None if start is None else f64(start, (k * (p + (1 if fit_intercept else 0)),))
    coef, icpt, hyp = np.empty((k, p)), np.empty(k), np.empty(3)
    check(h.shira_logreg_multi_lbfgs_fused(ptr(x), ptr(y), ptr(w), n, p, k, float(C), int(bool(fit_intercept)),
                                           float(tol), int(max_iter), int(threads), None if x0 is None else ptr(x0),
                                           ptr(coef), ptr(icpt), ptr(hyp)), "shira_logreg_multi_lbfgs_fused")
    return {"coef": coef, "intercept": icpt, "n_iter": int(hyp[0]), "objective": float(hyp[1]), "n_fev": int(hyp[2])}
