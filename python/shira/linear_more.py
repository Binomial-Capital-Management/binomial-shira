from __future__ import annotations

import ctypes

import numpy as np

from ._lib import check, f64, lib, ptr
from .linear import Totals, pack

_P = ctypes.c_void_p
_I32, _I64, _D = ctypes.c_int32, ctypes.c_int64, ctypes.c_double
_INT = ctypes.c_int
_SIGS = {
    "shira_fit_bayes_ridge": ([_P, _I32, _P, _D, _P, _P, _P, _P, _P], _INT),
    "shira_fit_ard": ([_P, _I32, _P, _D, _P, _P, _P, _P, _P, _P], _INT),
    "shira_fit_linreg": ([_P, _I32, _I32, _I64, _P, _D, _P, _P, _P, _P], _INT),
    "shira_f_regression": ([_P, _I32, _I32, _P, _P, _P], _INT),
    "shira_f_sf1": ([_D, _D], _D),
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


def _nan(v: float | None) -> float:
    return float("nan") if v is None else float(v)


def fit_bayes_ridge(t: Totals, ref: np.ndarray, ref_y: float, max_iter: int = 300, tol: float = 1e-3,
                    alpha_1: float = 1e-6, alpha_2: float = 1e-6, lambda_1: float = 1e-6,
                    lambda_2: float = 1e-6, alpha_init: float | None = None,
                    lambda_init: float | None = None) -> dict:
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    prm = f64([max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, _nan(alpha_init), _nan(lambda_init)])
    coef, icpt, hyp, sigma = np.empty(p), np.empty(1), np.empty(3), np.empty((p, p))
    check(_l().shira_fit_bayes_ridge(ptr(tot), p, ptr(refa), float(ref_y), ptr(prm), ptr(coef),
                                     ptr(icpt), ptr(hyp), ptr(sigma)), "shira_fit_bayes_ridge")
    return {"coef": coef, "intercept": float(icpt[0]), "alpha": float(hyp[0]), "lambda": float(hyp[1]),
            "sigma": sigma, "n_iter": int(hyp[2])}


def fit_ard(t: Totals, ref: np.ndarray, ref_y: float, max_iter: int = 300, tol: float = 1e-3,
            alpha_1: float = 1e-6, alpha_2: float = 1e-6, lambda_1: float = 1e-6,
            lambda_2: float = 1e-6, threshold_lambda: float = 1e4) -> dict:
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    prm = f64([max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, threshold_lambda])
    coef, icpt, lam, hyp, sigma = np.empty(p), np.empty(1), np.empty(p), np.empty(3), np.empty(p * p)
    check(_l().shira_fit_ard(ptr(tot), p, ptr(refa), float(ref_y), ptr(prm), ptr(coef), ptr(icpt),
                             ptr(lam), ptr(hyp), ptr(sigma)), "shira_fit_ard")
    m = int(hyp[2])
    return {"coef": coef, "intercept": float(icpt[0]), "alpha": float(hyp[0]), "lambda": lam,
            "sigma": sigma[:m * m].reshape(m, m).copy(), "n_iter": int(hyp[1])}


def fit_linreg(t: Totals, ref: np.ndarray, ref_y: float, n_rows: int,
               fit_intercept: bool = True) -> dict:
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    coef, icpt, rank, sing = np.empty(p), np.empty(1), np.zeros(1, dtype=np.int32), np.empty(p)
    check(_l().shira_fit_linreg(ptr(tot), p, int(bool(fit_intercept)), int(n_rows), ptr(refa),
                                float(ref_y), ptr(coef), ptr(icpt), ptr(rank), ptr(sing)),
          "shira_fit_linreg")
    return {"coef": coef, "intercept": float(icpt[0]), "rank": int(rank[0]), "singular": sing}


def f_sf_1(f, dof: float) -> np.ndarray:
    fa = np.atleast_1d(np.asarray(f, dtype=np.float64))
    return np.array([_l().shira_f_sf1(float(v), float(dof)) for v in fa])


def f_regression(t: Totals, force_finite: bool = True) -> dict:
    p = t.sx.size
    tot = pack(t)
    r, f, pv = np.empty(p), np.empty(p), np.empty(p)
    check(_l().shira_f_regression(ptr(tot), p, int(bool(force_finite)), ptr(r), ptr(f), ptr(pv)),
          "shira_f_regression")
    return {"r": r, "f": f, "pvalue": pv}


_LARS_SIG = ([_P, _I32, _P, _D, _D, _I32, _I32, _P, _P, _P, _P, _P, _P], _INT)


_LARS_PATH_SIG = ([_P, _I32, _P, _D, _D, _I32, _I32, _P, _P, _P, _P, _P, _P, _I32, _P, _P, _P], _INT)


LARS_DEGENERATE = -4


def fit_lars(t: Totals, ref: np.ndarray, ref_y: float, alpha_min: float = 0.0, method: str = "lasso",
             max_iter: int = 500, path: bool = False) -> dict:
    assert method in ("lasso", "lar")
    h = _l()
    h.shira_fit_lars_path.argtypes, h.shira_fit_lars_path.restype = _LARS_PATH_SIG
    p = t.sx.size
    tot, refa = pack(t), f64(ref, (p,))
    coef, icpt, alpha = np.empty(p), np.empty(1), np.empty(1)
    nit, act, nact = np.zeros(1, np.int32), np.zeros(p, np.int32), np.zeros(1, np.int32)
    cap = int(max_iter) + 1 if path else 0
    alphas, cpath, npath = np.empty(max(cap, 1)), np.empty((max(cap, 1), p)), np.zeros(1, np.int32)
    rc = h.shira_fit_lars_path(ptr(tot), p, ptr(refa), float(ref_y), float(alpha_min), int(method == "lasso"),
                               int(max_iter), ptr(coef), ptr(icpt), ptr(nit), ptr(act), ptr(nact), ptr(alpha),
                               cap, ptr(alphas), ptr(cpath), ptr(npath))
    if rc == LARS_DEGENERATE:
        return {"degenerate": True}
    check(rc, "shira_fit_lars_path")
    out = {"coef": coef, "intercept": float(icpt[0]), "n_iter": int(nit[0]),
           "active": [int(a) for a in act[:int(nact[0])]], "alpha": float(alpha[0])}
    if path:
        k = int(npath[0])
        out["alphas"], out["coef_path"] = alphas[:k].copy(), np.ascontiguousarray(cpath[:k].T)
    return out


def linreg_tsqr(x: np.ndarray, y: np.ndarray, threads: int) -> dict:
    x = f64(x)
    n, p = x.shape
    yv = f64(y, (n,))
    assert threads >= 1 and n > p
    coef, icpt, sing, rank = np.empty(p), np.empty(1), np.empty(p), np.zeros(1, dtype=np.int32)
    colss = np.empty(p)
    check(lib().shira_linreg_tsqr(ptr(x), ptr(yv), n, p, int(threads), ptr(coef), ptr(icpt), ptr(sing), ptr(rank),
                                  ptr(colss)), "shira_linreg_tsqr")
    return {"coef": coef, "intercept": float(icpt[0]), "singular": sing, "rank": int(rank[0]), "colss": colss}


def bayes_ridge_tsqr(x: np.ndarray, y: np.ndarray, threads: int, prm: np.ndarray) -> dict:
    x = f64(x)
    n, p = x.shape
    yv = f64(y, (n,))
    prm = f64(prm, (8,))
    assert threads >= 1 and n > p
    coef, icpt, hyp, sigma = np.empty(p), np.empty(1), np.empty(3), np.empty((p, p))
    mean, colss = np.empty(p), np.empty(p)
    check(lib().shira_bayes_ridge_tsqr(ptr(x), ptr(yv), n, p, int(threads), ptr(prm), ptr(coef), ptr(icpt), ptr(hyp),
                                       ptr(sigma), ptr(mean), ptr(colss)), "shira_bayes_ridge_tsqr")
    return {"coef": coef, "intercept": float(icpt[0]), "alpha": float(hyp[0]), "lambda": float(hyp[1]),
            "n_iter": int(hyp[2]), "sigma": sigma, "mean": mean, "colss": colss}


def qda_tsqr(x: np.ndarray, yi: np.ndarray, k: int, threads: int, reg_param: float, tol: float) -> dict | None:
    x = f64(x)
    n, p = x.shape
    yv = np.ascontiguousarray(yi, dtype=np.int64)
    assert yv.shape == (n,) and k >= 2 and threads >= 1
    means, scal, rot = np.empty((k, p)), np.empty((k, p)), np.empty((k, p, p))
    priors, colss = np.empty(k), np.empty((k, p))
    rc = lib().shira_qda_tsqr(ptr(x), yv.ctypes.data, n, p, k, int(threads), float(reg_param), float(tol), ptr(means),
                              ptr(scal), ptr(rot), ptr(priors), ptr(colss))
    if rc == -2:
        return None
    check(rc, "shira_qda_tsqr")
    return {"means": means, "scalings": list(scal), "rotations": list(rot), "priors": priors, "colss": colss}


def eq_cond_capped(t: Totals, alpha: float, limit: float) -> float:
    p = t.sx.size
    tot, out = pack(t), np.empty(1)
    assert alpha >= 0.0 and p > 0 and limit > 0.0
    check(lib().shira_eq_cond_capped(ptr(tot), p, float(alpha), float(limit), ptr(out)), "shira_eq_cond_capped")
    return float(out[0])


def eq_cond(t: Totals, alpha: float = 0.0) -> float:
    p = t.sx.size
    tot, out = pack(t), np.empty(1)
    assert alpha >= 0.0 and p > 0
    check(lib().shira_eq_cond(ptr(tot), p, float(alpha), ptr(out)), "shira_eq_cond")
    return float(out[0])
