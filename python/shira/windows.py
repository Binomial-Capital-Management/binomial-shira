from __future__ import annotations

import ctypes

import numpy as np

from ._lib import check, f64, lib, ptr

_P = ctypes.c_void_p
_I32, _D, _INT = ctypes.c_int32, ctypes.c_double, ctypes.c_int
_W = [_P, _I32, _I32, _P, _P, _I32]
_SIGS = {
    "shira_block_totals_many": ([_P, _P, _P, _I32, _P, _I32, _P, _D, _I32, _P], _INT),
    "shira_class_totals_many": ([_P, _P, _P, _I32, _P, _I32, _P, _I32, _P, _D, _I32, _P], _INT),
    "shira_class_diag_many": ([_P, _P, _I32, _P, _I32, _P, _I32, _P, _I32, _P], _INT),
    "shira_ridge_windows": (_W + [_D, _P, _D, _I32, _I32, _P, _P], _INT),
    "shira_enet_windows": (_W + [_D, _D, _P, _D, _D, _I32, _I32, _I32, _P, _P, _P], _INT),
    "shira_cov_windows": (_W + [_I32, _I32, _I32, _P, _P], _INT),
    "shira_pca_windows": (_W + [_I32, _P, _I32, _I32, _I32, _P, _P, _P, _P, _P], _INT),
    "shira_gnb_windows": ([_P, _I32, _I32, _I32, _P, _P, _I32, _P, _D, _I32, _I32, _P, _P, _P], _INT),
    "shira_lda_lsqr_windows": ([_P, _I32, _I32, _I32, _P, _P, _I32, _P, _I32, _I32, _P, _P], _INT),
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


def _win(lo, hi):
    lo, hi = np.ascontiguousarray(lo, dtype=np.int32), np.ascontiguousarray(hi, dtype=np.int32)
    assert lo.shape == hi.shape and lo.ndim == 1
    return lo, hi, lo.size


def block_totals_many(x: np.ndarray, y: np.ndarray, w: np.ndarray, starts: np.ndarray, ref: np.ndarray,
                      ref_y: float, threads: int = 1) -> np.ndarray:
    x = f64(x)
    n, p = x.shape
    y, w, ref = f64(y, (n,)), f64(w, (n,)), f64(ref, (p,))
    st = np.ascontiguousarray(starts, dtype=np.int64)
    nb = st.size - 1
    out = np.empty((nb, int(lib().shira_totals_len(p))))
    check(_l().shira_block_totals_many(ptr(x), ptr(y), ptr(w), p, ptr(st), nb, ptr(ref), float(ref_y),
                                       int(threads), ptr(out)), "shira_block_totals_many")
    return out


def class_totals_many(x: np.ndarray, y: np.ndarray, labels: np.ndarray, starts: np.ndarray, classes,
                      ref: np.ndarray, ref_y: float, threads: int = 1) -> np.ndarray:
    x = f64(x)
    n, p = x.shape
    y, ref = f64(y, (n,)), f64(ref, (p,))
    lab, cls = np.ascontiguousarray(labels, dtype=np.int64), np.ascontiguousarray(classes, dtype=np.int64)
    st = np.ascontiguousarray(starts, dtype=np.int64)
    nb, k = st.size - 1, cls.size
    out = np.empty((nb, k, int(lib().shira_totals_len(p))))
    check(_l().shira_class_totals_many(ptr(x), ptr(y), ptr(lab), p, ptr(st), nb, ptr(cls), k, ptr(ref),
                                       float(ref_y), int(threads), ptr(out)), "shira_class_totals_many")
    return out


def class_diag_many(x: np.ndarray, labels: np.ndarray, starts: np.ndarray, classes, ref: np.ndarray,
                    threads: int = 1) -> np.ndarray:
    x = f64(x)
    n, p = x.shape
    ref = f64(ref, (p,))
    lab, cls = np.ascontiguousarray(labels, dtype=np.int64), np.ascontiguousarray(classes, dtype=np.int64)
    st = np.ascontiguousarray(starts, dtype=np.int64)
    nb, k = st.size - 1, cls.size
    out = np.empty((nb, k + 2 * k * p))
    check(_l().shira_class_diag_many(ptr(x), ptr(lab), p, ptr(st), nb, ptr(cls), k, ptr(ref), int(threads),
                                     ptr(out)), "shira_class_diag_many")
    return out


def ridge(blocks: np.ndarray, p: int, lo, hi, alpha: float, ref, ref_y: float, sum_mode: int = 1,
          threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(blocks), f64(ref, (p,))
    coef, icpt = np.empty((nw, p)), np.empty(nw)
    check(_l().shira_ridge_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, float(alpha), ptr(refa),
                                   float(ref_y), int(sum_mode), int(threads), ptr(coef), ptr(icpt)),
          "shira_ridge_windows")
    return coef, icpt


def enet(blocks: np.ndarray, p: int, lo, hi, alpha: float, l1_ratio: float, ref, ref_y: float,
         tol: float = 1e-4, sum_mode: int = 1, warm: bool = False, threads: int = 1):
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(blocks), f64(ref, (p,))
    coef, icpt, it = np.empty((nw, p)), np.empty(nw), np.empty(nw, dtype=np.int32)
    check(_l().shira_enet_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, float(alpha), float(l1_ratio),
                                  ptr(refa), float(ref_y), float(tol), int(sum_mode), int(bool(warm)),
                                  int(threads), ptr(coef), ptr(icpt), ptr(it)), "shira_enet_windows")
    return coef, icpt, it


def covariance(blocks: np.ndarray, p: int, lo, hi, ddof: int = 0, sum_mode: int = 1, threads: int = 1):
    lo, hi, nw = _win(lo, hi)
    b = f64(blocks)
    mean, cov = np.empty((nw, p)), np.empty((nw, p, p))
    check(_l().shira_cov_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, int(ddof), int(sum_mode),
                                 int(threads), ptr(mean), ptr(cov)), "shira_cov_windows")
    return mean, cov


def pca(blocks: np.ndarray, p: int, lo, hi, k: int, ref, sum_mode: int = 1, top_only: bool = True,
        threads: int = 1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(blocks), f64(ref, (p,))
    mean, comp, ev, evr, noise = (np.empty((nw, p)), np.empty((nw, k, p)), np.empty((nw, k)),
                                  np.empty((nw, k)), np.empty(nw))
    check(_l().shira_pca_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, int(k), ptr(refa), int(sum_mode),
                                 int(bool(top_only)), int(threads), ptr(mean), ptr(comp), ptr(ev), ptr(evr),
                                 ptr(noise)), "shira_pca_windows")
    return {"mean": mean, "components": comp, "explained_variance": ev, "explained_variance_ratio": evr,
            "noise_variance": noise}


def gaussian_nb(diag_blocks: np.ndarray, k: int, p: int, lo, hi, ref, var_smoothing: float = 1e-9,
                sum_mode: int = 1, threads: int = 1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(diag_blocks), f64(ref, (p,))
    theta, var, prior = np.empty((nw, k, p)), np.empty((nw, k, p)), np.empty((nw, k))
    check(_l().shira_gnb_windows(ptr(b), b.shape[0], k, p, ptr(lo), ptr(hi), nw, ptr(refa), float(var_smoothing),
                                 int(sum_mode), int(threads), ptr(theta), ptr(var), ptr(prior)),
          "shira_gnb_windows")
    return {"theta": theta, "var": var, "class_prior": prior}


def lda_lsqr(class_blocks: np.ndarray, k: int, p: int, lo, hi, ref, sum_mode: int = 1,
             threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(class_blocks), f64(ref, (p,))
    rows = 1 if k == 2 else k
    coef, icpt = np.empty((nw, rows, p)), np.empty((nw, rows))
    check(_l().shira_lda_lsqr_windows(ptr(b), b.shape[0], k, p, ptr(lo), ptr(hi), nw, ptr(refa), int(sum_mode),
                                      int(threads), ptr(coef), ptr(icpt)), "shira_lda_lsqr_windows")
    return coef, icpt


_I64P = _P
_SIGS_MORE = {
    "shira_bayes_ridge_windows": (_W + [_P, _D, _P, _I32, _I32, _P, _P, _P], _INT),
    "shira_ard_windows": (_W + [_P, _D, _P, _I32, _I32, _P, _P, _P, _P], _INT),
    "shira_linreg_windows": (_W + [_I32, _I64P, _P, _D, _I32, _I32, _P, _P, _P, _P], _INT),
    "shira_freg_windows": (_W + [_I32, _I32, _I32, _P, _P, _P], _INT),
    "shira_lars_windows": (_W + [_P, _D, _D, _I32, _I32, _I32, _I32, _P, _P, _P, _P, _P, _P], _INT),
    "shira_tsvd_windows": (_W + [_I32, _P, _I32, _I32, _P, _P, _P, _P], _INT),
    "shira_fa_windows": (_W + [_I32, _P, _P, _I32, _I32, _P, _P, _P, _P], _INT),
    "shira_qda_windows": ([_P, _I32, _I32, _I32, _P, _P, _I32, _P, _I32, _D, _D, _D, _I32, _I32, _P, _P, _P, _P], _INT),
    "shira_lda_svd_windows": ([_P, _I32, _I32, _I32, _P, _P, _I32, _P, _D, _I32, _I32, _P, _P], _INT),
    "shira_lda_eigen_windows": ([_P, _I32, _I32, _I32, _P, _P, _I32, _P, _D, _I32, _I32, _P, _P], _INT),
    "shira_logreg_lbfgs_windows": ([_P, _P, _P, _I32, _P, _I32, _P, _P, _I32, _D, _I32, _D, _I32, _I32, _P, _P, _P],
                                   _INT),
}
_DONE_MORE: list[bool] = []


def _m() -> ctypes.CDLL:
    h = _l()
    if not _DONE_MORE:
        for name, (args, res) in _SIGS_MORE.items():
            fn = getattr(h, name)
            fn.argtypes, fn.restype = args, res
        _DONE_MORE.append(True)
    return h


def _nan(v):
    return float("nan") if v is None else float(v)


def bayes_ridge(blocks, p, lo, hi, ref, ref_y, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
                lambda_2=1e-6, alpha_init=None, lambda_init=None, sum_mode=1, threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(blocks), f64(ref, (p,))
    prm = f64([max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, _nan(alpha_init), _nan(lambda_init)])
    coef, icpt, hyp = np.empty((nw, p)), np.empty(nw), np.empty((nw, 3))
    check(_m().shira_bayes_ridge_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, ptr(refa), float(ref_y), ptr(prm),
                                         int(sum_mode), int(threads), ptr(coef), ptr(icpt), ptr(hyp)),
          "shira_bayes_ridge_windows")
    return {"coef": coef, "intercept": icpt, "alpha": hyp[:, 0], "lambda": hyp[:, 1], "n_iter": hyp[:, 2].astype(int)}


def ard(blocks, p, lo, hi, ref, ref_y, max_iter=300, tol=1e-3, alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6,
        lambda_2=1e-6, threshold_lambda=1e4, sum_mode=1, threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(blocks), f64(ref, (p,))
    prm = f64([max_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, threshold_lambda])
    coef, icpt, lam, hyp = np.empty((nw, p)), np.empty(nw), np.empty((nw, p)), np.empty((nw, 3))
    check(_m().shira_ard_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, ptr(refa), float(ref_y), ptr(prm),
                                 int(sum_mode), int(threads), ptr(coef), ptr(icpt), ptr(lam), ptr(hyp)),
          "shira_ard_windows")
    return {"coef": coef, "intercept": icpt, "lambda": lam, "alpha": hyp[:, 0], "n_iter": hyp[:, 1].astype(int)}


def linreg(blocks, p, lo, hi, n_rows, ref, ref_y, fit_intercept=True, sum_mode=1, threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa, nr = f64(blocks), f64(ref, (p,)), np.ascontiguousarray(n_rows, dtype=np.int64)
    coef, icpt, rank, sing = np.empty((nw, p)), np.empty(nw), np.empty(nw, np.int32), np.empty((nw, p))
    check(_m().shira_linreg_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, int(bool(fit_intercept)), ptr(nr),
                                    ptr(refa), float(ref_y), int(sum_mode), int(threads), ptr(coef), ptr(icpt),
                                    ptr(rank), ptr(sing)), "shira_linreg_windows")
    return {"coef": coef, "intercept": icpt, "rank": rank, "singular": sing}


def f_regression(blocks, p, lo, hi, force_finite=True, sum_mode=1, threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b = f64(blocks)
    r, f, pv = np.empty((nw, p)), np.empty((nw, p)), np.empty((nw, p))
    check(_m().shira_freg_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, int(bool(force_finite)),
                                  int(sum_mode), int(threads), ptr(r), ptr(f), ptr(pv)), "shira_freg_windows")
    return {"r": r, "f": f, "pvalue": pv}


def lars(blocks, p, lo, hi, ref, ref_y, alpha_min=0.0, method="lasso", max_iter=500, sum_mode=1, threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(blocks), f64(ref, (p,))
    coef, icpt, nit = np.empty((nw, p)), np.empty(nw), np.empty(nw, np.int32)
    act, nact, al = np.empty((nw, p), np.int32), np.empty(nw, np.int32), np.empty(nw)
    check(_m().shira_lars_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, ptr(refa), float(ref_y),
                                  float(alpha_min), int(method == "lasso"), int(max_iter), int(sum_mode), int(threads),
                                  ptr(coef), ptr(icpt), ptr(nit), ptr(act), ptr(nact), ptr(al)), "shira_lars_windows")
    return {"coef": coef, "intercept": icpt, "n_iter": nit, "active": [list(a[:k]) for a, k in zip(act, nact)],
            "alpha": al}


def truncated_svd(blocks, p, lo, hi, k, ref, sum_mode=1, threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(blocks), f64(ref, (p,))
    comp, sv, ev, evr = np.empty((nw, k, p)), np.empty((nw, k)), np.empty((nw, k)), np.empty((nw, k))
    check(_m().shira_tsvd_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, int(k), ptr(refa), int(sum_mode),
                                  int(threads), ptr(comp), ptr(sv), ptr(ev), ptr(evr)), "shira_tsvd_windows")
    return {"components": comp, "singular_values": sv, "explained_variance": ev, "explained_variance_ratio": evr}


def factor_analysis(blocks, p, lo, hi, k, ref, max_iter=1000, tol=1e-2, sum_mode=1, threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa, prm = f64(blocks), f64(ref, (p,)), f64([max_iter, tol])
    mean, comp, noise, nit = np.empty((nw, p)), np.empty((nw, k, p)), np.empty((nw, p)), np.empty(nw, np.int32)
    check(_m().shira_fa_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, int(k), ptr(refa), ptr(prm),
                                int(sum_mode), int(threads), ptr(mean), ptr(comp), ptr(noise), ptr(nit)),
          "shira_fa_windows")
    return {"mean": mean, "components": comp, "noise_variance": noise, "n_iter": nit}


def qda(class_blocks, k, p, lo, hi, ref, solver="svd", reg_param=0.0, shrinkage=None, tol=1e-4, sum_mode=1,
        threads=1) -> dict:
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(class_blocks), f64(ref, (p,))
    means, scal, rot, pri = np.empty((nw, k, p)), np.empty((nw, k, p)), np.empty((nw, k, p, p)), np.empty((nw, k))
    check(_m().shira_qda_windows(ptr(b), b.shape[0], k, p, ptr(lo), ptr(hi), nw, ptr(refa), 0 if solver == "svd" else 1,
                                 float(reg_param), _nan(shrinkage), float(tol), int(sum_mode), int(threads),
                                 ptr(means), ptr(scal), ptr(rot), ptr(pri)), "shira_qda_windows")
    return {"means": means, "scalings": scal, "rotations": rot, "priors": pri}


def lda_svd(class_blocks, k, p, lo, hi, ref, tol=1e-4, sum_mode=1, threads=1):
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(class_blocks), f64(ref, (p,))
    rows = 1 if k == 2 else k
    coef, icpt = np.empty((nw, rows, p)), np.empty((nw, rows))
    check(_m().shira_lda_svd_windows(ptr(b), b.shape[0], k, p, ptr(lo), ptr(hi), nw, ptr(refa), float(tol),
                                     int(sum_mode), int(threads), ptr(coef), ptr(icpt)), "shira_lda_svd_windows")
    return coef, icpt


def lda_eigen(class_blocks, k, p, lo, hi, ref, shrinkage=None, sum_mode=1, threads=1):
    lo, hi, nw = _win(lo, hi)
    b, refa = f64(class_blocks), f64(ref, (p,))
    rows = 1 if k == 2 else k
    coef, icpt = np.empty((nw, rows, p)), np.empty((nw, rows))
    check(_m().shira_lda_eigen_windows(ptr(b), b.shape[0], k, p, ptr(lo), ptr(hi), nw, ptr(refa), _nan(shrinkage),
                                       int(sum_mode), int(threads), ptr(coef), ptr(icpt)), "shira_lda_eigen_windows")
    return coef, icpt


def logistic_lbfgs(x, y, w, starts, lo, hi, C=1.0, fit_intercept=True, tol=1e-4, max_iter=100, threads=1):
    x = f64(x)
    n, p = x.shape
    y, w = f64(y, (n,)), f64(w, (n,))
    st = np.ascontiguousarray(starts, dtype=np.int64)
    lo, hi, nw = _win(lo, hi)
    coef, icpt, nit = np.empty((nw, p)), np.empty(nw), np.empty(nw, np.int32)
    check(_m().shira_logreg_lbfgs_windows(ptr(x), ptr(y), ptr(w), p, ptr(st), st.size - 1, ptr(lo), ptr(hi), nw,
                                          float(C), int(bool(fit_intercept)), float(tol), int(max_iter), int(threads),
                                          ptr(coef), ptr(icpt), ptr(nit)), "shira_logreg_lbfgs_windows")
    return {"coef": coef, "intercept": icpt, "n_iter": nit}


def ridge_path(blocks, p, lo, hi, alphas, ref, ref_y, sum_mode=1, threads=1):
    h = _l()
    h.shira_ridge_path_windows.argtypes = _W + [_P, _I32, _P, _D, _I32, _I32, _P, _P]
    h.shira_ridge_path_windows.restype = _INT
    lo, hi, nw = _win(lo, hi)
    b, refa, al = f64(blocks), f64(ref, (p,)), f64(alphas)
    coef, icpt = np.empty((nw, al.size, p)), np.empty((nw, al.size))
    check(h.shira_ridge_path_windows(ptr(b), b.shape[0], p, ptr(lo), ptr(hi), nw, ptr(al), al.size, ptr(refa),
                                     float(ref_y), int(sum_mode), int(threads), ptr(coef), ptr(icpt)),
          "shira_ridge_path_windows")
    return coef, icpt


_SIG_LOCK = ([_P, _P, _P, _I32, _P, _P, _I32, _D, _I32, _D, _I32, _I32, _I32, _P, _P, _P, _P], _INT)


def logistic_lbfgs_lockstep(x, y, w, row_lo, row_hi, C=1.0, fit_intercept=True, tol=1e-4, max_iter=100, threads=1,
                            mode=0) -> dict:
    h = _l()
    h.shira_logreg_lbfgs_lockstep.argtypes, h.shira_logreg_lbfgs_lockstep.restype = _SIG_LOCK
    x = f64(x)
    n, p = x.shape
    y, w = f64(y, (n,)), f64(w, (n,))
    rl = np.ascontiguousarray(row_lo, dtype=np.int64)
    rh = np.ascontiguousarray(row_hi, dtype=np.int64)
    nf = int(rl.size)
    assert rh.size == nf and nf > 0, "row_lo / row_hi must have one entry per fold"
    assert int(rl.min()) >= 0 and int(rh.max()) <= n, "fold rows outside x"
    coef, icpt = np.empty((nf, p)), np.empty(nf)
    nit, info = np.empty(nf, np.int32), np.zeros((nf, 4), np.int32)
    check(h.shira_logreg_lbfgs_lockstep(ptr(x), ptr(y), ptr(w), p, ptr(rl), ptr(rh), nf, float(C),
                                        int(bool(fit_intercept)), float(tol), int(max_iter), int(threads), int(mode),
                                        ptr(coef), ptr(icpt), ptr(nit), ptr(info)), "shira_logreg_lbfgs_lockstep")
    return {"coef": coef, "intercept": icpt, "n_iter": nit, "info": info}


def _logistic_start(r: dict, K: int, fit_intercept: bool) -> np.ndarray:
    if K == 2:
        return np.append(r["coef"], r["intercept"]) if fit_intercept else np.asarray(r["coef"], dtype=np.float64)
    w = np.ascontiguousarray(np.asarray(r["coef"]).T).ravel()
    return np.concatenate([w, np.asarray(r["intercept"])]) if fit_intercept else w


def logistic_walk(X: np.ndarray, y: np.ndarray, folds: list, C: float = 1.0, fit_intercept: bool = True, tol: float = 1e-4,
                  max_iter: int = 100, warm: bool = True, nthread: int | None = None) -> dict:
    import time

    from . import _threads
    from . import logistic as lg

    assert X.ndim == 2 and y.shape == (X.shape[0],) and len(folds) >= 1 and C > 0
    classes = np.unique(y)
    K = int(classes.size)
    assert K >= 2, "one class in y"
    Xf = np.ascontiguousarray(X, dtype=np.float64)
    yk = np.searchsorted(classes, y).astype(np.float64)
    th = int(nthread or _threads.env_budget())
    out: dict = {"classes": classes, "proba": [], "coef": [], "intercept": [], "n_iter": []}
    start = None
    t0 = time.perf_counter()
    for lo, hi, a, b in folds:
        assert 0 <= lo < hi <= a < b <= X.shape[0], (lo, hi, a, b)
        assert np.unique(yk[lo:hi]).size == K, f"fold [{lo}, {hi}) does not hold every class of y"
        xs, ys, w = Xf[lo:hi], yk[lo:hi], np.ones(hi - lo)
        if K == 2:
            r = lg.fit_logistic_lbfgs_fused(xs, ys, w, C, fit_intercept, tol, max_iter, th, start=start)
        else:
            r = lg.fit_logistic_multi_lbfgs_fused(xs, ys, w, K, C, fit_intercept, tol, max_iter, th, start=start)
        start = _logistic_start(r, K, fit_intercept) if warm else None
        z = Xf[a:b] @ np.atleast_2d(r["coef"]).T + np.atleast_1d(r["intercept"])
        if K == 2:
            p1 = 1.0 / (1.0 + np.exp(-z[:, 0]))
            proba = np.column_stack([1.0 - p1, p1])
        else:
            e = np.exp(z - z.max(axis=1, keepdims=True))
            proba = e / e.sum(axis=1, keepdims=True)
        for key, v in (("proba", proba), ("coef", r["coef"]), ("intercept", r["intercept"]), ("n_iter", r["n_iter"])):
            out[key].append(v)
    out["seconds"] = time.perf_counter() - t0
    return out


TABLE_MAX_LEAVES = 4096
BOOST_FAMILIES = ("xgboost", "lightgbm", "sklearn_gb", "sklearn_hgb")
_MONO = {"xgboost": "monotone_constraints", "lightgbm": "monotone_constraints", "sklearn_gb": "monotonic_cst",
         "sklearn_hgb": "monotonic_cst"}
FOREST_MODELS = ("random_forest", "extra_trees")
SINGLE_TREE_MODELS = ("extra_tree",)
SINGLE_TREE_CLASSIFIERS = ("extra_tree_classifier",)
_LGB_L2 = ("regression", "regression_l2", "l2", "mean_squared_error", "mse")


def _id_array(leaf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert leaf.ndim == 2 and leaf.shape[0] >= 1 and int(leaf.min()) >= 0
    nl = (leaf.max(axis=0) + 1).astype(np.int32)
    assert int(nl.max()) <= 65536, f"{int(nl.max())} leaves in one tree: over the 2-byte leaf id"
    dt = np.uint8 if int(nl.max()) <= 256 else np.uint16
    return np.ascontiguousarray(leaf.T, dtype=dt), nl


def _sk_ids(model, trees: list, Xf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(model.apply(Xf)).reshape(Xf.shape[0], -1).astype(np.int64)
    assert a.shape[1] == len(trees)
    out = np.empty(a.shape, dtype=np.int64)
    for t, est in enumerate(trees):
        leaf = est.tree_.children_left == -1
        lut = np.where(leaf, np.cumsum(leaf) - 1, -1)
        out[:, t] = lut[a[:, t]]
    return _id_array(out)


def _hgb_ids(m, Xf: np.ndarray, th: int) -> tuple[np.ndarray, np.ndarray]:
    from shira_cpp import product_cpp

    preds = [pr[0] for pr in m._predictors]
    assert len(preds) >= 1 and all(len(pr) == 1 for pr in m._predictors), "one tree per iteration (regression)"
    assert not any(bool(pr.nodes["is_categorical"].any()) for pr in preds), "categorical splits are not supported"
    count = np.array([pr.nodes.size for pr in preds], dtype=np.int32)
    nodes = np.zeros((len(preds), int(count.max()), 5), dtype=np.int32)
    for t, pr in enumerate(preds):
        nd, c = pr.nodes, int(count[t])
        nodes[t, :c, 0] = np.where(nd["is_leaf"].astype(bool), -1, nd["feature_idx"])
        nodes[t, :c, 1] = nd["bin_threshold"].astype(np.int32) + 1
        nodes[t, :c, 2] = nd["missing_go_to_left"]
        nodes[t, :c, 3] = nd["left"]
        nodes[t, :c, 4] = nd["right"]
    binned = m._bin_mapper.transform(Xf)
    codes = binned.astype(np.uint16) + 1
    codes[binned == m._bin_mapper.missing_values_bin_idx_] = 0
    return product_cpp.leaf_ids(np.ascontiguousarray(codes), nodes, count, nthread=th)


def _grow_shapes(Xf: np.ndarray, yf: np.ndarray, lo: int, hi: int, p: dict, th: int,
                 family: str = "xgboost", w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    sw = {} if w is None else {"sample_weight": w[lo:hi]}
    import os
    import warnings

    from shira_cpp import product_cpp

    if family == "lightgbm":
        from . import lightgbm as slg

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = slg.LGBMRegressor(**p).fit(Xf[lo:hi], yf[lo:hi], **sw)
            return _id_array(np.asarray(m.predict(Xf, pred_leaf=True)).reshape(Xf.shape[0], -1).astype(np.int64))
    if family == "sklearn_hgb":
        from . import sklearn as ssk

        saved = os.environ.get("SHIRA_HGB_MODE")
        os.environ["SHIRA_HGB_MODE"] = "ordinary"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _hgb_ids(ssk.HistGradientBoostingRegressor(**p).fit(Xf[lo:hi], yf[lo:hi], **sw), Xf, th)
        finally:
            if saved is None:
                os.environ.pop("SHIRA_HGB_MODE", None)
            else:
                os.environ["SHIRA_HGB_MODE"] = saved
    if family in ("sklearn_gb", *FOREST_MODELS, *SINGLE_TREE_MODELS, *SINGLE_TREE_CLASSIFIERS):
        from . import sklearn as ssk

        cls = {"sklearn_gb": ssk.GradientBoostingRegressor, "random_forest": ssk.RandomForestRegressor,
               "extra_trees": ssk.ExtraTreesRegressor, "extra_tree": ssk.ExtraTreeRegressor,
               "extra_tree_classifier": ssk.ExtraTreeClassifier}[family]
        m = cls(**p).fit(Xf[lo:hi], yf[lo:hi], **sw)
        single = family in SINGLE_TREE_MODELS + SINGLE_TREE_CLASSIFIERS
        trees = [e[0] for e in m.estimators_] if family == "sklearn_gb" else [m] if single \
            else list(m.estimators_)
        return _sk_ids(m, trees, Xf)
    from . import xgboost as sx

    saved = os.environ.get("SHIRA_XGB_CUTS")
    os.environ["SHIRA_XGB_CUTS"] = "stock"
    try:
        m = sx.XGBRegressor(**p).fit(Xf[lo:hi], yf[lo:hi], **sw)
    finally:
        if saved is None:
            os.environ.pop("SHIRA_XGB_CUTS", None)
        else:
            os.environ["SHIRA_XGB_CUTS"] = saved
    sm = m.get_booster().shira_model
    assert sm is not None, "the shape fit ran on stock XGBoost (no SHIRA model)"
    ids, nl = product_cpp.leaf_ids(sm.codes(Xf, clamp=True), sm.trees.nodes, sm.trees.count, nthread=th)
    assert ids.shape == (nl.size, Xf.shape[0])
    return ids, nl


def _leaf_rule(family: str, p: dict) -> tuple[str, float, float, float, float]:
    mono = p.get(_MONO[family])
    flat = list(mono.values()) if isinstance(mono, dict) else (
        [v for v in str(mono).strip("()[] ").split(",") if v.strip()] if isinstance(mono, str) else list(mono or []))
    assert all(float(v) == 0 for v in flat), "monotone constraints: re-solved leaf values would not keep them"
    if family == "sklearn_hgb":
        assert p.get("loss", "squared_error") == "squared_error", "boosted_walk takes HistGradientBoosting's squared error only"
        return "reg:squarederror", float(p.get("learning_rate", 0.1)), float(p.get("l2_regularization", 0.0)), 0.0, 0.0
    if family == "lightgbm":
        assert p.get("objective", "regression") in _LGB_L2, "boosted_walk takes LightGBM's squared error only"
        assert float(p.get("reg_alpha", p.get("lambda_l1", 0.0))) == 0.0, "L1 leaf shrinkage is not supported"
        mds = float(p.get("max_delta_step", 0.0))
        return ("reg:squarederror", float(p.get("learning_rate", 0.1)), float(p.get("reg_lambda", p.get("lambda_l2", 0.0))),
                float(p.get("min_child_weight", p.get("min_sum_hessian_in_leaf", 1e-3))), mds if mds > 0 else 0.0)
    if family == "sklearn_gb":
        assert p.get("loss", "squared_error") == "squared_error", "boosted_walk takes GradientBoosting's squared error only"
        assert float(p.get("subsample", 1.0)) == 1.0, "a row-subsampled GradientBoosting has no fixed-shape value rule here"
        return "reg:squarederror", float(p.get("learning_rate", 0.1)), 0.0, 0.0, 0.0
    obj = p.get("objective", "reg:squarederror")
    return (obj, float(p.get("learning_rate", p.get("eta", 0.3))), float(p.get("reg_lambda", p.get("lambda", 1.0))),
            float(p.get("min_child_weight", 1.0)), float(p.get("max_delta_step", 0.7 if obj == "count:poisson" else 0.0)))


def _family_params(family: str, p: dict) -> str:
    if family != "sklearn_hgb":
        return ""
    asked = p.get("early_stopping", "auto")
    if asked is False:
        return ""
    if asked is True:
        return (f"HistGradientBoosting: early stopping as asked (True) by scikit-learn's own rule in every regrow fit; "
                f"that fit's tree count (at most max_iter = {p.get('max_iter', 100)}) is kept until the next regrow, and "
                f"every fold re-solves the values of those trees on its window")
    p["early_stopping"] = False
    return (f"HistGradientBoosting: early stopping off inside the walk (asked: {asked!r}); every fold re-solves the "
            f"values of all max_iter = {p.get('max_iter', 100)} trees on its window; early_stopping=True turns it on")


def _window_base(family: str, obj: str, y: np.ndarray, th: int, p: dict, w: np.ndarray | None = None) -> float:
    from . import _xgb_fit

    if family == "xgboost":
        b = _xgb_fit.intercept(obj, y, w, th, float(p.get("huber_slope", 1.0)))
        return float(_xgb_fit.engine_base_margin(obj, float(b)))
    if w is None:
        return float(np.float32(np.mean(y, dtype=np.float64)))
    tot = float(np.sum(w, dtype=np.float64))
    assert tot > 0.0, "the window's row weights sum to 0"
    return float(np.float32(np.dot(y.astype(np.float64), w.astype(np.float64)) / tot))


def _move_tables(ids, nl, yf, tab, win, lo, hi, th, pairs=True):
    from shira_cpp import product_cpp

    upd = product_cpp.pair_update if pairs else product_cpp.leaf_sums
    if tab is None or lo < win[0] or hi < win[1] or lo > win[1] or (lo - win[0]) + (hi - win[1]) > hi - lo:
        L = int(nl.sum())
        tab = product_cpp.pair_tables(nl) if pairs else (np.zeros(L), np.zeros(L))
        upd(ids, lo, hi, nl, yf[lo:hi], tab, 1.0, nthread=th)
        return tab
    upd(ids, win[1], hi, nl, yf[win[1]:hi], tab, 1.0, nthread=th)
    upd(ids, win[0], lo, nl, yf[win[0]:lo], tab, -1.0, nthread=th)
    return tab


def _walk_args(X, y, folds, regrow, params, nthread):
    import os

    p = dict(params or {})
    assert X.ndim == 2 and y.shape == (X.shape[0],) and len(folds) >= 1 and regrow >= 0
    Xf = np.ascontiguousarray(X, dtype=np.float32)
    yf = np.ascontiguousarray(y, dtype=np.float32)
    assert np.isfinite(yf).all(), "y has NaN / inf: the leaf sums would be NaN"
    for lo, hi, a, b in folds:
        assert 0 <= lo < hi <= a < b <= X.shape[0], (lo, hi, a, b)
    return p, Xf, yf, int(nthread or p.get("n_jobs") or os.cpu_count() or 1)


def _walk_weights(sample_weight: object, n: int, method: str) -> tuple[np.ndarray | None, str]:
    if sample_weight is None:
        return None, ""
    wf = np.ascontiguousarray(sample_weight, dtype=np.float32)
    assert wf.shape == (n,) and bool(np.isfinite(wf).all()) and bool((wf >= 0).all()), "sample_weight: one finite weight >= 0 per row"
    assert method != "table", "row weights: the table route holds unweighted sums; use method='auto' or 'product'"
    return wf, "row weights: values by the product over the window's rows (no tables)"


def boosted_walk(X: np.ndarray, y: np.ndarray, folds: list, params: dict | None = None, regrow: int = 26,
                 nthread: int | None = None, method: str = "auto", family: str = "xgboost",
                 sample_weight: object = None) -> dict:
    import time

    from shira_cpp import product_cpp

    assert family in BOOST_FAMILIES and method in ("auto", "product", "table"), (family, method)
    p, Xf, yf, th = _walk_args(X, y, folds, regrow, params, nthread)
    wf, note = _walk_weights(sample_weight, X.shape[0], method)
    reason = "; ".join(v for v in (_family_params(family, p), note) if v)
    obj, eta, lam, mcw, mds = _leaf_rule(family, p)
    assert obj in product_cpp.OBJ, f"objective {obj!r}"
    assert method != "table" or obj == "reg:squarederror", "the table route is squared error only"
    preds, ids, nl, tab, win, used = [], None, None, None, (0, 0), method
    t_regrow = t_folds = t_solve = 0.0
    regrows = 0
    for k, (lo, hi, a, b) in enumerate(folds):
        t0 = time.perf_counter()
        grow = ids is None or (regrow > 0 and k % regrow == 0)
        if grow:
            ids, nl = _grow_shapes(Xf, yf, lo, hi, p, th, family, wf)
            used = method if method != "auto" else (
                "table" if wf is None and obj == "reg:squarederror" and int(nl.sum()) <= TABLE_MAX_LEAVES else "product")
            tab, regrows = None, regrows + 1
        bm = _window_base(family, obj, yf[lo:hi], th, p, None if wf is None else wf[lo:hi])
        if used == "table":
            tab = _move_tables(ids, nl, yf, tab, win, lo, hi, th)
            win = (lo, hi)
            t1 = time.perf_counter()
            leaf = product_cpp.pair_solve(tab, nl, bm, eta, lam, mcw, mds)
        else:
            t1 = time.perf_counter()
            leaf = product_cpp.product_window(ids, lo, hi, nl, yf[lo:hi], bm, obj, eta=eta, reg_lambda=lam,
                                              min_child_weight=mcw, max_delta_step=mds,
                                              weight=None if wf is None else wf[lo:hi], nthread=th)
        preds.append(bm + leaf[np.arange(leaf.shape[0])[:, None], ids[:, a:b]].astype(np.float64).sum(axis=0))
        t2 = time.perf_counter()
        if grow:
            t_regrow += t2 - t0
        else:
            t_folds += t2 - t0
            t_solve += t2 - t1
    steps = max(1, len(folds) - regrows)
    return {"pred": preds, "regrows": regrows, "method": used, "t_regrow": t_regrow, "t_folds": t_folds, "t_solve": t_solve,
            "t_fold_ms": 1e3 * t_folds / steps, "reason": reason}


def forest_walk(X: np.ndarray, y: np.ndarray, folds: list, model: str = "random_forest", params: dict | None = None,
                regrow: int = 0, nthread: int | None = None) -> dict:
    import time

    assert model in FOREST_MODELS + SINGLE_TREE_MODELS, model
    p, Xf, yf, th = _walk_args(X, y, folds, regrow, params, nthread)
    msl = p.get("min_samples_leaf", 1)
    assert model in SINGLE_TREE_MODELS or (isinstance(msl, (int, np.integer)) and msl >= 20), \
        "forest_walk needs an integer min_samples_leaf >= 20 for a forest"
    assert p.get("criterion", "squared_error") == "squared_error", "forest_walk takes squared error only"
    if model in FOREST_MODELS:
        p.setdefault("n_jobs", th)
    preds, ids, nl, tab, win, off = [], None, None, None, (0, 0), None
    t_regrow = t_folds = 0.0
    regrows = 0
    for k, (lo, hi, a, b) in enumerate(folds):
        t0 = time.perf_counter()
        grow = ids is None or (regrow > 0 and k % regrow == 0)
        if grow:
            ids, nl = _grow_shapes(Xf, yf, lo, hi, p, th, model)
            off, tab, regrows = np.concatenate([[0], np.cumsum(nl)]).astype(np.int64), None, regrows + 1
        tab = _move_tables(ids, nl, yf, tab, win, lo, hi, th, pairs=False)
        win = (lo, hi)
        base = float(np.mean(yf[lo:hi], dtype=np.float64))
        n_leaf, s_leaf = tab
        val = np.where(n_leaf > 0, s_leaf / np.maximum(n_leaf, 1e-300), base)
        preds.append(val[off[:-1, None] + ids[:, a:b]].mean(axis=0))
        if grow:
            t_regrow += time.perf_counter() - t0
        else:
            t_folds += time.perf_counter() - t0
    steps = max(1, len(folds) - regrows)
    return {"pred": preds, "regrows": regrows, "t_regrow": t_regrow, "t_folds": t_folds, "t_fold_ms": 1e3 * t_folds / steps}


def _class_fractions(tabs: list, ind: list, lo: int, hi: int) -> np.ndarray:
    n_leaf = tabs[0][0]
    base = np.array([float(np.mean(v[lo:hi], dtype=np.float64)) for v in ind])
    assert n_leaf.ndim == 1 and base.size == len(tabs)
    cols = [np.where(n_leaf > 0, t[1] / np.maximum(n_leaf, 1e-300), base[c]) for c, t in enumerate(tabs)]
    return np.stack(cols, axis=1)


def tree_classifier_walk(X: np.ndarray, y: np.ndarray, folds: list, model: str = "extra_tree_classifier",
                         params: dict | None = None, regrow: int = 0, nthread: int | None = None) -> dict:
    import time

    assert model in SINGLE_TREE_CLASSIFIERS, model
    classes = np.unique(y)
    assert classes.size >= 2, "one class in y"
    p, Xf, yf, th = _walk_args(X, np.searchsorted(classes, y).astype(np.float32), folds, regrow, params, nthread)
    ind = [np.ascontiguousarray((yf == c).astype(np.float32)) for c in range(classes.size)]
    proba, ids, nl, tabs, win, regrows = [], None, None, None, (0, 0), 0
    t0 = time.perf_counter()
    for k, (lo, hi, a, b) in enumerate(folds):
        if ids is None or (regrow > 0 and k % regrow == 0):
            ids, nl = _grow_shapes(Xf, yf, lo, hi, p, th, model)
            tabs, win, regrows = [None] * classes.size, (0, 0), regrows + 1
        tabs = [_move_tables(ids, nl, v, t, win, lo, hi, th, pairs=False) for v, t in zip(ind, tabs)]
        win = (lo, hi)
        proba.append(_class_fractions(tabs, ind, lo, hi)[ids[0, a:b]])
    return {"classes": classes, "proba": proba, "regrows": regrows, "seconds": time.perf_counter() - t0}
