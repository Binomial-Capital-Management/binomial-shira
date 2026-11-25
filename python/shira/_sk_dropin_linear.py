from __future__ import annotations

import functools
from typing import Any

import numpy as np
from sklearn.utils.validation import validate_data as _sk_validate_data


class PreEngineInputError(ValueError):

    def __init__(self, exc: Exception) -> None:
        super().__init__(str(exc))
        self.kind = type(exc).__name__


def validate_data(*args: Any, **kwargs: Any) -> Any:
    try:
        return _sk_validate_data(*args, **kwargs)
    except (ValueError, TypeError) as exc:
        raise PreEngineInputError(exc) from exc

REF_ROWS = 512
EARLY_ROWS = 2048
EARLY_SMALL_CELLS = 1 << 22
EARLY_SAMPLE_CELLS = EARLY_ROWS * 32
EARLY_ROWS_SMALL = 256
EARLY_ROWS_LINREG = 512
PCA_SMALL_CELLS = 2_000_000


def _dense(X: Any) -> str | None:
    if hasattr(X, "tocsr") or hasattr(X, "tocsc"):
        return "sparse X"
    return None


@functools.cache
def _param_defaults(cls: type) -> tuple[tuple[str, Any], ...]:
    d = cls().get_params(deep=False)
    return tuple((k, d[k]) for k in cls._get_param_names())


def _defaults_except(est: Any, free: tuple[str, ...]) -> str | None:
    for k, dv in _param_defaults(type(est)):
        v = getattr(est, k)
        if k not in free and not (v is dv or v == dv):
            return f"parameter {k}={v!r}"
    return None


def _machine() -> str:
    import platform
    import sys
    if sys.platform == "darwin":
        return "mac"
    return "x86" if platform.machine().lower() in ("x86_64", "amd64") else "other"


MACHINE = _machine()
STOCK_BY_MACHINE = {"mac": set(),
               "x86": set()}
STOCK_BY_MACHINE["other"] = set(STOCK_BY_MACHINE["mac"])
COND_LIMIT = {"LinearRegression": 1e7}


def stock_by_machine(est: Any) -> str | None:
    name = type(est).__name__
    if name in STOCK_BY_MACHINE.get(MACHINE, STOCK_BY_MACHINE["other"]):
        if name == "PCA":
            return f"PCA runs stock on {MACHINE}: SHIRA's path measured slower than stock (0.56-0.74x)"
        return f"estimator {name} is not within the accuracy bar of stock on {MACHINE} after its fix pass"
    return None


def _xyw(est: Any, X: Any, y: Any, sw: Any, min_samples: int = 1,
         finite: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = validate_data(est, X, y, dtype=np.float64, y_numeric=True, reset=True,
                         ensure_min_samples=min_samples,
                         ensure_all_finite=finite)
    assert X.ndim == 2 and y.shape[0] == X.shape[0]
    w = np.ones(X.shape[0]) if sw is None else np.asarray(sw, dtype=np.float64)
    return np.ascontiguousarray(X), np.ascontiguousarray(y, dtype=np.float64), w


def _totals(X: np.ndarray, y: np.ndarray, w: np.ndarray, accurate: bool = True) -> tuple:
    from . import linear as sl

    k = min(REF_ROWS, X.shape[0])
    ref, ref_y = X[:k].mean(axis=0), float(y[:k].mean())
    fn = sl.block_totals_acc if accurate else sl.block_totals_mt
    return fn(X, y, w, ref, ref_y), ref, ref_y


def _xyw_lazy(est: Any, X: Any, y: Any, sw: Any, min_samples: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _xyw(est, X, y, sw, min_samples, finite=False)
    except ValueError:
        _xyw(est, X, y, sw, min_samples)
        raise


def _totals_checked(est: Any, X: np.ndarray, y: np.ndarray, w: np.ndarray, accurate: bool = True) -> tuple:
    try:
        return _totals(X, y, w, accurate)
    except AssertionError:
        _finite_check(est, X)
        raise


def _early_exit(X: Any, sample_weight: Any, row: str, max_rows: int = EARLY_ROWS) -> str | None:
    from . import linear as sl
    from ._sk_dropin_more import _scale_span

    if not (isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float64 and X.flags.c_contiguous):
        return None
    rows = EARLY_ROWS
    if X.shape[0] * X.shape[1] <= EARLY_SMALL_CELLS:
        rows = max(EARLY_ROWS_SMALL, min(EARLY_ROWS, EARLY_SAMPLE_CELLS // max(1, X.shape[1])))
    rows = min(rows, max_rows)
    step = max(1, X.shape[0] // rows)
    block = np.ascontiguousarray(X[::step][:rows])
    k = block.shape[0]
    if sample_weight is None:
        w = np.ones(k)
    elif isinstance(sample_weight, np.ndarray) and sample_weight.dtype == np.float64 and sample_weight.ndim == 1:
        w = np.ascontiguousarray(sample_weight[::step][:EARLY_ROWS])
    else:
        return None
    if k < 2 or w.shape[0] != k:
        return None
    if sample_weight is not None and not (np.all(np.isfinite(w)) and np.all(w >= 0.0) and w.sum() > 0.0):
        return None
    sw, s1, s2 = sl.diag_totals(block, w, block.mean(axis=0))
    why = _scale_span(s2 - s1 ** 2 / sw, sw, row)
    return f"early exit: {k} rows sampled every {step}: {why}" if why else None


def x11_guard(est: Any, X: Any, sample_weight: Any = None, row: str | None = None) -> str | None:
    from . import linear as sl
    from ._sk_dropin_more import SCALE_SPAN_BY_CLASS, _scale_span

    row = row or type(est).__name__
    if row not in SCALE_SPAN_BY_CLASS:
        return None
    why = _early_exit(X, sample_weight, row)
    if why:
        return why
    try:
        Xc = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    except (TypeError, ValueError):
        return None
    if Xc.ndim != 2 or Xc.shape[0] < 2 or not np.isfinite(Xc).all():
        return None
    sw, s1, s2 = sl.diag_totals(Xc, np.ones(Xc.shape[0]), Xc[:min(REF_ROWS, Xc.shape[0])].mean(axis=0))
    return _scale_span(s2 - s1 ** 2 / sw, sw, row)


def _span_guard(t: Any, row: str) -> str | None:
    from ._sk_dropin_more import _scale_span

    return _scale_span(np.diag(np.asarray(t.xx)) - np.asarray(t.sx) ** 2 / float(t.w), float(t.w), row)


EQ_COND_LIMIT = 1e5


def eq_cond_guard(t: Any, alpha: float, row: str) -> str | None:
    from . import linear_more as lm

    c = lm.eq_cond_capped(t, alpha, EQ_COND_LIMIT)
    return f"equilibrated cond {c:.1e} > {EQ_COND_LIMIT:.0e} ({row}: collinear columns)" if c > EQ_COND_LIMIT else None


def _finite_check(est: Any, X: np.ndarray) -> None:
    from sklearn.utils.validation import _assert_all_finite

    try:
        _assert_all_finite(X, input_name="X", estimator_name=type(est).__name__)
    except ValueError as exc:
        raise PreEngineInputError(exc) from exc


def _reuse_or(kind: str, est: Any, X: Any, y: Any, check: bool = True) -> tuple[np.ndarray, np.ndarray, Any]:
    from . import _reuse

    shp = np.shape(X)
    fast = len(shp) == 2 and _reuse.enabled(int(shp[0]), int(shp[1]))
    Xv, yv, _ = _xyw(est, X, y, None, finite=not fast) if check else _xyw_lazy(est, X, y, None)
    if not fast:
        return Xv, yv, None
    r = _reused(kind, Xv, yv)
    if r is None and check:
        _finite_check(est, Xv)
    return Xv, yv, r


def _reused(kind: str, X: np.ndarray, y: np.ndarray) -> Any:
    from . import _reuse

    try:
        return _reuse.linreg(X, y, _tsqr_threads()) if kind == "qr" else _reuse.ridge_totals(X, y, _tsqr_threads())
    except (RuntimeError, AssertionError) as exc:
        import logging
        logging.getLogger("shira.sklearn").debug("reuse: miss (%s)", exc)
        return None


def _tsqr_threads() -> int:
    from . import _threads

    return _threads.env_budget()


def fit_linear_regression(est: Any, X: Any, y: Any, sample_weight: Any = None) -> str | None:
    from . import linear_more as lm

    why = _dense(X) or (np.ndim(y) != 1 and "y is not one column") or (est.positive and "positive=True")
    if why:
        return why
    if sample_weight is None and est.fit_intercept:
        why = _early_exit(X, None, "LinearRegression", max_rows=EARLY_ROWS_LINREG)
        if why:
            return why
        X, y, r = _reuse_or("qr", est, X, y)
        if X.shape[0] <= X.shape[1]:
            return "fewer rows than features"
        r = r or lm.linreg_tsqr(X, y, _tsqr_threads())
        cond = float(r["singular"][0] / r["singular"][r["rank"] - 1]) if r["rank"] > 0 else 1.0
        if cond > COND_LIMIT["LinearRegression"]:
            return f"kept condition number {cond:.1e} > {COND_LIMIT['LinearRegression']:.0e} (stock's own error)"
        est.coef_, est.rank_, est.singular_ = r["coef"], r["rank"], r["singular"]
        est.intercept_ = r["intercept"]
        return None
    why = _early_exit(X, sample_weight, "LinearRegression")
    if why:
        return why
    X, y, w = _xyw_lazy(est, X, y, sample_weight)
    if X.shape[0] <= X.shape[1]:
        return "fewer rows than features"
    t, ref, ref_y = _totals_checked(est, X, y, w)
    why = _span_guard(t, "LinearRegression")
    if why:
        return why
    r = lm.fit_linreg(t, ref, ref_y, X.shape[0], fit_intercept=bool(est.fit_intercept))
    est.coef_, est.rank_, est.singular_ = r["coef"], r["rank"], r["singular"]
    est.intercept_ = r["intercept"] if est.fit_intercept else 0.0
    return None


def fit_ridge(est: Any, X: Any, y: Any, sample_weight: Any = None) -> str | None:
    from . import linear as sl

    why = _dense(X) or (np.ndim(y) != 1 and "y is not one column") or (est.positive and "positive=True")
    why = why or (not est.fit_intercept and "fit_intercept=False")
    why = why or (est.solver not in ("auto", "cholesky") and f"solver {est.solver}")
    why = why or (np.ndim(est.alpha) != 0 and "alpha per target")
    if why:
        return why
    row = "Ridge" if sample_weight is None else "Ridge+sample_weight"
    why = _early_exit(X, sample_weight, row)
    if why:
        return why
    if sample_weight is None:
        X, y, tr = _reuse_or("acc", est, X, y, check=False)
        w = np.ones(X.shape[0]) if tr is None else None
    else:
        X, y, w = _xyw_lazy(est, X, y, sample_weight)
        tr = None
    t, ref, ref_y = tr or _totals_checked(est, X, y, w)
    if float(est.alpha) > 0.0:
        coef, icpt, c = sl.fit_ridge_capped(t, float(est.alpha), ref, ref_y, EQ_COND_LIMIT)
        if c > EQ_COND_LIMIT:
            return f"equilibrated cond {c:.1e} > {EQ_COND_LIMIT:.0e} ({row}: collinear columns)"
        why = _span_guard(t, row)
        if why:
            return why
        est.coef_, est.intercept_ = coef, icpt
        est.n_iter_, est.solver_ = None, "cholesky"
        return None
    why = eq_cond_guard(t, float(est.alpha), row)
    if why:
        return why
    why = _span_guard(t, row)
    if why:
        return why
    est.coef_, est.intercept_ = sl.fit_ridge(t, float(est.alpha), ref, ref_y)
    est.n_iter_, est.solver_ = None, "cholesky"
    return None


def _warn_max_iter(n_iter: int, max_iter: int) -> None:
    if n_iter < max_iter:
        return
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model._logistic import _LOGISTIC_SOLVER_CONVERGENCE_MSG

    msg = (f"lbfgs failed to converge after {n_iter} iteration(s) (status=1):\n"
           "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT\n"
           f"\nIncrease the number of iterations to improve the convergence (max_iter={max_iter})."
           "\nYou might also want to scale the data as shown in:\n"
           "    https://scikit-learn.org/stable/modules/preprocessing.html\n" + _LOGISTIC_SOLVER_CONVERGENCE_MSG)
    warnings.warn(msg, ConvergenceWarning, stacklevel=3)


def fit_logistic(est: Any, X: Any, y: Any, sample_weight: Any = None) -> str | None:
    from . import logistic as lg

    why = _dense(X) or _defaults_except(est, ("C", "fit_intercept", "tol", "max_iter", "random_state",
                                              "n_jobs", "verbose"))
    if why or est.verbose:
        return why or "verbose output"
    X, y = validate_data(est, X, y, dtype=np.float64, reset=True)
    classes = np.unique(y)
    if classes.size < 2:
        return f"{classes.size} class"
    w = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if classes.size >= 3:
        r = lg.fit_logistic_multi_lbfgs_fused(np.ascontiguousarray(X), np.searchsorted(classes, y).astype(np.float64), w,
                                              classes.size, C=float(est.C), fit_intercept=bool(est.fit_intercept),
                                              tol=float(est.tol), max_iter=int(est.max_iter), threads=_tsqr_threads())
        est.classes_, est.coef_ = classes, r["coef"]
        est.intercept_ = r["intercept"] if est.fit_intercept else np.zeros(classes.size)
        est.n_iter_ = np.array([r["n_iter"]], dtype=np.int32)
        _warn_max_iter(r["n_iter"], int(est.max_iter))
        return None
    r = lg.fit_logistic_lbfgs_fused(np.ascontiguousarray(X), (y == classes[1]).astype(np.float64), w,
                                    C=float(est.C), fit_intercept=bool(est.fit_intercept), tol=float(est.tol),
                                    max_iter=int(est.max_iter), threads=_tsqr_threads())
    est.classes_ = classes
    est.coef_ = r["coef"].reshape(1, -1)
    est.intercept_ = np.array([r["intercept"] if est.fit_intercept else 0.0])
    est.n_iter_ = np.array([r["n_iter"]], dtype=np.int32)
    _warn_max_iter(r["n_iter"], int(est.max_iter))
    return None


def _pca_solver(est: Any, n: int, p: int, k: int) -> str | None:
    s = est.svd_solver
    if s == "auto":
        if max(n, p) <= 500:
            s = "full"
        elif p <= 1000 and n >= 10 * p:
            s = "covariance_eigh"
        elif 1 <= k < 0.8 * min(n, p):
            s = "randomized"
        else:
            s = "full"
    return s if s in ("full", "covariance_eigh") else None


def _pca_small_guard(X: Any) -> str | None:
    from . import _machine

    shp = np.shape(X)
    if len(shp) != 2:
        return None
    lo, hi = int(_machine.value("pca_stock_min_rows", int(shp[1]))), int(_machine.value("pca_stock_below_rows", int(shp[1])))
    if lo <= int(shp[0]) < hi:
        return f"PCA on {shp[0]} rows x {shp[1]}: stock is faster from {lo} to below {hi} rows on this machine (never slower than stock)"
    return None


def fit_pca(est: Any, X: Any, y: Any = None) -> str | None:
    from . import linear as sl
    from . import stats_more as sm

    why = _pca_small_guard(X)
    if why:
        return why

    small = isinstance(X, np.ndarray) and X.size <= PCA_SMALL_CELLS
    why = stock_by_machine(est) or _dense(X) or _defaults_except(est, ("n_components", "svd_solver", "whiten", "copy", "random_state"))
    why = why or _early_exit(X, None, "PCA", max_rows=EARLY_ROWS_LINREG if small else EARLY_ROWS)
    if why:
        return why
    X = np.ascontiguousarray(validate_data(est, X, dtype=np.float64, reset=True, ensure_all_finite=False))
    n, p = X.shape
    nc = est.n_components
    if not (nc is None or (isinstance(nc, (int, np.integer)) and 1 <= nc <= min(n, p))) or n <= p:
        return f"n_components {nc!r} or fewer rows than features"
    k = min(n, p) if nc is None else int(nc)
    solver = _pca_solver(est, n, p, k)
    if solver is None:
        return f"svd_solver {est.svd_solver} resolves to a randomized / arpack solver"
    ref = X[:min(REF_ROWS, n)].mean(axis=0)
    t = sl.block_totals_acc(X, np.zeros(n), np.ones(n), ref, 0.0)
    why = _span_guard(t, "PCA")
    if why:
        return why
    r = sm.pca_full(t, k, ref)
    r["singular_values"] = np.sqrt(r["explained_variance"] * (n - 1))
    est.mean_, est.components_ = r["mean"], r["components"]
    est.explained_variance_, est.explained_variance_ratio_ = r["explained_variance"], r["explained_variance_ratio"]
    est.singular_values_ = r["singular_values"]
    est.noise_variance_ = r["noise_variance"]
    est.n_components_, est.n_samples_, est._fit_svd_solver = k, n, solver
    return None


def fit_gaussian_nb(est: Any, X: Any, y: Any, sample_weight: Any = None) -> str | None:
    from . import stats as ss

    why = _dense(X) or (sample_weight is not None and "sample_weight") or (est.priors is not None and "priors")
    if why:
        return why
    X, y = validate_data(est, X, y, dtype=np.float64, reset=True)
    classes, yi = np.unique(y, return_inverse=True)
    ref = X[:min(REF_ROWS, X.shape[0])].mean(axis=0)
    d = ss.class_diag_totals(np.ascontiguousarray(X), yi, np.arange(classes.size), ref)
    r = ss.gaussian_nb(d, ref, float(est.var_smoothing))
    est.classes_, est.class_count_ = classes, np.bincount(yi, minlength=classes.size).astype(np.float64)
    est.theta_, est.var_, est.class_prior_, est.epsilon_ = r["theta"], r["var"], r["class_prior"], r["epsilon"]
    return None
