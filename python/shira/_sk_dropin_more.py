from __future__ import annotations

from typing import Any

import numpy as np

from ._sk_dropin_linear import (REF_ROWS, validate_data, _dense, _defaults_except, _early_exit, _span_guard,
                                _totals_checked, _xyw_lazy, eq_cond_guard, stock_by_machine, x11_guard)


def _narrow(X: Any) -> str | None:
    dts = [getattr(X, "dtype", None)] if hasattr(X, "dtype") else list(getattr(X, "dtypes", []))
    return "float32 / float16 X (sklearn keeps that dtype)" if any(d in (np.float32, np.float16) for d in dts) else None


def fit_enet(est: Any, X: Any, y: Any, sample_weight: Any = None, check_input: bool = True) -> str | None:
    from . import linear as sl

    why = _dense(X) or _narrow(X) or (np.ndim(y) != 1 and "y is not one column")
    why = why or (sample_weight is not None and "sample_weight")
    why = why or (not est.fit_intercept and "fit_intercept=False") or (est.positive and "positive=True")
    why = why or (est.selection != "cyclic" and f"selection {est.selection}") or (est.warm_start and "warm_start")
    why = why or (est.precompute is not False and f"precompute {est.precompute!r} (sklearn's Gram CD)")
    l1r = float(getattr(est, "l1_ratio", 1.0))
    why = why or (not (float(est.alpha) > 0.0 and 0.0 < l1r <= 1.0) and "alpha <= 0 or l1_ratio outside (0, 1]")
    if why:
        return why
    why = _early_exit(X, None, type(est).__name__)
    if why:
        return why
    X, y, w = _xyw_lazy(est, X, y, None)
    t, ref, ref_y = _totals_checked(est, X, y, w)
    why = _span_guard(t, type(est).__name__)
    if why:
        return why
    r = sl.fit_enet_sk(t, float(est.alpha), l1r, ref, ref_y, float(est.tol), int(est.max_iter))
    if not r["converged"]:
        return f"max_iter {est.max_iter} reached (stock raises its ConvergenceWarning)"
    est.coef_, est.intercept_, est.n_iter_ = r["coef"], np.float64(r["intercept"]), int(r["n_iter"])
    est.dual_gap_ = np.float64(r["gap"] / X.shape[0])
    _draw_like_stock(est.random_state)
    return None


RAND_R_MAX = 2147483647


def _draw_like_stock(random_state: Any) -> None:
    if random_state is None or random_state is np.random:
        np.random.mtrand._rand.randint(0, RAND_R_MAX)
    elif isinstance(random_state, np.random.RandomState):
        random_state.randint(0, RAND_R_MAX)


SCALE_SPAN_MAX = 1e8


INF = float("inf")
SCALE_SPAN_BY_CLASS = {"Ridge": INF, "Ridge+sample_weight": INF, "LinearRegression": 1e7, "Lars": INF,
                       "LassoLars": 1e11, "Lasso": 1e9, "ElasticNet": 1e11, "BayesianRidge": 1e7, "BayesianRidge+sample_weight": 1e3,
                       "ARDRegression": 1e3,
                       "PCA": 1e5, "TruncatedSVD": 1e12,
                       "QuadraticDiscriminantAnalysis": 1e7, "QuadraticDiscriminantAnalysis+eigen": 1e1,
                       "FactorAnalysis": 1e3}


def _scale_span(diag: np.ndarray, n: float, row: str | None = None) -> str | None:
    limit = SCALE_SPAN_BY_CLASS.get(row, SCALE_SPAN_MAX) if row else SCALE_SPAN_MAX
    sd = np.sqrt(np.maximum(np.asarray(diag, dtype=np.float64), 0.0) / float(n))
    live = sd[sd > 0.0]
    if live.size and float(live.max()) > limit * float(live.min()):
        return f"column scales span {float(live.max()) / float(live.min()):.1e} > {limit:.0e} (Gram conditioning)"
    return None


def _bayes_why(est: Any, X: Any) -> str | None:
    return _dense(X) or _narrow(X) or (not est.fit_intercept and "fit_intercept=False") or (
        est.compute_score and "compute_score")


def _bayes_ridge_tsqr(est: Any, X: Any, y: Any) -> str | None:
    from . import linear_more as lm
    from ._sk_dropin_linear import _tsqr_threads, _xyw

    X, y, _ = _xyw(est, X, y, None)
    if X.shape[0] <= X.shape[1]:
        return "fewer rows than features"
    prm = np.array([est.max_iter, est.tol, est.alpha_1, est.alpha_2, est.lambda_1, est.lambda_2,
                    lm._nan(est.alpha_init), lm._nan(est.lambda_init)], dtype=np.float64)
    r = lm.bayes_ridge_tsqr(X, y, _tsqr_threads(), prm)
    why = _scale_span(r["colss"], X.shape[0], "BayesianRidge")
    if why:
        return why
    est.coef_, est.intercept_ = r["coef"], np.float64(r["intercept"])
    est.alpha_, est.lambda_, est.sigma_ = np.float64(r["alpha"]), np.float64(r["lambda"]), r["sigma"]
    est.n_iter_, est.scores_ = int(r["n_iter"]), []
    est.X_offset_, est.X_scale_ = r["mean"], np.ones(X.shape[1])
    return None


def fit_bayes_ridge(est: Any, X: Any, y: Any, sample_weight: Any = None) -> str | None:
    from . import linear_more as lm

    why = _bayes_why(est, X) or (np.ndim(y) != 1 and "y is not one column") or (est.verbose and "verbose")
    if why:
        return why
    row = "BayesianRidge" if sample_weight is None else "BayesianRidge+sample_weight"
    why = _early_exit(X, sample_weight, row)
    if why:
        return why
    if sample_weight is None:
        return _bayes_ridge_tsqr(est, X, y)
    X, y, w = _xyw_lazy(est, X, y, sample_weight)
    t, ref, ref_y = _totals_checked(est, X, y, w)
    why = _span_guard(t, row)
    if why:
        return why
    r = lm.fit_bayes_ridge(t, ref, ref_y, max_iter=int(est.max_iter), tol=float(est.tol),
                           alpha_1=float(est.alpha_1), alpha_2=float(est.alpha_2), lambda_1=float(est.lambda_1),
                           lambda_2=float(est.lambda_2), alpha_init=est.alpha_init, lambda_init=est.lambda_init)
    est.coef_, est.intercept_ = r["coef"], np.float64(r["intercept"])
    est.alpha_, est.lambda_, est.sigma_ = np.float64(r["alpha"]), np.float64(r["lambda"]), r["sigma"]
    est.n_iter_, est.scores_ = int(r["n_iter"]), []
    est.X_offset_, est.X_scale_ = np.asarray(t.sx) / float(t.w) + ref, np.ones(X.shape[1])
    return None


def fit_ard(est: Any, X: Any, y: Any) -> str | None:
    why = stock_by_machine(est) or x11_guard(est, X)
    if why:
        return why
    from . import linear_more as lm

    why = _bayes_why(est, X) or (np.ndim(y) != 1 and "y is not one column") or (est.verbose and "verbose")
    if why:
        return why
    X, y, w = _xyw_lazy(est, X, y, None, min_samples=2)
    t, ref, ref_y = _totals_checked(est, X, y, w)
    why = eq_cond_guard(t, 0.0, "ARDRegression")
    if why:
        return why
    r = lm.fit_ard(t, ref, ref_y, max_iter=int(est.max_iter), tol=float(est.tol), alpha_1=float(est.alpha_1),
                   alpha_2=float(est.alpha_2), lambda_1=float(est.lambda_1), lambda_2=float(est.lambda_2),
                   threshold_lambda=float(est.threshold_lambda))
    est.coef_, est.intercept_ = r["coef"], np.float64(r["intercept"])
    est.alpha_, est.lambda_, est.sigma_ = np.float64(r["alpha"]), r["lambda"], r["sigma"]
    est.n_iter_, est.scores_ = int(r["n_iter"]), []
    est.X_offset_, est.X_scale_ = np.asarray(t.sx) / float(t.w) + ref, np.ones(X.shape[1])
    return None


_PREFIX = 1024


def _fingerprint(U: np.ndarray) -> np.ndarray:
    w = np.random.default_rng(0x5EED).integers(0, 2**63, size=U.shape[0], dtype=np.uint64) | np.uint64(1)
    return U.T.dot(w)


def _rank_deficient_columns(X: np.ndarray) -> str | None:
    assert X.ndim == 2 and X.dtype == np.float64
    n, p = X.shape
    if n == 0 or p < 2:
        return None
    head = X[:min(n, _PREFIX)]
    cand = np.flatnonzero(head.min(axis=0) == head.max(axis=0))
    if cand.size and bool(np.any(X[:, cand].min(axis=0) == X[:, cand].max(axis=0))):
        return "a constant column (LARS degenerate branch; stock decides)"
    fp = _fingerprint(np.ascontiguousarray(head).view(np.uint64))
    order = np.argsort(fp, kind="stable")
    same = np.flatnonzero(fp[order][1:] == fp[order][:-1])
    if same.size == 0:
        return None
    cols = np.unique(np.concatenate([order[same], order[same + 1]]))
    full = _fingerprint(np.ascontiguousarray(X[:, cols]).view(np.uint64))
    by: dict[int, list[int]] = {}
    for c, f in zip(cols.tolist(), full.tolist()):
        by.setdefault(int(f), []).append(int(c))
    for group in by.values():
        for k in range(1, len(group)):
            if np.array_equal(X[:, group[0]].view(np.uint64), X[:, group[k]].view(np.uint64)):
                return "a duplicate column (LARS degenerate branch; stock decides)"
    return None


_DUP_TOL = 1e-9


def _lars_head(X: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]] | None:
    assert X.ndim == 2 and X.dtype == np.float64
    n, p = X.shape
    if n == 0 or p < 2:
        return np.zeros(0, dtype=np.int64), []
    head = X[:min(n, _PREFIX)]
    cand = np.flatnonzero(head.min(axis=0) == head.max(axis=0))
    hb = np.ascontiguousarray(head).view(np.uint64)
    fp = _fingerprint(hb)
    order = np.argsort(fp, kind="stable")
    cuts = np.flatnonzero(fp[order][1:] != fp[order][:-1]) + 1
    groups = [g for g in np.split(order, cuts) if g.size > 1]
    for g in groups:
        if not bool(np.all(hb[:, g] == hb[:, g[:1]])):
            return None
    return cand, groups


def _lars_totals(X: np.ndarray, y: np.ndarray, w: np.ndarray, head: tuple, accurate: bool) -> tuple:
    from . import linear as sl

    cand, groups = head
    k = min(REF_ROWS, X.shape[0])
    ref, ref_y = X[:k].mean(axis=0), float(y[:k].mean())
    ref[cand] = X[0, cand]
    for g in groups:
        ref[g] = ref[g[0]]
    fn = sl.block_totals_acc if accurate else sl.block_totals_mt
    return fn(X, y, w, ref, ref_y), ref, ref_y


def _lars_degenerate(X: np.ndarray, t: Any, head: tuple, lasso_pos: bool) -> tuple[str | None, np.ndarray]:
    cand, groups = head
    xx, sx, xy = np.asarray(t.xx), np.asarray(t.sx), np.asarray(t.xy)
    d = np.diag(xx)
    const = cand[d[cand] == 0.0]
    if const.size and not lasso_pos:
        return "a constant column (LARS degenerate branch; stock decides)", const
    n, yy = float(t.w), max(float(t.yy), 0.0)
    for g in groups:
        live = g[d[g] != 0.0]
        live = live[np.argsort(d[live], kind="stable")]
        top = np.searchsorted(d[live], d[live] * (1.0 + 2 * _DUP_TOL), side="right")
        for i in range(live.size):
            a = int(live[i])
            for b in live[i + 1:top[i]].tolist():
                ok = abs(xx[a, b] - d[a]) <= _DUP_TOL * d[a] and abs(sx[a] - sx[b]) <= _DUP_TOL * np.sqrt(n * d[a])
                ok = ok and abs(xy[a] - xy[b]) <= _DUP_TOL * np.sqrt(d[a] * yy)
                if ok and np.array_equal(X[:, a].view(np.uint64), X[:, b].view(np.uint64)):
                    return "a duplicate column (LARS degenerate branch; stock decides)", const
    return None, const


LASSO_CONST_SAFETY = 1e3


def _lasso_const_bound(X: np.ndarray, t: Any, ref: np.ndarray, ref_y: float, const: np.ndarray, path: np.ndarray,
                       alpha_min: float) -> str | None:
    assert const.size and alpha_min > 0.0
    v = float(np.max(np.abs(X[0, const])))
    if v == 0.0:
        return None
    n, eps = float(t.w), float(np.finfo(np.float64).eps)
    d = np.maximum(np.diag(np.asarray(t.xx)), 0.0)
    xmax = 2.0 * (np.sqrt(d) + np.abs(np.asarray(t.sx)) / n + np.abs(ref))
    ymax = 2.0 * (np.sqrt(max(float(t.yy), 0.0)) + abs(float(t.sy)) / n + abs(ref_y))
    reach = ymax + float(np.max(np.abs(path).T @ xmax)) if path.size else ymax
    bound = LASSO_CONST_SAFETY * n * n * eps * eps * v * reach
    if np.isfinite(bound) and bound < alpha_min:
        return None
    return f"a constant column whose centring residue in stock could reach alpha {alpha_min:g} (bound {bound:.1e})"


def fit_lars(est: Any, X: Any, y: Any, Xy: Any = None) -> str | None:
    from . import linear_more as lm

    lasso = getattr(est, "method", "lar") == "lasso"
    free = ("alpha", "max_iter", "n_nonzero_coefs", "copy_X", "random_state")
    why = _dense(X) or _narrow(X) or (np.ndim(y) != 1 and "y is not one column") or (Xy is not None and "Xy given")
    why = why or _defaults_except(est, free) or (lasso and not float(est.alpha) >= 0.0 and "alpha < 0")
    if why:
        return why
    why = _early_exit(X, None, "LassoLars" if lasso else "Lars")
    if why:
        return why
    X, y = validate_data(est, X, y, dtype=np.float64, y_numeric=True, reset=True, ensure_all_finite=False)
    X, y, w = np.ascontiguousarray(X), np.ascontiguousarray(y, dtype=np.float64), np.ones(X.shape[0])
    if X.shape[0] <= X.shape[1]:
        return "fewer rows than features (sklearn's precompute='auto' takes the X path)"
    head = _lars_head(X)
    if head is None:
        why = _rank_deficient_columns(X)
        if why:
            return why
        head = (np.zeros(0, dtype=np.int64), [])
    pos = lasso and float(est.alpha) > 0.0
    t, ref, ref_y = _lars_totals(X, y, w, head, accurate=not lasso)
    why, const = _lars_degenerate(X, t, head, pos)
    why = why or _span_guard(t, "LassoLars" if lasso else "Lars")
    if why:
        return why
    mi = int(est.max_iter) if lasso else int(est.n_nonzero_coefs)
    r = lm.fit_lars(t, ref, ref_y, alpha_min=float(est.alpha) if lasso else 0.0, method="lasso" if lasso else "lar",
                    max_iter=mi, path=True)
    if r.get("degenerate"):
        return "degenerate regressor in the LARS path (stock decides)"
    why = _lasso_const_bound(X, t, ref, ref_y, const, r["coef_path"], float(est.alpha)) if const.size else None
    if why:
        return why
    est.coef_, est.intercept_, est.n_iter_ = r["coef"], np.float64(r["intercept"]), int(r["n_iter"])
    est.alphas_, est.active_, est.coef_path_ = r["alphas"], list(r["active"]), r["coef_path"]
    return None


def _x_only(est: Any, X: Any) -> np.ndarray:
    return np.ascontiguousarray(validate_data(est, X, dtype=np.float64, reset=True))


def _x_totals(X: np.ndarray) -> tuple[Any, np.ndarray]:
    from . import linear as sl

    n = X.shape[0]
    ref = X[:min(REF_ROWS, n)].mean(axis=0)
    return sl.block_totals_acc(X, np.zeros(n), np.ones(n), ref, 0.0), ref


def _raw_gram(t: Any, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, sx, xx = float(t.w), np.asarray(t.sx), np.asarray(t.xx)
    g = xx + np.outer(ref, sx) + np.outer(sx, ref) + n * np.outer(ref, ref)
    return g, sx + n * ref, xx - np.outer(sx, sx) / n


def _randomized_core(g: np.ndarray, k: int, n_iter: int, n_over: int, rng: Any) -> tuple:
    from scipy import linalg

    p = g.shape[0]
    om = rng.normal(size=(p, k + n_over))
    t_map = None
    if k + n_over >= p:
        s_mat = g
    else:
        q, _ = linalg.qr(om, mode="economic")
        for _ in range(int(n_iter)):
            q, _ = linalg.qr(g @ q, mode="economic")
        gm = g @ q
        sol = linalg.solve(q.T @ gm, gm.T, assume_a="pos")
        s_mat = gm @ sol
        s_mat = 0.5 * (s_mat + s_mat.T)
        t_map = q @ sol
    ev, vec = linalg.eigh(s_mat)
    top = np.argsort(ev)[::-1][:k]
    return np.sqrt(np.maximum(ev[top], 0.0)), vec[:, top].T, t_map


def _randomized_right(g: np.ndarray, k: int, n_iter: int, n_over: int, rng: Any) -> tuple[np.ndarray, np.ndarray]:
    sv, vt, _ = _randomized_core(g, k, n_iter, n_over, rng)
    vt *= np.sign(vt[np.arange(k), np.argmax(np.abs(vt), axis=1)])[:, None]
    return sv, vt


def fit_truncated_svd(est: Any, X: Any, y: Any = None) -> str | None:
    why = stock_by_machine(est) or x11_guard(est, X)
    if why:
        return why
    from sklearn.utils import check_random_state

    from . import stats_more as sm

    why = _dense(X) or _narrow(X) or (est.algorithm not in ("arpack", "randomized") and f"algorithm {est.algorithm!r}")
    why = why or (est.algorithm == "arpack" and float(est.tol) != 0.0 and f"arpack tol {est.tol} (approximate)")
    if why:
        return why
    X = _x_only(est, X)
    k = int(est.n_components)
    n, p = X.shape
    if est.algorithm == "arpack" and not 1 <= k < min(n, p):
        return f"n_components {k} outside arpack's [1, min(n, p))"
    if est.algorithm == "randomized" and not (1 <= k <= p and n >= p and est.n_iter != "auto"):
        return "randomized: n_components > p, fewer rows than features (sklearn transposes) or n_iter='auto'"
    from . import stats as ss

    gr = ss.gram_np(X)
    g, gc = gr["g"], gr["gc"]
    t = ref = None
    why = _scale_span(np.diag(g), n)
    if why:
        return why
    if est.algorithm == "arpack":
        t, ref = _x_totals(X)
        r = sm.truncated_svd(t, k, ref)
        est.components_, est.singular_values_ = r["components"], r["singular_values"]
        est.explained_variance_, est.explained_variance_ratio_ = r["explained_variance"], r["explained_variance_ratio"]
        return None
    rng = check_random_state(est.random_state)
    sv, vt = _randomized_right(g, k, int(est.n_iter), int(est.n_oversamples), rng)
    ev = np.einsum("ij,jk,ik->i", vt, gc, vt) / n
    est.components_, est.singular_values_ = vt, sv
    est.explained_variance_, est.explained_variance_ratio_ = ev, ev / (np.trace(gc) / n)
    return None


def _fa_randomized(est: Any, X: np.ndarray, t: Any, ref: np.ndarray, k: int) -> str | None:
    from math import log, pi, sqrt

    from sklearn.utils import check_random_state

    from . import stats as ss

    n, p = X.shape
    gr = ss.gram_np(X, raw=False)
    gc, mean = gr["gc"], gr["mean"]
    var, nsqrt, small = gr["var"], sqrt(n), 1e-12
    llconst = p * log(2.0 * pi) + k
    rng = check_random_state(est.random_state)
    psi, loglike, old_ll, w = np.ones(p), [], -np.inf, None
    for i in range(int(est.max_iter)):
        sq = np.sqrt(psi) + small
        d = 1.0 / (sq * nsqrt)
        gz = gc * np.outer(d, d)
        sv, vt, t_map = _randomized_core(gz, k, int(est.iterated_power), 10, rng)
        unexp = float(np.trace(gz)) - float(np.sum(sv * sv))
        s2 = sv * sv
        w = np.sqrt(np.maximum(s2 - 1.0, 0.0))[:, None] * vt * sq
        ll = (llconst + np.sum(np.log(s2)) + unexp + np.sum(np.log(psi))) * (-n / 2.0)
        loglike.append(float(ll))
        if (ll - old_ll) < est.tol:
            break
        old_ll = ll
        psi = np.maximum(var - np.sum(w ** 2, axis=0), small)
    else:
        import warnings

        from sklearn.exceptions import ConvergenceWarning

        warnings.warn("FactorAnalysis did not converge. You might want to increase the number of iterations.",
                      ConvergenceWarning)
    a = vt.T if t_map is None else t_map @ vt.T
    u = (X - mean) @ (a * d[:, None])
    sg = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(k)])
    est.components_, est.noise_variance_, est.mean_ = w * sg[:, None], psi, mean
    est.loglike_, est.n_iter_ = loglike, i + 1
    return None


FA_TIED_CELLS = 1e8
FA_TIED_LEVELS = 16
FA_TIED_SAMPLE = 4096


def _fa_large_tied(X: np.ndarray) -> str | None:
    n, p = X.shape
    if float(n) * float(p) < FA_TIED_CELLS:
        return None
    s = np.sort(X[:FA_TIED_SAMPLE], axis=0)
    distinct = 1 + np.count_nonzero(np.diff(s, axis=0), axis=0)
    assert distinct.shape == (p,)
    if np.all(distinct <= FA_TIED_LEVELS):
        return (f"tied data at n x p = {float(n) * float(p):.2g} >= {FA_TIED_CELLS:.0e} (every column <= {FA_TIED_LEVELS} "
                "distinct values in the first rows): stock runs (the 1e-9 rail at this scale)")
    return None


def fit_factor_analysis(est: Any, X: Any, y: Any = None) -> str | None:
    why = stock_by_machine(est) or x11_guard(est, X)
    if why:
        return why
    from . import stats_more as sm

    why = _dense(X) or (est.svd_method not in ("lapack", "randomized") and f"svd_method {est.svd_method!r}")
    why = why or (est.rotation is not None and f"rotation {est.rotation!r}")
    why = why or (est.noise_variance_init is not None and "noise_variance_init")
    if why:
        return why
    X = _x_only(est, X)
    k = X.shape[1] if est.n_components is None else int(est.n_components)
    if not 1 <= k <= X.shape[1] or X.shape[0] <= X.shape[1]:
        return f"n_components {k} or fewer rows than features"
    why = _fa_large_tied(X)
    if why:
        return why
    t, ref = _x_totals(X)
    cv = np.diag(np.asarray(t.xx)) - np.asarray(t.sx) ** 2 / float(t.w)
    why = _scale_span(cv, float(t.w))
    if why:
        return why
    if np.any(cv <= 0.0):
        return "a constant column (stock's EM log-likelihood is -inf; stock runs)"
    why = eq_cond_guard(t, 0.0, "FactorAnalysis")
    if why:
        return why
    if est.svd_method == "randomized":
        return _fa_randomized(est, X, t, ref, k)
    r = sm.factor_analysis(t, k, ref, max_iter=int(est.max_iter), tol=float(est.tol))
    if r["n_iter"] >= int(est.max_iter):
        return "engine reached max_iter (stock's non-converged iterate decides)"
    est.components_, est.noise_variance_, est.mean_ = r["components"], r["noise_variance"], r["mean"]
    est.loglike_, est.n_iter_ = [float(v) for v in r["loglike"]], int(r["n_iter"])
    est.components_ = _lapack_signs(X, est.mean_, est.noise_variance_, est.components_)
    return None


def _lapack_signs(X: np.ndarray, mean: np.ndarray, psi: np.ndarray, w: np.ndarray) -> np.ndarray:
    from math import sqrt

    from scipy import linalg

    z = (X - mean) / ((np.sqrt(psi) + 1e-12) * sqrt(X.shape[0]))
    _, _, vt = linalg.svd(z, full_matrices=False, check_finite=False)
    sg = np.sign(np.einsum("ij,ij->i", w, vt[: w.shape[0]]))
    sg[sg == 0] = 1.0
    return w * sg[:, None]


def _classes(est: Any, X: Any, y: Any, min_samples: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
    from . import linear as sl

    X, y = validate_data(est, X, y, dtype=np.float64, reset=True, ensure_min_samples=min_samples)
    X = np.ascontiguousarray(X)
    classes, yi = np.unique(y, return_inverse=True)
    ref = X[:min(REF_ROWS, X.shape[0])].mean(axis=0)
    cts = []
    for c in range(classes.size):
        xc = np.ascontiguousarray(X[yi == c])
        nc = xc.shape[0]
        cts.append(sl.block_totals_acc(xc, np.zeros(nc), np.ones(nc), ref, 0.0))
    return X, classes, yi, cts, ref


def _shrunk_total_cov(cts: list, shrinkage: float) -> np.ndarray:
    w = sum(float(t.w) for t in cts)
    sx = sum(np.asarray(t.sx) for t in cts)
    xx = sum(np.asarray(t.xx) for t in cts)
    emp = (xx - np.outer(sx, sx) / w) / w
    mu = np.trace(emp) / emp.shape[0]
    out = (1.0 - shrinkage) * emp
    out.flat[:: emp.shape[0] + 1] += shrinkage * mu
    return out


def fit_lda(est: Any, X: Any, y: Any) -> str | None:
    why = stock_by_machine(est) or x11_guard(est, X)
    if why:
        return why
    from scipy import linalg

    from . import stats as ss
    from . import stats_more as sm

    why = _dense(X) or _narrow(X) or _defaults_except(est, ("solver", "shrinkage", "tol"))
    s = est.solver
    why = why or (s not in ("svd", "lsqr", "eigen") and f"solver {s!r}")
    why = why or (s in ("svd", "lsqr") and est.shrinkage is not None and f"shrinkage {est.shrinkage!r} with {s} (engine: none)")
    why = why or (s == "eigen" and isinstance(est.shrinkage, str) and f"shrinkage {est.shrinkage!r}")
    why = why or (s == "eigen" and est.shrinkage is None and "solver eigen without shrinkage: stock's scalings_ "
                  "past n_classes - 1 columns is an arbitrary basis of a zero-eigenvalue subspace (route-all r1)")
    if why:
        return why
    X, classes, yi, cts, ref = _classes(est, X, y, min_samples=2)
    k, p = classes.size, X.shape[1]
    if k < 2 or X.shape[0] <= k:
        return f"{k} classes / rows"
    mc = min(k - 1, p)
    if s == "svd":
        r = sm.lda_svd(cts, ref, tol=float(est.tol))
        est.xbar_, est.scalings_ = r["xbar"], r["scalings"]
        est.explained_variance_ratio_ = r["explained_variance_ratio"][:mc]
    elif s == "lsqr":
        r = ss.lda_lsqr(cts, ref)
        est.covariance_ = r["covariance"]
    else:
        shr = None if est.shrinkage is None else float(est.shrinkage)
        r = sm.lda_eigen(cts, ref, shrinkage=shr)
        est.covariance_ = r["covariance"]
        st = _shrunk_total_cov(cts, 0.0 if shr is None else shr)
        evals, evecs = linalg.eigh(st - r["covariance"], r["covariance"])
        est.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][:mc]
        est.scalings_ = evecs[:, np.argsort(evals)[::-1]]
    est.classes_, est.priors_, est.means_ = classes, r["priors"], r["means"]
    est.coef_, est.intercept_ = r["coef"], r["intercept"]
    est._max_components = mc
    return None


def _qda_tsqr(est: Any, X: Any, y: Any) -> str | None:
    from . import linear_more as lm
    from ._sk_dropin_linear import _tsqr_threads

    X, y = validate_data(est, X, y, dtype=np.float64, reset=True)
    X = np.ascontiguousarray(X)
    classes, yi = np.unique(y, return_inverse=True)
    if classes.size < 2 or np.min(np.bincount(yi)) < 2:
        return "fewer than 2 classes or a class with one row"
    r = lm.qda_tsqr(X, yi, int(classes.size), _tsqr_threads(), float(est.reg_param), float(est.tol))
    if r is None:
        return "a class covariance is rank deficient (stock raises its LinAlgError)"
    for c in range(classes.size):
        why = _scale_span(r["colss"][c], float(np.sum(yi == c)), "QuadraticDiscriminantAnalysis")
        if why:
            return why
    est.classes_, est.priors_, est.means_ = classes, r["priors"], r["means"]
    est.scalings_, est.rotations_ = r["scalings"], r["rotations"]
    return None


def fit_qda(est: Any, X: Any, y: Any) -> str | None:
    row = "QuadraticDiscriminantAnalysis"
    why = stock_by_machine(est) or (_early_exit(X, None, row) if getattr(est, "solver", "svd") == "svd"
                               else x11_guard(est, X, row=row + "+eigen"))
    if why:
        return why
    from . import stats_more as sm

    why = _dense(X) or _narrow(X) or (est.priors is not None and "priors") or (est.store_covariance and "store_covariance")
    why = why or (getattr(est, "covariance_estimator", None) is not None and "covariance_estimator")
    solver = getattr(est, "solver", "svd")
    shr = getattr(est, "shrinkage", None)
    why = why or (solver not in ("svd", "eigen") and f"solver {solver!r}") or (isinstance(shr, str) and f"shrinkage {shr!r}")
    if why:
        return why
    if solver == "svd":
        return _qda_tsqr(est, X, y)
    X, classes, yi, cts, ref = _classes(est, X, y)
    if classes.size < 2 or np.min(np.bincount(yi)) < 2:
        return "fewer than 2 classes or a class with one row"
    r = sm.qda(cts, ref, solver=solver, reg_param=float(est.reg_param),
               shrinkage=None if shr is None else float(shr), tol=float(est.tol))
    if any(int(np.sum(np.asarray(sc) > float(est.tol))) < X.shape[1] for sc in r["scalings"]):
        return "a class covariance is rank deficient (stock raises its LinAlgError)"
    est.classes_, est.priors_, est.means_ = classes, r["priors"], r["means"]
    est.scalings_, est.rotations_ = r["scalings"], r["rotations"]
    return None
