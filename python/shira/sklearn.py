from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import logging
from typing import Any

from . import _hgb_first_read as _hgb_fr
from . import _sk_first_read as _sk_fr
from . import _sk_dropin_hgb as _hgb
from . import _sk_dropin_linear as _lin
from . import _sk_dropin_more as _more
from . import _sk_dropin_trees as _trees

LOG = logging.getLogger("shira.sklearn")
last_backend: str | None = None
backend_history: list[str] = []

TREES = {"sklearn.tree": ("DecisionTreeRegressor", "DecisionTreeClassifier", "ExtraTreeRegressor",
                          "ExtraTreeClassifier"),
         "sklearn.ensemble": ("RandomForestRegressor", "RandomForestClassifier", "ExtraTreesRegressor",
                              "ExtraTreesClassifier", "IsolationForest")}
HGB = {"sklearn.ensemble": ("HistGradientBoostingRegressor", "HistGradientBoostingClassifier")}
GB = {"sklearn.ensemble": ("GradientBoostingRegressor", "GradientBoostingClassifier")}
ATTRS = {"sklearn.linear_model": {"LinearRegression": _lin.fit_linear_regression, "Ridge": _lin.fit_ridge,
                                  "LogisticRegression": _lin.fit_logistic, "Lasso": _more.fit_enet,
                                  "ElasticNet": _more.fit_enet, "BayesianRidge": _more.fit_bayes_ridge,
                                  "ARDRegression": _more.fit_ard, "Lars": _more.fit_lars,
                                  "LassoLars": _more.fit_lars},
         "sklearn.decomposition": {"PCA": _lin.fit_pca, "TruncatedSVD": _more.fit_truncated_svd,
                                   "FactorAnalysis": _more.fit_factor_analysis},
         "sklearn.naive_bayes": {"GaussianNB": _lin.fit_gaussian_nb},
         "sklearn.discriminant_analysis": {"LinearDiscriminantAnalysis": _more.fit_lda,
                                           "QuadraticDiscriminantAnalysis": _more.fit_qda}}
NOT_ROUTED: dict[str, tuple[str, ...]] = {}
WHY_NOT_ROUTED = "not routed to SHIRA yet"
_CLASSES: dict[str, tuple[str, Any, Any]] = {}


def _record(backend: str, name: str, reason: str | None = None) -> None:
    global last_backend
    assert backend in ("shira", "sklearn")
    if reason:
        LOG.warning("shira.sklearn: running stock %s: %s", name, reason)
    last_backend = backend
    backend_history.append(backend)


def _sparse(X: Any) -> bool:
    return hasattr(X, "tocsr") or hasattr(X, "tocsc")


def _fit_x(args: tuple, kwargs: dict) -> Any:
    assert isinstance(args, tuple) and isinstance(kwargs, dict)
    return args[0] if args else kwargs.get("X")


ISO_MIN_ROWS = 300_000


def _iso_stock_below(x_in: Any) -> int:
    from . import _machine

    shape = getattr(x_in, "shape", None)
    features = int(shape[1]) if shape is not None and len(shape) == 2 and int(shape[1]) >= 1 else 1
    out = int(_machine.value("iso_stock_below_rows", features))
    assert out >= 0
    return out


def _fit_trees(self: Any, base: Any, args: tuple, kwargs: dict, gb: bool = False) -> Any:
    name = base.__name__

    def fit_any(est: Any, xs: Any, ys: Any, ws: Any = None) -> bool:
        from sklearn import config_context

        with config_context(skip_parameter_validation=True):
            _fit_trees_ordinary(est, base, (xs, ys) if ws is None else (xs, ys, ws), {}, gb)
        return last_backend == "shira"

    snap = _snapshot(self)
    try:
        if gb:
            ok = _sk_fr.gb_why_not(self, name, args, kwargs) is None and _sk_fr.gb_fit(
                self, *_sk_fr.fit_args(args, kwargs), fit_any, _env_cores())
        elif _sk_fr.why_not(self, name, args, kwargs) is not None:
            ok = False
        else:
            threads = _forest_threads(self, _default_cores(self)) if hasattr(self, "n_jobs") else _env_cores()
            ok = _sk_fr.fit(self, name, *_sk_fr.fit_args(args, kwargs), fit_any, threads)
    except FIRST_READ_ERRORS as e:
        _restore(self, snap)
        LOG.warning("shira.sklearn: %s: the first read did not apply (%s: %s); today's route", name, type(e).__name__, e)
        ok = False
    if not ok:
        return _fit_trees_ordinary(self, base, args, kwargs, gb)
    LOG.info("shira.sklearn: %s: %s", name, _sk_fr.log[-1]["line"])
    return self


FIRST_READ_ERRORS = (AssertionError, ValueError, RuntimeError, AttributeError, TypeError, ArithmeticError)


def _snapshot(est: Any) -> tuple:
    import numpy as np

    rs = getattr(est, "random_state", None)
    kept = None
    if getattr(est, "warm_start", False):
        kept = {k: (list(v) if isinstance(v, list) else v) for k, v in est.__dict__.items()}
    snap = (np.random.get_state(), rs.get_state() if isinstance(rs, np.random.RandomState) else None, kept)
    assert len(snap) == 3
    return snap


def _restore(est: Any, snap: tuple) -> None:
    import numpy as np

    assert len(snap) == 3
    np.random.set_state(snap[0])
    if snap[1] is not None:
        est.random_state.set_state(snap[1])
    if snap[2] is not None:
        est.__dict__.clear()
        est.__dict__.update(snap[2])


def _fit_trees_ordinary(self: Any, base: Any, args: tuple, kwargs: dict, gb: bool = False) -> Any:
    self.__dict__.pop("_shira_fr_rows", None)
    if _sparse(_fit_x(args, kwargs)):
        base.fit(self, *args, **kwargs)
        _record("sklearn", base.__name__, "sparse X")
        return self
    x_in = _fit_x(args, kwargs)
    rows = int(x_in.shape[0]) if hasattr(x_in, "shape") else (len(x_in) if x_in is not None else 0)
    iso_lo = _iso_stock_below(x_in) if base.__name__ == "IsolationForest" else 0
    if rows < iso_lo:
        base.fit(self, *args, **kwargs)
        _record("sklearn", base.__name__, f"IsolationForest with {rows} rows < {iso_lo} (switch iso_stock_below_rows)")
        return self
    cores = 1 if gb else _default_cores(self)
    single = not hasattr(self, "n_jobs")
    pre = _env_cores() if single else 1
    with _trees.presort_threads(pre), _trees.routed(), _trees.default_mode(base.__name__), \
            (_trees.gb_session() if gb else contextlib.nullcontext()), \
            (contextlib.nullcontext() if single else _trees.forest_session(_forest_threads(self, cores))):
        if cores > 1:
            import joblib

            with joblib.parallel_backend("threading", n_jobs=cores):
                base.fit(self, *args, **kwargs)
        else:
            base.fit(self, *args, **kwargs)
        shira, stock, why = _trees.counts()
    _record("shira" if shira else "sklearn", base.__name__, why if stock else None)
    return self


def _forest_threads(est: Any, cores: int) -> int:
    assert cores >= 1 and hasattr(est, "n_jobs")
    if cores > 1:
        return min(cores, 64)
    import joblib

    trees = max(1, int(getattr(est, "n_estimators", 1)))
    try:
        jobs = int(joblib.effective_n_jobs(est.n_jobs))
    except ValueError:
        jobs = 1
    return max(1, min(jobs, trees, 64))


def _default_cores(est: Any) -> int:
    if getattr(est, "n_jobs", 0) is not None or not hasattr(est, "estimators_") and not hasattr(est, "n_estimators"):
        return 1
    return max(1, min(_env_cores(), int(getattr(est, "n_estimators", 1))))


def _env_cores() -> int:
    from . import _threads

    return _threads.env_budget()


def _fit_hgb(self: Any, base: Any, X: Any, y: Any, sample_weight: Any, kwargs: dict) -> Any:
    classifier = base.__name__.endswith("Classifier")
    extra = {k: v for k, v in kwargs.items() if v is not None}

    def stock_fit(est: Any, x: Any, yy: Any, sw: Any) -> Any:
        return base.fit(est, x, yy, sample_weight=sw)

    def ordinary(est: Any, sf: Any, x: Any, yy: Any, sw: Any, cl: bool) -> str | None:
        return _hgb.x86_guard(x) or _hgb.fit_on_shira(est, sf, x, yy, sw, cl)

    frame = _hgb_fr.numeric_frame(self, X)
    if frame is None or _hgb.why_not(self, frame, y, extra) is not None:
        frame = None
    Xr = X if frame is None else frame
    why, line, used = _hgb.why_not(self, Xr, y, extra), "", False
    snap = _snapshot(self)
    try:
        if why and _hgb_fr.why_stock_grown(self, Xr, y, why, classifier, extra) is None:
            used, (why, line) = True, _hgb_fr.fit_stock_grown(self, base.fit, Xr, y, sample_weight, classifier, extra)
        elif not why and _hgb_fr.why_not(self, Xr, classifier) is None:
            used, (why, line) = True, _hgb_fr.fit(self, stock_fit, Xr, y, sample_weight, classifier, ordinary)
    except FIRST_READ_ERRORS as e:
        _restore(self, snap)
        LOG.warning("shira.sklearn: %s: the first read did not apply (%s: %s); today's route", base.__name__,
                    type(e).__name__, e)
        used, line = False, ""
    if not why and not used:
        why = ordinary(self, stock_fit, Xr, y, sample_weight, classifier)
    if line:
        LOG.info("shira.sklearn: %s: %s", base.__name__, line)
    if why:
        self.__dict__.pop("_shira_fr_rows", None)
        base.fit(self, X, y, sample_weight=sample_weight, **kwargs)
    elif frame is not None:
        _hgb_fr.set_frame_names(self, X)
    _record("sklearn" if why else "shira", base.__name__, why)
    return self


def _float32(X: Any) -> bool:
    import numpy as np

    dt = getattr(X, "dtype", None)
    if isinstance(dt, np.dtype):
        return dt == np.float32
    dts = getattr(X, "dtypes", None)
    try:
        return dts is not None and len(dts) > 0 and all(np.dtype(d) == np.float32 for d in dts)
    except TypeError:
        return False


_ARGS: dict[Any, tuple[int, frozenset, bool]] = {}


def _args_fit(fn: Any, n_pos: int, kwargs: dict) -> bool:
    if fn not in _ARGS:
        import inspect

        ps = list(inspect.signature(fn).parameters.values())[1:]
        pos = sum(1 for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD))
        names = frozenset(p.name for p in ps if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
        _ARGS[fn] = (pos, names, any(p.kind == p.VAR_KEYWORD for p in ps))
    pos, names, var_kw = _ARGS[fn]
    return n_pos <= pos and (var_kw or all(k in names for k in kwargs))


def _fit_attrs(self: Any, base: Any, fn: Any, args: tuple, kwargs: dict) -> Any:
    from ._sk_dropin_linear import PreEngineInputError

    if not _args_fit(fn, len(args), kwargs):
        why = "SHIRA's input check raised TypeError; stock re-run"
    elif _float32(_fit_x(args, kwargs)):
        why = "float32 input (dtype and float32 rounding follow stock)"
    else:
        try:
            why = fn(self, *args, **kwargs)
        except PreEngineInputError as exc:
            why = f"SHIRA's input check raised {exc.kind}; stock re-run"
        except (RuntimeError, AssertionError) as exc:
            why = f"engine: {exc}"
    if why:
        _stock_fit_validated(self, base, args, kwargs)
    _record("sklearn" if why else "shira", base.__name__, why)
    return self


@functools.cache
def _fit_skips_nested(base: Any) -> bool:
    f = base.fit
    if not hasattr(f, "__wrapped__"):
        return False
    return inspect.getclosurevars(f).nonlocals.get("prefer_skip_nested_validation") is True


def _stock_fit_validated(self: Any, base: Any, args: tuple, kwargs: dict) -> None:
    from sklearn import config_context, get_config

    if get_config().get("skip_parameter_validation", False) or not _fit_skips_nested(base):
        base.fit(self, *args, **kwargs)
        return
    with config_context(skip_parameter_validation=True):
        base.fit(self, *args, **kwargs)


def _validate_params_like_stock(est: Any) -> None:
    from sklearn import get_config

    if not get_config().get("skip_parameter_validation", False) and hasattr(est, "_validate_params"):
        est._validate_params()


def _estimators_samples(self: Any) -> list:
    import numpy as np

    samples = self._shira_base.estimators_samples_.fget(self)
    rows = self.__dict__.get("_shira_fr_rows")
    if rows is None:
        return samples
    assert all(int(s.max(initial=-1)) < rows.size for s in samples)
    out = [np.asarray(rows, dtype=np.int64)[s] for s in samples]
    assert len(out) == len(samples)
    return out


def _make(module: str, name: str, kind: str, fn: Any = None) -> Any:
    base = getattr(importlib.import_module(module), name)

    @functools.wraps(base.fit)
    def fit(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kind != "stock":
            _validate_params_like_stock(self)
        if kind in ("trees", "gb"):
            return _fit_trees(self, base, args, kwargs, gb=kind == "gb")
        if kind == "hgb" and 2 <= len(args) <= 3:
            sw = args[2] if len(args) == 3 else kwargs.pop("sample_weight", None)
            return _fit_hgb(self, base, args[0], args[1], sw, kwargs)
        if kind == "attrs":
            return _fit_attrs(self, base, fn, args, kwargs)
        base.fit(self, *args, **kwargs)
        _record("sklearn", name, WHY_NOT_ROUTED if kind == "stock" else "X and y must be positional")
        return self

    extra = {"estimators_samples_": property(_estimators_samples, doc=_estimators_samples.__doc__)} \
        if kind == "trees" and isinstance(getattr(base, "estimators_samples_", None), property) else {}
    cls = type(name, (base,), {"fit": fit, "__module__": __name__, "__doc__": base.__doc__,
                                "_shira_base": base, "_shira_kind": kind, **extra})
    _CLASSES[name] = (module, base, cls)
    return cls


def _build() -> None:
    for module, names in TREES.items():
        for n in names:
            globals()[n] = _make(module, n, "trees")
    for module, names in HGB.items():
        for n in names:
            globals()[n] = _make(module, n, "hgb")
    for module, names in GB.items():
        for n in names:
            globals()[n] = _make(module, n, "gb")
    for module, table in ATTRS.items():
        for n, fn in table.items():
            globals()[n] = _make(module, n, "attrs", fn)
    for module, names in NOT_ROUTED.items():
        for n in names:
            globals()[n] = _make(module, n, "stock")
    assert len(_CLASSES) >= 25


_build()


from . import _sk_dropin_cv as _cv

cross_val_score = _cv.cross_val_score
cross_validate = _cv.cross_validate


_PATCHED: dict[tuple[str, str], Any] = {}
_CV_NAMES = ("cross_val_score", "cross_validate")


def patch() -> None:
    for name, (module, base, cls) in _CLASSES.items():
        mod = importlib.import_module(module)
        _PATCHED.setdefault((module, name), base)
        setattr(mod, name, cls)
    ms = importlib.import_module("sklearn.model_selection")
    for n in _CV_NAMES:
        _PATCHED.setdefault(("sklearn.model_selection", n), _cv.STOCK[n])
        setattr(ms, n, getattr(_cv, n))
    assert _PATCHED


def unpatch() -> None:
    for (module, name), stock in _PATCHED.items():
        setattr(importlib.import_module(module), name, stock)
    _PATCHED.clear()


def supported() -> dict[str, str]:
    return {n: ("stock" if cls._shira_kind == "stock" else "shira") for n, (_, _, cls) in _CLASSES.items()}


def stock_class(est_or_cls: Any) -> Any:
    cls = est_or_cls if isinstance(est_or_cls, type) else type(est_or_cls)
    return getattr(cls, "_shira_base", cls)


__all__ = sorted(_CLASSES) + ["cross_val_score", "cross_validate", "patch", "unpatch", "supported",
                              "stock_class", "last_backend", "backend_history"]
