from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shira_cpp import cat_cpp

MAX_ITERATIONS = 100_000
MIN_ROWS_DEFAULT_SUBSAMPLE = 100
EXACT_RANDOM_ROWS = 880_000

ALIASES = {"n_estimators": "iterations", "num_boost_round": "iterations", "num_trees": "iterations",
           "eta": "learning_rate", "reg_lambda": "l2_leaf_reg", "max_depth": "depth",
           "random_state": "random_seed", "objective": "loss_function", "max_bin": "border_count",
           "colsample_bylevel": "rsm", "min_child_samples": "min_data_in_leaf", "num_leaves": "max_leaves"}

LOSSES = {"RMSE": (("regressor", "generic"), True, True),
          "MAE": (("regressor", "generic"), True, False),
          "Quantile": (("regressor", "generic"), True, False),
          "Poisson": (("regressor", "generic"), False, False),
          "Tweedie": (("regressor", "generic"), False, False),
          "Logloss": (("classifier", "generic"), False, True),
          "CrossEntropy": (("generic",), False, True)}
DEFAULT_LOSS = {"regressor": "RMSE", "classifier": "Logloss", "generic": "RMSE"}
CLASSIFIER_LOSSES = ("Logloss", "CrossEntropy")
CLASSIFIER_REASON = ("classifier results are not bit-identical to stock at the default "
                     "iteration count")
LENGTH_BROKEN = {"Poisson": "Poisson results are not bit-identical to stock at 500 iterations and more (default "
                            "1000)"}


def _is(*vals: Any) -> Callable[[Any], bool]:
    return lambda v: v in vals


def _num(lo: float, hi: float, lo_open: bool = False) -> Callable[[Any], bool]:
    return lambda v: (lo < float(v) if lo_open else lo <= float(v)) and float(v) <= hi


def _int_in(lo: int, hi: int) -> Callable[[Any], bool]:
    return lambda v: int(v) == v and lo <= int(v) <= hi


def _anything(_: Any) -> bool:
    return True


SUPPORTED: dict[str, Callable[[Any], bool]] = {
    "iterations": _int_in(1, MAX_ITERATIONS), "depth": _int_in(1, 16),
    "learning_rate": _num(0.0, 10.0, lo_open=True), "l2_leaf_reg": _num(0.0, 1e9),
    "loss_function": lambda v: isinstance(v, str) and v.partition(":")[0] in LOSSES,
    "score_function": _is("L2", "Cosine"), "boosting_type": _is("Plain", "Ordered"),
    "bootstrap_type": _is("No", "Bayesian", "Bernoulli", "MVS"), "random_strength": _num(0.0, 1e9),
    "boost_from_average": _is(True, False), "leaf_estimation_iterations": _is(1),
    "leaf_estimation_method": _is("Newton", "Exact"), "border_count": _int_in(1, 254),
    "random_seed": _int_in(0, 2**63 - 1), "grow_policy": _is("SymmetricTree"),
    "feature_border_type": _is("GreedyLogSum"), "nan_mode": _is("Min", "Max", "Forbidden"),
    "random_score_type": _is("NormalWithModelSizeDecrease", "Gumbel"),
    "bagging_temperature": _num(0.0, 1e9), "subsample": _num(0.0, 1.0, lo_open=True),
    "mvs_reg": _num(0.0, 1e30), "sampling_frequency": _is("PerTree"),
    "permutation_count": _int_in(1, 64), "fold_len_multiplier": _num(1.0, 1e9, lo_open=True),
    "fold_permutation_block": _int_in(0, 2**31 - 1),
    "use_best_model": _is(True, False), "early_stopping_rounds": _int_in(1, MAX_ITERATIONS),
    "od_type": _is("Iter"), "od_wait": _int_in(1, MAX_ITERATIONS), "eval_metric": lambda v: isinstance(v, str),
    "thread_count": lambda v: v is None or int(v) == -1 or 1 <= int(v) <= 4096,
    "verbose": _anything, "silent": _anything, "logging_level": _anything, "metric_period": _anything,
    "allow_writing_files": _anything, "train_dir": _anything, "name": _anything, "used_ram_limit": _anything,
    "task_type": _is("CPU"),
    "rsm": _is(1, 1.0), "min_data_in_leaf": _is(1), "model_shrink_rate": _is(0, 0.0),
    "langevin": _is(False), "posterior_sampling": _is(False), "penalties_coefficient": _is(1, 1.0),
    "has_time": _is(False), "approx_on_full_history": _is(False),
    "leaf_estimation_backtracking": _is("AnyImprovement"), "scale_pos_weight": _is(1, 1.0),
    "best_model_min_trees": _is(1), "save_snapshot": _is(False),
}
ENGINE_OPTIONAL = ("random_score_type", "bagging_temperature", "subsample", "mvs_reg", "permutation_count",
                   "fold_len_multiplier", "fold_permutation_block")


def normalize(params: Any) -> tuple[dict[str, Any], str | None]:
    if params is None:
        return {}, None
    if not isinstance(params, dict):
        return {}, f"params of type {type(params).__name__}"
    out: dict[str, Any] = {}
    for key, val in params.items():
        name = ALIASES.get(key, key)
        if val is None:
            continue
        if name in out and out[name] != val:
            return {}, f"parameter {name} given twice with different values"
        out[name] = val
    assert len(out) <= len(params)
    return out, None


def _table_reason(p: dict[str, Any]) -> str | None:
    for key, val in p.items():
        check = SUPPORTED.get(key)
        if check is None:
            return f"parameter {key} is not in SHIRA's CatBoost table"
        try:
            ok = bool(check(val))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return f"{key}={val!r} is not supported by SHIRA's CatBoost engine"
    return None


def _loss(p: dict[str, Any], kind: str) -> tuple[str, str, str | None]:
    spec = str(p.get("loss_function", DEFAULT_LOSS[kind]))
    name = spec.partition(":")[0]
    if name not in LOSSES:
        return spec, name, f"loss_function {spec} is not supported by SHIRA's CatBoost engine"
    if kind not in LOSSES[name][0]:
        return spec, name, f"loss_function {spec} with a {kind} estimator is not routed to SHIRA"
    if kind == "classifier" or name in CLASSIFIER_LOSSES:
        return spec, name, CLASSIFIER_REASON
    if name in LENGTH_BROKEN:
        return spec, name, LENGTH_BROKEN[name]
    return spec, name, None


def _bootstrap(p: dict[str, Any], n_rows: int | None) -> tuple[str, str | None]:
    bt = p.get("bootstrap_type")
    if bt is not None:
        return bt, None
    if "bagging_temperature" in p or "subsample" in p:
        return "", "bagging_temperature / subsample without bootstrap_type (CatBoost then chooses the type)"
    if n_rows is not None and n_rows < MIN_ROWS_DEFAULT_SUBSAMPLE:
        return "", "fewer than 100 rows with the default bootstrap (CatBoost's default subsample changes)"
    return "MVS", None


def _learning_rate(p: dict[str, Any], name: str, auto_lr: bool, bfa: bool, n_rows: int | None,
                   has_eval: bool | None) -> tuple[float, str | None]:
    if "learning_rate" in p:
        return float(p["learning_rate"]), None
    tuned = [k for k in ("l2_leaf_reg", "leaf_estimation_method", "leaf_estimation_iterations") if k in p]
    if not auto_lr or tuned:
        return 0.03, None
    if name == "CrossEntropy":
        return 0.0, "learning_rate is not set (CrossEntropy: not verified against CatBoost's choice)"
    if n_rows is None or has_eval is None:
        return 0.0, f"learning_rate is not set (CatBoost picks it from the row count for {name})"
    best = bool(has_eval) and p.get("use_best_model") is not False
    lr = cat_cpp.auto_learning_rate(name, best, bfa, int(p.get("iterations", 1000)), int(n_rows))
    assert lr is None or 0.0 < lr <= 0.5
    return (0.03 if lr is None else lr), None


def random_rows_reason(p: dict[str, Any], bootstrap: str, n_rows: int | None) -> str | None:
    random_on = float(p.get("random_strength", 1.0)) > 0.0 or bootstrap != "No"
    if not random_on or n_rows is None or n_rows <= EXACT_RANDOM_ROWS:
        return None
    return (f"random_strength / bootstrap with {n_rows:,} rows: SHIRA matches CatBoost bit for bit only up to "
            f"{EXACT_RANDOM_ROWS:,} rows with random options")


TESTED_VERSION = "1.2.10"


def version_reason() -> str | None:
    import catboost

    v = str(getattr(catboost, "__version__", "unknown"))
    assert TESTED_VERSION.count(".") == 2
    return None if v == TESTED_VERSION else f"CatBoost {v}: SHIRA's exact engine is tested on {TESTED_VERSION} only (stock runs)"


def resolve(params: Any, kind: str, n_rows: int | None = None,
            has_eval: bool | None = None) -> tuple[dict | None, dict | None, str | None]:
    assert kind in DEFAULT_LOSS, f"unknown estimator kind {kind}"
    why = version_reason()
    if why:
        return None, None, why
    p, why = normalize(params)
    why = why or _table_reason(p)
    if why:
        return None, None, why
    spec, name, why = _loss(p, kind)
    if why:
        return None, None, why
    _, bfa_default, auto_lr = LOSSES[name]
    score = p.get("score_function", "Cosine")
    bfa = bool(p.get("boost_from_average", bfa_default))
    boosting = p.get("boosting_type", "Plain")
    if bfa and (name not in ("RMSE", "MAE", "Quantile") or boosting != "Plain"):
        return None, None, (f"boost_from_average is on for {name} / {boosting} "
                            "(SHIRA: RMSE, MAE, Quantile with Plain boosting)")
    if score == "Cosine" and boosting != "Plain":
        return None, None, "score_function Cosine with Ordered boosting (SHIRA: Plain boosting)"
    lr, why = _learning_rate(p, name, auto_lr, bfa, n_rows, has_eval)
    if why:
        return None, None, why
    lr = float(np.float32(lr))
    method = "Exact" if name in ("MAE", "Quantile") else "Newton"
    if p.get("leaf_estimation_method", method) != method:
        return None, None, f"leaf_estimation_method {p['leaf_estimation_method']} with {name}"
    bt, why = _bootstrap(p, n_rows)
    if why:
        return None, None, why
    why = random_rows_reason(p, bt, n_rows)
    if why:
        return None, None, why
    eng = {"iterations": int(p.get("iterations", 1000)), "depth": int(p.get("depth", 6)),
           "learning_rate": lr, "l2_leaf_reg": float(p.get("l2_leaf_reg", 3.0)),
           "loss_function": spec, "score_function": score, "boosting_type": boosting,
           "bootstrap_type": bt, "random_strength": float(p.get("random_strength", 1.0)),
           "boost_from_average": bfa, "leaf_estimation_iterations": 1, "leaf_estimation_method": method,
           "border_count": int(p.get("border_count", 254)), "random_seed": int(p.get("random_seed", 0)),
           "nan_mode": p.get("nan_mode", "Min")}
    eng.update({k: p[k] for k in ENGINE_OPTIONAL if k in p})
    try:
        cat_cpp.check_params(eng)
    except AssertionError as err:
        return None, None, f"engine refused the parameters: {err}"
    ctl, why = _control(p, spec)
    if ctl is not None:
        ctl["learning_rate"] = lr
        tc = p.get("thread_count")
        if (tc is None or int(tc) == -1) and n_rows is not None:
            from . import _threads

            ctl["threads"] = _threads.default_threads("catboost", int(n_rows), int(ctl["threads"]))
        elif tc is not None and int(tc) != -1:
            from . import _threads

            ctl["threads"] = _threads.explicit_threads(int(ctl["threads"]))
    return (None, None, why) if why else (eng, ctl, None)


def _control(p: dict[str, Any], spec: str) -> tuple[dict | None, str | None]:
    metric = p.get("eval_metric", spec)
    if metric != spec:
        return None, f"eval_metric {metric} differs from the loss {spec}"
    if "od_type" in p and "od_wait" not in p and "early_stopping_rounds" not in p:
        return None, "od_type without od_wait"
    wait = p.get("early_stopping_rounds", p.get("od_wait"))
    tc = p.get("thread_count")
    threads = (os.cpu_count() or 1) if tc is None or int(tc) == -1 else int(tc)
    ctl = {"od_wait": None if wait is None else int(wait), "use_best_model": p.get("use_best_model"),
           "threads": max(1, min(threads, cat_cpp.MAX_THREADS)), "metric": spec,
           "seed": int(p.get("random_seed", 0)), "learning_rate": float(p.get("learning_rate", 0.03))}
    assert ctl["threads"] >= 1
    return ctl, None


def why_not(params: Any, kind: str = "regressor", n_rows: int | None = None) -> str | None:
    return resolve(params, kind, n_rows)[2]


def _is_pool(obj: Any) -> bool:
    import catboost

    return isinstance(obj, catboost.Pool)


def _pool_reason(pool: Any) -> str | None:
    if pool.is_quantized():
        return "a quantized Pool"
    if len(pool.get_cat_feature_indices()) or len(pool.get_text_feature_indices()) \
            or len(pool.get_embedding_feature_indices()):
        return "categorical, text or embedding features (not wired into the drop-in)"
    if np.size(pool.get_baseline()) or pool.num_pairs():
        return "a Pool with a baseline or pairs"
    if not pool.has_label():
        return "a Pool without a label"
    return None


def _features(X: Any) -> tuple[np.ndarray | None, list[str] | None, str | None]:
    names = None
    if hasattr(X, "columns") and hasattr(X, "dtypes"):
        if not all(d.kind in "biuf" for d in X.dtypes):
            return None, None, "a DataFrame with non-numeric columns"
        names = [str(c) for c in X.columns]
        X = X.to_numpy()
    arr = X if isinstance(X, np.ndarray) else None
    if arr is None or arr.ndim != 2 or arr.dtype.kind not in "biuf":
        return None, None, f"features of type {type(X).__name__} (numeric 2-D numpy / pandas only)"
    xf = np.ascontiguousarray(arr, dtype=np.float32)
    if np.isinf(xf).any():
        return None, None, "infinite feature values"
    return xf, names, None


def _label(y: Any, kind: str, name: str) -> tuple[np.ndarray | None, list | None, str | None]:
    arr = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1 or arr.dtype.kind not in "biuf":
        return None, None, "labels that are not a numeric 1-D array"
    if kind == "classifier":
        if arr.dtype.kind not in "iu" or not np.array_equal(np.unique(arr), [0, 1]):
            return None, None, "classifier labels other than integer 0 / 1 (multiclass and other label types)"
        return arr.astype(np.float32), [0, 1], None
    y32 = arr.astype(np.float32)
    if not np.isfinite(y32).all():
        return None, None, "labels that are not finite"
    if name == "Logloss" and not np.isin(y32, (0.0, 1.0)).all():
        return None, None, "Logloss labels other than 0 / 1"
    return y32, None, None


def _weight(w: Any, n: int) -> tuple[np.ndarray | None, str | None]:
    if w is None:
        return None, None
    w32 = np.asarray(w, dtype=np.float32)
    if w32.shape != (n,) or not np.isfinite(w32).all() or not (w32 > 0).all():
        return None, "sample weights that are not finite and > 0 for every row"
    return (None if bool(np.all(w32 == 1.0)) else w32), None


def extract(X: Any, y: Any, sample_weight: Any, kind: str, loss: str) -> tuple[dict | None, str | None]:
    names, from_pool = None, _is_pool(X)
    if from_pool:
        why = "y / sample_weight given together with a Pool" if (y is not None or sample_weight is not None) \
            else _pool_reason(X)
        if why:
            return None, why
        feats, y, sample_weight = X.get_features(), X.get_label(), X.get_weight()
        pool_names = [str(v) for v in X.get_feature_names()]
        names = pool_names if any(pool_names) else None
        X = np.asarray(feats)
    elif y is None:
        return None, "no label"
    xf, df_names, why = _features(X)
    if why:
        return None, why
    y32, classes, why = _label(y, kind, loss)
    if why:
        return None, why
    if y32.shape[0] != xf.shape[0] or xf.shape[0] < 2 or xf.shape[1] < 1:
        return None, "features and labels of different length (or fewer than 2 rows)"
    w32, why = _weight(sample_weight, xf.shape[0])
    if why:
        return None, why
    return {"x": xf, "y": y32, "w": w32, "names": df_names or names, "classes": classes}, None


def engine(data: dict, eng: dict, threads: int, folds: list | None = None) -> list:
    n = data["x"].shape[0]
    fl = folds if folds is not None else [(0, n)]
    assert len(fl) >= 1 and all(0 <= a < b <= n for a, b in fl), "folds must be row ranges of the data"
    if folds is None:
        eng = dict(eng, shira_fit_quantization=True)
    res = cat_cpp.train(data["x"], data["y"], fl, eng, threads=threads, trees_out=True, weight=data["w"])
    assert len(res) == len(fl)
    return res
