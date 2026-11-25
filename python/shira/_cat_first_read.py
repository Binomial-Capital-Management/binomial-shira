from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

import numpy as np

log: list[dict] = []
VERIFIED_VERSIONS = ("1.2.10",)


def version_why() -> str | None:
    import catboost

    v = str(getattr(catboost, "__version__", ""))
    assert isinstance(VERIFIED_VERSIONS, tuple) and VERIFIED_VERSIONS
    return None if v in VERIFIED_VERSIONS else (f"CatBoost {v or '(no version)'}: the first read is verified on "
                                                f"{', '.join(VERIFIED_VERSIONS)} only")
AUTO_LR = ("RMSE", "Logloss", "MultiClass")
AUTO_LR_STOCK = {("RMSE", False, False): (0.158, -4.287, -0.813, 2.571), ("RMSE", False, True): (0.157, -4.062, -0.61, 1.557),
                 ("RMSE", True, False): (0.178, -4.473, -0.76, 2.133), ("RMSE", True, True): (0.189, -4.383, -0.623, 1.439),
                 ("Logloss", False, False): (0.427, -7.525, -0.917, 2.63), ("Logloss", False, True): (0.247, -5.158, -0.435, 0.934),
                 ("Logloss", True, False): (0.408, -7.299, -0.928, 2.701), ("Logloss", True, True): (0.246, -5.127, -0.451, 0.978),
                 ("MultiClass", False, False): (0.051, -2.889, -0.845, 2.928),
                 ("MultiClass", False, True): (0.02, -2.364, -0.382, 0.924)}
ALLOWED = frozenset(("iterations", "learning_rate", "depth", "l2_leaf_reg", "random_seed", "thread_count", "verbose",
                     "logging_level", "silent", "allow_writing_files", "loss_function", "bootstrap_type",
                     "bagging_temperature", "subsample", "random_strength", "border_count", "boost_from_average",
                     "train_dir", "metric_period", "used_ram_limit", "class_weights", "auto_class_weights",
                     "scale_pos_weight",
                     "use_best_model", "early_stopping_rounds", "od_wait", "od_type", "eval_metric",
                     "cat_features", "one_hot_max_size", "max_ctr_complexity",
                     "grow_policy", "max_leaves", "num_leaves", "min_data_in_leaf", "min_child_samples",
                     "boosting_type",
                     "rsm", "colsample_bylevel", "score_function", "feature_border_type", "model_size_reg", "max_bin",
                     "per_float_feature_quantization", "nan_mode", "custom_metric", "sampling_frequency",
                     "leaf_estimation_method", "leaf_estimation_iterations", "leaf_estimation_backtracking",
                     "n_estimators", "num_boost_round", "num_trees", "eta", "max_depth", "reg_lambda", "random_state"))
ALIASES = {"iterations": ("n_estimators", "num_boost_round", "num_trees"), "learning_rate": ("eta",),
           "depth": ("max_depth",), "l2_leaf_reg": ("reg_lambda",), "random_seed": ("random_state",)}
CLASS_WEIGHTS = ("class_weights", "auto_class_weights", "scale_pos_weight")
FIT_PASSIVE = ("verbose", "logging_level", "plot", "metric_period", "silent", "verbose_eval")
FIT_TAKEN = ("sample_weight", "eval_set", "early_stopping_rounds", "use_best_model", "cat_features")


def _p(model: Any, name: str, default: Any = None) -> Any:
    p = model._init_params
    for k in (name,) + ALIASES.get(name, ()):
        if p.get(k) is not None:
            return p[k]
    return default


def logloss_steps(model: Any) -> int:
    given = model._init_params.get("leaf_estimation_iterations")
    steps = int(given) if given is not None else (10 if int(_p(model, "iterations", 1000)) >= 200 else 1)
    assert steps >= 1
    return steps


def value_steps(model: Any, loss: str | None, steps: int) -> int:
    if loss != "Logloss" or steps <= 1 or model._init_params.get("leaf_estimation_iterations") is not None:
        return steps
    cap = _switch("cat_first_read_logloss_steps", "SHIRA_CAT_FIRST_READ_LOGLOSS_STEPS", len(model.feature_names_))
    assert cap >= 1, "cat_first_read_logloss_steps must be at least 1"
    return min(steps, cap)


def mode_on() -> bool:
    return os.environ.get("SHIRA_CAT_MODE", "") != "ordinary"


def _switch(name: str, env: str, features: int) -> int:
    from . import _machine

    raw = os.environ.get(env, "")
    v = int(raw) if raw else int(_machine.value(name, int(features)))
    assert v >= 0
    return v


def _sample_loss(model: Any, loss: str | None) -> str | None:
    if loss == "Poisson" and reg_steps(model, loss) > 1:
        return None
    assert loss is None or isinstance(loss, str)
    return loss


def sample_rows(features: int, cat: bool = False, loss: str | None = None) -> int:
    if not os.environ.get("SHIRA_CAT_FIRST_READ_ROWS", ""):
        if cat and loss in ("RMSE", "Quantile") + REG_LOSSES:
            return _switch("cat_first_read_rows_cat_reg", "SHIRA_CAT_FIRST_READ_ROWS_CAT_REG", features)
        if cat:
            return _switch("cat_first_read_rows_cat", "SHIRA_CAT_FIRST_READ_ROWS_CAT", features)
        if loss == "Poisson":
            return _switch("cat_first_read_rows_poisson", "SHIRA_CAT_FIRST_READ_ROWS_POISSON", features)
    return _switch("cat_first_read_rows", "SHIRA_CAT_FIRST_READ_ROWS", features)


def _cat_data(model: Any, kw: dict, X: Any) -> bool:
    return bool(_cats(model, kw)) or (_is_pool(X) and len(X.get_cat_feature_indices()) > 0)


def _small_band(n: int, features: int) -> bool:
    from . import _machine

    assert n >= 0 and features >= 1
    if min_rows(features) >= _machine.OPEN:
        return False
    hi = _switch("cat_first_read_small_below_rows", "SHIRA_CAT_FIRST_READ_SMALL_BELOW_ROWS", features)
    return _switch("cat_first_read_small_min_rows", "SHIRA_CAT_FIRST_READ_SMALL_MIN_ROWS", features) < n < hi


def small_sample(n: int, features: int, m: int) -> int:
    assert m >= 0
    explicit = ("SHIRA_CAT_FIRST_READ_ROWS", "SHIRA_CAT_FIRST_READ_ROWS_CAT", "SHIRA_CAT_FIRST_READ_ROWS_CAT_REG",
                "SHIRA_CAT_FIRST_READ_ROWS_POISSON")
    if any(os.environ.get(v, "") for v in explicit):
        return m
    return min(m, n // 4) if _small_band(n, features) else m


def small_min_rows(n: int, features: int, lo: int) -> int:
    assert lo >= 0
    return min(lo, _switch("cat_first_read_small_min_rows", "SHIRA_CAT_FIRST_READ_SMALL_MIN_ROWS", features)) if _small_band(n, features) else lo


def small_floor_why(n: int, features: int) -> str | None:
    from . import _machine

    assert n >= 0 and features >= 1
    lo = min_rows(features)
    floor = _switch("cat_first_read_small_min_rows", "SHIRA_CAT_FIRST_READ_SMALL_MIN_ROWS", features)
    if lo >= _machine.OPEN or n >= lo or n > floor:
        return None
    return (f"{n} rows: under the small-data floor ({floor}; switch cat_first_read_small_min_rows; "
            "SHIRA_CAT_FIRST_READ_SMALL_MIN_ROWS lowers it)")


def min_rows(features: int) -> int:
    return _switch("cat_first_read_min_rows", "SHIRA_CAT_FIRST_READ_MIN_ROWS", features)


EXPLICIT_ROWS = ("SHIRA_CAT_FIRST_READ_ROWS", "SHIRA_CAT_FIRST_READ_ROWS_CAT", "SHIRA_CAT_FIRST_READ_ROWS_CAT_REG",
                 "SHIRA_CAT_FIRST_READ_ROWS_POISSON")


def choose_rows(model: Any, d: tuple, kw: dict, loss: str | None, refused: bool = False) -> tuple[int, str | None]:
    n, F = d[3], d[4]
    assert n >= 1 and F >= 1
    if (why := small_floor_why(n, F)):
        return 0, why
    m = small_sample(n, F, sample_rows(F, _cat_data(model, kw, d[0]), _sample_loss(model, loss)))
    lo = small_min_rows(n, F, min_rows(F))
    if m < 1 or n <= m or n < lo:
        return 0, f"{n} rows: under the first read's row bound ({max(lo, m + 1)})"
    if any(os.environ.get(v, "") for v in EXPLICIT_ROWS):
        return m, None
    k = _switch("cat_first_read_rows_per_feature", "SHIRA_CAT_FIRST_READ_ROWS_PER_FEATURE", F)
    mf = max(m, k * F)
    if mf <= n // 2:
        return mf, None
    if not refused:
        return 0, (f"{n} rows x {F} features: a sample of {k} rows per feature ({k * F}) would be more than half the rows "
                   "(switch cat_first_read_rows_per_feature; SHIRA_CAT_FIRST_READ_ROWS_PER_FEATURE=0 turns it "
                   "off): SHIRA's exact engine")
    if _engine_shapes_why(model, d, str(loss), kw) is None:
        return n, None
    return n // 2, None


def _loss(model: Any, y: Any = None) -> str | None:
    given = model._init_params.get("loss_function")
    kind = getattr(model, "_shira_kind", "generic")
    if kind == "generic":
        if given in ("Logloss", "CrossEntropy", "MultiClass"):
            return given
        kind = "regressor"
    if kind == "regressor":
        if given in (None, "RMSE"):
            return "RMSE"
        if _quantile_params(given) is not None:
            return "Quantile"
        r = _reg_params(given)
        return None if r is None else r[0]
    if kind == "classifier" and given == "MultiClass":
        return "MultiClass"
    if kind == "classifier" and given == "CrossEntropy":
        return "CrossEntropy"
    if kind == "classifier" and given in (None, "Logloss"):
        many = given is None and y is not None and np.unique(np.asarray(y)).size > 2
        return "MultiClass" if many else "Logloss"
    return None


def _quantile_params(given: Any) -> tuple[float, float] | None:
    if not isinstance(given, str):
        return None
    name, _, rest = given.partition(":")
    if name not in ("MAE", "Quantile") or (name == "MAE" and rest):
        return None
    kv = dict(part.split("=", 1) for part in rest.split(";") if "=" in part)
    if set(kv) - {"alpha", "delta"} or (rest and len(kv) != len(rest.split(";"))):
        return None
    alpha, delta = float(kv.get("alpha", 0.5)), float(kv.get("delta", 1e-6))
    return (alpha, delta) if 0.0 < alpha < 1.0 and delta >= 0.0 else None


REG_LOSSES = ("Huber", "Poisson", "Tweedie")
REG_LOSSES = REG_LOSSES + ("Lq", "Expectile")
REG_LOSSES = REG_LOSSES + ("LogCosh",)


def _reg_params(given: Any) -> tuple[str, float] | None:
    if not isinstance(given, str):
        return None
    name, _, rest = given.partition(":")
    if name == "LogCosh":
        return ("LogCosh", 0.0) if not rest else None
    if name in ("Lq", "Expectile"):
        return _reg_params_4a(name, rest)
    key = {"Huber": "delta", "Tweedie": "variance_power", "Poisson": ""}.get(name)
    if key is None:
        return None
    if name == "Poisson":
        return ("Poisson", 0.0) if not rest else None
    k, _, v = rest.partition("=")
    if k != key or not v or ";" in v:
        return None
    try:
        p = float(v)
    except ValueError:
        return None
    ok = (p > 0.0) if name == "Huber" else (1.0 < p < 2.0)
    assert isinstance(ok, bool)
    return (name, p) if ok and np.isfinite(p) else None


def _reg_params_4a(name: str, rest: str) -> tuple[str, float] | None:
    assert name in ("Lq", "Expectile")
    k, _, v = rest.partition("=")
    if name == "Expectile" and not rest:
        return ("Expectile", float(np.float32(0.5)))
    if k != ("q" if name == "Lq" else "alpha") or not v or ";" in v:
        return None
    try:
        p = float(v)
    except ValueError:
        return None
    ok = bool(np.isfinite(p)) and ((p >= 1.0) if name == "Lq" else (0.0 < p < 1.0))
    return (name, p if name == "Lq" else float(np.float32(p))) if ok else None


def reg_steps(model: Any, loss: str) -> int:
    assert loss in REG_LOSSES
    given = model._init_params.get("leaf_estimation_iterations")
    steps = int(given) if given is not None else (10 if loss == "Poisson" and int(_p(model, "iterations", 1000)) >= 200 else 1)
    if given is None and loss == "Expectile" and int(_p(model, "iterations", 1000)) >= 200:
        steps = 5
    assert steps >= 1
    return steps


def _r_why(model: Any, d: tuple, F: int, depth: int, loss: str) -> str | None:
    assert loss in REG_LOSSES and depth >= 1
    p = model._init_params
    cap = _switch("cat_first_read_reg_max_depth", "SHIRA_CAT_FIRST_READ_REG_MAX_DEPTH", F)
    if depth > min(cap, 12):
        return (f"{loss} with depth {depth}: over the depth measured for these losses on this machine ({min(cap, 12)}; switch "
                "cat_first_read_reg_max_depth; SHIRA_CAT_FIRST_READ_REG_MAX_DEPTH=12 forces it)")
    if int(p.get("max_leaves") or p.get("num_leaves") or 0) > 4096:
        return f"{loss} with more than 4,096 leaves (the 16-bit kernels' bound)"
    if p.get("boost_from_average") is True:
        return f"{loss} with boost_from_average (stock's start not pinned for it)"
    steps = reg_steps(model, loss)
    if loss == "LogCosh" or (loss == "Expectile" and 1 < steps <= 64):
        return _walk_why(model, d, F, loss, steps)
    if steps != 1 and (loss != "Poisson" or steps > 64):
        return f"{loss} with {steps} leaf steps (stock's multi-step walk for it: not yet)"
    if p.get("leaf_estimation_backtracking") not in (None, "AnyImprovement", "No"):
        return f"leaf_estimation_backtracking {p['leaf_estimation_backtracking']} (AnyImprovement or No)"
    name = f"cat_first_read_{loss.lower()}_min_rows" if steps == 1 else "cat_first_read_poisson_walk_min_rows"
    lo = _switch(name, "SHIRA_" + name.upper(), F)
    return None if d[3] >= lo else (f"{loss} on {d[3]} rows: not measured faster than stock there on this machine (switch "
                                    f"{name}; SHIRA_{name.upper()}=0 forces it)")


def _walk_why(model: Any, d: tuple, F: int, loss: str, steps: int) -> str | None:
    assert loss in ("Expectile", "LogCosh") and steps >= 1
    p = model._init_params
    if p.get("leaf_estimation_backtracking") not in (None, "AnyImprovement", "No"):
        return f"leaf_estimation_backtracking {p['leaf_estimation_backtracking']} (AnyImprovement or No)"
    name = "cat_first_read_logcosh_min_rows" if loss == "LogCosh" else "cat_first_read_expectile_walk_min_rows"
    lo = _switch(name, "SHIRA_" + name.upper(), F)
    return None if d[3] >= lo else (f"{loss} on {d[3]} rows: under its switch's rows on this machine (switch {name}; "
                                    f"SHIRA_{name.upper()}=0 forces it)")


def _ce_why(model: Any, d: tuple, F: int) -> str | None:
    p = model._init_params
    y = np.asarray(d[1], dtype=np.float64)
    assert y.ndim == 1 and y.size == d[3]
    if not (bool(np.isfinite(y).all()) and float(y.min()) >= 0.0 and float(y.max()) <= 1.0):
        return "CrossEntropy labels outside [0, 1] (stock's own error)"
    if any(k in p for k in CLASS_WEIGHTS) or p.get("boost_from_average") is True:
        return "CrossEntropy with class weights or boost_from_average (not yet)"
    if logloss_steps(model) > 1:
        if not bool(np.all((y == 0.0) | (y == 1.0))):
            return "CrossEntropy on soft labels with several leaf steps: the soft-label walk is not measured a clear win"
        walk = _switch("cat_first_read_walk_min_rows", "SHIRA_CAT_FIRST_READ_WALK_MIN_ROWS", F)
        if d[3] < walk:
            return (f"CrossEntropy at 200 or more iterations on {d[3]} rows: under the walk's switch "
                    "(cat_first_read_walk_min_rows)")
    lo = _switch("cat_first_read_crossentropy_min_rows", "SHIRA_CAT_FIRST_READ_CROSSENTROPY_MIN_ROWS", F)
    return None if d[3] >= lo else (f"CrossEntropy on {d[3]} rows: not measured faster than stock there on this machine "
                                    "(switch cat_first_read_crossentropy_min_rows; "
                                    "SHIRA_CAT_FIRST_READ_CROSSENTROPY_MIN_ROWS=0 forces it)")


def _q_why(model: Any, d: tuple, F: int, depth: int) -> str | None:
    if depth > 12 or model._init_params.get("leaf_estimation_iterations") not in (None, 1):
        return "MAE / Quantile deeper than 12 or with several leaf steps"
    lo = _switch("cat_first_read_quantile_min_rows", "SHIRA_CAT_FIRST_READ_QUANTILE_MIN_ROWS", F)
    return None if d[3] >= lo else (f"MAE / Quantile on {d[3]} rows: not measured faster than stock there on this "
                                    "machine (switch cat_first_read_quantile_min_rows; SHIRA_CAT_FIRST_READ_QUANTILE_MIN_ROWS=0 "
                                    "forces it)")


def _mc_why(model: Any, y: np.ndarray, n: int, F: int, depth: int, cat: bool = False) -> str | None:
    K = int(np.unique(y).size)
    if K < 2 or K > 32:
        return f"MultiClass with {K} classes (2 to 32)"
    if "scale_pos_weight" in model._init_params or depth > 8:
        return "MultiClass with scale_pos_weight or depth over 8"
    if int(model._init_params.get("leaf_estimation_iterations") or 1) != 1:
        return "MultiClass with several leaf steps (not yet)"
    if model._init_params.get("boost_from_average") is True:
        return "MultiClass with boost_from_average (stock's own error)"
    if int(np.unique(y[_stride(n, sample_rows(F, cat))]).size) != K:
        return "a class missing from the stride sample"
    lo = _switch("cat_first_read_mc_min_rows", "SHIRA_CAT_FIRST_READ_MC_MIN_ROWS", F)
    return None if n >= lo else (f"MultiClass on {n} rows: not measured faster than stock there on this machine (switch "
                                 "cat_first_read_mc_min_rows; SHIRA_CAT_FIRST_READ_MC_MIN_ROWS=0 forces it)")


def why_not(model: Any, X: Any, y: Any, kw: dict, refused: bool = False) -> str | None:
    if not mode_on():
        return "SHIRA_CAT_MODE=ordinary"
    why = version_why()
    if why:
        return why
    if _loss(model) is None:
        return "not RMSE / Logloss on the regressor / classifier"
    extra = sorted(set(model._init_params) - ALLOWED)
    if extra:
        return f"parameters {extra}"
    other = sorted(k for k, v in kw.items() if v is not None and k not in FIT_PASSIVE and k not in FIT_TAKEN)
    if other:
        return f"fit arguments {other}"
    d, why = _data(X, y, kw.get("sample_weight"), _cats(model, kw))
    why = why or _params_why(model) or _eval_why(kw.get("eval_set"), _cats(model, kw))
    if why or d is None:
        return why or "no data"
    why = _acw_why(model, d)
    if why:
        return why
    n, F = d[3], d[4]
    depth = int(_p(model, "depth", 6))
    if depth > 16:
        return "depth over 16 (CatBoost's own limit)"
    cap = _switch("cat_first_read_max_depth", "SHIRA_CAT_FIRST_READ_MAX_DEPTH", F)
    if depth > cap:
        return (f"depth {depth}: over the depth the first read is measured to hold on this machine ({cap}; switch "
                "cat_first_read_max_depth; SHIRA_CAT_FIRST_READ_MAX_DEPTH=16 forces it)")
    cat = _cat_data(model, kw, d[0])
    m, why = choose_rows(model, d, kw, _loss(model, d[1]), refused)
    if why:
        return why
    assert 1 <= m <= n
    grow = model._init_params.get("grow_policy")
    if grow not in (None, "SymmetricTree", "Depthwise", "Lossguide"):
        return f"grow_policy {grow}"
    if grow in ("Depthwise", "Lossguide"):
        lo_grow = _switch("cat_first_read_grow_min_rows", "SHIRA_CAT_FIRST_READ_GROW_MIN_ROWS", F)
        if n < lo_grow:
            return (f"grow_policy {grow} on {n} rows: not measured faster than stock there on this machine (switch "
                    "cat_first_read_grow_min_rows; SHIRA_CAT_FIRST_READ_GROW_MIN_ROWS=0 forces it)")
    why = _ordered_why(model, grow, n, F, cat)
    if why:
        return why
    if cat:
        name = "cat_first_read_cat_clf_min_rows" if _loss(model, d[1]) in ("Logloss", "MultiClass") else "cat_first_read_cat_min_rows"
        lo_cat = _switch(name, "SHIRA_" + name.upper(), F)
        if n < lo_cat:
            return (f"categorical columns on {n} rows: not measured to hold held-out at a win on this machine (switch "
                    f"{name}; SHIRA_{name.upper()}=0 forces it)")
    loss = _loss(model, d[1])
    if loss == "MultiClass":
        return _mc_why(model, d[1], n, F, depth, cat)
    if loss == "Quantile":
        return _q_why(model, d, F, depth)
    if loss in REG_LOSSES:
        return _r_why(model, d, F, depth, loss)
    if loss == "CrossEntropy":
        return _ce_why(model, d, F)
    if _loss(model) == "Logloss" and logloss_steps(model) > 1:
        walk = _switch("cat_first_read_walk_min_rows", "SHIRA_CAT_FIRST_READ_WALK_MIN_ROWS", F)
        if n < walk:
            return (f"Logloss at 200 or more iterations on {n} rows: the multi-step leaf walk is not measured faster than "
                    "stock there on this machine (switch cat_first_read_walk_min_rows; "
                    "SHIRA_CAT_FIRST_READ_WALK_MIN_ROWS=0 forces it)")
    if _loss(model) == "Logloss" and np.unique(d[1]).size != 2:
        return "not two classes"
    return None


def _ordered_why(model: Any, grow: Any, n: int, F: int, cat: bool) -> str | None:
    bt = model._init_params.get("boosting_type")
    assert bt is None or isinstance(bt, str)
    assert n >= 1 and F >= 1
    if bt not in (None, "Plain", "Ordered"):
        return f"boosting_type {bt}"
    if bt != "Ordered":
        return None
    if grow not in (None, "SymmetricTree"):
        return f"boosting_type Ordered with grow_policy {grow} (stock's own error)"
    if cat:
        return "boosting_type Ordered with categorical columns: held-out not measured on the first read"
    lo = _switch("cat_first_read_ordered_min_rows", "SHIRA_CAT_FIRST_READ_ORDERED_MIN_ROWS", F)
    if n < lo:
        return (f"boosting_type Ordered on {n} rows: not measured faster than stock there on this machine (switch "
                "cat_first_read_ordered_min_rows; SHIRA_CAT_FIRST_READ_ORDERED_MIN_ROWS=0 forces it)")
    return None


def _params_why(model: Any) -> str | None:
    p = model._init_params
    if getattr(model, "_shira_kind", "") != "classifier" and any(k in p for k in CLASS_WEIGHTS):
        return "class weights outside the classifier"
    if p.get("od_type") not in (None, "Iter"):
        return f"od_type {p['od_type']} (only Iter)"
    if p.get("eval_metric") not in (None, _loss(model)):
        return f"eval_metric {p['eval_metric']} (only the loss)"
    if p.get("leaf_estimation_method") not in (None, "Newton"):
        return f"leaf_estimation_method {p['leaf_estimation_method']} (only Newton)"
    if p.get("leaf_estimation_backtracking") not in (None, "AnyImprovement", "No"):
        return f"leaf_estimation_backtracking {p['leaf_estimation_backtracking']} (AnyImprovement or No)"
    steps = p.get("leaf_estimation_iterations")
    if steps is not None and (int(steps) < 1 or int(steps) > 64 or (_loss(model) == "RMSE" and int(steps) != 1)):
        return f"leaf_estimation_iterations {steps} (RMSE: 1; Logloss: 1 to 64)"
    return None


def _acw_why(model: Any, d: tuple) -> str | None:
    p = model._init_params
    assert isinstance(p, dict) and len(d) == 5
    if (getattr(model, "_shira_kind", "") == "classifier" and p.get("boost_from_average") is True
            and p.get("auto_class_weights") not in (None, "None") and d[2] is not None):
        return "boost_from_average True with auto_class_weights and sample weights (stock's weighted class totals not reproduced)"
    return None


def _is_pool(X: Any) -> bool:
    return type(X).__name__ == "Pool" and hasattr(X, "get_features") and hasattr(X, "slice")


def _pool_why(p: Any) -> str | None:
    if p.is_quantized():
        return "a quantized Pool"
    if len(p.get_text_feature_indices()) or len(p.get_embedding_feature_indices()):
        return "a Pool with text / embedding columns"
    if p.get_group_id_hash() is not None:
        return "a Pool with groups"
    b = p.get_baseline()
    if b is not None and np.size(b) > 0:
        return "a Pool with a baseline"
    return None if p.has_label() else "a Pool without labels"


def _cats(model: Any, kw: dict) -> list:
    c = kw.get("cat_features")
    c = model._init_params.get("cat_features") if c is None else c
    return [] if c is None else list(c)


def _symmetric(model: Any) -> bool:
    return model.get_all_params().get("grow_policy", "SymmetricTree") == "SymmetricTree"


def _has_cat(model: Any) -> bool:
    return len(model.get_cat_feature_indices()) > 0


def _data(X: Any, y: Any, sw: Any, cats: list | None = None) -> tuple[tuple | None, str | None]:
    if _is_pool(X):
        why = _pool_why(X) or (None if y is None and sw is None else "a Pool with y or sample_weight beside it")
        if why:
            return None, why
        w = np.asarray(X.get_weight(), dtype=np.float64)
        w = None if bool(np.all(w == 1.0)) else w
        why = _w_why(w, X.num_row())
        return (None, why) if why else ((X, np.asarray(X.get_label()), w, X.num_row(), X.num_col()), None)
    why = _x_why(X, y, cats or []) or _w_why(sw, len(y) if np.ndim(y) == 1 else -1)
    if why:
        return None, why
    assert X.ndim == 2 and len(y) == X.shape[0]
    return (X, np.asarray(y), None if sw is None else np.asarray(sw, dtype=np.float64), X.shape[0], X.shape[1]), None


def _eval_why(ev: Any, cats: list | None = None) -> str | None:
    if ev is None:
        return None
    if isinstance(ev, list) and len(ev) != 1:
        return "more than one eval set"
    e = ev[0] if isinstance(ev, list) else ev
    if _is_pool(e):
        return _pool_why(e)
    if isinstance(e, tuple) and len(e) == 2:
        return _x_why(e[0], e[1], cats or [])
    return "eval_set is not a Pool or an (X, y) pair"


def _eval_pool(ev: Any, model: Any = None) -> Any:
    from catboost import Pool

    e = ev[0] if isinstance(ev, list) else ev
    cats = list(model.get_cat_feature_indices()) if model is not None and _has_cat(model) else None
    return e if _is_pool(e) else Pool(e[0], np.asarray(e[1]), cat_features=cats)


def _use_best(model: Any, kw: dict) -> bool:
    if kw.get("eval_set") is None:
        return False
    u = kw.get("use_best_model")
    u = model._init_params.get("use_best_model") if u is None else u
    return u is None or bool(u)


def auto_lr(loss: str, rows: int, iterations: int, best: bool = False, bfa: bool | None = None) -> float:
    flip = (loss == "RMSE" and bfa is False) or (loss == "Logloss" and bfa is True)
    a, b, c, d = AUTO_LR_STOCK[(loss, flip, best)]
    assert rows >= 1 and iterations >= 1
    lr = min(0.5, math.exp(a * math.log(rows) + b) * math.exp(c * math.log(iterations) + d) / math.exp(c * math.log(1000) + d))
    return float(np.float32(round(lr, 6)))


def _stride(n: int, m: int) -> np.ndarray:
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    assert rows.size == m and int(rows[0]) == 0 and int(rows[-1]) == n - 1
    return rows


def _is_frame(X: Any) -> bool:
    return type(X).__name__ == "DataFrame" and hasattr(X, "dtypes") and hasattr(X, "iloc")


def _x_why(X: Any, y: Any, cats: list | None = None) -> str | None:
    cats = cats or []
    if isinstance(X, np.ndarray):
        why = None if X.ndim == 2 and X.dtype.kind in "fiu" and not cats else "X is not a numeric 2-D array (or has cat_features)"
    elif _is_frame(X):
        odd = [c for i, (c, d) in enumerate(zip(X.columns, X.dtypes)) if d.kind not in "fiub" and c not in cats and i not in cats]
        why = None if not odd else f"DataFrame columns {odd[:3]} are not numeric and not in cat_features"
    else:
        return "X is not a numpy array, a numeric DataFrame or a numeric Pool"
    return why or (None if np.ndim(y) == 1 and len(y) == len(X) else "y is not one column with one value per row")


def _w_why(w: Any, n: int) -> str | None:
    if w is None:
        return None
    a = np.asarray(w)
    if a.ndim != 1 or a.shape[0] != n or a.dtype.kind not in "fiu":
        return "sample_weight is not one number per row"
    a = a.astype(np.float64)
    return None if bool(np.isfinite(a).all()) and float(a.min()) >= 0.0 and float(a.sum()) > 0.0 else "sample_weight values"


def _rows(X: Any, rows: np.ndarray) -> Any:
    if _is_pool(X):
        return X.slice(rows)
    return X.iloc[rows] if _is_frame(X) else np.ascontiguousarray(X[rows])


def _xv(X: Any) -> np.ndarray:
    if _is_pool(X):
        return np.ascontiguousarray(X.get_features(), dtype=np.float32)
    return np.ascontiguousarray(X.to_numpy(dtype=np.float64) if _is_frame(X) else X, dtype=np.float64)


def _class_row_weights(model: Any, yk: np.ndarray, w: np.ndarray | None) -> np.ndarray | None:
    p = model._init_params
    if not any(k in p for k in CLASS_WEIGHTS):
        return w
    K = len(model.classes_)
    base = np.ones(yk.size) if w is None else w
    cw = np.ones(K)
    if p.get("class_weights") is not None:
        g = p["class_weights"]
        cw = np.array([g[c] for c in model.classes_] if isinstance(g, dict) else list(g), dtype=np.float64)
    elif p.get("auto_class_weights") not in (None, "None"):
        tot = np.array([base[yk == k].sum() for k in range(K)])
        cw = tot.max() / tot
        cw = np.sqrt(cw.astype(np.float32)).astype(np.float64) if p["auto_class_weights"] == "SqrtBalanced" else cw
    if p.get("scale_pos_weight") is not None:
        assert K == 2
        cw[1] = cw[1] * float(p["scale_pos_weight"])
    assert cw.shape == (K,) and np.isfinite(cw).all()
    return (base.astype(np.float32) * cw.astype(np.float32)[yk]).astype(np.float64)


def _sample_fit(model: Any, stock_fit: Any, d: tuple, rows: np.ndarray, lr: float | None, kw: dict) -> None:
    keep = dict(model._init_params)
    extra = {k: kw[k] for k in ("eval_set", "early_stopping_rounds") if kw.get(k) is not None}
    if "eval_set" in extra:
        extra["use_best_model"] = False
    if d[2] is not None and not _is_pool(d[0]):
        extra["sample_weight"] = d[2][rows]
    if kw.get("cat_features") is not None and not _is_pool(d[0]):
        extra["cat_features"] = kw["cat_features"]
    try:
        if lr is not None:
            model._init_params["learning_rate"] = lr
        stock_fit(model, _rows(d[0], rows), None if _is_pool(d[0]) else d[1][rows], **extra)
    finally:
        model._init_params = keep
    assert model.is_fitted() and model._init_params == keep


ENGINE_SHAPE_ERRORS = (AssertionError, ValueError, RuntimeError, TypeError, ArithmeticError)
LOG = logging.getLogger("shira.catboost")


def _engine_shapes_why(model: Any, d: tuple, loss: str, kw: dict) -> str | None:
    sw = _switch("cat_first_read_shapes_engine", "SHIRA_CAT_FIRST_READ_SHAPES_ENGINE", d[4])
    if sw not in (1, 2):
        return "switch cat_first_read_shapes_engine off"
    if sw == 2 and loss != "Logloss":
        return "switch cat_first_read_shapes_engine 2: the two-class classifier only"
    if loss not in ("RMSE", "Logloss") or _cat_data(model, kw, d[0]) or not isinstance(d[0], np.ndarray):
        return "loss, categorical columns or X the engine's shapes do not take"
    if (kw.get("eval_set") is not None or kw.get("early_stopping_rounds") is not None) and _switch(
            "cat_first_read_shapes_engine_eval", "SHIRA_CAT_FIRST_READ_SHAPES_ENGINE_EVAL", d[4]) != 1:
        return "eval set or early stopping with the engine's shapes: switch cat_first_read_shapes_engine_eval off"
    from . import catboost as scb

    if scb._version_info() is None:
        return "the CatBoost library's build text was not found once: stock's sample fit"
    if loss == "Logloss" and not np.array_equal(np.unique(d[1]), np.array([0, 1])):
        return "Logloss labels other than 0 / 1 (not yet)"
    return None


def _engine_sample_fit(model: Any, d: tuple, rows: np.ndarray, lr: float | None, loss: str, th: int,
                       kw: dict | None = None) -> bool:
    from . import _cat_fit as cf
    from . import catboost as scb
    from . import export_cat

    kw = kw or {}
    params = dict(model._init_params)
    if lr is not None:
        params["learning_rate"] = lr
    if kw.get("early_stopping_rounds") is not None:
        params["early_stopping_rounds"] = kw["early_stopping_rounds"]
    if loss == "Logloss":
        params.pop("loss_function", None)
        params.pop("eval_metric", None)
    eng, ctl, why = cf.resolve(params, "regressor", int(rows.size), kw.get("eval_set") is not None)
    if why is not None or eng is None or ctl is None:
        LOG.info("shira.catboost: stock's sample fit (%s)", why)
        return False
    if loss == "Logloss":
        eng = dict(eng, loss_function="Logloss", boost_from_average=False, learning_rate=float(np.float32(eng["learning_rate"])))
    y_s = d[1][rows].astype(np.float64)
    try:
        data, why = cf.extract(np.ascontiguousarray(d[0][rows]), y_s, None if d[2] is None else d[2][rows], "generic", loss)
        assert why is None, why
        fm = cf.engine(data, eng, th)[0]
        doc = export_cat.to_doc(fm.borders, fm.splits, fm.leaves, [0, 1] if loss == "Logloss" else None, data["names"],
                                loss=loss)
    except ENGINE_SHAPE_ERRORS as e:
        LOG.warning("shira.catboost: the engine's shapes did not apply (%s: %s); stock's sample fit", type(e).__name__, e)
        return False
    lw = export_cat.leaf_weights(data["x"], data["w"], doc)
    for tree, v in zip(doc["oblivious_trees"], lw):
        tree["leaf_weights"] = [float(x) for x in v]
    export_cat.load_into(model, doc, fm.leaves)
    scb._stamp(model, ctl)
    run = len(fm.splits)
    if kw.get("eval_set") is not None and ctl["od_wait"] is not None:
        run = scb._detector(_eval_curve(model.copy(), kw["eval_set"], loss, th), int(ctl["od_wait"]))[1]
        if run < int(model.tree_count_):
            model.shrink(run)
            model._set_trained_model_attributes()
            scb._stamp(model, ctl)
    _engine_record(model, d, rows, lr)
    assert model.is_fitted() and int(model.tree_count_) == run
    return True


def _engine_record(model: Any, d: tuple, rows: np.ndarray, lr: float | None) -> None:
    from . import catboost as scb

    keep = dict(model._init_params)
    try:
        if lr is not None:
            model._init_params["learning_rate"] = lr
        rec = scb._nofit_param_record(model, (np.ascontiguousarray(d[0][rows]), d[1][rows], {}))
    finally:
        model._init_params = keep
    meta = model.get_metadata()
    for key, text in rec.items():
        meta[key] = text
    assert model._init_params == keep and "params" in model.get_metadata()


def _logloss_start(yv: np.ndarray, w: np.ndarray | None) -> float:
    ww = np.ones(yv.size) if w is None else np.asarray(w, dtype=np.float64)
    q = float(np.float32(np.sum(ww * yv) / np.sum(ww)))
    assert ww.shape == yv.shape and 0.0 < q < 1.0, "boost_from_average needs both classes with weight"
    b = -math.log(1.0 / q - 1.0)
    assert math.isfinite(b)
    return b


def _target(model: Any, y: np.ndarray, loss: str, w: np.ndarray | None, n: int) -> tuple:
    prm = model.get_all_params()
    if loss == "RMSE":
        yv = np.asarray(y, dtype=np.float64)
        mean = float(np.average(yv, weights=w))
        bias, obj = (mean if prm.get("boost_from_average", True) else 0.0), "reg:squarederror"
    elif loss == "CrossEntropy":
        yv = np.asarray(y, dtype=np.float32).astype(np.float64)
        bias, obj = 0.0, "binary:logistic"
        if w is None and not bool(np.all((yv == 0.0) | (yv == 1.0))):
            w = np.ones(n)
    else:
        cls = list(model.classes_)
        yv = (np.asarray(y) == cls[1]).astype(np.float64)
        w = _class_row_weights(model, yv.astype(np.int64), w)
        bias = _logloss_start(yv, w) if prm.get("boost_from_average") is True else 0.0
        obj = "binary:logistic"
    w32 = None if w is None else np.ascontiguousarray(w, dtype=np.float32)
    scale = 1.0 if w32 is None else float(w32.sum(dtype=np.float64)) / n
    assert yv.shape == (n,) and np.isfinite(scale)
    return yv, bias, obj, w32, float(prm["l2_leaf_reg"]) * scale


CORR_SWEEPS = 10
CORR_HOLD = 10
corr_log: list[dict] = []


def _corr_switch(name: str, default: int, lo: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() and int(v) >= lo else default


def _corrections(ids8: np.ndarray, nl: np.ndarray, yv: np.ndarray, w32: np.ndarray | None, obj: str, lam: float,
                 base: float, flat: np.ndarray, shape_rows: np.ndarray | None, th: int) -> np.ndarray:
    from shira_cpp import product_cpp

    sweeps = _corr_switch("SHIRA_CAT_CORR_SWEEPS", CORR_SWEEPS, 0)
    if shape_rows is None or sweeps == 0:
        if shape_rows is not None:
            corr_log.append({"sweeps": 0, "reason": "SHIRA_CAT_CORR_SWEEPS=0"})
        return flat
    T, offs = int(nl.size), np.concatenate([[0], np.cumsum(nl)]).astype(np.int64)
    mat = np.zeros((T, int(nl.max())), dtype=np.float32)
    for t in range(T):
        mat[t, :int(nl[t])] = flat[offs[t]:offs[t + 1]]
    lv, _, clog = product_cpp.corr_drive(ids8, nl, yv, w32, obj, lam, 0.0, base, mat, shape_rows, sweeps,
                                         _corr_switch("SHIRA_CAT_CORR_HOLD", CORR_HOLD, 2), nthread=th)
    corr_log.append(clog)
    LOG.info("shira.catboost: amortised corrections: %d of at most %d sweeps kept by the held-back loss (steps "
             "%s); SHIRA_CAT_CORR_SWEEPS=0 turns them off", clog["sweeps"], sweeps, clog.get("damps"))
    if lv is None:
        return flat
    out = np.concatenate([lv[t, :int(nl[t])] for t in range(T)]).astype(np.float64)
    assert out.size == flat.size and np.isfinite(out).all()
    return out


def _values(model: Any, X: np.ndarray, y: np.ndarray, loss: str, th: int,
            w: np.ndarray | None = None, shape_rows: np.ndarray | None = None) -> tuple[np.ndarray, float, float, float]:
    from shira_cpp import node_stats_cpp, product_cpp

    if loss == "MultiClass":
        return _values_mc(model, X, y, th, w)
    if loss == "Quantile":
        return _values_qx(model, X, y, th, w)
    if loss in REG_LOSSES:
        return _values_r(model, X, y, loss, th, w)
    yv, bias, obj, w32, lam = _target(model, y, loss, w, len(y))
    if int(np.max(model.get_tree_leaf_counts())) > 256:
        t0 = time.perf_counter()
        flat, t_ids = _values_deep(model, X, yv, bias, obj, w32, lam, th)
        b_out = bias if obj == "binary:logistic" else float(np.float32(bias))
        return flat, b_out, t_ids, time.perf_counter() - t0 - t_ids
    t0 = time.perf_counter()
    ids8, nl = _ids8(model, X, th)
    T = model.tree_count_
    assert ids8.shape == (T, len(y)) and int(nl.sum()) == model.get_leaf_values().size
    t1 = time.perf_counter()
    prm = model.get_all_params()
    steps = int(prm.get("leaf_estimation_iterations", 1))
    steps = value_steps(model, loss, steps)
    if loss in ("Logloss", "CrossEntropy") and steps > 1:
        flat, taken = node_stats_cpp.cat_logloss_newton(ids8, nl, yv, None if w32 is None else w32.astype(np.float64), bias,
                                                        float(model.learning_rate_), lam, steps, th,
                                                        force=prm.get("leaf_estimation_backtracking") == "No")
        assert flat.size == int(nl.sum()) and np.isfinite(flat).all() and int(taken.max()) <= steps
        flat = _corrections(ids8, nl, yv, w32, obj, lam, bias, flat, shape_rows, th)
        return flat, bias, t1 - t0, time.perf_counter() - t1
    if loss in ("Logloss", "CrossEntropy"):
        flat = node_stats_cpp.cat_logloss_onestep(ids8, nl, yv, None if w32 is None else w32.astype(np.float64), bias,
                                                  float(model.learning_rate_), lam, th)
        assert flat.size == int(nl.sum()) and np.isfinite(flat).all()
        flat = _corrections(ids8, nl, yv, w32, obj, lam, bias, flat, shape_rows, th)
        return flat, bias, t1 - t0, time.perf_counter() - t1
    val = product_cpp.product_window(ids8, 0, len(y), nl, yv.astype(np.float32), float(np.float32(bias)), obj,
                                     eta=float(model.learning_rate_), reg_lambda=lam, min_child_weight=0.0,
                                     max_delta_step=0.0, nthread=th, weight=w32)
    flat = np.concatenate([val[t, :int(nl[t])] for t in range(T)]).astype(np.float64)
    assert flat.size == int(nl.sum()) and np.isfinite(flat).all()
    return flat, float(np.float32(bias)), t1 - t0, time.perf_counter() - t1


ONESTEP16_MAX_LEAVES = 4096


def _deep_logloss_block(ids: np.ndarray, nl: np.ndarray, yv: np.ndarray, w: np.ndarray | None, eta: float, lam: float,
                        steps: int, th: int, F: np.ndarray, force: bool) -> np.ndarray:
    from shira_cpp import node_stats_cpp

    assert ids.ndim == 2 and ids.shape[0] == nl.size and F.shape == (ids.shape[1],)
    if steps == 1 and int(nl.max()) <= ONESTEP16_MAX_LEAVES:
        return node_stats_cpp.cat_logloss_onestep(ids, nl, yv, w, 0.0, eta, lam, th, F0=F)
    flat, taken = node_stats_cpp.cat_logloss_newton(ids, nl, yv, w, 0.0, eta, lam, steps, th, F0=F, force=force)
    assert int(taken.max()) <= steps
    return flat


def _values_deep(model: Any, X: np.ndarray, yv: np.ndarray, bias: float, obj: str, w32: np.ndarray | None, lam: float,
                 th: int) -> tuple[np.ndarray, float]:
    from shira_cpp import product_cpp

    n, T = yv.size, int(model.tree_count_)
    nl = np.asarray(model.get_tree_leaf_counts(), dtype=np.int32)
    assert nl.size == T and int(nl.max()) <= 65536
    doc = None if _has_cat(model) else _json_doc(model)
    block = max(1, min(T, (1 << 25) // max(1, n)))
    prm = model.get_all_params()
    steps = int(prm.get("leaf_estimation_iterations", 1)) if obj == "binary:logistic" else 1
    steps = value_steps(model, _loss(model), steps)
    pred = np.full(n, np.float32(bias), dtype=np.float32)
    F = np.full(n, float(bias)) if obj == "binary:logistic" else None
    y32, out, t_ids = yv.astype(np.float32), [], 0.0
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        a = time.perf_counter()
        ids = _ids_block(model, X, th, t0, t1, doc)
        t_ids += time.perf_counter() - a
        if F is not None:
            flat_b = _deep_logloss_block(ids, nl[t0:t1], yv, None if w32 is None else w32.astype(np.float64),
                                         float(model.learning_rate_), lam, steps, th, F,
                                         prm.get("leaf_estimation_backtracking") == "No")
            offs = np.concatenate([[0], np.cumsum(nl[t0:t1])]).astype(np.int64)
            for k in range(t1 - t0):
                F += flat_b[offs[k]:offs[k + 1]][ids[k].astype(np.int64)]
            out.extend(flat_b[offs[k]:offs[k + 1]] for k in range(t1 - t0))
            continue
        val, pred = product_cpp.product(ids, nl[t0:t1], y32, float(np.float32(bias)), obj, eta=float(model.learning_rate_),
                                        reg_lambda=lam, min_child_weight=0.0, max_delta_step=0.0, weight=w32, nthread=th,
                                        pred=pred)
        out.extend(val[k, :int(nl[t0 + k])] for k in range(t1 - t0))
    flat = np.concatenate(out).astype(np.float64)
    assert flat.size == int(nl.sum()) and np.isfinite(flat).all()
    return flat, t_ids


def _values_mc(model: Any, X: np.ndarray, y: np.ndarray, th: int, w: np.ndarray | None) -> tuple:
    from shira_cpp import node_stats_cpp

    t0 = time.perf_counter()
    ids8, nl = _ids8(model, X, th)
    t1 = time.perf_counter()
    cls = np.asarray(model.classes_)
    order = np.argsort(cls, kind="stable")
    yk = order[np.clip(np.searchsorted(cls[order], np.asarray(y)), 0, cls.size - 1)]
    assert np.array_equal(cls[yk], np.asarray(y)) and ids8.shape[0] == model.tree_count_
    w = _class_row_weights(model, yk, w)
    w32 = None if w is None else np.ascontiguousarray(w, dtype=np.float32)
    scale = 1.0 if w32 is None else float(w32.sum(dtype=np.float64)) / len(y)
    lam = float(np.float32(float(model.get_all_params()["l2_leaf_reg"]) * scale))
    flat = node_stats_cpp.cat_multiclass_values(ids8, nl, yk, None if w32 is None else w32.astype(np.float64), int(cls.size),
                                                float(model.learning_rate_), lam, th)
    assert flat.size == model.get_leaf_values().size and np.isfinite(flat).all()
    return flat, [0.0] * int(cls.size), t1 - t0, time.perf_counter() - t1


def _nudged_quantile(v: np.ndarray, alpha: float, delta: float) -> float:
    v32 = np.sort(np.asarray(v, dtype=np.float32))
    am = alpha * v32.size
    k = min(v32.size - 1, max(0, math.ceil(am) - 1))
    assert v32.size >= 1 and 0 <= k < v32.size
    return float(np.float32(float(v32[k]) + (delta if am == math.floor(am) else -delta)))


def _values_q(model: Any, X: np.ndarray, y: np.ndarray, th: int) -> tuple:
    from shira_cpp import node_stats_cpp

    alpha, delta = _quantile_params(model._init_params.get("loss_function"))
    t0 = time.perf_counter()
    ids8, nl = _ids8(model, X, th)
    t1 = time.perf_counter()
    yv = np.asarray(y, dtype=np.float64)
    bias = _nudged_quantile(yv, alpha, delta) if model.get_all_params().get("boost_from_average", True) else 0.0
    flat = node_stats_cpp.cat_quantile_values(ids8, nl, yv, bias, alpha, delta, float(model.learning_rate_), th)
    assert flat.size == model.get_leaf_values().size and np.isfinite(flat).all()
    return flat, bias, t1 - t0, time.perf_counter() - t1


def _weighted_nudged_quantile(v: np.ndarray, w: np.ndarray | None, alpha: float, delta: float) -> float:
    v32 = np.asarray(v, dtype=np.float32)
    w64 = np.ones(v32.size) if w is None else np.asarray(w, dtype=np.float32).astype(np.float64)
    assert v32.size >= 1 and w64.size == v32.size and float(w64.sum()) > 0.0
    o = np.argsort(v32, kind="stable")
    cw = np.cumsum(w64[o])
    k = min(int(np.searchsorted(cw, alpha * cw[-1], side="left")), v32.size - 1)
    past = alpha * cw[-1] > (cw[k - 1] if k > 0 else 0.0) + alpha * w64[o][k]
    up = past and k + 1 < v32.size and v32[o][k + 1] > v32[o][k]
    d32 = np.float32(delta if up else -delta)
    return float(np.float32(v32[o][k] + d32))


def _values_qx(model: Any, X: np.ndarray, y: np.ndarray, th: int, w: np.ndarray | None) -> tuple:
    from shira_cpp import node_stats_cpp

    alpha, delta = _quantile_params(model._init_params.get("loss_function"))
    yv = np.asarray(y, dtype=np.float32).astype(np.float64)
    bias = _weighted_nudged_quantile(yv, w, alpha, delta) if model.get_all_params().get("boost_from_average", True) else 0.0
    nl = np.asarray(model.get_tree_leaf_counts(), dtype=np.int32)
    eta = float(model.learning_rate_)
    t0 = time.perf_counter()
    if int(nl.max()) <= 256:
        ids8, nl8 = _ids8(model, X, th)
        t_ids = time.perf_counter() - t0
        flat = node_stats_cpp.cat_quantile_values_x(ids8, nl8, yv, bias, alpha, delta, eta, th, f32=True)
    else:
        flat, t_ids = _values_q_deep(model, X, yv, bias, alpha, delta, eta, nl, th)
    assert flat.size == model.get_leaf_values().size and np.isfinite(flat).all()
    return flat, bias, t_ids, time.perf_counter() - t0 - t_ids


def _values_q_deep(model: Any, X: np.ndarray, yv: np.ndarray, bias: float, alpha: float, delta: float, eta: float,
                   nl: np.ndarray, th: int) -> tuple[np.ndarray, float]:
    from shira_cpp import node_stats_cpp

    n, T = yv.size, int(model.tree_count_)
    assert int(nl.max()) <= 4096 and nl.size == T
    doc = None if _has_cat(model) else _json_doc(model)
    block = max(1, min(T, (1 << 25) // max(1, n)))
    F, out, t_ids = np.full(n, float(bias)), [], 0.0
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        a = time.perf_counter()
        ids = _ids_block(model, X, th, t0, t1, doc)
        t_ids += time.perf_counter() - a
        flat_b = node_stats_cpp.cat_quantile_values_x(ids, nl[t0:t1], yv, bias, alpha, delta, eta, th, F0=F, f32=True)
        offs = np.concatenate([[0], np.cumsum(nl[t0:t1])]).astype(np.int64)
        for k in range(t1 - t0):
            F += flat_b[offs[k]:offs[k + 1]][ids[k].astype(np.int64)]
        out.append(flat_b)
    return np.concatenate(out), t_ids


def _values_r(model: Any, X: np.ndarray, y: np.ndarray, loss: str, th: int, w: np.ndarray | None) -> tuple:
    name, param = _reg_params(model._init_params.get("loss_function"))
    assert name == loss
    yv = np.asarray(y, dtype=np.float32).astype(np.float64)
    w32 = None if w is None else np.ascontiguousarray(w, dtype=np.float32)
    scale = 1.0 if w32 is None else float(w32.sum(dtype=np.float64)) / yv.size
    prm = model.get_all_params()
    k = {"loss": loss, "param": param, "y": yv, "w": None if w32 is None else w32.astype(np.float64),
         "lam": float(prm["l2_leaf_reg"]) * scale, "eta": float(model.learning_rate_),
         "steps": int(prm.get("leaf_estimation_iterations", 1)), "th": th,
         "force": prm.get("leaf_estimation_backtracking") == "No"}
    nl = np.asarray(model.get_tree_leaf_counts(), dtype=np.int32)
    t0 = time.perf_counter()
    if int(nl.max()) <= 256:
        ids8, nl8 = _ids8(model, X, th)
        t_ids = time.perf_counter() - t0
        flat = _reg_block(ids8, nl8, None, k)
    else:
        flat, t_ids = _values_r_deep(model, X, nl, k)
    assert flat.size == model.get_leaf_values().size and np.isfinite(flat).all()
    return flat, 0.0, t_ids, time.perf_counter() - t0 - t_ids


def _reg_block(ids: np.ndarray, nl: np.ndarray, F0: np.ndarray | None, k: dict) -> np.ndarray:
    from shira_cpp import node_stats_cpp

    assert ids.ndim == 2 and ids.shape[0] == nl.size
    if k["loss"] == "LogCosh" or (k["loss"] == "Expectile" and k["steps"] > 1):
        w4b = None if k["loss"] == "LogCosh" else k["w"]
        flat, taken = node_stats_cpp.cat_reg_walk(k["loss"], k["param"], ids, nl, k["y"], w4b, 0.0, k["eta"], k["lam"],
                                                  max(1, k["steps"]), k["th"], F0=F0, force=k["force"])
        assert int(taken.min()) >= 1
        return flat
    if k["loss"] == "Poisson" and k["steps"] > 1:
        flat, taken = node_stats_cpp.cat_poisson_walk(ids, nl, k["y"], k["w"], 0.0, k["eta"], k["lam"], k["steps"], k["th"],
                                                      F0=F0, force=k["force"])
        assert int(taken.max()) <= k["steps"]
        return flat
    return node_stats_cpp.cat_reg_onestep(k["loss"], k["param"], ids, nl, k["y"], k["w"], 0.0, k["eta"], k["lam"], k["th"], F0=F0)


def _values_r_deep(model: Any, X: Any, nl: np.ndarray, k: dict) -> tuple[np.ndarray, float]:
    n, T = k["y"].size, nl.size
    assert int(nl.max()) <= 4096 and T == int(model.tree_count_)
    doc = None if _has_cat(model) else _json_doc(model)
    block = max(1, min(T, (1 << 25) // max(1, n)))
    F, out, t_ids = np.zeros(n), [], 0.0
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        a = time.perf_counter()
        ids = _ids_block(model, X, k["th"], t0, t1, doc)
        t_ids += time.perf_counter() - a
        flat_b = _reg_block(ids, nl[t0:t1], F, k)
        offs = np.concatenate([[0], np.cumsum(nl[t0:t1])]).astype(np.int64)
        for j in range(t1 - t0):
            F += flat_b[offs[j]:offs[j + 1]][ids[j].astype(np.int64)]
        out.extend(flat_b[offs[j]:offs[j + 1]] for j in range(t1 - t0))
    return np.concatenate(out), t_ids


def _pool_all(model: Any, X: Any) -> Any:
    from catboost import Pool

    if _is_pool(X):
        return X
    if _has_cat(model):
        return Pool(X, cat_features=list(model.get_cat_feature_indices()))
    return Pool(np.ascontiguousarray(X, dtype=np.float32))


def _ids8(model: Any, X: Any, th: int) -> tuple[np.ndarray, np.ndarray]:
    from shira_cpp import node_stats_cpp

    if not _has_cat(model) and _symmetric(model):
        return node_stats_cpp.cat_oblivious_ids(np.ascontiguousarray(X, dtype=np.float32), _json_doc(model), th)
    if not _has_cat(model):
        T = int(model.tree_count_)
        ids16 = _ids_block(model, X, th, 0, T, _json_doc(model))
        nl = np.asarray(model.get_tree_leaf_counts(), dtype=np.int32)
        assert int(nl.max()) <= 256 and ids16.shape[0] == T
        return ids16.astype(np.uint8), nl
    pool = _pool_all(model, X)
    nl = np.asarray(model.get_tree_leaf_counts(), dtype=np.int32)
    n, T = int(pool.num_row()), int(model.tree_count_)
    assert nl.size == T and int(nl.max()) <= 256
    out = np.empty((T, n), dtype=np.uint8)
    block = max(1, min(T, (1 << 25) // max(1, n)))
    for t0 in range(0, T, block):
        t1 = min(T, t0 + block)
        out[t0:t1] = np.asarray(model.calc_leaf_indexes(pool, ntree_start=t0, ntree_end=t1, thread_count=th)).T
    return out, nl


def _json_doc(model: Any) -> dict:
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.json")
        model.save_model(path, format="json")
        with open(path) as fh:
            doc = json.load(fh)
    key = "oblivious_trees" if "oblivious_trees" in doc else "trees"
    assert key in doc and len(doc[key]) == model.tree_count_
    return doc


def _ids_block(model: Any, X: Any, th: int, t0: int, t1: int, doc: dict | None) -> np.ndarray:
    from shira_cpp import node_stats_cpp

    if _has_cat(model):
        li = model.calc_leaf_indexes(_pool_all(model, X), ntree_start=t0, ntree_end=t1, thread_count=th)
        return np.ascontiguousarray(np.asarray(li).T, dtype=np.uint16)
    assert doc is not None and t1 > t0
    x32 = np.ascontiguousarray(X, dtype=np.float32)
    if _symmetric(model):
        return node_stats_cpp.cat_oblivious_ids16(x32, doc, th, t0, t1)[0]
    return node_stats_cpp.cat_tree_ids(x32, doc, model, th, t0, t1)[0]


def _engine_history(model: Any, d: tuple, rows: np.ndarray, metric: str, th: int, curve: np.ndarray) -> dict:
    from catboost import Pool

    w = None if d[2] is None else d[2][rows]
    learn = np.asarray(model.eval_metrics(Pool(np.ascontiguousarray(d[0][rows]), d[1][rows], weight=w), [metric],
                                          thread_count=th)[metric], dtype=np.float64)
    assert learn.size == curve.size == model.tree_count_ and np.isfinite(learn).all()
    return {"metric": metric, "validation": curve, "learn": learn, "best": int(np.argmin(curve))}


def _eval_curve(model: Any, ev: Any, loss: str, th: int) -> np.ndarray:
    curve = np.asarray(model.eval_metrics(_eval_pool(ev, model), [loss], thread_count=th)[loss], dtype=np.float64)
    assert curve.size == model.tree_count_ and np.isfinite(curve).all()
    return curve


def _cut(model: Any, ev: Any, loss: str, th: int, curve: np.ndarray | None = None) -> int:
    curve = _eval_curve(model, ev, loss, th) if curve is None else curve
    assert curve.size == model.tree_count_
    best = int(np.argmin(curve))
    if best + 1 < model.tree_count_:
        model.shrink(best + 1)
    assert model.tree_count_ == best + 1
    return best


def fit(model: Any, stock_fit: Any, X: Any, y: Any, kw: dict | None = None, refused: bool = False) -> str:
    from ._threads import env_budget

    kw = kw or {}
    d, why = _data(X, y, kw.get("sample_weight"), _cats(model, kw))
    loss = _loss(model, None if d is None else d[1])
    assert why is None and d is not None and loss is not None
    n, best_on = d[3], _use_best(model, kw)
    m, why = choose_rows(model, d, kw, loss, refused)
    assert why is None and 1 <= m <= n, why
    rows = _stride(n, m)
    given = _p(model, "learning_rate")
    iters = int(_p(model, "iterations", 1000))
    lr = None if given is not None or loss not in AUTO_LR else auto_lr(loss, n, iters, best_on,
                                                                       model._init_params.get("boost_from_average"))
    th = int(model._init_params.get("thread_count", -1))
    th = max(1, min(64, env_budget() if th < 1 else th))
    t0 = time.perf_counter()
    engine = _engine_shapes_why(model, d, loss, kw) is None and _engine_sample_fit(model, d, rows, lr, loss, th, kw)
    if not engine and rows.size >= n:
        raise RuntimeError("SHIRA's engine did not grow the all-row shapes")
    if not engine:
        _sample_fit(model, stock_fit, d, rows, lr, kw)
    grower = "SHIRA's CatBoost engine" if engine else "stock CatBoost"
    t1 = time.perf_counter()
    src, yv, wv = (d[0] if _has_cat(model) else _xv(d[0])), d[1], d[2]
    if _has_cat(model):
        out = np.ones(n, dtype=bool)
        out[rows] = False
        keep = np.flatnonzero(out)
        assert keep.size == n - rows.size and keep.size > 0
        src, yv, wv = _rows(d[0], keep), yv[keep], None if wv is None else wv[keep]
    srows = np.empty(0, dtype=np.int64) if len(yv) < n else rows
    val, bias, t_ids, t_val = _values(model, src, yv, loss, th, wv, shape_rows=srows)
    model.set_leaf_values(val)
    model.set_scale_and_bias(1.0, bias)
    grown = int(model.tree_count_)
    metric = model._init_params.get("loss_function") if loss == "Quantile" or loss in REG_LOSSES else loss
    curve = None
    if engine and kw.get("eval_set") is not None:
        curve = _eval_curve(model, kw["eval_set"], metric, th)
        model._shira_fr = _engine_history(model, d, rows, metric, th, curve)
    best = _cut(model, kw["eval_set"], metric, th, curve) if best_on else None
    cut = "" if best is None else f"; cut at the all-row eval curve's best iteration: {best + 1} of {grown} trees"
    line = (f"fitted the first-read way (SHIRA's default): {grown} tree shapes grown by {grower} on a "
            f"stride sample of {rows.size} of {n} rows at the all-row learning rate {model.learning_rate_:.6g} (not stock's trees; the "
            f"tree count, borders, learn history and leaf weights are the sample fit's), every leaf value solved on "
            f"{'the ' + str(len(yv)) + ' rows outside the sample (the CTRs hold the sample rows own targets)' if len(yv) < n else 'all ' + str(n) + ' rows'}"
            f"{cut} ({time.perf_counter() - t0:.2f} s); SHIRA_CAT_MODE=ordinary turns it off")
    log.append({"n": n, "m": int(rows.size), "trees": int(model.tree_count_), "grown": grown, "best": best,
                "lr": float(model.learning_rate_), "lr_formula": lr, "t_fit": t1 - t0, "t_ids": t_ids, "t_values": t_val,
                "line": line, "grower": "shira" if engine else "stock"})
    del log[:-64]
    return line
