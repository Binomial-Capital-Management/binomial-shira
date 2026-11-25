from __future__ import annotations

import copy
import logging
import os
import platform
import sys
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from operator import attrgetter
from types import SimpleNamespace
from typing import Any

import lightgbm as _lgb
import numpy as np
from lightgbm import basic as _basic
from lightgbm import callback as _cb
from lightgbm import engine as _engine
from lightgbm import sklearn as _sk

from . import _lgb_first_read as _first_read
from . import _machine
from . import _lgb_fit as _fit
from . import export_lgb as export

LOG = logging.getLogger("shira.lightgbm")
STOCK_VERSION = "4.6.0"


def version_reason() -> str | None:
    v = str(getattr(_lgb, "__version__", "unknown"))
    if v == STOCK_VERSION:
        return None
    return f"lightgbm {v} is installed; SHIRA's engines and first reads reproduce lightgbm {STOCK_VERSION} only"
MAX_ROUNDS = 100_000
X86_GUARD: bool = sys.platform == "linux" and platform.machine().lower() in ("x86_64", "amd64")
X86_ALLOWED = ("regression", "regression_l1", "huber", "quantile", "poisson", "multiclass", "lambdarank",
               "binary", "cross_entropy", "tweedie", "gamma", "multiclassova")
last_backend: str | None = None
first_read_reason: str | None = None
backend_history: list[str] = []

_STOCK: dict[str, Any] = {"train": _engine.train, "LGBMRegressor": _sk.LGBMRegressor,
                          "LGBMClassifier": _sk.LGBMClassifier, "LGBMRanker": _sk.LGBMRanker}
Dataset = _basic.Dataset
Unsupported = _fit.Unsupported


def _is(*vals: str) -> Callable[[str], bool]:
    return lambda v: v in vals


def _anything(_: str) -> bool:
    return True


OBJECTIVES = ("regression", "binary", "multiclass", "multiclassova", "regression_l1", "huber", "quantile",
              "poisson", "gamma", "tweedie", "cross_entropy", "lambdarank")
_ENGINE = ("learning_rate", "num_leaves", "max_depth", "min_data_in_leaf", "min_sum_hessian_in_leaf",
           "bagging_fraction", "bagging_freq", "bagging_seed", "feature_fraction", "feature_fraction_bynode",
           "feature_fraction_seed", "lambda_l1", "lambda_l2", "min_gain_to_split", "drop_rate", "max_drop",
           "skip_drop", "xgboost_dart_mode", "uniform_drop", "drop_seed", "top_rate", "other_rate",
           "path_smooth", "min_data_in_bin", "bin_construct_sample_cnt", "data_random_seed", "use_missing",
           "zero_as_missing", "num_class", "sigmoid", "boost_from_average", "alpha", "poisson_max_delta_step",
           "tweedie_variance_power", "lambdarank_truncation_level", "lambdarank_norm", "label_gain",
           "max_delta_step",
           "scale_pos_weight",
           "is_unbalance")
_LAYER = ("num_iterations", "metric", "early_stopping_round", "early_stopping_min_delta", "first_metric_only",
          "eval_at", "multi_error_top_k", "auc_mu_weights", "feature_pre_filter", "enable_bundle", "seed",
          "verbosity", "num_threads")
_NO_EFFECT = ("deterministic", "histogram_pool_size", "is_enable_sparse", "objective_seed",
              "extra_seed", "saved_feature_importance_type", "min_data_per_group", "max_cat_threshold",
              "cat_l2", "cat_smooth", "max_cat_to_onehot")
SUPPORTED: dict[str, Callable[[str], bool]] = {
    "objective": _is(*OBJECTIVES),
    "boosting": lambda v: v in ("gbdt", "dart") or (v == "rf" and _first_read.mode()),
    "monotone_constraints": lambda v: v == "" or _first_read.mode(),
    "monotone_constraints_method": lambda v: v in ("", "basic"), "monotone_penalty": lambda v: float(v or 0) == 0.0,
    "linear_tree": lambda v: v == "0" or _first_read.mode(), "linear_lambda": _anything,
    "data_sample_strategy": _is("bagging", "goss"),
    "max_bin": lambda v: 2 <= int(v) <= 65535,
    "extra_trees": lambda v: v == "0" or _first_read.mode(), "extra_seed": _anything,
    "force_col_wise": _anything, "force_row_wise": _anything,
    **{k: _anything for k in _ENGINE},
    **{k: _anything for k in _LAYER},
    **{k: _anything for k in _NO_EFFECT},
}

_DEFAULTS: dict[tuple[str, str], dict[str, str]] = {}


def _defaults(cfg: dict[str, str]) -> dict[str, str]:
    key = (cfg["objective"], cfg["num_class"])
    if key not in _DEFAULTS:
        base: dict[str, Any] = {"objective": key[0]}
        if int(key[1]) > 1:
            base["num_class"] = int(key[1])
        _DEFAULTS[key] = _fit.resolve(base, ["Column_0"]).cfg
    assert _DEFAULTS[key]["objective"] == key[0]
    return _DEFAULTS[key]


SHAPE_ONLY = ("interaction_constraints", "cegb_tradeoff", "cegb_penalty_split", "cegb_penalty_feature_lazy",
              "cegb_penalty_feature_coupled")


BUILTIN_ROUTE = {"fair": "fair", "mape": "mape", "mean_absolute_percentage_error": "mape",
                 "rank_xendcg": "xendcg", "xendcg": "xendcg", "xe_ndcg": "xendcg", "xe_ndcg_mart": "xendcg",
                 "xendcg_mart": "xendcg"}


def _shape_free(eff: dict[str, Any], names: list[str]) -> tuple[Any, list[str]]:
    amap = _fit.alias_to_main()
    keys = [k for k in eff if amap.get(k, k) in SHAPE_ONLY]
    if not keys:
        return None, []
    rs = _fit.resolve({k: v for k, v in eff.items() if k not in keys}, names)
    if _table_reason(rs.cfg) is not None:
        return None, []
    return rs, sorted({amap.get(k, k) for k in keys})


def _table_reason(cfg: dict[str, str]) -> str | None:
    if X86_GUARD and cfg.get("objective") not in X86_ALLOWED:
        return f"x86 guard: objective {cfg.get('objective')} is not on the Linux x86 allow-list"
    dflt = _defaults(cfg)
    for k, v in cfg.items():
        check = SUPPORTED.get(k)
        if check is None:
            if v != dflt.get(k, v):
                return f"param {k}={v} is not in SHIRA's table"
            continue
        try:
            ok = check(v)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return f"param {k}={v} is outside what SHIRA reproduces exactly"
    if cfg["boosting"] == "dart" and cfg["early_stopping_round"] not in ("0", ""):
        return "dart with early stopping"
    assert cfg["objective"] in OBJECTIVES
    return None


def why_not(params: dict[str, Any], n_features: int = 1) -> str | None:
    assert n_features >= 1
    if version_reason() is not None:
        return version_reason()
    try:
        rs = _fit.resolve(dict(params or {}), [f"Column_{j}" for j in range(n_features)])
        reason = _table_reason(rs.cfg)
        if reason is None:
            _fit.check_engine(_fit.engine_params(rs, dict(params or {})))
    except Unsupported as e:
        return str(e)
    return reason


_BAD_NAME = set(' \t\n",:[]{}')


def _matrix(data: Any) -> tuple[np.ndarray, list[str] | None]:
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.dtype.kind in "fiub":
        keep = data.dtype in (np.float32, np.float64)
        return np.ascontiguousarray(data if keep else data.astype(np.float32), dtype=np.float64), None
    if type(data).__name__ == "DataFrame" and hasattr(data, "dtypes"):
        dts = list(data.dtypes)
        if dts and all(isinstance(dt, np.dtype) and dt.kind in "fiub" for dt in dts):
            target = np.result_type(*[dt.type for dt in dts], np.float32)
            x = np.ascontiguousarray(data.to_numpy(dtype=target), dtype=np.float64)
            return x, [str(c) for c in data.columns]
    if hasattr(data, "tocsr") and hasattr(data, "toarray") and len(getattr(data, "shape", ())) == 2:
        if data.dtype not in (np.float32, np.float64):
            raise Unsupported(f"sparse input of dtype {data.dtype} (stock raises too: float32 / float64 only)")
        return np.ascontiguousarray(data.toarray(), dtype=np.float64), None
    raise Unsupported(f"input type {type(data).__name__} (numpy arrays, pandas frames and scipy sparse matrices only)")


def _frame_codes(data: Any, pc: list | None) -> tuple[Any, list[int] | None]:
    if type(data).__name__ != "DataFrame" or not hasattr(data, "dtypes"):
        return data, None
    cat_cols = [c for c, t in zip(data.columns, data.dtypes) if isinstance(t, _lgb.compat.pd_CategoricalDtype)]
    if not cat_cols:
        return data, None
    data = data.copy(deep=False)
    auto = [j for j, c in enumerate(data.columns) if c in cat_cols and not data[c].cat.ordered]
    if pc is not None:
        if len(cat_cols) != len(pc):
            raise Unsupported("train and valid dataset categorical_feature do not match (stock raises too)")
        for col, category in zip(cat_cols, pc):
            if list(data[col].cat.categories) != list(category):
                data[col] = data[col].cat.set_categories(category)
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes).replace({-1: np.nan})
    assert len(auto) <= len(cat_cols)
    return data, auto


def _vector(v: Any, dtype: Any, what: str) -> np.ndarray:
    a = np.asarray(v.to_numpy() if hasattr(v, "to_numpy") else v)
    if a.ndim != 1 or a.dtype.kind not in "fiub":
        raise Unsupported(f"{what} is not a 1-D numeric array")
    if not np.isfinite(a.astype(np.float64)).all():
        raise Unsupported(f"{what} has non-finite values")
    return np.ascontiguousarray(a, dtype=dtype)


def _names(ds: Any, from_frame: list[str] | None, nf: int) -> list[str]:
    fn = ds.feature_name
    names = from_frame if (isinstance(fn, str) and fn == "auto") else (None if fn is None else list(fn))
    names = [f"Column_{j}" for j in range(nf)] if names is None else [str(s) for s in names]
    if len(names) != nf or len(set(names)) != nf or any(set(s) & _BAD_NAME or not s for s in names):
        raise Unsupported("feature names LightGBM would rewrite or reject")
    return names


def _cat_columns(ds: Any, name: str) -> list:
    cf = ds.categorical_feature
    if cf is None or (isinstance(cf, str) and cf == "auto") or len(cf) == 0:
        return []
    if isinstance(cf, str):
        raise Unsupported(f"{name}: categorical_feature as a string")
    vals = list(cf)
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in vals):
        return sorted({int(v) for v in vals})
    if all(isinstance(v, str) for v in vals):
        return sorted(set(vals))
    raise Unsupported(f"{name}: categorical_feature mixes names and indices")


def _init_rows(ds: Any, n: int) -> np.ndarray | None:
    if ds.init_score is None:
        return None
    a = np.asarray(ds.init_score, dtype=np.float64)
    if a.ndim == 2:
        out = np.ascontiguousarray(a.reshape(n, -1))
    else:
        c = a.size // max(1, n)
        assert c >= 1 and c * n == a.size, "init_score: n x num_class values"
        out = np.ascontiguousarray(a.reshape(c, n).T)
    if not bool(np.all(np.isfinite(out))):
        raise Unsupported("init_score with non-finite values")
    return out


def _extract(ds: Any, name: str, train: bool, init_ok: bool = False, pc: list | None = None) -> SimpleNamespace:
    if not isinstance(ds, _basic.Dataset):
        raise Unsupported(f"{name} is not a lightgbm.Dataset")
    if ds.data is None:
        raise Unsupported(f"{name} holds no raw data: it was constructed with free_raw_data=True (LightGBM's default), "
                          "so LightGBM kept only its bins; pass the Dataset before it is constructed, or build it with "
                          "free_raw_data=False")
    checks = (("init_score", ds.init_score is not None and not init_ok),
              ("position", getattr(ds, "position", None) is not None),
              ("a row subset", ds.used_indices is not None), ("a reference Dataset", train and ds.reference is not None))
    for what, bad in checks:
        if bad:
            raise Unsupported(f"{name}: {what}")
    cat = _cat_columns(ds, name)
    if ds.label is None:
        raise Unsupported(f"{name}: no label")
    data, auto_cat = _frame_codes(ds.data, pc)
    x, frame_names = _matrix(data)
    cf = ds.categorical_feature
    if auto_cat and (cf is None or (isinstance(cf, str) and cf == "auto")):
        cat = list(auto_cat)
    y = _vector(ds.label, np.float32, f"{name} label")
    w = None if ds.weight is None else _vector(ds.weight, np.float32, f"{name} weight")
    g = None if ds.group is None else _vector(ds.group, np.int32, f"{name} group")
    if y.size != x.shape[0] or (w is not None and (w.size != y.size or (w < 0).any())):
        raise Unsupported(f"{name}: label / weight sizes or negative weights")
    if g is not None and (int(g.sum()) != y.size or (g <= 0).any()):
        raise Unsupported(f"{name}: group sizes do not sum to the row count")
    names = _names(ds, frame_names, x.shape[1])
    if cat and isinstance(cat[0], str):
        if any(c not in names for c in cat):
            raise Unsupported(f"{name}: categorical_feature names a column the data does not have")
        cat = sorted({names.index(c) for c in cat})
    if any(j < 0 or j >= x.shape[1] for j in cat):
        raise Unsupported(f"{name}: categorical_feature index out of range")
    return SimpleNamespace(x=x, y=y, w=w, g=g, names=names, ds=ds, cat=cat,
                           init=_init_rows(ds, x.shape[0]) if init_ok else None)


_DATASET_KEYS: frozenset = frozenset()
_CALLBACKS = (_cb._EarlyStoppingCallback, _cb._LogEvaluationCallback, _cb._RecordEvaluationCallback)
_BIN_KEYS = ("data_random_seed", "bin_construct_sample_cnt", "max_bin", "min_data_in_bin", "use_missing",
             "zero_as_missing")


def _bins_fixed(ds: Any) -> bool:
    return ds._handle is not None or bool(getattr(ds, "_shira_bins_pending", False))


def _stock_keeps_bins(p: dict[str, Any], ds: Any) -> bool:
    ret = _basic._LIB.LGBM_DatasetUpdateParamChecking(_basic._c_str(_basic._param_dict_to_str(ds.params or {})),
                                                       _basic._c_str(_basic._param_dict_to_str(p)))
    return int(ret) == 0


def _bin_pin(p: dict[str, Any], ds: Any, q: dict[str, Any], names: list[str]) -> dict[str, Any]:
    if not _bins_fixed(ds):
        return {}
    if not _stock_keeps_bins(p, ds):
        if ds._handle is None and ds.free_raw_data:
            raise Unsupported("Dataset reused with new binning params after its raw data was freed")
        return {}
    amap = _fit.alias_to_main()
    own = {k: v for k, v in (ds.params or {}).items() if amap.get(k, k) in _DATASET_KEYS | {"seed"}}
    cfg = _fit.resolve(own, names).cfg
    built = {k: (cfg[k] == "1") if k in _fit.BOOL_KEYS else int(cfg[k]) for k in _BIN_KEYS}
    other = [k for k in _BIN_KEYS[1:] if built[k] != q[k]]
    if other:
        raise Unsupported(f"pre-built Dataset: its {other[0]}={built[other[0]]}, the train call's {q[other[0]]}")
    assert set(built) == set(_BIN_KEYS)
    return {"data_random_seed": built["data_random_seed"]}


def _mirror_dataset(ds: Any, p: dict[str, Any]) -> None:
    ds._update_params(p)
    if ds._handle is None:
        ds._shira_bins_pending = True
    assert _bins_fixed(ds)


def _effective(p: dict[str, Any], ds: Any) -> dict[str, Any]:
    global _DATASET_KEYS
    if not _DATASET_KEYS:
        probe = _basic.Dataset(None, params={k: 1 for k in _fit.alias_to_main()})
        _DATASET_KEYS = frozenset(probe.get_params())
    if _bins_fixed(ds):
        main_p, main_d = _fit.user_values(dict(p)), _fit.user_values(dict(ds.params or {}))
        for k, v in main_p.items():
            if k in _DATASET_KEYS and k in main_d and str(main_d[k]) != str(v):
                raise Unsupported(f"constructed Dataset: train sets {k}={v}, the Dataset was built with {main_d[k]}")
    eff = dict(p)
    for k, v in (ds.params or {}).items():
        if k in _DATASET_KEYS and k not in eff:
            eff[k] = v
    assert set(p) <= set(eff)
    return eff


def _processed(params: Any, num_boost_round: int) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise Unsupported("params is not a dict")
    p = _basic._choose_param_value("objective", copy.deepcopy(params), None)
    if callable(p["objective"]) and not _first_read.mode():
        raise Unsupported("callable objective")
    given = {str(params[a]) for a in _basic._ConfigAliases.get("num_iterations") if a in params}
    if len(given) > 1:
        raise Unsupported("conflicting num_iterations aliases")
    p = _basic._choose_param_value("num_iterations", p, num_boost_round)
    p = _basic._choose_param_value("early_stopping_round", p, None)
    if p["early_stopping_round"] is None:
        p.pop("early_stopping_round")
    rounds = p["num_iterations"]
    if not isinstance(rounds, (int, np.integer)) or not 1 <= int(rounds) <= MAX_ROUNDS:
        raise Unsupported(f"num_boost_round {rounds!r}")
    return p


def _valid_plan(valid_sets: Any, valid_names: Any, train_set: Any, nf: int) -> tuple[list, str, bool]:
    out: list = []
    train_name, has_train = "training", False
    if valid_sets is None:
        return out, train_name, has_train
    sets = [valid_sets] if isinstance(valid_sets, _basic.Dataset) else list(valid_sets)
    names = [valid_names] if isinstance(valid_names, str) else valid_names
    assert len(sets) < 1000
    for i, vd in enumerate(sets):
        if vd is train_set:
            has_train = True
            if names is not None:
                train_name = names[i]
            continue
        nm = names[i] if names is not None and len(names) > i else f"valid_{i}"
        arr = _extract(vd, f"valid set {nm!r}", train=False, init_ok=True,
                       pc=_pandas_categorical(train_set))
        if arr.x.shape[1] != nf:
            raise Unsupported(f"valid set {nm!r} feature count differs")
        out.append((nm, arr))
    return out, train_name, has_train


def _plan(params: Any, train_set: Any, num_boost_round: int, kw: dict[str, Any]) -> SimpleNamespace:
    if version_reason() is not None:
        raise Unsupported(str(version_reason()))
    if kw.get("feval") not in (None, [], ()):
        raise Unsupported("feval is set")
    if kw.get("keep_training_booster"):
        raise Unsupported("keep_training_booster=True")
    cbs = list(kw.get("callbacks") or [])
    for cb in cbs:
        if not isinstance(cb, _CALLBACKS):
            raise Unsupported(f"callback {type(cb).__name__}")
    p = _processed(params, num_boost_round)
    tr = _extract(train_set, "train_set", train=True, init_ok=True)
    if tr.x.shape[0] == 0:
        raise Unsupported("no rows (stock decides)")
    eff = _effective(p, train_set)
    fobj = eff["objective"] if callable(eff.get("objective")) else None
    builtin = None if fobj is not None else BUILTIN_ROUTE.get(str(eff.get("objective", "")).lower())
    eff_r = {**eff, "objective": "regression"} if fobj is not None else eff
    if builtin is not None and _first_read.mode():
        amap = _fit.alias_to_main()
        fc = [float(v) for k, v in eff.items() if amap.get(k, k) == "fair_c"]
        sd = [int(v) for k, v in eff.items() if amap.get(k, k) == "objective_seed"]
        builtin = (builtin, fc[0] if fc else 1.0) if builtin != "xendcg" else (builtin, sd[0] if sd else 5)
        eff_r = {k: v for k, v in eff.items() if amap.get(k, k) not in ("fair_c", "objective_seed")}
        eff_r["objective"] = "regression" if builtin[0] != "xendcg" else "lambdarank"
    else:
        builtin = None
    rs = _fit.resolve(eff_r, tr.names)
    reason = _table_reason(rs.cfg)
    shape: list[str] = []
    if reason and _first_read.mode():
        rs2, shape = _shape_free(eff_r, tr.names)
        if rs2 is not None:
            rs, reason = rs2, None
    if reason:
        raise Unsupported(reason)
    if builtin is not None:
        if int(rs.cfg.get("num_class", "1")) > 1 or float(rs.cfg["path_smooth"]) != 0.0 or tr.cat:
            raise Unsupported(f"objective {builtin[0]} with num_class > 1, path_smooth or categorical features "
                              "(LightGBM's general route does not take them yet)")
        shape = shape + [f"objective {builtin[0]}"]
    if fobj is not None:
        if int(rs.cfg.get("num_class", "1")) > 1 or float(rs.cfg["path_smooth"]) != 0.0 or tr.cat:
            raise Unsupported("a custom objective with num_class > 1, path_smooth or categorical features "
                              "(LightGBM's general route does not take them yet)")
        shape = shape + ["a custom objective"]
    xe = builtin is not None and builtin[0] == "xendcg"
    es = any(isinstance(cb, _cb._EarlyStoppingCallback) for cb in cbs) or "early_stopping_round" in p
    if xe and (tr.g is None or tr.w is not None or es):
        raise Unsupported("rank_xendcg without query groups, with sample weights or with early stopping "
                          "(LightGBM's general route does not take them yet)")
    if shape and ((tr.g is not None and not xe) or rs.cfg.get("linear_tree", "0") == "1" or rs.cfg["boosting"] in ("rf", "dart")
                  or _first_read.mono_signs(rs.cfg, tr.x.shape[1]) is not None):
        raise Unsupported(f"{' / '.join(shape)} with query groups, linear_tree, rf, dart or monotone constraints "
                          "(LightGBM's general route does not take them yet)")
    q = _fit.engine_params(rs, {k: v for k, v in eff_r.items() if _fit.alias_to_main().get(k, k) not in SHAPE_ONLY})
    mono = _first_read.mono_signs(rs.cfg, tr.x.shape[1])
    if mono is not None:
        why = _first_read.mono_reason(rs.cfg, tr)
        if why:
            raise Unsupported(why)
        q["monotone_constraints"] = [int(v) for v in mono]
    linear = rs.cfg.get("linear_tree", "0") == "1"
    if linear:
        why = _first_read.linear_reason(rs.cfg, tr, kw, mono)
        if why:
            raise Unsupported(why)
    rf = rs.cfg["boosting"] == "rf"
    if rf:
        why = _first_read.rf_reason(rs.cfg, tr, kw)
        if why:
            raise Unsupported(why)
        q["boosting"], q["learning_rate"] = "gbdt", _first_read.RF_ETA
    if tr.cat:
        q["categorical_feature"] = list(tr.cat)
        for k in ("cat_l2", "cat_smooth"):
            q[k] = float(rs.cfg[k])
        for k in ("max_cat_threshold", "max_cat_to_onehot", "min_data_per_group"):
            q[k] = int(rs.cfg[k])
    pin = _bin_pin(p, train_set, q, tr.names)
    q.update(pin)
    _fit.check_engine(q)
    valid, train_name, has_train = _valid_plan(kw.get("valid_sets"), kw.get("valid_names"), train_set,
                                               tr.x.shape[1])
    if rs.cfg["boosting"] == "dart" and (valid or has_train):
        raise Unsupported("dart with evaluation sets (tree weights change after each round)")
    assert int(p["num_iterations"]) >= 1
    old = _init_booster(kw.get("init_model"), p)
    if old is not None or tr.init is not None:
        if shape or fobj is not None or rf or linear or mono is not None or rs.cfg["boosting"] != "gbdt":
            raise Unsupported("init_score / init_model with LightGBM's general route, a custom objective, rf, dart, "
                              "linear_tree or monotone constraints")
        if old is not None:
            tr.init = _raw_scores(old, tr.x)
            for _, arr in valid:
                arr.init = _raw_scores(old, arr.x)
        q["_shira_init_score"], q["_shira_continued"] = tr.init, 1 if old is not None else 0
        _fit.check_engine(q)
    return SimpleNamespace(p=p, eff=eff, stock_params={**eff, **pin}, rs=rs, q=q, tr=tr, valid=valid, old=old,
                           r0=0 if old is None else old.current_iteration(),
                           train_name=train_name,
                           has_train=has_train, callbacks=cbs, rounds=int(p["num_iterations"]),
                           has_valid_arg=kw.get("valid_sets") is not None, rf=rf, mono=mono, linear=linear,
                           shape_route=shape, custom_obj=fobj, builtin_obj=builtin)


class Booster(_basic.Booster):

    def __init__(self, params: Any = None, train_set: Any = None, model_file: Any = None,
                 model_str: Any = None, *, shira_model: Any = None) -> None:
        super().__init__(params=params, train_set=train_set, model_file=model_file, model_str=model_str)
        self.shira_model = shira_model
        self.backend = "shira" if shira_model is not None else "lightgbm"

    def predict(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "shira_model", None) is not None and _pred_contrib(args, kwargs) and \
                getattr(self.shira_model, "stats", None) is None:
            raise NotImplementedError("this SHIRA-trained model carries no node counts; SHAP values need them")
        if getattr(self, "shira_model", None) is not None and not (_THREAD_KEYS & set(kwargs)):
            nt = _train_threads(self.params)
            if nt > 0:
                kwargs = dict(kwargs, num_threads=nt)
        return super().predict(data, *args, **kwargs)


def _train_threads(params: Any) -> int:
    for k in _THREAD_KEYS:
        v = (params or {}).get(k)
        if v is not None:
            try:
                return max(0, int(v))
            except (TypeError, ValueError):
                return 0
    return 0


_PREDICT_SIG: list[Any] = []


def _pred_contrib(args: tuple, kwargs: dict) -> bool:
    assert isinstance(args, tuple) and isinstance(kwargs, dict)
    if not _PREDICT_SIG:
        import inspect

        _PREDICT_SIG.append(inspect.signature(_basic.Booster.predict))
    try:
        bound = _PREDICT_SIG[0].bind(None, None, *args, **kwargs)
    except TypeError:
        return False
    return bool(bound.arguments.get("pred_contrib", False))


def _pandas_categorical(ds: Any) -> list | None:
    pc = getattr(ds, "pandas_categorical", None)
    if pc is not None:
        return pc
    data = getattr(ds, "data", None)
    if not isinstance(data, _lgb.compat.pd_DataFrame):
        return None
    cats = [c for c, t in zip(data.columns, data.dtypes) if isinstance(t, _lgb.compat.pd_CategoricalDtype)]
    return [list(data[c].cat.categories) for c in cats]


def _final_booster(plan: SimpleNamespace, model: Any, n_iter: int | None, best_iteration: int,
                   best_score: Any, names_valid: list[str]) -> Booster:
    text = export.to_text(model, plan.rs, plan.tr.names, n_iter, _pandas_categorical(plan.tr.ds))
    bst = Booster(model_str=text, shira_model=model)
    bst.params = copy.deepcopy(plan.p)
    bst.best_iteration, bst.best_score = best_iteration, best_score
    bst._train_data_name = plan.train_name
    bst.name_valid_sets = list(names_valid)
    assert bst.num_trees() >= 1 and bst.num_feature() == len(plan.tr.names)
    return bst


class _Scorer:

    def __init__(self, name: str, arr: SimpleNamespace, eff: dict[str, Any], full: Booster, n_class: int) -> None:
        self.name, self.x, self.full = name, arr.x, full
        self.score = np.zeros((arr.x.shape[0], n_class), dtype=np.float64)
        if getattr(arr, "init", None) is not None:
            self.score = np.array(arr.init, dtype=np.float64).reshape(self.score.shape)
        self.params = {k: v for k, v in eff.items() if _fit.alias_to_main().get(k, k) != "verbosity"}
        self.params["verbosity"] = -1
        self.ds = _basic.Dataset(np.zeros((arr.x.shape[0], 1)), label=arr.y, weight=arr.w, group=arr.g,
                                 params=dict(self.params), free_raw_data=False)
        self.ds.set_init_score(self.score[:, 0] if n_class == 1 else self.score)
        self.ds.construct()

    def advance(self, i: int) -> list:
        add = self.full.predict(self.x, raw_score=True, start_iteration=i, num_iteration=1)
        self.score = self.score + np.asarray(add, dtype=np.float64).reshape(self.score.shape)
        self.ds.set_init_score(self.score[:, 0] if self.score.shape[1] == 1 else self.score)
        res = _basic.Booster(params=dict(self.params), train_set=self.ds).eval_train()
        assert isinstance(res, list)
        return [(self.name, metric, value, hib) for _, metric, value, hib in res]


def _callback_lists(plan: SimpleNamespace) -> tuple[list, list]:
    params, cbs = plan.p, plan.callbacks
    for i, cb in enumerate(cbs):
        cb.__dict__.setdefault("order", i - len(cbs))
    cb_set = set(cbs)
    if _cb._should_enable_early_stopping(params.get("early_stopping_round", 0)):
        verbose = _basic._choose_param_value("verbosity", params, 1).pop("verbosity") > 0
        cb_set.add(_cb.early_stopping(stopping_rounds=params["early_stopping_round"],
                                      first_metric_only=params.get("first_metric_only", False),
                                      min_delta=params.get("early_stopping_min_delta", 0.0), verbose=verbose))
    before = {cb for cb in cb_set if getattr(cb, "before_iteration", False)}
    return sorted(before, key=attrgetter("order")), sorted(cb_set - before, key=attrgetter("order"))


def _replay(plan: SimpleNamespace, model: Any) -> Booster:
    names_valid = [nm for nm, _ in plan.valid]
    full = _final_booster(plan, model, None, 0, {}, names_valid)
    n_c = model.num_class
    scorers = ([_Scorer(plan.train_name, plan.tr, plan.eff, full, n_c)] if plan.has_train else []) + \
              [_Scorer(nm, arr, plan.eff, full, n_c) for nm, arr in plan.valid]
    before, after = _callback_lists(plan)
    best_iteration, results = 0, []
    assert plan.rounds * n_c == model.n_trees
    r0 = getattr(plan, "r0", 0)
    for i in range(plan.rounds):
        env = {"model": full, "params": plan.p, "iteration": r0 + i, "begin_iteration": r0,
               "end_iteration": r0 + plan.rounds}
        for cb in before:
            cb(_cb.CallbackEnv(evaluation_result_list=None, **env))
        results = [r for s in scorers for r in s.advance(i)] if plan.has_valid_arg else []
        try:
            for cb in after:
                cb(_cb.CallbackEnv(evaluation_result_list=results, **env))
        except _cb.EarlyStopException as stop:
            best_iteration, results = stop.best_iteration + 1, stop.best_score
            break
    best_score: Any = defaultdict(OrderedDict)
    for ds_name, eval_name, score, _ in results:
        best_score[ds_name][eval_name] = score
    if not best_iteration:
        full.best_iteration, full.best_score = 0, best_score
        return _merged(plan, full)
    return _merged(plan, _final_booster(plan, model, best_iteration - r0, best_iteration, best_score, names_valid))


def _init_booster(init_model: Any, params: dict[str, Any]) -> Any:
    if init_model in (None, [], (), False):
        return None
    if isinstance(init_model, _basic.Booster):
        return init_model
    return _basic.Booster(model_file=init_model)


def _raw_scores(old: Any, x: np.ndarray) -> np.ndarray:
    r = np.asarray(old.predict(x, raw_score=True, num_iteration=-1), dtype=np.float64)
    return np.ascontiguousarray(r.reshape(x.shape[0], -1))


def _merged(plan: SimpleNamespace, bst: Booster) -> Booster:
    old = getattr(plan, "old", None)
    if old is None:
        return bst
    n0 = bst.num_trees()
    _basic._safe_call(_basic._LIB.LGBM_BoosterMerge(bst._handle, old._handle))
    assert bst.num_trees() == n0 + old.num_trees()
    bst.shira_model, bst.backend = None, "shira"
    return bst


def _fallback(reason: str, args: tuple, kwargs: dict[str, Any]) -> Any:
    global last_backend
    LOG.warning("shira.lightgbm: running stock LightGBM: %s", reason)
    last_backend = "lightgbm"
    backend_history.append(last_backend)
    ts = args[1]
    if getattr(ts, "_shira_bins_pending", False) and ts._handle is None:
        ts.construct()
    return _STOCK["train"](*args, **kwargs)


def _rank_guard(plan: SimpleNamespace) -> str | None:
    if plan.tr.g is None:
        return None
    n, nf = int(plan.tr.x.shape[0]), int(plan.tr.x.shape[1])
    env = os.environ.get("SHIRA_LGB_RANK_ENGINE_MIN_ROWS", "").strip()
    lim = int(env) if env.isdigit() else _machine.value("lgb_rank_engine_min_rows", nf)
    if n >= lim:
        return None
    return (f"lambdarank at {n} rows x {nf}: SHIRA's engine is not measured faster than stock on this machine "
            f"(switch lgb_rank_engine_min_rows in _machine.py; SHIRA_LGB_RANK_ENGINE_MIN_ROWS=0 forces it)")


FIRST_READ_ERRORS = (AssertionError, ValueError, RuntimeError, AttributeError, TypeError, ArithmeticError)


def _run(plan: SimpleNamespace) -> Booster:
    global first_read_reason
    tr = plan.tr
    n = tr.x.shape[0]
    why = _rank_guard(plan)
    if why is not None:
        raise Unsupported(why)
    if plan.linear:
        try:
            out = _first_read.fit_linear(plan)
        except FIRST_READ_ERRORS as e:
            LOG.warning("shira.lightgbm: the first read failed (%s: %s); stock LightGBM fits this call", type(e).__name__, e)
            raise Unsupported(f"linear_tree: the first read failed ({type(e).__name__}: {e})") from e
        if out is None:
            raise Unsupported("linear_tree: the sample would be every row (stock's fit is the same model)")
        first_read_reason = out[1]
        LOG.info("shira.lightgbm: %s", out[1])
        if not isinstance(out[0], str):
            return _final_booster(plan, out[0], None, 0, defaultdict(OrderedDict), [])
        bst = Booster(model_str=out[0], shira_model=SimpleNamespace(linear_tree=True))
        bst.params = copy.deepcopy(plan.p)
        bst.best_iteration, bst.best_score = 0, defaultdict(OrderedDict)
        bst._train_data_name, bst.name_valid_sets = plan.train_name, []
        return bst
    try:
        starts = getattr(plan, "old", None) is not None or tr.init is not None
        fr = _first_read.fit(plan) if _first_read.mode() and not starts else None
    except FIRST_READ_ERRORS as e:
        LOG.warning("shira.lightgbm: the first read failed (%s: %s); the ordinary fit decides", type(e).__name__, e)
        fr = None
    first_read_reason = None if fr is None else fr[1]
    if fr is None and plan.rf:
        raise Unsupported("boosting=rf: the first-read fit did not apply")
    if fr is None and plan.mono is not None:
        raise Unsupported("monotone constraints: the sample would be every row (stock's fit is the same model)")
    if fr is None and plan.shape_route:
        raise Unsupported(f"{' / '.join(plan.shape_route)}: the general route did not apply (at or under the small-data "
                          "floor, or the first read refused the call: its reason is logged above)")
    if fr is not None:
        LOG.info("shira.lightgbm: %s", fr[1])
    model = fr[0] if fr is not None else _fit.fit_folds(plan.q, plan.rs, tr.x, tr.y, [(0, n)], plan.rounds, tr.w, tr.g,
                                                        stock_params=plan.stock_params,
                                                        order_exact=True)[0]
    assert model.n_trees == plan.rounds * model.num_class
    if plan.has_valid_arg or plan.callbacks or "early_stopping_round" in plan.p:
        return _replay(plan, model)
    return _merged(plan, _final_booster(plan, model, None, 0, defaultdict(OrderedDict), []))


def train(params: dict[str, Any], train_set: Any, num_boost_round: int = 100, valid_sets: Any = None,
          valid_names: Any = None, feval: Any = None, init_model: Any = None,
          keep_training_booster: bool = False, callbacks: Any = None) -> Any:
    global last_backend, first_read_reason
    first_read_reason = None
    kw = {"valid_sets": valid_sets, "valid_names": valid_names, "feval": feval, "init_model": init_model,
          "keep_training_booster": keep_training_booster, "callbacks": callbacks}
    try:
        plan = _plan(params, train_set, num_boost_round, kw)
        bst = _run(plan)
    except Unsupported as e:
        return _fallback(str(e), (params, train_set, num_boost_round), kw)
    assert isinstance(bst, Booster) and bst.backend == "shira"
    _mirror_dataset(train_set, plan.p)
    last_backend = "shira"
    backend_history.append(last_backend)
    return bst


def train_folds(params: dict[str, Any], X: Any, y: Any, folds: list[tuple[int, int]], num_boost_round: int = 100,
                weight: Any = None, group: Any = None) -> list[Any]:
    global last_backend
    assert len(folds) >= 1 and all(0 <= lo < hi for lo, hi in folds)
    try:
        ds = _basic.Dataset(X, label=y, weight=weight, group=group, free_raw_data=False)
        plan = _plan(params, ds, num_boost_round, {})
        if plan.rf:
            raise Unsupported("boosting=rf with folds")
        if plan.mono is not None:
            raise Unsupported("monotone constraints with folds")
        if plan.linear:
            raise Unsupported("linear_tree with folds")
        tr = plan.tr
        models = _fit.fit_folds(plan.q, plan.rs, tr.x, tr.y, [(int(a), int(b)) for a, b in folds], plan.rounds,
                                tr.w, tr.g, stock_params=plan.eff,
                                order_exact=True)
        out = [_final_booster(plan, m, None, 0, defaultdict(OrderedDict), []) for m in models]
    except Unsupported as e:
        LOG.warning("shira.lightgbm: running stock LightGBM: %s", e)
        last_backend = "lightgbm"
        backend_history.append(last_backend)
        return [_STOCK["train"](params, _stock_fold(X, y, weight, group, lo, hi), num_boost_round) for lo, hi in folds]
    last_backend = "shira"
    backend_history.append(last_backend)
    return out


def _stock_fold(X: Any, y: Any, weight: Any, group: Any, lo: int, hi: int) -> Any:
    from shira_cpp import lgb_cpp
    g = None if group is None else lgb_cpp.fold_groups(group, lo, hi)
    return _basic.Dataset(X[lo:hi], label=y[lo:hi], weight=None if weight is None else weight[lo:hi], group=g)


_RANK_OBJECTIVES = {"lambdarank", "rank_xendcg", "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}


def _cv_folds(tr: SimpleNamespace, params: dict[str, Any], folds: Any, nfold: int, stratified: bool, shuffle: bool,
              seed: int) -> list:
    from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold
    num_data = tr.x.shape[0]
    if folds is not None:
        if not hasattr(folds, "__iter__") and not hasattr(folds, "split"):
            raise Unsupported("folds is neither an iterable nor a splitter")
        if hasattr(folds, "split"):
            flat = (np.repeat(range(len(tr.g)), repeats=tr.g) if tr.g is not None
                    else np.zeros(num_data, dtype=np.int32))
            folds = folds.split(X=np.empty(num_data), y=tr.y, groups=flat)
        return [(np.asarray(a), np.asarray(b)) for a, b in folds]
    if any(params.get(a, "") in _RANK_OBJECTIVES for a in _basic._ConfigAliases.get("objective")):
        if not SKLEARN_INSTALLED or tr.g is None:
            raise Unsupported("ranking cv without scikit-learn or group")
        flat = np.repeat(range(len(tr.g)), repeats=tr.g)
        return list(_LGBMGroupKFold(n_splits=nfold).split(X=np.empty(num_data), groups=flat))
    if stratified:
        if not SKLEARN_INSTALLED:
            raise Unsupported("stratified cv without scikit-learn")
        skf = _LGBMStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
        return list(skf.split(X=np.empty(num_data), y=tr.y))
    randidx = np.random.RandomState(seed).permutation(num_data) if shuffle else np.arange(num_data)
    kstep = int(num_data / nfold)
    test_id = [randidx[i: i + kstep] for i in range(0, num_data, kstep)]
    train_id = [np.concatenate([test_id[i] for i in range(nfold) if k != i]) for k in range(nfold)]
    return list(zip(train_id, test_id))


def _cv_plan(params: Any, train_set: Any, num_boost_round: int, kw: dict[str, Any]) -> SimpleNamespace:
    if version_reason() is not None:
        raise Unsupported(str(version_reason()))
    cbs = list(kw.get("callbacks") or [])
    resets = [cb for cb in cbs if type(cb).__name__ == "_ResetParameterCallback"]
    general = bool(resets) and (len(resets) > 1 or any(set(cb.kwargs) != {"learning_rate"} for cb in resets))
    cbs = [cb for cb in cbs if cb not in resets]
    for cb in cbs:
        if not isinstance(cb, _CALLBACKS):
            raise Unsupported(f"callback {getattr(cb, '__name__', type(cb).__name__)}: a callback can change the training "
                              "each round (reset_parameter, a learning-rate schedule), which folds trained ahead of the loop "
                              "cannot replay; early_stopping, log_evaluation and record_evaluation are taken")
    p = _processed(params, num_boost_round)
    if kw.get("metrics") is not None:
        for alias in _basic._ConfigAliases.get("metric"):
            p.pop(alias, None)
        p["metric"] = kw["metrics"]
    tr = _extract(train_set, "train_set", train=True, init_ok=True)
    eff = _effective(p, train_set)
    if callable(eff.get("objective")) or tr.x.shape[0] == 0:
        raise Unsupported("a custom objective or no rows")
    old = _init_booster(kw.get("init_model"), p)
    if old is not None:
        tr.init = _raw_scores(old, tr.x)
    rs = _fit.resolve(eff, tr.names)
    reason = _table_reason(rs.cfg)
    if reason:
        raise Unsupported(reason)
    rf = rs.cfg["boosting"] == "rf"
    if rf:
        why = _first_read.rf_reason(rs.cfg, tr, {})
        if why:
            raise Unsupported(why)
    q = _fit.engine_params(rs, eff)
    if rf:
        q["boosting"], q["learning_rate"] = "gbdt", _first_read.RF_ETA
    if tr.cat:
        q["categorical_feature"] = list(tr.cat)
        for k in ("cat_l2", "cat_smooth"):
            q[k] = float(rs.cfg[k])
        for k in ("max_cat_threshold", "max_cat_to_onehot", "min_data_per_group"):
            q[k] = int(rs.cfg[k])
    by_round = general or (bool(resets) and float(rs.cfg["feature_fraction"]) < 1.0)
    if by_round and (rf or rs.cfg["boosting"] != "gbdt" or kw.get("fpreproc") is not None
                     or kw.get("return_cvbooster")):
        raise Unsupported("reset_parameter of other parameters (or with feature_fraction < 1) with boosting rf / dart, "
                          "fpreproc or return_cvbooster")
    round_p = _cv_round_params(resets, int(p["num_iterations"]), p) if by_round else None
    lr_sched = None if by_round else _cv_lr_schedule(resets, int(p["num_iterations"]), rs.cfg, rf)
    if lr_sched is not None:
        q["learning_rates"] = lr_sched
    if tr.init is not None:
        if rf or rs.cfg["boosting"] != "gbdt" or kw.get("fpreproc") is not None:
            raise Unsupported("init_score / init_model with boosting rf / dart or with fpreproc")
        q["_shira_init_score"], q["_shira_continued"] = tr.init, 1 if old is not None else 0
    _fit.check_engine(q)
    pairs = _cv_folds(tr, p, kw.get("folds"), int(kw["nfold"]), bool(kw["stratified"]), bool(kw["shuffle"]),
                      kw["seed"])
    if not 1 <= len(pairs) <= _fit.MAX_FOLDS:
        raise Unsupported(f"{len(pairs)} cv folds")
    n = tr.x.shape[0]
    rows = []
    for a, b in pairs:
        ra, rb = np.sort(np.asarray(a, dtype=np.int64)), np.sort(np.asarray(b, dtype=np.int64))
        for r in (ra, rb):
            if r.size < 1 or int(r[0]) < 0 or int(r[-1]) >= n or bool((np.diff(r) == 0).any()):
                raise Unsupported("cv fold indices out of range, empty or repeated")
        rows.append((ra, rb))
    assert int(p["num_iterations"]) >= 1
    plan = SimpleNamespace(p=p, eff=eff, rs=rs, q=q, tr=tr, rows=rows, callbacks=cbs, rounds=int(p["num_iterations"]),
                           train_metric=bool(kw["eval_train_metric"]),
                           first_metric_only=p.get("first_metric_only", False), train_name="training",
                           feval=kw.get("feval"), ret_cvb=bool(kw.get("return_cvbooster")), subs=None, rf=rf,
                           lr_sched=lr_sched, old=old, round_p=round_p)
    if kw.get("fpreproc") is not None:
        plan.subs = _cv_fpreproc(plan, train_set, kw["fpreproc"])
    return plan


def _cv_round_params(resets: list, rounds: int, p: dict[str, Any]) -> list:
    amap, cur, out = _fit.alias_to_main(), dict(p), []
    for r in range(rounds):
        n_reset = 0
        for cb in resets:
            changed = False
            for key, value in cb.kwargs.items():
                if isinstance(value, list):
                    if len(value) != rounds:
                        raise ValueError(f"Length of list {key!r} has to be equal to 'num_boost_round'.")
                    new = value[r]
                elif callable(value):
                    new = value(r)
                else:
                    raise ValueError("Only list and callable values are supported as a mapping from boosting round "
                                     "index to new parameter value.")
                if amap.get(key, key) in _DATASET_KEYS:
                    raise Unsupported(f"reset_parameter of the dataset parameter {key} (stock refuses it too)")
                if new != cur.get(key, None):
                    cur[key] = new
                    changed = True
            n_reset += 1 if changed else 0
        out.append((dict(cur), n_reset))
    assert len(out) == rounds
    return out


def _round_plan(plan: SimpleNamespace, pr: dict[str, Any]) -> SimpleNamespace:
    amap = _fit.alias_to_main()
    eff = dict(plan.eff)
    for k, v in pr.items():
        if plan.p.get(k, None) != v or k not in plan.p:
            main = amap.get(k, k)
            eff = {a: b for a, b in eff.items() if amap.get(a, a) != main}
            eff[k] = v
    rs = _fit.resolve(eff, plan.tr.names)
    reason = _table_reason(rs.cfg)
    if reason or rs.cfg["boosting"] != "gbdt":
        raise Unsupported(f"reset_parameter gives a round outside the table: {reason or rs.cfg['boosting']}")
    q = _fit.engine_params(rs, eff)
    _fit.check_engine(q)
    sub = copy.copy(plan)
    sub.p, sub.eff, sub.rs, sub.q = dict(pr), eff, rs, q
    return sub


class _RoundScorer(_Scorer):

    def __init__(self, name: str, arr: SimpleNamespace, eff: dict[str, Any], boosters: list, n_class: int) -> None:
        super().__init__(name, arr, eff, boosters[0], n_class)
        self.boosters = boosters

    def advance(self, i: int) -> list:
        add = self.boosters[i].predict(self.x, raw_score=True)
        self.score = self.score + np.asarray(add, dtype=np.float64).reshape(self.score.shape)
        self.ds.set_init_score(self.score[:, 0] if self.score.shape[1] == 1 else self.score)
        res = _basic.Booster(params=dict(self.params), train_set=self.ds).eval_train(getattr(self, "feval", None))
        assert isinstance(res, list)
        return [(self.name, metric, value, hib) for _, metric, value, hib in res]


def _cv_by_round(plan: SimpleNamespace) -> list:
    tr, rows = plan.tr, [a for a, _ in plan.rows]
    cat_n = sum(int(r.size) for r in rows)
    per_fold: list = [[] for _ in rows]
    cur = None
    extra = np.array([nr for _, nr in plan.round_p], dtype=np.int32)
    for r, (pr, _) in enumerate(plan.round_p):
        sub = _round_plan(plan, pr)
        q = dict(sub.q)
        q["_shira_iter0"], q["_shira_col_extra"] = r, extra[:r + 1]
        if r == 0 and tr.init is not None:
            q["_shira_init_score"], q["_shira_continued"] = tr.init, 1 if plan.old is not None else 0
        if r > 0:
            q["_shira_continued"] = 1
        models = _fit.fit_subsets(q, sub.rs, tr.x, tr.y, rows, 1, tr.w, tr.g, init_cat=cur)
        cur = np.concatenate([np.asarray(m.scores, dtype=np.float64).reshape(m.num_class, -1).T for m in models])
        assert cur.shape[0] == cat_n
        for k, m in enumerate(models):
            per_fold[k].append(_final_booster(sub, m, None, 0, {}, []))
    LOG.info("shira.lightgbm: cv with reset_parameter of other parameters: one round at a time on each fold's scores so "
             "far, the random streams where stock's are (stock's model)")
    out = []
    for (ra, rb), bsts in zip(plan.rows, per_fold):
        n_c = int(bsts[0].num_model_per_iteration())
        fold = [_RoundScorer("train", _cv_subset(tr, ra), plan.eff, bsts, n_c)] if plan.train_metric else []
        out.append(fold + [_RoundScorer("valid", _cv_subset(tr, rb), plan.eff, bsts, n_c)])
    return out


def _cv_lr_schedule(resets: list, rounds: int, cfg: dict, rf: bool) -> list | None:
    if not resets:
        return None
    if rf or cfg["boosting"] != "gbdt":
        raise Unsupported("reset_parameter with boosting rf / dart (one rate per run there)")
    if float(cfg["feature_fraction"]) < 1.0:
        raise Unsupported("reset_parameter with feature_fraction < 1: stock's ResetConfig re-seeds the tree-level column "
                          "sampler on every reset; not built")
    v = resets[0].kwargs["learning_rate"]
    if isinstance(v, list) and len(v) != rounds:
        raise ValueError("Length of list 'learning_rate' has to be equal to 'num_boost_round'.")
    if not isinstance(v, list) and not callable(v):
        raise ValueError("Only list and callable values are supported as a mapping from boosting round index to new "
                         "parameter value.")
    out = [v[r] if isinstance(v, list) else v(r) for r in range(rounds)]
    assert len(out) == rounds
    return out


def _cv_fpreproc(plan: SimpleNamespace, train_set: Any, fp: Any) -> list:
    tr = plan.tr
    if tr.g is not None:
        raise Unsupported("fpreproc with query groups")
    amap, base_main, subs = _fit.alias_to_main(), _fit.user_values(dict(plan.eff)), []
    for ra, rb in plan.rows:
        ts, vs, tp = fp(train_set.subset(ra.tolist()), train_set.subset(rb.tolist()), dict(plan.p))
        if not isinstance(tp, dict):
            raise Unsupported(f"fpreproc returned params of type {type(tp).__name__}, not a dict")
        own = _cv_fold_own(plan, train_set, ts, vs, tp, ra, rb)
        if own is not None:
            subs.append(own)
            continue
        y, w = np.array(tr.y, dtype=np.float32), None if tr.w is None else np.array(tr.w, dtype=np.float32)
        for ds, rows, side in ((ts, ra, "train"), (vs, rb, "valid")):
            if not isinstance(ds, _basic.Dataset) or ds.reference is not train_set or ds.used_indices is None \
                    or not np.array_equal(np.asarray(ds.used_indices, dtype=np.int64), rows):
                raise Unsupported(f"fpreproc returned a {side} Dataset that is not the fold's subset of train_set")
            if ds.group is not None or getattr(ds, "position", None) is not None:
                raise Unsupported(f"fpreproc set a group / position on the fold's {side} Dataset")
            if ds._handle is None:
                continue
            if ds.get_field("init_score") is not None:
                raise Unsupported(f"fpreproc set an init_score on the fold's {side} Dataset")
            y[rows] = np.asarray(ds.get_field("label"), dtype=np.float32)
            fw = ds.get_field("weight")
            if fw is not None:
                w = np.ones(y.size, dtype=np.float32) if w is None else w
                w[rows] = np.asarray(fw, dtype=np.float32)
        tpm = _fit.user_values(dict(tp))
        for k, v in tpm.items():
            if k in _DATASET_KEYS and str(base_main.get(k)) != str(v):
                raise Unsupported(f"fpreproc changed the dataset param {amap.get(k, k)}={v} (stock bins every fold with the "
                                  "full data's)")
        eff = dict(tp)
        for k, v in plan.eff.items():
            if k in _DATASET_KEYS and amap.get(k, k) not in tpm:
                eff[k] = v
        rs = _fit.resolve(eff, tr.names)
        reason = _table_reason(rs.cfg)
        if reason or rs.cfg["boosting"] not in ("gbdt", "dart"):
            raise Unsupported(f"fpreproc's fold params: {reason or 'boosting ' + rs.cfg['boosting']}")
        q = _fit.engine_params(rs, eff)
        if plan.lr_sched is not None:
            q["learning_rates"] = plan.lr_sched
        _fit.check_engine(q)
        trk = copy.copy(tr)
        trk.y, trk.w = y, w
        sub = copy.copy(plan)
        sub.p, sub.eff, sub.rs, sub.q, sub.tr, sub.rows, sub.subs = tp, eff, rs, q, trk, [(ra, rb)], None
        subs.append(sub)
    assert len(subs) == len(plan.rows)
    thr = getattr(subs[-1], "vthreads", None)
    if thr is not None:
        for sub in subs:
            _cv_rethread(sub, thr)
    return subs


def _cv_rethread(sub: SimpleNamespace, thr: int) -> None:
    alias = set(_basic._ConfigAliases.get("num_threads"))
    eff = dict({k: v for k, v in sub.eff.items() if k not in alias}, num_threads=thr)
    rs = _fit.resolve(eff, sub.tr.names)
    q = _fit.engine_params(rs, eff)
    if sub.lr_sched is not None:
        q["learning_rates"] = sub.lr_sched
    _fit.check_engine(q)
    sub.eff, sub.rs, sub.q = eff, rs, q


def _is_fold_subset(ds: Any, train_set: Any, rows: np.ndarray) -> bool:
    return (isinstance(ds, _basic.Dataset) and ds.reference is train_set and ds.used_indices is not None
            and np.array_equal(np.asarray(ds.used_indices, dtype=np.int64), rows))


def _ds_threads(ds: Any) -> int:
    prm = dict(ds.params or {})
    for a in _basic._ConfigAliases.get("num_threads"):
        if a in prm:
            return int(prm[a])
    return 0


def _cv_fold_own(plan: SimpleNamespace, train_set: Any, ts: Any, vs: Any, tp: dict, ra: np.ndarray,
                 rb: np.ndarray) -> SimpleNamespace | None:
    if _is_fold_subset(ts, train_set, ra) and _is_fold_subset(vs, train_set, rb):
        return None
    if not isinstance(ts, _basic.Dataset) or not isinstance(vs, _basic.Dataset):
        raise Unsupported("fpreproc returned a fold that is not a lightgbm.Dataset")
    if _is_fold_subset(ts, train_set, ra):
        if ts._handle is not None:
            raise Unsupported("fpreproc built the fold's train subset and replaced its valid Dataset")
        src, base, rows_t = train_set, plan.tr, ra
        tpm, eff = _fit.user_values(dict(tp)), dict(tp)
        for k, v in plan.eff.items():
            if k in _DATASET_KEYS and _fit.alias_to_main().get(k, k) not in tpm:
                eff[k] = v
    else:
        if ts.reference is not None or ts.used_indices is not None:
            raise Unsupported("fpreproc returned a train Dataset that is a subset of, or references, another Dataset")
        src, base = ts, _extract(ts, "fpreproc's train Dataset", train=True)
        rows_t, eff = np.arange(base.x.shape[0], dtype=np.int64), _effective(dict(tp), ts)
    if base.g is not None or vs.group is not None:
        raise Unsupported("fpreproc with query groups")
    if vs.reference is not src:
        raise Unsupported("fpreproc returned a valid Dataset whose reference is not the fold's train Dataset (stock: "
                          "different bin mappers)")
    v = _extract(vs, "fpreproc's valid Dataset", train=False) if vs.used_indices is None else None
    if v is None:
        vr = np.asarray(vs.used_indices, dtype=np.int64)
        v = SimpleNamespace(x=base.x[vr], y=base.y[vr], w=None if base.w is None else base.w[vr])
    if v.x.shape[1] != base.x.shape[1]:
        raise Unsupported("fpreproc's valid Dataset has another column count than its train Dataset")
    n0, n1 = base.x.shape[0], v.x.shape[0]
    w = None if base.w is None and v.w is None else np.concatenate(
        [np.ones(n0, np.float32) if base.w is None else base.w, np.ones(n1, np.float32) if v.w is None else v.w])
    trk = copy.copy(base)
    trk.x, trk.y, trk.w, trk.g = np.concatenate([base.x, v.x]), np.concatenate([base.y, v.y]), w, None
    rs = _fit.resolve(eff, base.names)
    reason = _table_reason(rs.cfg)
    if reason or rs.cfg["boosting"] not in ("gbdt", "dart"):
        raise Unsupported(f"fpreproc's fold params: {reason or 'boosting ' + rs.cfg['boosting']}")
    q = _fit.engine_params(rs, eff)
    if plan.lr_sched is not None:
        q["learning_rates"] = plan.lr_sched
    _fit.check_engine(q)
    sub = copy.copy(plan)
    sub.p, sub.eff, sub.rs, sub.q, sub.tr, sub.subs = tp, eff, rs, q, trk, None
    sub.rows, sub.fit = [(rows_t, n0 + np.arange(n1, dtype=np.int64))], (base.x, base.y, base.w)
    sub.vthreads = _ds_threads(vs)
    assert trk.x.shape[0] == trk.y.shape[0] == n0 + n1
    return sub


class _LeafScorer(_Scorer):

    def __init__(self, name: str, arr: SimpleNamespace, eff: dict[str, Any], full: Booster, n_class: int) -> None:
        super().__init__(name, arr, eff, full, n_class)
        leaves = np.asarray(full.predict(arr.x, pred_leaf=True), dtype=np.int64)
        self.leaves = leaves.reshape(arr.x.shape[0], -1)
        self.n_class = n_class
        assert self.leaves.shape[1] == full.shira_model.n_trees and not full.shira_model.average

    def advance(self, i: int) -> list:
        m = self.full.shira_model
        add = np.stack([m.leaf[i * self.n_class + c][self.leaves[:, i * self.n_class + c]]
                        for c in range(self.n_class)], axis=1)
        self.score = self.score + add
        self.ds.set_init_score(self.score[:, 0] if self.score.shape[1] == 1 else self.score)
        res = _basic.Booster(params=dict(self.params), train_set=self.ds).eval_train(getattr(self, "feval", None))
        assert isinstance(res, list)
        return [(self.name, metric, value, hib) for _, metric, value, hib in res]


def _cv_subset(tr: SimpleNamespace, rows: np.ndarray) -> SimpleNamespace:
    g = None if tr.g is None else _fit.subset_groups(tr.g, rows, tr.x.shape[0])
    init = getattr(tr, "init", None)
    return SimpleNamespace(x=tr.x[rows], y=tr.y[rows], w=None if tr.w is None else tr.w[rows], g=g,
                           init=None if init is None else init[rows])


def _cv_callbacks(plan: SimpleNamespace) -> tuple[list, list]:
    params, cbs = plan.p, plan.callbacks
    for i, cb in enumerate(cbs):
        cb.__dict__.setdefault("order", i - len(cbs))
    cb_set = set(cbs)
    if _cb._should_enable_early_stopping(params.get("early_stopping_round", 0)):
        verbose = _basic._choose_param_value("verbosity", params, 1).pop("verbosity") > 0
        cb_set.add(_cb.early_stopping(stopping_rounds=params["early_stopping_round"],
                                      first_metric_only=plan.first_metric_only,
                                      min_delta=params.get("early_stopping_min_delta", 0.0), verbose=verbose))
    before = {cb for cb in cb_set if getattr(cb, "before_iteration", False)}
    return sorted(before, key=attrgetter("order")), sorted(cb_set - before, key=attrgetter("order"))


last_cv_form: str | None = None


class _MatrixScorer(_Scorer):

    def __init__(self, name: str, arr: SimpleNamespace, eff: dict[str, Any], ids: np.ndarray, rows: np.ndarray,
                 leaf: np.ndarray, rf: bool = False) -> None:
        super().__init__(name, arr, eff, None, 1)
        self.ids, self.rows, self.leaf, self.rf = ids, rows, leaf, rf
        assert ids.shape[0] == leaf.shape[0] and rows.size == arr.x.shape[0]
        if rf:
            amap = _fit.alias_to_main()
            self.params = {k: v for k, v in self.params.items() if amap.get(k, k) != "boosting"}
            self.params["boosting"] = "gbdt"

    def advance(self, i: int) -> list:
        add = self.leaf[i][self.ids[i][self.rows]]
        if self.rf:
            self.score[:, 0] = (self.score[:, 0] * float(i) + add) * (1.0 / (i + 1))
        else:
            self.score[:, 0] = self.score[:, 0] + add
        self.ds.set_init_score(self.score[:, 0])
        res = _basic.Booster(params=dict(self.params), train_set=self.ds).eval_train(getattr(self, "feval", None))
        assert isinstance(res, list)
        return [(self.name, metric, value, hib) for _, metric, value, hib in res]


cv_scored_log: list[dict] = []


def cv_form() -> str:
    v = os.environ.get("SHIRA_CV_FORM", "").strip().lower()
    assert v in ("", "matrix_scored", "perfold"), f"SHIRA_CV_FORM={v!r}: '', matrix_scored or perfold"
    return v


def _cv_first_read(plan: SimpleNamespace, shared: bool, score_sample: bool = False) -> list | None:
    tr = plan.tr
    n, nf = int(tr.x.shape[0]), int(tr.x.shape[1])
    cfg = _fit.check_engine(plan.q)
    rf = bool(getattr(plan, "rf", False))
    if (not _first_read.mode() or int(cfg["objective"]) not in _first_read.OBJ or _first_read._lgbg_kind(cfg) is not None
            or bool(cfg.get("dart")) or float(cfg["path_smooth"]) != 0.0 or tr.g is not None):
        return None
    th = _fit.engine_threads(plan.rs, n)
    spec = SimpleNamespace(tr=tr, q=plan.q, rs=plan.rs, rounds=plan.rounds, stock_params=plan.eff, eff=plan.eff)

    def grow(pool: np.ndarray, band_rows: int) -> tuple | None:
        m = _first_read.fixed_rows(nf, "", 0 if (tr.cat or rf) else int(band_rows))
        if rf and (m is None or m >= pool.size):
            m = int(pool.size)
        if m is None or m >= pool.size and not rf:
            return None
        rows = pool if m >= pool.size else pool[(pool.size - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)]
        sm = _first_read._shapes(spec, rows, th, th)
        if int(np.min(sm.nleaves)) < 2 or int(sm.n_trees) != plan.rounds:
            return None
        ids, nl = _first_read._leaf_ids(sm, tr.x, th)
        return sm, ids, nl, rows

    band = int(np.median([np.asarray(a).size for a, _ in plan.rows]))
    common = grow(np.arange(n, dtype=np.int64), band) if shared else None
    if shared and common is None:
        return None
    w_all = np.ones(n, dtype=np.float64) if tr.w is None else np.asarray(tr.w, dtype=np.float64)
    scorers = []
    for ra, rb in plan.rows:
        got = common if shared else grow(ra, int(ra.size))
        if got is None:
            return None
        sm, ids, nl, srows = got
        wk = np.zeros(n, dtype=np.float64)
        wk[ra] = w_all[ra]
        vf = _first_read._values_rf if rf else _first_read._values
        leaf, _, _ = vf(sm, ids, nl, tr.y, wk, cfg, th)
        rbs = rb if (not shared or score_sample) else np.setdiff1d(rb, srows)
        if rbs.size == 0:
            return None
        cv_scored_log.append({"fold": len(scorers), "scored": int(rbs.size), "held_out": int(rb.size)})
        fold = ([_MatrixScorer("train", _cv_subset(tr, ra), plan.eff, ids, ra, leaf, rf)] if plan.train_metric else [])
        scorers.append(fold + [_MatrixScorer("valid", _cv_subset(tr, rbs), plan.eff, ids, rbs, leaf, rf)])
    return scorers


def _cv_run(plan: SimpleNamespace) -> dict:
    global last_cv_form
    tr = plan.tr
    form = cv_form()
    cv_scored_log.clear()
    if plan.subs is not None:
        return _cv_loop_run(plan, *_cv_fold_scorers(plan))
    if getattr(plan, "round_p", None) is not None:
        last_cv_form = "rounds"
        return _cv_loop_run(plan, _cv_by_round(plan), None)
    scorers = None if (plan.ret_cvb or plan.lr_sched is not None or tr.init is not None) else _cv_first_read(plan, form != "perfold",
                                                                                    form == "matrix_scored")
    last_cv_form = ((form or "matrix") if scorers is not None else "exact")
    if scorers is None and plan.rf and not plan.ret_cvb and plan.lr_sched is None and form != "perfold":
        cv_scored_log.clear()
        scorers = _cv_first_read(plan, False)
        last_cv_form = "perfold" if scorers is not None else "exact"
    if scorers is not None and form != "perfold":
        LOG.info("shira.lightgbm: cv the matrix form: tree shapes grown once on the first read's sample of all "
                 "rows, leaf values per fold on its training rows; the held-out metric scored on %d of %d held-out rows "
                 "(%s)", sum(d["scored"] for d in cv_scored_log), sum(d["held_out"] for d in cv_scored_log),
                 "the sample's rows scored too: SHIRA_CV_FORM=matrix_scored" if form == "matrix_scored"
                 else "the shape sample's rows are never scored")
    models = None
    if scorers is None and plan.rf:
        raise Unsupported("boosting rf on exact folds: SHIRA has no exact rf engine (rf runs the first read only, as lgb.train; "
                          "the first read did not apply to this call)")
    if scorers is None and tr.cat:
        models = [_fit.fit_subsets(plan.q, plan.rs, tr.x, tr.y, [a], plan.rounds, tr.w, tr.g)[0] for a, _ in plan.rows]
        scorers = [_cv_exact_scorers(plan, m, ra, rb) for (ra, rb), m in zip(plan.rows, models)]
    if scorers is None:
        models = _fit.fit_subsets(plan.q, plan.rs, tr.x, tr.y, [a for a, _ in plan.rows], plan.rounds, tr.w, tr.g)
        scorers = [_cv_exact_scorers(plan, m, ra, rb) for (ra, rb), m in zip(plan.rows, models)]
    return _cv_loop_run(plan, scorers, None if models is None else [(m, plan) for m in models])


def _cv_exact_scorers(plan: SimpleNamespace, m: Any, ra: np.ndarray, rb: np.ndarray) -> list:
    full = _final_booster(plan, m, None, 0, {}, [])
    if m.dart:
        xgb_mode = plan.rs.cfg.get("xgboost_dart_mode", "0") == "1"
        mk = lambda name, rows: _DartScorer(name, _cv_subset(plan.tr, rows), plan.eff, full, m, xgb_mode,
                                            float(m.learning_rate))
    else:
        mk = lambda name, rows: _LeafScorer(name, _cv_subset(plan.tr, rows), plan.eff, full, m.num_class)
    fold = [mk("train", ra)] if plan.train_metric else []
    return fold + [mk("valid", rb)]


K_ZERO = 1e-35
K_EPS = 1e-15


def _rtz(v: np.ndarray) -> np.ndarray:
    return np.where(np.abs(v) > K_ZERO, v, 0.0)


class _DartScorer(_Scorer):

    def __init__(self, name: str, arr: SimpleNamespace, eff: dict[str, Any], full: Booster, m: Any, xgb_mode: bool,
                 lr: float) -> None:
        super().__init__(name, arr, eff, full, m.num_class)
        assert m.leaf0 is not None and m.drops is not None, "the engine's DART record"
        self.leaves = np.asarray(full.predict(arr.x, pred_leaf=True), dtype=np.int64).reshape(arr.x.shape[0], -1)
        self.m, self.xgb_mode, self.lr = m, xgb_mode, lr
        self.C = int(m.num_class)
        self.vals: dict[int, np.ndarray] = {}
        for c in range(self.C):
            if abs(float(m.init[c])) > K_EPS:
                self.score[:, c] = self.score[:, c] + float(m.init[c])
        assert self.leaves.shape[1] == int(m.nleaves.size)

    def _add(self, c: int, t: int, v: np.ndarray) -> None:
        self.score[:, c] = self.score[:, c] + v[self.leaves[:, t]]

    def advance(self, i: int) -> list:
        m, C = self.m, self.C
        drops = m.drops[i]
        k = float(len(drops))
        for d in drops:
            for c in range(C):
                t = d * C + c
                self.vals[t] = _rtz(-self.vals[t])
        for c in range(C):
            t = i * C + c
            nl = int(m.nleaves[t])
            assert nl > 1, "a one-leaf DART tree"
            pre = np.asarray(m.leaf0[t, :nl], dtype=np.float64)
            self._add(c, t, pre)
            bias = float(m.init[c]) if (i == 0 and abs(float(m.init[c])) > K_EPS) else 0.0
            self.vals[t] = _rtz(pre + bias) if bias != 0.0 else pre
        for d in drops:
            for c in range(C):
                t = d * C + c
                first = (self.lr / (self.lr + k)) if self.xgb_mode else (1.0 / (k + 1.0))
                v = _rtz(self.vals[t] * first)
                self._add(c, t, v)
                self.vals[t] = _rtz(v * ((-k / self.lr) if self.xgb_mode else -k))
        self.ds.set_init_score(self.score[:, 0] if C == 1 else self.score)
        res = _basic.Booster(params=dict(self.params), train_set=self.ds).eval_train(getattr(self, "feval", None))
        assert isinstance(res, list)
        return [(self.name, metric, value, hib) for _, metric, value, hib in res]


def _cv_fold_scorers(plan: SimpleNamespace) -> tuple[list, list | None]:
    global last_cv_form
    own = any(getattr(sub, "fit", None) is not None for sub in plan.subs)
    hw = None if (plan.ret_cvb or plan.lr_sched is not None or own) else [_cv_first_read(sub, False) for sub in plan.subs]
    if hw is not None and all(h is not None for h in hw):
        last_cv_form = "perfold"
        return [h[0] for h in hw], None
    cv_scored_log.clear()
    last_cv_form = "exact"
    if any(sub.tr.cat for sub in plan.subs):
        raise Unsupported("categorical columns on exact folds (fpreproc): SHIRA's multi-fold binning takes no "
                          "categorical feature yet")
    models = []
    for sub in plan.subs:
        ra, rb = sub.rows[0]
        fx, fy, fw = sub.fit if getattr(sub, "fit", None) is not None else (sub.tr.x, sub.tr.y, sub.tr.w)
        models.append((_fit.fit_subsets(sub.q, sub.rs, fx, fy, [ra], plan.rounds, fw, None)[0], sub))
    return [_cv_exact_scorers(sub, m, *sub.rows[0]) for m, sub in models], models


def _cv_loop_run(plan: SimpleNamespace, scorers: list, models: list | None) -> dict:
    tr = plan.tr
    for fold in scorers:
        for sc in fold:
            sc.feval = plan.feval
    before, after = _cv_callbacks(plan)
    cvb = _engine.CVBooster()
    results: dict = defaultdict(list)
    trained = plan.rounds
    for i in range(plan.rounds):
        env = {"model": cvb, "params": plan.p, "iteration": i, "begin_iteration": 0, "end_iteration": plan.rounds}
        for cb in before:
            cb(_cb.CallbackEnv(evaluation_result_list=None, **env))
        res = _engine._agg_cv_result([[r for s in fold for r in s.advance(i)] for fold in scorers])
        for ds_name, metric, mean, _, stdv in res:
            results[f"{ds_name} {metric}-mean"].append(mean)
            results[f"{ds_name} {metric}-stdv"].append(stdv)
        try:
            for cb in after:
                cb(_cb.CallbackEnv(evaluation_result_list=res, **env))
        except _cb.EarlyStopException as stop:
            cvb.best_iteration = stop.best_iteration + 1
            for k in results:
                results[k] = results[k][: cvb.best_iteration]
            trained = i + 1
            break
    if plan.ret_cvb:
        assert models is not None
        for m, mp in models:
            if mp is None:
                cut = trained < plan.rounds
                cvb.boosters.append(Booster(model_str=m.model_to_string(num_iteration=trained)) if cut else m)
                continue
            m.sched = mp.lr_sched is not None
            bst = _merged(mp, Booster(model_str=export.to_text(m, mp.rs, tr.names, trained, _pandas_categorical(tr.ds)),
                                      shira_model=m))
            bst.params = copy.deepcopy(mp.p)
            if cvb.best_iteration > 0:
                bst.best_iteration = cvb.best_iteration
            cvb.boosters.append(bst)
        results["cvbooster"] = cvb
    return dict(results)


class _FoldScorer(_Scorer):

    def advance(self, i: int) -> list:
        add = self.full.predict(self.x, raw_score=True, start_iteration=i, num_iteration=1)
        self.score = self.score + np.asarray(add, dtype=np.float64).reshape(self.score.shape)
        self.ds.set_init_score(self.score[:, 0] if self.score.shape[1] == 1 else self.score)
        res = _basic.Booster(params=dict(self.params), train_set=self.ds).eval_train(getattr(self, "feval", None))
        assert isinstance(res, list)
        return [(self.name, metric, value, hib) for _, metric, value, hib in res]


_ES_KEYS = ("early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change")


def _cv_custom(params: Any, train_set: Any, num_boost_round: int, kw: dict[str, Any]) -> dict:
    global last_cv_form
    if not _first_read.mode():
        raise Unsupported("a custom objective under SHIRA_LGB_MODE=ordinary: no exact engine takes a caller's objective; "
                          "the default runs it on the first read per fold (not stock's trees)")
    cbs = list(kw.get("callbacks") or [])
    for cb in cbs:
        if not isinstance(cb, _CALLBACKS):
            raise Unsupported(f"callback {getattr(cb, '__name__', type(cb).__name__)} with a custom objective")
    if kw.get("fpreproc") is not None or kw.get("init_model") is not None:
        raise Unsupported("a custom objective with fpreproc or init_model")
    p = _processed(params, num_boost_round)
    if kw.get("metrics") is not None:
        for alias in _basic._ConfigAliases.get("metric"):
            p.pop(alias, None)
        p["metric"] = kw["metrics"]
    tr = _extract(train_set, "train_set", train=True)
    if tr.x.shape[0] == 0:
        raise Unsupported("no rows")
    eff = _effective(p, train_set)
    if int(eff.get("num_class", 1)) > 1:
        raise Unsupported("a custom objective with num_class > 1 (LightGBM's general route does not take it yet)")
    pairs = _cv_folds(tr, p, kw.get("folds"), int(kw["nfold"]), bool(kw["stratified"]), bool(kw["shuffle"]), kw["seed"])
    if not 1 <= len(pairs) <= _fit.MAX_FOLDS:
        raise Unsupported(f"{len(pairs)} cv folds")
    amap, rounds = _fit.alias_to_main(), int(p["num_iterations"])
    tp = {k: v for k, v in eff.items() if amap.get(k, k) not in _ES_KEYS and k not in _ES_KEYS
          and amap.get(k, k) != "num_iterations"}
    eff_s = {**{k: v for k, v in eff.items() if amap.get(k, k) != "objective"}, "objective": "none"}
    scorers, models, rows = [], [], []
    for a, b in pairs:
        ra, rb = np.sort(np.asarray(a, dtype=np.int64)), np.sort(np.asarray(b, dtype=np.int64))
        sub = _cv_subset(tr, ra)
        ds = _basic.Dataset(sub.x, label=sub.y, weight=sub.w, group=sub.g, feature_name=list(tr.names),
                            categorical_feature=list(tr.cat) if tr.cat else "auto", free_raw_data=False)
        bst = train(dict(tp), ds, rounds)
        if last_backend != "shira" or not isinstance(bst, Booster):
            raise Unsupported("a fold's lgb.train with the caller's objective did not run on SHIRA (its reason is in "
                              "the log)")
        fold = [_FoldScorer("train", _cv_subset(tr, ra), eff_s, bst, 1)] if kw["eval_train_metric"] else []
        scorers.append(fold + [_FoldScorer("valid", _cv_subset(tr, rb), eff_s, bst, 1)])
        models.append((bst, None))
        rows.append((ra, rb))
    plan = SimpleNamespace(p=p, eff=eff, tr=tr, rows=rows, callbacks=cbs, rounds=rounds,
                           train_metric=bool(kw["eval_train_metric"]), first_metric_only=p.get("first_metric_only", False),
                           train_name="training", feval=kw.get("feval"), ret_cvb=bool(kw.get("return_cvbooster")),
                           lr_sched=None)
    last_cv_form = "perfold"
    cv_scored_log.clear()
    assert len(scorers) == len(pairs)
    return _cv_loop_run(plan, scorers, models)


def _cv_is_custom(params: Any) -> bool:
    if not isinstance(params, dict):
        return False
    return any(callable(params.get(a)) for a in _basic._ConfigAliases.get("objective"))


def _dump_codes(path: str, n: int, nf: int) -> np.ndarray:
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    nl = np.flatnonzero(raw == 10)
    if nl.size < n:
        raise Unsupported("the Dataset's text dump has fewer lines than rows")
    b = raw[int(nl[-n]) + 1:]
    if bool((b == 78).any()):
        raise Unsupported("the Dataset holds a feature LightGBM filtered out (no bins to read)")
    is_d = (b >= 48) & (b <= 57)
    starts = np.flatnonzero(is_d & ~np.concatenate([[False], is_d[:-1]]))
    ends = np.flatnonzero(is_d & ~np.concatenate([is_d[1:], [False]]))
    if starts.size != n * nf:
        raise Unsupported("the Dataset's text dump does not hold rows x features codes")
    length = ends - starts + 1
    vals = np.zeros(starts.size, dtype=np.int64)
    for k in range(int(length.max())):
        m = length > k
        vals[m] = vals[m] * 10 + (b[starts[m] + k].astype(np.int64) - 48)
    return vals.reshape(n, nf)


def _bin_space_twin(ds: Any, params: Any) -> tuple[Any, dict[str, Any]]:
    import tempfile

    if not isinstance(params, dict):
        raise Unsupported("params is not a dict")
    amap = _fit.alias_to_main()
    truthy = lambda v: str(v).strip().lower() in ("true", "1", "+")
    for src in (dict(ds.params or {}), params):
        if any(amap.get(k, k) == "zero_as_missing" and truthy(v) for k, v in src.items()):
            raise Unsupported("zero_as_missing (the zero bin is the missing bin)")
        if any(amap.get(k, k) == "linear_tree" and truthy(v) for k, v in src.items()):
            raise Unsupported("linear_tree (its leaves fit the raw values)")
    cf = ds.categorical_feature
    if (cf is not None and not (isinstance(cf, str) and cf == "auto") and len(cf) > 0) or getattr(ds, "pandas_categorical", None):
        raise Unsupported("categorical features (their bins are not ordered values)")
    if ds.used_indices is not None or ds.reference is not None:
        raise Unsupported("a subset or a Dataset with a reference")
    n, nf = int(ds.num_data()), int(ds.num_feature())
    lab = ds.get_label()
    if lab is None or n < 1 or nf < 1:
        raise Unsupported("no label or no rows")
    with tempfile.TemporaryDirectory() as td:
        ds._dump_text(os.path.join(td, "d.txt"))
        codes = _dump_codes(os.path.join(td, "d.txt"), n, nf)
        probe = _basic.Dataset(np.vstack([np.full((1, nf), np.nan), np.zeros((1, nf))]), reference=ds, free_raw_data=False)
        probe.construct()
        probe._dump_text(os.path.join(td, "p.txt"))
        pc = _dump_codes(os.path.join(td, "p.txt"), 2, nf)
    nb = np.array([int(ds.feature_num_bin(j)) for j in range(nf)], dtype=np.int64)
    xp = codes.astype(np.float64)
    for j in range(nf):
        if int(pc[0, j]) == int(nb[j]) - 1 and int(pc[0, j]) != int(pc[1, j]):
            xp[codes[:, j] == nb[j] - 1, j] = np.nan
    twin = _basic.Dataset(xp, label=lab, weight=ds.get_weight(), group=ds.get_group(), init_score=ds.get_init_score(),
                          feature_name=[str(v) for v in ds.get_feature_name()], free_raw_data=False)
    drop = {"max_bin", "min_data_in_bin", "bin_construct_sample_cnt", "max_bin_by_feature", "forcedbins_filename",
            "feature_pre_filter"}
    p2 = {k: v for k, v in params.items() if amap.get(k, k) not in drop}
    p2.update(max_bin=max(2, int(nb.max())), min_data_in_bin=1, bin_construct_sample_cnt=max(n, 2), feature_pre_filter=False)
    assert xp.shape == (n, nf)
    return twin, p2


def cv(params: dict[str, Any], train_set: Any, num_boost_round: int = 100, folds: Any = None, nfold: int = 5,
       stratified: bool = True, shuffle: bool = True, metrics: Any = None, feval: Any = None,
       init_model: Any = None, fpreproc: Any = None, seed: int = 0, callbacks: Any = None,
       eval_train_metric: bool = False, return_cvbooster: bool = False) -> dict:
    global last_backend
    kw = {"folds": folds, "nfold": nfold, "stratified": stratified, "shuffle": shuffle, "metrics": metrics,
          "feval": feval, "init_model": init_model, "fpreproc": fpreproc, "seed": seed, "callbacks": callbacks,
          "eval_train_metric": eval_train_metric, "return_cvbooster": return_cvbooster}
    if folds is not None and not hasattr(folds, "split") and hasattr(folds, "__iter__") \
            and not isinstance(folds, (list, tuple)):
        kw["folds"] = list(folds)
    if isinstance(train_set, _basic.Dataset) and train_set.data is None and train_set._handle is not None:
        global last_cv_form
        why = ("SHIRA_LGB_MODE=ordinary (stock's cv is the exact answer)" if not _first_read.mode() else
               "return_cvbooster / fpreproc / init_model need the raw values" if (
                   return_cvbooster or fpreproc is not None or init_model is not None or _cv_is_custom(params)) else None)
        twin = None
        if why is None:
            try:
                twin = _bin_space_twin(train_set, params)
            except Unsupported as e:
                why = str(e)
        if twin is not None:
            LOG.info("shira.lightgbm: cv on a constructed Dataset whose raw data was freed: SHIRA's folds run on "
                     "LightGBM's own bin numbers of every row (as the first read)")
            return cv(twin[1], twin[0], num_boost_round, folds=kw["folds"], nfold=nfold, stratified=stratified,
                      shuffle=shuffle, metrics=metrics, feval=feval, seed=seed, callbacks=callbacks,
                      eval_train_metric=eval_train_metric)
        LOG.warning("shira.lightgbm: running stock LightGBM's cv: train_set holds no raw data (it was constructed with "
                    "free_raw_data=True, so only LightGBM's bins remain) and %s; pass the Dataset before it is "
                    "constructed, or build it with free_raw_data=False, for SHIRA's folds", why)
        out = _engine.cv(params, train_set, num_boost_round, folds=kw["folds"], nfold=nfold, stratified=stratified,
                         shuffle=shuffle, metrics=metrics, feval=feval, init_model=init_model, fpreproc=fpreproc,
                         seed=seed, callbacks=callbacks, eval_train_metric=eval_train_metric,
                         return_cvbooster=return_cvbooster)
        last_backend, last_cv_form = "lightgbm", "stock"
        backend_history.append(last_backend)
        return out
    try:
        out = (_cv_custom(params, train_set, num_boost_round, kw) if _cv_is_custom(params)
               else _cv_run(_cv_plan(params, train_set, num_boost_round, kw)))
    except Unsupported as e:
        if _first_read.mode():
            raise NotImplementedError(f"shira.lightgbm.cv: no SHIRA route for this call: {e}") from e
        LOG.warning("shira.lightgbm: SHIRA_LGB_MODE=ordinary and no exact SHIRA engine takes this cv call (%s): running "
                    "stock LightGBM's cv (the exact answer); the default mode runs it on SHIRA", e)
        out = _engine.cv(params, train_set, num_boost_round, folds=kw["folds"], nfold=nfold, stratified=stratified,
                         shuffle=shuffle, metrics=metrics, feval=feval, init_model=init_model, fpreproc=fpreproc,
                         seed=seed, callbacks=callbacks, eval_train_metric=eval_train_metric,
                         return_cvbooster=return_cvbooster)
        last_backend, last_cv_form = "lightgbm", "stock"
        backend_history.append(last_backend)
        return out
    last_backend = "shira"
    backend_history.append(last_backend)
    return out


_THREAD_KEYS = frozenset(("num_threads", "num_thread", "nthread", "nthreads", "n_jobs"))


class _SklearnRoute:

    def __init__(self, est: Any = None) -> None:
        extra = set(getattr(est, "_other_params", {}) or {})
        self.default = est is not None and getattr(est, "n_jobs", 0) is None and not (extra & _THREAD_KEYS)

    def __enter__(self) -> None:
        self.saved = _sk.train
        _sk.train = train
        self.token = _fit.SK_DEFAULT_THREADS.set(self.default)

    def __exit__(self, *exc: Any) -> None:
        _sk.train = self.saved
        _fit.SK_DEFAULT_THREADS.reset(self.token)


class LGBMRegressor(_sk.LGBMRegressor):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> LGBMRegressor:
        with _SklearnRoute(self):
            super().fit(X, y, **kwargs)
        return self


class LGBMClassifier(_sk.LGBMClassifier):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> LGBMClassifier:
        with _SklearnRoute(self):
            super().fit(X, y, **kwargs)
        return self


class LGBMRanker(_sk.LGBMRanker):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> LGBMRanker:
        with _SklearnRoute(self):
            super().fit(X, y, **kwargs)
        return self


_PATCHED: dict[str, Any] = {}


def patch() -> None:
    mine = {"train": train, "LGBMRegressor": LGBMRegressor, "LGBMClassifier": LGBMClassifier,
            "LGBMRanker": LGBMRanker, "cv": cv}
    if not _PATCHED:
        _PATCHED.update({k: getattr(_lgb, k) for k in mine})
        _PATCHED["sklearn.train"] = _sk.train
    for k, v in mine.items():
        setattr(_lgb, k, v)
    _sk.train = train
    assert _lgb.train is train


def unpatch() -> None:
    if "sklearn.train" in _PATCHED:
        _sk.train = _PATCHED.pop("sklearn.train")
    for k, v in _PATCHED.items():
        setattr(_lgb, k, v)
    _PATCHED.clear()
    assert _lgb.train is _STOCK["train"]


def __getattr__(name: str) -> Any:
    return getattr(_lgb, name)
