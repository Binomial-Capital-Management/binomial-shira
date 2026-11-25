from __future__ import annotations

import copy
import ctypes
import json
import logging
import math
import os
import re
import sys
import warnings
import weakref
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import xgboost as _xgb
from xgboost import callback as _cb
from xgboost import core as _core
from xgboost import sklearn as _sk
from xgboost import training as _training

from . import _machine
from . import _threads
from . import _xgb_fit as _fit
from . import export
from . import trees as _trees

LOG = logging.getLogger("shira.xgboost")
STOCK_VERSION = "3.2.0"


def version_reason() -> str | None:
    v = str(getattr(_xgb, "__version__", "unknown"))
    if v == STOCK_VERSION:
        return None
    return f"xgboost {v} is installed; SHIRA's engines and first reads reproduce xgboost {STOCK_VERSION} only"
F32 = np.float32
MAX_ROUNDS = 100_000
MAX_SLOTS = 65_535
FAST_MAX_FEATURES = 1_000
MAX_WIDE_BINS = 4_086
last_backend: str | None = None


def default_engine(n: int | None = None, F: int | None = None) -> str:
    from ._lib import lib

    if n is not None and F is not None:
        return "fast" if lib().shira_xgb_mode_for(int(n), int(F)) == 1 else "exact"
    return "fast" if lib().shira_xgb_mode() == 1 else "exact"
backend_history: list[str] = []

_STOCK: dict[str, Any] = {"train": _training.train, "DMatrix": _core.DMatrix,
                          "QuantileDMatrix": _core.QuantileDMatrix, "XGBRegressor": _sk.XGBRegressor,
                          "XGBClassifier": _sk.XGBClassifier}


ALIASES = {"learning_rate": "eta", "reg_lambda": "lambda", "reg_alpha": "alpha", "min_split_loss": "gamma",
           "random_state": "seed", "n_jobs": "nthread"}


def _is(*vals: Any) -> Callable[[Any], bool]:
    return lambda v: v in vals


def _num(lo: float, hi: float, lo_open: bool = False) -> Callable[[Any], bool]:
    return lambda v: (lo < float(v) if lo_open else lo <= float(v)) and float(v) <= hi


def _anything(_: Any) -> bool:
    return True


def _int_in(lo: int, hi: int) -> Callable[[Any], bool]:
    return lambda v: int(v) == v and lo <= int(v) <= hi


def _int_list(v: Any) -> bool:
    vals = [x for x in v.strip("()").split(",") if x.strip()] if isinstance(v, str) else list(v)
    return all(int(x) in (-1, 0, 1) for x in vals)


def _index_sets(v: Any) -> bool:
    sets = json.loads(v) if isinstance(v, str) else None
    return sets is not None and all(isinstance(f, int) for s in sets for f in s)


OBJECTIVES = ("reg:pseudohubererror", "binary:logistic", "reg:squarederror", "reg:logistic", "binary:logitraw",
              "count:poisson", "reg:tweedie", "reg:gamma", "reg:absoluteerror", "reg:quantileerror",
              "survival:cox", "multi:softprob", "multi:softmax")

SUPPORTED: dict[str, Callable[[Any], bool]] = {
    "objective": _is(*OBJECTIVES),
    "huber_slope": _num(0.0, math.inf, lo_open=True),
    "quantile_alpha": lambda v: np.ndim(v) == 0 and 0.0 < float(v) < 1.0,
    "tweedie_variance_power": _num(1.0, 2.0),
    "num_class": _int_in(2, 10_000),
    "tree_method": _is("hist", "auto"),
    "grow_policy": _is("depthwise", "lossguide"),
    "max_depth": _int_in(0, 12),
    "max_leaves": _int_in(0, 1 << 16),
    "eta": _num(0.0, math.inf, lo_open=True),
    "lambda": _num(0.0, math.inf, lo_open=True),
    "alpha": _num(0.0, math.inf),
    "max_delta_step": _num(0.0, math.inf),
    "min_child_weight": _num(0.0, math.inf),
    "gamma": _num(0.0, math.inf),
    "max_bin": _int_in(2, 65534),
    "base_score": lambda v: math.isfinite(float(v)),
    "subsample": _num(0.0, 1.0, lo_open=True),
    "colsample_bytree": _num(0.0, 1.0, lo_open=True),
    "colsample_bylevel": _num(0.0, 1.0, lo_open=True),
    "colsample_bynode": _num(0.0, 1.0, lo_open=True),
    "seed": lambda v: int(v) == v,
    "nthread": _int_in(1, 1024),
    "monotone_constraints": _int_list,
    "interaction_constraints": _index_sets,
    "verbosity": _anything, "validate_parameters": _anything, "disable_default_eval_metric": _anything,
    "max_cat_to_onehot": _anything, "max_cat_threshold": _anything,
    "fail_on_invalid_gpu_id": _anything, "sparse_threshold": _anything, "max_cached_hist_node": _anything,
    "debug_synchronize": _anything,
    "_shira_etas": _anything, "_shira_base_margin": _anything,
    "_shira_start_margin": _anything, "_shira_eval_start_margin": _anything,
    "booster": _is("gbtree"), "device": _is("cpu"), "num_parallel_tree": _is(1),
    "scale_pos_weight": _num(0.0, 3.0e38, lo_open=True),
    "sampling_method": _is("uniform"),
    "multi_strategy": _is("one_output_per_tree"), "seed_per_iteration": _is(False, 0),
}
NO_ENGINE_KEYS = ("verbosity", "validate_parameters", "disable_default_eval_metric", "max_cat_to_onehot",
                  "max_cat_threshold", "booster", "device", "num_parallel_tree", "scale_pos_weight",
                  "sampling_method", "multi_strategy", "seed_per_iteration", "tree_method")


def normalize_params(params: Any) -> tuple[dict[str, Any], list[str], str | None]:
    items = list(params.items()) if isinstance(params, dict) else list(params or [])
    assert len(items) < 10_000
    out: dict[str, Any] = {}
    metrics: list[str] = []
    for k, v in items:
        if v is None:
            continue
        if k == "eval_metric":
            metrics += [v] if isinstance(v, str) else list(v)
            continue
        key = ALIASES.get(k, k)
        if key in out and out[key] != v:
            return out, metrics, f"param {key} given twice with different values"
        out[key] = v
    assert not set(out) & set(ALIASES), "aliases are canonical after normalisation"
    return out, metrics, None


_IGNORED_CACHE: dict[str, frozenset] = {}
_UNUSED = re.compile(r"Parameters: \{ (.*) \} are not used")


def _stock_ignores(p: dict[str, Any], keys: list[str]) -> set[str]:
    obj = str(p.get("objective", "reg:squarederror"))
    sub = {k: p[k] for k in ("objective", "booster", "num_class", "multi_strategy", "tree_method") if k in p}
    sub.update({k: p[k] for k in keys})
    sub["nthread"] = 1
    ck = json.dumps(sorted((k, repr(v)) for k, v in sub.items()))
    if ck not in _IGNORED_CACHE:
        _IGNORED_CACHE[ck] = frozenset(_stock_unused_probe(sub, obj))
    out = set(_IGNORED_CACHE[ck]) & set(keys)
    if out:
        warnings.warn("Parameters: { " + ", ".join(f'"{k}"' for k in sorted(out)) + " } are not used.", UserWarning,
                      stacklevel=3)
    return out


def _stock_unused_probe(sub: dict[str, Any], obj: str) -> set[str]:
    n = 8
    X = np.arange(n * 2, dtype=F32).reshape(n, 2)
    if obj.startswith("survival:") or obj.startswith("rank:"):
        return set()
    if obj.startswith("multi:"):
        y = np.arange(n, dtype=F32) % max(2, int(sub.get("num_class", 2)))
    elif obj.startswith("binary:") or obj == "reg:logistic":
        y = np.arange(n, dtype=F32) % 2
    else:
        y = np.arange(1, n + 1, dtype=F32) / n
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            _STOCK["train"](dict(sub), _STOCK["DMatrix"](X, label=y), 1)
    except _core.XGBoostError:
        return set()
    for w in wl:
        m = _UNUSED.search(str(w.message))
        if m:
            return {t.strip().strip('"') for t in m.group(1).split(",")}
    return set()


def why_not(params: Any) -> str | None:
    if version_reason() is not None:
        return version_reason()
    p, metrics, reason = normalize_params(params)
    assert isinstance(p, dict) and isinstance(metrics, list)
    if reason:
        return reason
    if not all(isinstance(m, str) for m in metrics):
        return "callable eval_metric"
    unknown = [k for k in p if k not in SUPPORTED]
    ignored = _stock_ignores(p, unknown) if unknown else set()
    for k, v in p.items():
        check = SUPPORTED.get(k)
        if check is None and k in ignored:
            continue
        if check is None:
            return f"param {k}={v!r} is not in SHIRA's table"
        try:
            ok = check(v)
        except (TypeError, ValueError, AttributeError):
            ok = False
        if not ok:
            return f"param {k}={v!r} is outside what SHIRA reproduces exactly"
    reason = _combo_reason(p)
    assert reason is not None or p.get("objective", "reg:squarederror") in OBJECTIVES
    return reason


DEEP_HIST_BYTES = 1073741824.0
MULTI_NO_BASE = "multi:* without base_score: XGBoost fits one intercept per class, the engine takes one"


SPW_OBJ = ("reg:squarederror", "reg:logistic", "binary:logistic", "binary:logitraw",
           "reg:gamma")


def _spw(p: dict[str, Any]) -> float:
    v = p.get("scale_pos_weight")
    return 1.0 if v is None else float(v)


def _spw_weights(p: dict[str, Any], y: np.ndarray, w: np.ndarray | None) -> np.ndarray | None:
    s = _spw(p)
    if s == 1.0 or p.get("objective", "reg:squarederror") not in SPW_OBJ:
        return w
    base = np.ones(y.size, dtype=F32) if w is None else np.asarray(w, dtype=F32)
    out = np.where(np.asarray(y, dtype=F32) == F32(1.0), base * F32(s), base).astype(F32)
    assert out.shape == y.shape and np.isfinite(out).all()
    return out


def _combo_reason(p: dict[str, Any]) -> str | None:
    obj = p.get("objective", "reg:squarederror")
    multi = obj.startswith("multi:")
    if multi != ("num_class" in p):
        return "num_class goes with multi:* objectives only"
    if multi and "base_score" not in p:
        return MULTI_NO_BASE
    if obj in ("binary:logistic", "reg:logistic") and "base_score" in p and not 0.0 < float(p["base_score"]) < 1.0:
        return f"{obj} base_score outside (0, 1)"
    lossguide = p.get("grow_policy", "depthwise") == "lossguide"
    if not lossguide and (int(p.get("max_leaves", 0)) != 0 or int(p.get("max_depth", 6)) == 0):
        return "depthwise growth needs max_leaves 0 and max_depth 1..12"
    if lossguide and int(p.get("max_leaves", 0)) == 0 and int(p.get("max_depth", 6)) == 0:
        return "lossguide with no max_leaves and no max_depth (unbounded tree)"
    if float(p.get("subsample", 1.0)) < 1.0 and "nthread" not in p:
        return "subsample < 1 without nthread: XGBoost's row-sampling blocks depend on its thread count"
    assert isinstance(obj, str)
    return None


def _raw_matrix(data: Any) -> tuple[np.ndarray | None, str | None]:
    x = None
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.dtype.kind in "fiub":
        x = np.ascontiguousarray(data, dtype=F32)
    elif type(data).__name__ == "DataFrame" and hasattr(data, "dtypes"):
        kinds = [getattr(dt, "kind", "?") if isinstance(dt, np.dtype) else "?" for dt in data.dtypes]
        if kinds and all(k in "fiu" for k in kinds):
            x = np.ascontiguousarray(data.to_numpy(dtype=F32))
    if x is None:
        return None, f"input type {type(data).__name__} (numpy arrays and numeric pandas frames only)"
    if np.isinf(x).any():
        return None, "infinite feature values"
    assert x.ndim == 2
    return x, None


def _capture(obj: Any, data: Any, kwargs: dict[str, Any]) -> None:
    miss = kwargs.get("missing")
    if miss is not None and not (isinstance(miss, float) and math.isnan(miss)):
        obj._shira_x, obj._shira_why = None, f"missing={miss!r} (NaN only)"
        return
    obj._shira_x, obj._shira_why = _raw_matrix(data)
    assert obj._shira_x is None or obj._shira_x.shape[0] == obj.num_row()


class DMatrix(_core.DMatrix):

    def __init__(self, data: Any, label: Any = None, **kwargs: Any) -> None:
        super().__init__(data, label, **kwargs)
        _capture(self, data, kwargs)
        if self._shira_x is None and kwargs.get("enable_categorical"):
            self.__dict__["_sx_src"] = (data, label, dict(kwargs))


_LAZY_NONE = ("weight", "base_margin", "feature_names", "feature_types", "ref", "group", "qid", "label_lower_bound",
              "label_upper_bound", "feature_weights", "max_quantile_batches")
_LAZY_KEYS = frozenset(_LAZY_NONE + ("missing", "silent", "nthread", "max_bin", "enable_categorical", "data_split_mode"))


def _pandas_matrix(data: Any) -> tuple[np.ndarray, tuple[Any, Any]] | None:
    from xgboost import data as _xd

    if not _xd._is_pandas_df(data) or data.shape[0] < 1 or data.shape[1] < 1:
        return None
    if not all(isinstance(t, np.dtype) and t.kind == "f" and t.itemsize in (4, 8) for t in data.dtypes):
        return None
    names, types = _xd.pandas_feature_info(data, None, None, None, False)
    mat = data.to_numpy()
    assert mat.ndim == 2 and mat.shape == data.shape and len(names) == mat.shape[1]
    return mat, (names, types)


def _lazy_inputs(data: Any, label: Any, kw: dict[str, Any]) -> tuple[Any, np.ndarray, Any, Any] | None:
    if version_reason() is not None:
        return None
    info = None
    if sampled_cuts() and not isinstance(data, np.ndarray):
        pm = _pandas_matrix(data)
        if pm is None:
            return None
        data, info = pm
    if not (isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] >= 1 and data.shape[1] >= 1):
        return None
    if set(kw) - _LAZY_KEYS or any(kw.get(k) is not None for k in _LAZY_NONE) or kw.get("enable_categorical", False):
        return None
    miss, nt, mb = kw.get("missing"), kw.get("nthread"), kw.get("max_bin")
    if miss is not None and not (isinstance(miss, float) and math.isnan(miss)):
        return None
    if (nt is not None and not (isinstance(nt, int) and nt >= 1)) or (mb is not None and not (isinstance(mb, int) and 2 <= mb <= 1024)):
        return None
    if kw.get("data_split_mode", _core.DataSplitMode.ROW) != _core.DataSplitMode.ROW or label is None:
        return None
    y = np.asarray(label)
    if y.ndim != 1 or y.shape[0] != data.shape[0] or y.dtype.kind not in "fiub":
        return None
    yf = np.ascontiguousarray(y, dtype=F32)
    if not np.isfinite(yf).all():
        return None
    if sampled_cuts() and _fast_input(data):
        return None, yf, data, info
    if info is not None:
        return None
    x, _ = _raw_matrix(data)
    if x is None:
        return None
    assert x.shape[0] == yf.shape[0]
    return x, yf, data, None


class QuantileDMatrix(_core.QuantileDMatrix):

    def __init__(self, data: Any, label: Any = None, **kwargs: Any) -> None:
        lazy = _lazy_inputs(data, label, kwargs)
        if lazy is None:
            super().__init__(data, label, **kwargs)
            _capture(self, data, kwargs)
            self.__dict__["_sx_src"] = (data, label, dict(kwargs))
            return
        self.__dict__["_lz"] = (data, label, dict(kwargs))
        self.max_bin = kwargs.get("max_bin")
        self.missing = np.nan
        self.nthread = kwargs.get("nthread") if kwargs.get("nthread") is not None else -1
        self._silent = bool(kwargs.get("silent", False))
        self._shira_why = None
        mat = lazy[2]
        self.__dict__["_lz_mat"], self.__dict__["_lz_fi"] = mat, lazy[3]
        if lazy[0] is None:
            self.__dict__["_sx_lazy64"] = mat
        else:
            self._shira_x = lazy[0]
        self.__dict__["_lz_shape"] = (int(mat.shape[0]), int(mat.shape[1]))
        self._lz_y, self._lz_cut = lazy[1], None

    @property
    def _shira_x(self) -> Any:
        v = self.__dict__.get("_sx")
        raw = self.__dict__.get("_sx_lazy64")
        if v is None and raw is not None:
            v, why = _raw_matrix(raw)
            self.__dict__["_sx_lazy64"] = None
            if v is None:
                self._shira_why = why
            self.__dict__["_sx"] = v
        return v

    @_shira_x.setter
    def _shira_x(self, value: Any) -> None:
        self.__dict__["_sx"] = value

    @property
    def handle(self) -> Any:
        h = self.__dict__.get("_h")
        if h is None:
            if "_lz" not in self.__dict__:
                raise AttributeError("handle")
            self._materialize()
            h = self.__dict__["_h"]
        return h

    @handle.setter
    def handle(self, value: Any) -> None:
        self.__dict__["_h"] = value

    @handle.deleter
    def handle(self) -> None:
        self.__dict__.pop("_h", None)

    def _lazy(self) -> bool:
        return self.__dict__.get("_h") is None and "_lz" in self.__dict__

    def _materialize(self) -> None:
        data, label, kwargs = self.__dict__.pop("_lz")
        materialized.append(1)
        _core.QuantileDMatrix.__init__(self, data, label, **kwargs)
        assert self.__dict__.get("_h") is not None

    def __del__(self) -> None:
        if self.__dict__.get("_h") is not None:
            super().__del__()

    def num_row(self) -> int:
        return self.__dict__["_lz_shape"][0] if self._lazy() else super().num_row()

    def num_col(self) -> int:
        return self.__dict__["_lz_shape"][1] if self._lazy() else super().num_col()

    def get_float_info(self, field: str) -> Any:
        if not self._lazy() or field not in ("label", "weight", "base_margin", "label_lower_bound", "label_upper_bound",
                                             "feature_weights"):
            return super().get_float_info(field)
        return self._lz_y.copy() if field == "label" else np.zeros(0, dtype=F32)

    def get_uint_info(self, field: str) -> Any:
        if not self._lazy() or field != "group_ptr":
            return super().get_uint_info(field)
        return np.zeros(0, dtype=np.uint32)

    def get_label(self) -> Any:
        return self.get_float_info("label")

    def get_weight(self) -> Any:
        return self.get_float_info("weight")

    @property
    def feature_names(self) -> Any:
        if self._lazy():
            return None if self.__dict__.get("_lz_fi") is None else list(self.__dict__["_lz_fi"][0])
        return _core.DMatrix.feature_names.fget(self)

    @feature_names.setter
    def feature_names(self, value: Any) -> None:
        _core.DMatrix.feature_names.fset(self, value)

    @property
    def feature_types(self) -> Any:
        if self._lazy():
            return None if self.__dict__.get("_lz_fi") is None else list(self.__dict__["_lz_fi"][1])
        return _core.DMatrix.feature_types.fget(self)

    @feature_types.setter
    def feature_types(self, value: Any) -> None:
        _core.DMatrix.feature_types.fset(self, value)

    def get_categories(self, export_to_arrow: bool = False) -> Any:
        if not self._lazy() or export_to_arrow:
            return super().get_categories(export_to_arrow)
        return _core.Categories((ctypes.c_void_p(), lambda: None), None)

    def get_quantile_cut(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._lazy():
            return super().get_quantile_cut()
        if self._lz_cut is None:
            nt = self.nthread if self.nthread >= 1 else _threads.xgb_default_nthread()
            raw = self.__dict__["_lz_mat"]
            if sampled_cuts() and _fast_input(raw) and raw.shape[0] > SAMPLE_ROWS:
                from shira_cpp import binning_cpp

                try:
                    ip, cv, codes = binning_cpp.xgb_sampled_cuts_codes(raw, int(self.max_bin or 256), SAMPLE_ROWS, nt,
                                                                       u8=raw.shape[0] >= U8_MIN_ROWS)
                except ValueError:
                    self.__dict__["_sx_lazy64"], self.__dict__["_sx"] = None, None
                    self._shira_why = "infinite feature values"
                    raise _Unroutable(self._shira_why) from None
                self._lz_cut, self._lz_codes = (ip, cv), codes
            else:
                if self._shira_x is None:
                    raise _Unroutable(str(self._shira_why))
                self._lz_cut = _fit.cuts(self._shira_x, int(self.max_bin or 256), nt)
        ip, cv = self._lz_cut
        return ip.copy(), cv.copy()


materialized: list[int] = []


class _Unroutable(Exception):
    pass


SAMPLE_ROWS = 200_000
U8_MIN_ROWS = 2_000_000


def _fast_input(data: Any) -> bool:
    from shira_cpp import binning_cpp

    return binning_cpp.xgb_view(data) is not None


def sampled_cuts() -> bool:
    return os.environ.get("SHIRA_XGB_CUTS", "").strip().lower() != "stock"


def _data_reason(dm: Any, name: str, groups_ok: bool = False, margin_ok: bool = False) -> str | None:
    if not isinstance(dm, (DMatrix, QuantileDMatrix)):
        return f"{name} was not built with shira.xgboost.DMatrix / QuantileDMatrix"
    lazy64 = dm.__dict__.get("_sx_lazy64") is not None
    if not lazy64 and dm._shira_x is None:
        return f"{name}: {dm._shira_why}"
    infos = (("base_margin", "base_margin"), ("label_lower_bound", "label bounds"),
             ("feature_weights", "feature_weights"))[1 if margin_ok else 0:]
    for key, what in infos:
        if dm.get_float_info(key).size > 0:
            return f"{name}: {what}"
    if dm.get_uint_info("group_ptr").size > 0 and not groups_ok:
        return f"{name}: query groups"
    if dm.feature_types is not None and any(t == "c" for t in dm.feature_types):
        return f"{name}: categorical features"
    y = dm.get_label()
    if y.size != dm.num_row() or not np.isfinite(y).all():
        return f"{name}: labels missing, non-finite or multi-target"
    w = dm.get_weight()
    if w.size and (w.size != dm.num_row() or not np.isfinite(w).all()):
        return f"{name}: weights not one finite value per row"
    if not lazy64:
        assert dm._shira_x.shape == (dm.num_row(), dm.num_col()), "raw matrix matches the DMatrix"
        assert dm._shira_x.dtype == F32
    return None


def _call_reason(params: Any, dtrain: Any, rounds: int, evals: list, extra: dict[str, Any]) -> str | None:
    assert isinstance(evals, list) and isinstance(extra, dict)
    assert len(evals) < 1000
    reason = why_not(params)
    if reason:
        return reason
    for key in ("obj", "custom_metric", "xgb_model"):
        if extra.get(key) is not None:
            return f"{key} is set"
    for cb in extra.get("callbacks") or []:
        if type(cb) not in (_cb.EarlyStopping, _cb.EvaluationMonitor):
            return f"callback {type(cb).__name__}"
    if not 1 <= int(rounds) <= MAX_ROUNDS:
        return f"num_boost_round {rounds}"
    rank = str(normalize_params(params)[0].get("objective", "")).startswith("rank:")
    reason = _data_reason(dtrain, "dtrain", groups_ok=not rank)
    for dm, name in evals:
        reason = reason or _data_reason(dm, f"eval set {name!r}")
        if not reason and dm.num_col() != dtrain.num_col():
            reason = f"eval set {name!r} feature count differs"
        if not reason and isinstance(dm, _core._RefMixIn) and dm.ref is not weakref.ref(dtrain) and dm is not dtrain:
            reason = f"eval set {name!r} is a QuantileDMatrix without ref=dtrain"
    return reason


PROBE_KEYS = ("objective", "max_bin", "nthread", "huber_slope", "base_score", "quantile_alpha", "scale_pos_weight",
              "tweedie_variance_power", "max_delta_step", "num_class",
              "lambdarank_pair_method", "lambdarank_num_pair_per_sample", "lambdarank_normalization",
              "lambdarank_score_normalization", "ndcg_exp_gain")


def _xgb_probe(p: dict[str, Any], dtrain: Any) -> str:
    probe = {k: p[k] for k in PROBE_KEYS if k in p}
    probe.update(tree_method="hist", max_depth=1)
    bst = _core.Booster(probe, [dtrain])
    bst.update(dtrain, 0)
    base = json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]
    vals = base.strip("[]").split(",")
    assert base.startswith("[") and base.endswith("]") and (len(vals) == 1 or "base_score" in p), base
    return vals[0]


def _objective_json(p: dict[str, Any], nf: int) -> dict:
    keys = [k for k in PROBE_KEYS if k in p and k not in ("max_bin", "base_score")]
    cfg = json.loads(_core.Booster(dict({k: p[k] for k in keys}, num_feature=nf)).save_config())
    obj = cfg["learner"]["objective"]
    assert obj["name"] == p.get("objective", "reg:squarederror"), obj
    return obj


_OBJ_CACHE: dict[str, tuple[dict, float]] = {}


def _objective_and_margin(p: dict[str, Any], nf: int, obj: str, base: float, C: int) -> tuple[dict, float]:
    key = json.dumps([sorted((k, str(p[k])) for k in PROBE_KEYS if k in p and k != "max_bin"), nf, obj,
                      float(F32(base)), C])
    if key not in _OBJ_CACHE:
        if len(_OBJ_CACHE) > 256:
            _OBJ_CACHE.clear()
        obj_json = _objective_json(p, nf)
        _OBJ_CACHE[key] = (obj_json, float(export.xgb_base_margin(obj, base, obj_json, C)))
    obj_json, margin = _OBJ_CACHE[key]
    assert obj_json["name"] == obj
    return copy.deepcopy(obj_json), margin


def _call_threads(p: dict[str, Any]) -> int:
    n = int(p["nthread"]) if "nthread" in p else _threads.xgb_default_nthread()
    assert n >= 1
    return n


def _abs_intercept(y: np.ndarray, w: np.ndarray | None, threads: int) -> Any:
    from shira_cpp import product_cpp

    yy = np.ascontiguousarray(y, dtype=F32)
    if yy.size == 0 or not bool(np.isfinite(yy).all()):
        return None
    ww = None if w is None else np.ascontiguousarray(w, dtype=F32)
    if ww is not None and not bool((ww != 0).any()):
        return None
    ids = np.zeros((1, yy.size), dtype=np.uint8)
    lv, _ = product_cpp.product_renew_xgb(ids, np.ones(1, dtype=np.int32), yy, 0.0, 0.5, eta=1.0, weight=ww, nthread=threads)
    assert lv.shape == (1, 1)
    return F32(lv[0, 0])


def _cuts_and_base(p: dict[str, Any], dtrain: Any, y: np.ndarray, w: np.ndarray | None) -> tuple:
    obj = p.get("objective", "reg:squarederror")
    is_qdm = isinstance(dtrain, _core.QuantileDMatrix)
    nt = _call_threads(p)
    ip = cv = None
    if is_qdm:
        ip, cv = dtrain.get_quantile_cut()
    elif w is None:
        ip, cv = _fit.cuts(dtrain._shira_x, int(p.get("max_bin", 256)), nt)
    spw_rule = _spw(p) != 1.0 and obj in SPW_OBJ
    base = F32(p["base_score"]) if "base_score" in p else None if spw_rule else _fit.intercept(
        obj, y, w, nt, float(p.get("huber_slope", 1.0)))
    if base is None and "base_score" not in p and obj == "reg:absoluteerror":
        base = _abs_intercept(y, w, nt)
    if ip is None or base is None:
        base_str = _xgb_probe(p, dtrain)
        base = F32(base_str) if base is None else base
        if ip is None:
            ip, cv = dtrain.get_quantile_cut()
    ip, cv = np.asarray(ip, dtype=np.int64), np.asarray(cv, dtype=F32)
    assert ip.size == dtrain.num_col() + 1 and cv.size == ip[-1]
    return ip, cv, float(base)


def _prepare(p: dict[str, Any], dtrain: Any) -> tuple[SimpleNamespace | None, str | None]:
    if dtrain.num_row() == 0:
        return None, "no rows (stock decides)"
    is_qdm = isinstance(dtrain, _core.QuantileDMatrix)
    if is_qdm and int(p.get("max_bin", 256)) != int(dtrain.max_bin or 256):
        return None, "max_bin differs from the QuantileDMatrix's"
    obj = p.get("objective", "reg:squarederror")
    y = np.ascontiguousarray(dtrain.get_label(), dtype=F32)
    w = dtrain.get_weight()
    w = np.ascontiguousarray(w, dtype=F32) if w.size else None
    if w is not None and not float(w.sum(dtype=np.float64)) > 0.0:
        return None, "sample weights do not sum to a positive number (stock decides)"
    try:
        ip, cv, base = _cuts_and_base(p, dtrain, y, w)
    except _Unroutable as e:
        return None, f"dtrain: {e}"
    slots = int(ip[-1]) + int(ip.size - 1)
    depth = int(p.get("max_depth", 6))
    if p.get("grow_policy", "depthwise") != "lossguide" and depth > 10 and (
            2.0 ** (depth - 1) * slots * 32.0 > DEEP_HIST_BYTES):
        return None, (f"max_depth {depth} on {slots} histogram bins: the level histograms would pass "
                      f"{DEEP_HIST_BYTES / 2**30:.0f} GiB")
    bins_max = int(np.diff(ip).max()) if ip.size > 1 else 0
    if slots > MAX_SLOTS and bins_max > MAX_WIDE_BINS:
        return None, f"wide data with {bins_max} bins on one feature > {MAX_WIDE_BINS} (stock decides)"
    nf, C = dtrain.num_col(), int(p.get("num_class", 1))
    if nf > FAST_MAX_FEATURES and default_engine() == "fast":
        LOG.info("shira.xgboost: fast grower not used for this call: %d features > %d (exact engine)", nf,
                 FAST_MAX_FEATURES)
    obj_json, margin = _objective_and_margin(p, nf, obj, base, C)
    if F32(_fit.engine_base_margin(obj, base)) != F32(margin):
        if C != 1:
            return None, f"base_score {base!r}: engine margin != XGBoost's {margin!r}"
        p["_shira_base_margin"] = float(F32(margin))
    sm = export.ShiraModel(obj, base, margin, nf, ip, cv, [], C, obj_json)
    pre = dtrain.__dict__.get("_lz_codes") if is_qdm else None
    codes = pre if pre is not None else _fit.codes(dtrain._shira_x, ip, cv, _call_threads(p))
    run = SimpleNamespace(sm=sm, codes=codes, y=y, w=_spw_weights(p, y, w))
    return run, None


def _engine_params(p: dict[str, Any], base: float) -> dict[str, Any]:
    q = {k: v for k, v in p.items() if k in _fit.ENGINE_KEYS and k not in NO_ENGINE_KEYS}
    q["objective"], q["base_score"] = p.get("objective", "reg:squarederror"), base
    if q["objective"] == "count:poisson":
        q.setdefault("max_delta_step", 0.7)
    for k in ("max_depth", "seed", "nthread", "max_bin", "num_class", "max_leaves"):
        if k in q:
            q[k] = int(q[k])
    assert "base_score" in q and "tree_method" not in q
    return q


def _engine_es(p: dict[str, Any], evals: list, kw: dict[str, Any]) -> int:
    esr = kw.get("early_stopping_rounds") or 0
    custom_es = any(type(cb) is _cb.EarlyStopping for cb in kw.get("callbacks") or [])
    if not esr or not evals or kw.get("maximize") or custom_es:
        return 0
    if p.get("objective") == "survival:cox":
        return 0
    assert int(esr) > 0
    return int(esr)


def _eval_tuple(run: SimpleNamespace, evals: list, threads: int = 1) -> tuple:
    dm = evals[-1][0]
    w = dm.get_weight()
    cd = _fit.codes(dm._shira_x, run.sm.ip, run.sm.cuts, threads)
    return (cd, np.ascontiguousarray(dm.get_label(), dtype=F32), np.ascontiguousarray(w, F32) if w.size else None,
            [(0, cd.shape[0])])


FIRST_READ_DEFAULT_ROWS = 200_000


def _mode() -> str:
    v = os.environ.get("SHIRA_XGB_MODE", "").strip().lower()
    return v if v in ("first_read", "first_read_auto") else ("ordinary" if v else "first_read")


def first_read() -> bool:
    return _mode() != "ordinary"


def _pget(p: dict[str, Any], keys: tuple[str, ...], default: float) -> float:
    for k in keys:
        if p.get(k) is not None:
            return float(p[k])
    return float(default)


def _adaptive_engine_on(p: dict[str, Any], rows: int, features: int) -> bool:
    if str(p.get("objective", "")) not in ADAPTIVE or _mode() != "first_read":
        return False
    v = os.environ.get("SHIRA_XGB_FIRST_READ_ADAPTIVE_MIN_ROWS", "").strip()
    lo = int(v) if v.isdigit() else _machine.value("xgb_first_read_adaptive_min_rows", features)
    return rows >= lo


def _first_read_reason(p: dict[str, Any], run: SimpleNamespace, esr: int) -> str | None:
    from shira_cpp import product_cpp

    fixed = _engine_rows(p, int(run.codes.shape[1]), int(run.codes.shape[0]))
    if fixed is not None and run.codes.shape[0] <= fixed:
        return ""
    if fixed is None and run.codes.shape[0] < FIRST_READ_AUTO_MIN:
        return (f"{run.codes.shape[0]} rows < {FIRST_READ_AUTO_MIN}: the sample probe would cost more than it saves "
                "(the ordinary fit; SHIRA_XGB_FIRST_READ_ROWS=<m> fixes the sample)")
    if esr and fixed is None:
        return "early stopping with the automatic sample choice (SHIRA_XGB_MODE=first_read_auto; the fixed sample takes it)"
    if run.sm.num_class != 1:
        why = _multi_reason(p, run, fixed, esr)
        if why:
            return why
    elif run.sm.objective in ADAPTIVE:
        if fixed is None or not _adaptive_engine_on(p, int(run.codes.shape[0]), int(run.codes.shape[1])):
            return f"objective {run.sm.objective!r} (xgb_first_read_adaptive_min_rows / the fixed sample)"
        if _mono_vector(p, int(run.codes.shape[1])) is not None:
            return f"objective {run.sm.objective!r} with monotone constraints"
    elif run.sm.objective not in product_cpp.OBJ:
        return f"objective {run.sm.objective!r}"
    if _mono_vector(p, int(run.codes.shape[1])) is not None:
        if fixed is None:
            return "monotone constraints with the automatic sample choice (SHIRA_XGB_MODE=first_read_auto)"
    return None


def _multi_reason(p: dict[str, Any], run: SimpleNamespace, fixed: int | None, esr: int) -> str | None:
    if run.sm.objective not in ("multi:softprob", "multi:softmax"):
        return f"multi-output objective {run.sm.objective!r}"
    if fixed is None:
        return "a multi-class objective with the automatic sample choice (SHIRA_XGB_MODE=first_read_auto)"
    return None


def _mono_vector(p: dict[str, Any], nf: int) -> np.ndarray | None:
    mono = p.get("monotone_constraints")
    if mono in (None, "", "()"):
        return None
    vals = [v for v in (str(mono).strip("()[] ").split(",") if isinstance(mono, str) else list(mono)) if str(v).strip()]
    vec = np.zeros(nf, dtype=np.int32)
    vec[: min(nf, len(vals))] = [int(v) for v in vals[:nf]]
    assert np.all(np.abs(vec) <= 1)
    return vec if np.any(vec != 0) else None


FIRST_READ_M0 = 100_000
FIRST_READ_GAIN = 0.005
FIRST_READ_HOLD = 20
FIRST_READ_HOLD_ROWS = 100_000
first_read_log: list[dict] = []


def _holdback_loss(obj: str, margin: np.ndarray, y: np.ndarray, w: np.ndarray | None, p: dict[str, Any]) -> float:
    m, yy = margin.astype(np.float64), y.astype(np.float64)
    if obj in ("binary:logistic", "reg:logistic", "binary:logitraw"):
        e = np.logaddexp(0.0, m) - yy * m
    elif obj == "count:poisson":
        e = np.exp(m) - yy * m
    elif obj == "reg:gamma":
        e = yy * np.exp(-m) + m
    elif obj == "reg:tweedie":
        r = _pget(p, ("tweedie_variance_power",), 1.5)
        e = -yy * np.exp((1 - r) * m) / (1 - r) + np.exp((2 - r) * m) / (2 - r)
    elif obj == "reg:pseudohubererror":
        sl = _pget(p, ("huber_slope",), 1.0)
        e = sl * sl * (np.sqrt(1 + ((m - yy) / sl) ** 2) - 1)
    else:
        e = (m - yy) ** 2
    return float(np.average(e, weights=w)) if w is not None else float(e.mean())


def _first_read_values(p: dict[str, Any], run: SimpleNamespace, ids: np.ndarray, nl: np.ndarray,
                       rows: np.ndarray | None) -> np.ndarray:
    from shira_cpp import product_cpp

    obj = run.sm.objective
    mds = _pget(p, ("max_delta_step",), 0.7 if obj == "count:poisson" else 0.0)
    obj_a = _pget(p, ("huber_slope",), 1.0) if obj == "reg:pseudohubererror" else _pget(p, ("tweedie_variance_power",), 1.5)
    sel = (lambda a: a) if rows is None else (lambda a: None if a is None else np.ascontiguousarray(a[rows]))
    if run.sm.num_class > 1 and _pget(p, ("alpha", "reg_alpha"), 0.0) != 0.0:
        leaf, _ = product_cpp.product_softmax_reg(ids, nl, run.sm.num_class, sel(run.y), run.sm.base_margin,
                                                  eta=_pget(p, ("eta", "learning_rate"), 0.3),
                                                  reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
                                                  reg_alpha=_pget(p, ("alpha", "reg_alpha"), 0.0),
                                                  min_child_weight=_pget(p, ("min_child_weight",), 1.0),
                                                  max_delta_step=mds, weight=sel(run.w), nthread=_call_threads(p))
        return leaf
    if run.sm.num_class > 1:
        leaf, _ = product_cpp.product_softmax(ids, nl, run.sm.num_class, sel(run.y), run.sm.base_margin,
                                              eta=_pget(p, ("eta", "learning_rate"), 0.3),
                                              reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
                                              min_child_weight=_pget(p, ("min_child_weight",), 1.0), max_delta_step=mds,
                                              weight=sel(run.w), nthread=_call_threads(p))
        return leaf
    alpha = _pget(p, ("alpha", "reg_alpha"), 0.0)
    if alpha != 0.0:
        leaf, _ = product_cpp.product_l1(ids, nl, sel(run.y), run.sm.base_margin, obj,
                                         eta=_pget(p, ("eta", "learning_rate"), 0.3),
                                         reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0), reg_alpha=alpha,
                                         min_child_weight=_pget(p, ("min_child_weight",), 1.0), max_delta_step=mds,
                                         weight=sel(run.w), obj_a=obj_a, obj_mds=_pget(p, ("max_delta_step",), 0.7),
                                         nthread=_call_threads(p))
        return leaf
    leaf, pred = product_cpp.product(ids, nl, sel(run.y), run.sm.base_margin, obj, eta=_pget(p, ("eta", "learning_rate"), 0.3),
                                  reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
                                  min_child_weight=_pget(p, ("min_child_weight",), 1.0), max_delta_step=mds,
                                  weight=sel(run.w), obj_a=obj_a, obj_mds=_pget(p, ("max_delta_step",), 0.7),
                                  nthread=_call_threads(p))
    run.fr_pred = pred if rows is None else None
    return leaf


def _first_read_values_adaptive(p: dict[str, Any], run: SimpleNamespace, ids: np.ndarray, nl: np.ndarray,
                                trees: Any) -> np.ndarray:
    from shira_cpp import product_cpp

    T, lmax = int(len(trees)), int(nl.max())
    own = np.zeros((T, lmax), dtype=np.float32)
    for t in range(T):
        c = int(trees.count[t])
        isl = np.flatnonzero(trees.nodes[t, :c, 0] < 0)
        own[t, : isl.size] = trees.leaf[t, isl]
    obj = run.sm.objective
    alpha = 0.5 if obj == "reg:absoluteerror" else _pget(p, ("quantile_alpha",), 0.5)
    leaf, _ = product_cpp.product_renew_xgb(ids, nl, run.y, run.sm.base_margin, alpha,
                                            eta=_pget(p, ("eta", "learning_rate"), 0.3), weight=run.w, fallback=own,
                                            nthread=_call_threads(p))
    return leaf


def _first_read_values_mono(p: dict[str, Any], run: SimpleNamespace, ids: np.ndarray, nl: np.ndarray, trees: Any,
                            mono: np.ndarray) -> np.ndarray:
    from shira_cpp import product_cpp

    obj = run.sm.objective
    T, maxn = int(trees.nodes.shape[0]), int(trees.nodes.shape[1])
    tree = np.zeros((T, maxn, 4), dtype=np.int32)
    for t in range(T):
        c = int(trees.count[t])
        nd = trees.nodes[t, :c]
        isl = nd[:, 0] < 0
        assert int(isl.sum()) == int(nl[t]) and c == 2 * int(nl[t]) - 1, "a full binary tree"
        tree[t, :c, 0] = -1
        tree[t, np.flatnonzero(isl), 0] = np.arange(int(nl[t]))
        inner = np.flatnonzero(~isl)
        tree[t, inner, 1], tree[t, inner, 2] = nd[inner, 3], nd[inner, 4]
        assert np.all(tree[t, inner, 1] > inner) and np.all(tree[t, inner, 2] > inner), "parents before children"
        tree[t, inner, 3] = mono[nd[inner, 0]]
    if run.sm.num_class > 1:
        leaf, _ = product_cpp.product_softmax_reg(ids, nl, run.sm.num_class, run.y, run.sm.base_margin, tree=tree,
                                                  count=np.ascontiguousarray(trees.count, dtype=np.int32),
                                                  eta=_pget(p, ("eta", "learning_rate"), 0.3),
                                                  reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
                                                  reg_alpha=_pget(p, ("alpha", "reg_alpha"), 0.0),
                                                  min_child_weight=_pget(p, ("min_child_weight",), 1.0),
                                                  max_delta_step=_pget(p, ("max_delta_step",), 0.0), weight=run.w,
                                                  nthread=_call_threads(p))
        return leaf
    leaf, _ = product_cpp.product_bounded(ids, nl, tree, np.ascontiguousarray(trees.count, dtype=np.int32), run.y,
                                          run.sm.base_margin, obj, eta=_pget(p, ("eta", "learning_rate"), 0.3),
                                          reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
                                          min_child_weight=_pget(p, ("min_child_weight",), 1.0), weight=run.w,
                                          nthread=_call_threads(p), obj_mds=_pget(p, ("max_delta_step",), 0.7),
                                          reg_alpha=_pget(p, ("alpha", "reg_alpha"), 0.0),
                                          max_delta_step=_pget(p, ("max_delta_step",), 0.7 if obj == "count:poisson" else 0.0),
                                          obj_a=_pget(p, ("huber_slope",), 1.0) if obj == "reg:pseudohubererror"
                                          else _pget(p, ("tweedie_variance_power",), 1.5) if obj == "reg:tweedie" else 1.0)
    return leaf


CORR_SWEEPS = 10
CORR_HOLD = 10
CORR_GAIN = 1e-4
corr_log: list[dict] = []


def _corr_switch(name: str, default: int, lo: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() and int(v) >= lo else default


def _corr_reason(p: dict[str, Any], run: SimpleNamespace) -> str | None:
    from shira_cpp import product_cpp

    if run.sm.num_class != 1:
        return "a multi-output objective"
    if run.sm.objective not in product_cpp.CORR_OBJ:
        return f"objective {run.sm.objective!r} (squared error and logistic so far)"
    if _pget(p, ("alpha", "reg_alpha"), 0.0) != 0.0:
        return "reg_alpha (the L1 leaf rule)"
    if _pget(p, ("max_delta_step",), 0.0) != 0.0:
        return "max_delta_step"
    return None


def _corrections(p: dict[str, Any], run: SimpleNamespace, ids: np.ndarray, nl: np.ndarray, leaf: np.ndarray,
                 shape_rows: np.ndarray) -> np.ndarray:
    from shira_cpp import product_cpp

    sweeps = _corr_switch("SHIRA_XGB_CORR_SWEEPS", CORR_SWEEPS, 0)
    why = "SHIRA_XGB_CORR_SWEEPS=0" if sweeps == 0 else _corr_reason(p, run)
    if why:
        corr_log.append({"sweeps": 0, "reason": why})
        return leaf
    k = _corr_switch("SHIRA_XGB_CORR_HOLD", CORR_HOLD, 2)
    lv, _, log = product_cpp.corr_drive(ids, nl, run.y, run.w, run.sm.objective, _pget(p, ("lambda", "reg_lambda"), 1.0),
                                        _pget(p, ("min_child_weight",), 1.0), float(run.sm.base_margin), leaf,
                                        shape_rows, sweeps, k, nthread=_call_threads(p), gain=CORR_GAIN,
                                        margin0=getattr(run, "fr_pred", None))
    corr_log.append(log)
    LOG.info("shira.xgboost: amortised corrections: %d of at most %d sweeps kept by the held-back loss "
             "(%s rows outside the shape sample; %s); SHIRA_XGB_CORR_SWEEPS=0 turns them off", log["sweeps"], sweeps,
             log.get("held_back", 0), " -> ".join(f"{v:.6g}" for v in log.get("losses", [])))
    return leaf if lv is None else lv


FIRST_READ_AUTO_MIN = 3_000_000
FIRST_READ_SCORE = 2
FIRST_READ_RICH = 0.015


def _engine_rows(p: dict[str, Any], features: int, rows: int = 0) -> int | None:
    v = first_read_rows(features)
    if v is None or os.environ.get("SHIRA_XGB_FIRST_READ_ROWS", "").strip():
        return v
    if p.get("grow_policy", "depthwise") != "lossguide" and int(p.get("max_depth", 6)) >= 11:
        d = _machine.value("xgb_first_read_rows_deep", features)
        v = d if d > 0 else v
    return _small_rows(rows, v, features) if rows else v


def _small_rows(n: int, m: int | None, features: int) -> int | None:
    assert n >= 0 and features >= 1
    if m is None or n > m or os.environ.get("SHIRA_XGB_FIRST_READ_ROWS", "").strip():
        return m
    floor = _machine.value("xgb_small_floor_rows", features)
    if n <= floor:
        return m
    s = max(floor, n * _machine.value("xgb_small_sample_pct", features) // 100)
    assert floor <= s < n
    return s


def first_read_rows(features: int) -> int | None:
    v = os.environ.get("SHIRA_XGB_FIRST_READ_ROWS", "").strip()
    if v:
        return int(v) if v.isdigit() and int(v) >= 1000 else None
    return None if _mode() == "first_read_auto" else _machine.value("xgb_first_read_rows", features)


def _fr_es_reason(evals: list, kw: dict[str, Any]) -> str | None:
    assert evals, "early stopping needs an eval set"
    if len(evals) != 1:
        return f"early stopping with {len(evals)} eval sets (the sample fit's history is the last set's only)"
    if kw.get("callbacks"):
        return "early stopping with user callbacks (they would read the sample fit's history)"
    return None


def _leaf_pass(p: dict[str, Any], codes: np.ndarray, trees: Any, th: int) -> tuple[np.ndarray, np.ndarray]:
    from shira_cpp import product_cpp

    env = os.environ.get("SHIRA_XGB_LEAF_LANES", "").strip()
    lim = _machine.value("xgb_leaf_lanes_min_depth", int(codes.shape[1]))
    on = env == "1" or (env != "0" and lim > 0 and p.get("grow_policy", "depthwise") != "lossguide"
                        and int(p.get("max_depth", 6)) >= lim)
    if on:
        return product_cpp.leaf_ids_lanes(codes, trees.nodes, trees.count, nthread=th)
    return product_cpp.leaf_ids(codes, trees.nodes, trees.count, nthread=th)


def _first_read_fit(p: dict[str, Any], q: dict[str, Any], run: SimpleNamespace, rounds: int,
                    ev: tuple | None = None, esr: int = 0) -> Any:
    from shira_cpp import product_cpp

    n = int(run.codes.shape[0])
    step = max(FIRST_READ_HOLD, n // FIRST_READ_HOLD_ROWS)
    hold = np.arange(step - 1, n, step, dtype=np.int64)
    nr = n - hold.size
    rest_of = lambda j: j + j // (step - 1)
    codes = run.codes
    th = _call_threads(p)
    hold_codes = np.ascontiguousarray(codes[hold])
    tried: list[dict] = []
    best = None
    m = FIRST_READ_M0
    fixed = _engine_rows(p, int(run.codes.shape[1]), n)
    fr_hist = None
    assert esr == 0 or (fixed is not None and ev is not None), "early stopping is the fixed sample's"
    if fixed is not None:
        m = min(fixed, n - 1)
        rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
        ws = None if run.w is None else np.ascontiguousarray(run.w[rows])
        trees, n_rounds, fr_hist = _fit.engine_es(q, np.ascontiguousarray(run.codes[rows]), run.sm.ip,
                                                  np.ascontiguousarray(run.y[rows]), rounds, ws, ev, esr)
        assert (n_rounds == rounds or (esr and 1 <= n_rounds < rounds)) and isinstance(trees, export.LazyTrees)
        best = (0.0, m, trees)
        tried.append({"m": m, "fixed": True, "rounds": n_rounds})
    for _ in range(0 if fixed is not None else 40):
        if 2 * m >= nr:
            first_read_log.append({"n": n, "tried": tried, "kept": None})
            return None
        rows = rest_of((nr * np.arange(m, dtype=np.int64)) // m)
        ws = None if run.w is None else np.ascontiguousarray(run.w[rows])
        trees, n_rounds, _ = _fit.engine_es(q, np.ascontiguousarray(run.codes[rows]), run.sm.ip,
                                            np.ascontiguousarray(run.y[rows]), rounds, ws, None, 0)
        assert n_rounds == rounds and isinstance(trees, export.LazyTrees)
        k = min(nr, FIRST_READ_SCORE * m)
        srows = rest_of((nr * np.arange(k, dtype=np.int64)) // k)
        sids, snl = product_cpp.leaf_ids(np.ascontiguousarray(codes[srows]), trees.nodes, trees.count, nthread=th)
        leaf = _first_read_values(p, run, sids, snl, srows)
        hids, _ = product_cpp.leaf_ids(hold_codes, trees.nodes, trees.count, nthread=th)
        margin = run.sm.base_margin + leaf[np.arange(leaf.shape[0])[:, None], hids].astype(np.float64).sum(axis=0)
        loss = _holdback_loss(run.sm.objective, margin, run.y[hold], None if run.w is None else run.w[hold], p)
        tried.append({"m": m, "scored_rows": int(k), "holdback_loss": loss})
        if best is not None and len(tried) == 2 and loss < best[0] * (1.0 - FIRST_READ_RICH):
            first_read_log.append({"n": n, "tried": tried, "kept": None, "exit": "signal-rich"})
            return None
        if best is not None and not loss < best[0] * (1.0 - FIRST_READ_GAIN):
            break
        best = (loss, m, trees)
        m *= 2
    assert best is not None
    _, m_kept, trees = best
    ids, nl = _leaf_pass(p, codes, trees, th)
    mono = _mono_vector(p, int(codes.shape[1]))
    if run.sm.objective in ADAPTIVE:
        leaf = _first_read_values_adaptive(p, run, ids, nl, trees)
    else:
        leaf = _first_read_values(p, run, ids, nl, None) if mono is None else _first_read_values_mono(p, run, ids, nl, trees, mono)
        if mono is None:
            srows = ((n - 1) * np.arange(m_kept, dtype=np.int64) // max(1, m_kept - 1) if fixed is not None
                     else rest_of((nr * np.arange(m_kept, dtype=np.int64)) // m_kept))
            leaf = _corrections(p, run, ids, nl, leaf, srows)
    for t in range(len(trees)):
        c = int(trees.count[t])
        isl = np.flatnonzero(trees.nodes[t, :c, 0] < 0)
        assert isl.size == int(nl[t])
        trees.leaf[t, isl] = leaf[t, : isl.size]
    first_read_log.append({"n": n, "tried": tried, "kept": m_kept, "hist": fr_hist})
    if esr:
        LOG.info("shira.xgboost: early stopping ran on the first read's sample fit (%d stride-sample rows, %d of %d rounds, "
                 "the engine's stock-exact rule on the eval set); the history is the sample fit's (best_iteration, "
                 "best_score, evals_result); the leaf values of its trees are solved on all %d rows", m_kept,
                 len(trees) // run.sm.num_class, rounds, n)
    LOG.info("shira.xgboost: fitted the first-read way (SHIRA's default): tree shapes on %d stride-sample rows (%s), leaf "
             "values on all %d rows; accuracy depends on the data (feature importances, gain and cover come from the sample "
             "fit); SHIRA_XGB_MODE=ordinary turns it off", m_kept,
             "SHIRA_XGB_FIRST_READ_ROWS" if os.environ.get("SHIRA_XGB_FIRST_READ_ROWS") else
             "the default sample" if fixed is not None else f"chosen by the hold-back loss from {[t['m'] for t in tried]}", n)
    return trees


def _custom_engine_reason(params: Any, dtrain: Any, rounds: int, evals: list, kw: dict[str, Any]) -> str | None:
    if kw.get("obj") is None:
        return "no custom objective"
    if os.environ.get("SHIRA_XGB_CUSTOM_ENGINE", "").strip() == "0":
        return "SHIRA_XGB_CUSTOM_ENGINE=0"
    if evals or any(kw.get(k) for k in ("early_stopping_rounds", "callbacks", "maximize")) or any(
            kw.get(k) is not None for k in ("xgb_model", "custom_metric")):
        return "eval sets, early stopping, callbacks, a starting model or a custom metric with the custom objective"
    p, _, _ = normalize_params(params)
    obj = str(p.get("objective", "reg:squarederror"))
    if obj.startswith(("multi:", "rank:", "survival:")) or obj in ADAPTIVE or int(p.get("num_class", 1)) > 1:
        return f"objective {obj} with a custom objective (one output only)"
    if int(dtrain.get_label().size) != int(dtrain.num_row()):
        return "more than one target with a custom objective"
    return _call_reason(p, dtrain, rounds, [], dict(kw, obj=None))


def _custom_engine_fit(params: Any, dtrain: Any, rounds: int, obj: Any,
                       all_rows: bool = False) -> tuple[export.ShiraModel | None, str | None]:
    p, _, _ = normalize_params(params)
    p = dict(p, base_score=float(p.get("base_score", 0.5)))
    run, why = _prepare(p, dtrain)
    if why:
        return None, why
    assert run is not None and rounds >= 1
    q = _engine_params(p, run.sm.base_score)
    n, nf = int(run.codes.shape[0]), int(run.codes.shape[1])
    m = n if all_rows else _custom_sample_rows(n, nf)
    if m >= n:
        trees, n_rounds, _ = _fit.engine_es(q, run.codes, run.sm.ip, run.y, int(rounds), None, None, 0,
                                            grad_fn=lambda pr: obj(pr, dtrain))
        assert n_rounds == rounds
        run.sm.trees = trees
        LOG.info("shira.xgboost: the custom objective on SHIRA's XGBoost engine: the caller's function once a round on "
                 "every row's margins, the trees grown from its gradients (stock's own model)")
        return run.sm, None
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    x = getattr(dtrain, "_shira_x", None)
    if x is None:
        return None, "the training matrix is not a plain array (the sample's DMatrix needs its rows)"
    w = run.w
    ds = _core.DMatrix(np.ascontiguousarray(x[rows]), label=run.y[rows], weight=None if w is None else w[rows],
                       nthread=_call_threads(p))
    trees, n_rounds, _ = _fit.engine_es(q, np.ascontiguousarray(run.codes[rows]), run.sm.ip, run.y[rows], int(rounds),
                                        None, None, 0, grad_fn=lambda pr: obj(pr, ds))
    assert n_rounds == rounds
    ids, nl = _leaf_pass(p, run.codes, trees, _call_threads(p))
    leaf = _custom_given_values(p, ids, nl, dtrain, float(run.sm.base_margin), obj, _call_threads(p))
    for t in range(len(trees)):
        c = int(trees.count[t])
        isl = np.flatnonzero(trees.nodes[t, :c, 0] < 0)
        assert isl.size == int(nl[t])
        trees.leaf[t, isl] = leaf[t, : isl.size]
    run.sm.trees = trees
    LOG.info("shira.xgboost: the custom objective the first-read way on SHIRA's XGBoost engine: tree shapes from the "
             "caller's function on %d stride-sample rows, every leaf solved on all %d rows from its gradients (not stock's trees); "
             "SHIRA_XGB_CUSTOM_MODE=all grows on every row (stock's own model)", m, n)
    return run.sm, None


def _custom_given_values(p: dict[str, Any], ids: np.ndarray, nl: np.ndarray, dtrain: Any, margin: float,
                         obj: Callable[..., Any], th: int) -> np.ndarray:
    from shira_cpp import product_cpp

    T, n = int(ids.shape[0]), int(ids.shape[1])
    assert T >= 1 and n >= 1
    kw = {"eta": _pget(p, ("eta", "learning_rate"), 0.3), "reg_lambda": _pget(p, ("lambda", "reg_lambda"), 1.0),
          "reg_alpha": _pget(p, ("alpha", "reg_alpha"), 0.0), "min_child_weight": _pget(p, ("min_child_weight",), 1.0),
          "max_delta_step": _pget(p, ("max_delta_step",), 0.0), "nthread": max(1, int(th))}
    pred = np.full(n, margin, dtype=F32)
    leaf = np.zeros((T, int(nl.max())), dtype=F32)
    for t in range(T):
        g, h = obj(pred.copy(), dtrain)
        leaf[t, : int(nl[t])] = product_cpp.xgb_given(ids[t], int(nl[t]), g, h, pred, **kw)
    return leaf


def _custom_sample_rows(n: int, nf: int) -> int:
    env = os.environ.get("SHIRA_XGB_CUSTOM_MODE", "").strip()
    top = _machine.value("xgb_custom_all_rows", nf)
    if env == "all" or (env != "sample" and (top == 0 or n <= top)):
        return n
    m = first_read_rows(nf)
    return n if m is None else int(min(n, _small_rows(n, m, nf)))


def _train_shira(p: dict[str, Any], dtrain: Any, rounds: int, evals: list, kw: dict[str, Any],
                 allow_fr: bool = True) -> tuple[export.ShiraModel | None, int, str | None, np.ndarray | None]:
    run, reason = _prepare(p, dtrain)
    if reason:
        return None, 0, reason, None
    assert run is not None
    q = _engine_params(p, run.sm.base_score)
    esr = _engine_es(p, evals, kw)
    fr_why = _first_read_reason(p, run, esr) if first_read() and allow_fr else "off"
    if fr_why is None and esr:
        fr_why = _fr_es_reason(evals, kw)
    if fr_why is None:
        failed = ""
        try:
            fr = _first_read_fit(p, q, run, int(rounds), _eval_tuple(run, evals, _call_threads(p)) if esr else None, esr)
        except FIRST_READ_ERRORS as e:
            LOG.warning("shira.xgboost: the first read failed (%s: %s); the ordinary fit runs", type(e).__name__, e)
            fr, failed = None, f"the first read failed ({type(e).__name__}: {e})"
        if fr is not None:
            run.sm.trees = fr
            return run.sm, esr, None, first_read_log[-1]["hist"]
        fr_why = failed or ("signal-rich data: the first sample doubling improved the hold-back loss by more than "
                  f"{100 * FIRST_READ_RICH:.1f} %" if first_read_log and first_read_log[-1].get("exit") == "signal-rich"
                  else "the sample doubling reached the data (no shape sample short of all rows was as good)")
    if first_read() and fr_why:
        LOG.info("shira.xgboost: the first-read fit does not apply to this call, the ordinary fit runs: %s", fr_why)
    ev = _eval_tuple(run, evals, _call_threads(p)) if esr else None
    trees, n_rounds, hist = _fit.engine_es(q, run.codes, run.sm.ip, run.y, int(rounds), run.w, ev, esr)
    assert 1 <= n_rounds <= rounds and (esr or n_rounds == rounds)
    run.sm.trees = trees
    return run.sm, esr, None, hist


class Booster(_core.Booster):

    def __init__(self, params: Any = None, cache: Sequence[Any] | None = None, model_file: Any = None,
                 *, shira_model: export.ShiraModel | None = None) -> None:
        self._skip_import = shira_model is not None or isinstance(self, (_ReplayBooster, _RouteBooster))
        super().__init__(params, cache, model_file)
        self._skip_import = False
        if shira_model is not None or not hasattr(self, "shira_model"):
            self.shira_model = shira_model
        self.backend = "shira" if shira_model is not None else "xgboost"

    def set_param(self, params: Any, value: Any = None) -> None:
        if self.__dict__.get("_shira_cv_locked"):
            raise NotImplementedError("shira.xgboost.cv: a callback changed a training parameter (set_param) during "
                                      "cv; the folds are trained ahead of the loop, so only callbacks that read the "
                                      "evaluation log are taken (a LearningRateScheduler is taken as a schedule)")
        super().set_param(params, value)

    def load_model(self, fname: Any) -> None:
        super().load_model(fname)
        self.shira_model = None
        if getattr(self, "_skip_import", False):
            return
        try:
            self.shira_model = export.from_dict(json.loads(self.save_raw("json")))
        except (AssertionError, KeyError, ValueError) as e:
            LOG.info("shira.xgboost: loaded model predicts on stock XGBoost: %s", e)
            self.shira_model = None

    def _shira_rows(self, data: Any, iteration_range: Any, flags: tuple[bool, ...]) -> tuple[Any, int | None]:
        sm = getattr(self, "shira_model", None)
        if sm is None or any(flags) or not isinstance(data, (DMatrix, QuantileDMatrix)) or data._shira_x is None:
            return None, None
        if data.get_float_info("base_margin").size > 0 or data.num_col() != sm.num_feature or sm.num_class != 1:
            return None, None
        a, b = int(iteration_range[0]), int(iteration_range[1])
        if a != 0 or not 0 <= b <= len(sm.trees) or len(sm.trees) != self.num_boosted_rounds():
            return None, None
        if not isinstance(data, _core.QuantileDMatrix) and sm.below_min_split(data._shira_x):
            return None, None
        return data._shira_x, (b or None)

    def predict(self, data: Any, *, output_margin: bool = False, pred_leaf: bool = False,
                pred_contribs: bool = False, approx_contribs: bool = False, pred_interactions: bool = False,
                validate_features: bool = True, training: bool = False, iteration_range: Any = (0, 0),
                strict_shape: bool = False) -> np.ndarray:
        sm = getattr(self, "shira_model", None)
        if sm is not None and (pred_contribs or pred_interactions):
            sb = self._stats_booster()
            if sb is None:
                raise NotImplementedError("this SHIRA-trained model carries no node covers; SHAP values need them")
            return sb.predict(data, output_margin=output_margin, pred_leaf=pred_leaf, pred_contribs=pred_contribs,
                              approx_contribs=approx_contribs, pred_interactions=pred_interactions,
                              validate_features=validate_features, training=training, iteration_range=iteration_range,
                              strict_shape=strict_shape)
        flags = (pred_leaf, training, strict_shape, not (output_margin or export.can_transform(
            sm.objective if sm else "")))
        assert len(flags) == 4
        x, n_trees = self._shira_rows(data, iteration_range, flags)
        if x is None:
            return super().predict(data, output_margin=output_margin, pred_leaf=pred_leaf,
                                   pred_contribs=pred_contribs, approx_contribs=approx_contribs,
                                   pred_interactions=pred_interactions, validate_features=validate_features,
                                   training=training, iteration_range=iteration_range, strict_shape=strict_shape)
        if validate_features:
            self._validate_features(data.feature_names)
        assert sm is not None
        return sm.predict(x, output_margin=output_margin, n_trees=n_trees,
                          clamp=isinstance(data, _core.QuantileDMatrix))

    def get_score(self, fmap: Any = "", importance_type: str = "weight") -> dict:
        if getattr(self, "shira_model", None) is not None and importance_type != "weight":
            sb = self._stats_booster()
            if sb is None:
                raise NotImplementedError("this SHIRA-trained model carries no gains / covers; use importance_type='weight'")
            return sb.get_score(fmap, importance_type)
        return super().get_score(fmap, importance_type)

    def _stats_booster(self) -> Any:
        sb = getattr(self, "_shira_stats_bst", None)
        if sb is not None:
            return sb
        sm = self.shira_model
        lt = sm.trees
        if not isinstance(lt, export.LazyTrees) or lt.cover is None or lt.gain is None:
            return None
        sb = _core.Booster()
        sb.load_model(bytearray(json.dumps(export.to_dict(sm)).encode()))
        sb.feature_names, sb.feature_types = self.feature_names, self.feature_types
        LOG.info("shira.xgboost: gain / cover importances and SHAP values run on stock XGBoost over SHIRA's trees and "
                 "the engine's node statistics (a first-read model's gains and covers are its sample fit's)")
        self._shira_stats_bst = sb
        return sb


class _RouteBooster(Booster):
    pass


class _ReplayBooster(Booster):

    def __init__(self, params: Any, sm: export.ShiraModel, evals: list, start: SimpleNamespace | None = None) -> None:
        raw = export.zero_tree_json(sm.objective, sm.base_score, 1, sm.num_class, sm.objective_json)
        super().__init__(params, None, bytearray(raw))
        self._sm, self._sets, self._full = sm, [], None
        self._r0 = 0 if start is None else int(start.r0)
        self._nt = int(normalize_params(params)[0].get("nthread", 0)) or None
        C = sm.num_class
        for dm, _ in evals:
            n = dm.num_row()
            qdm = isinstance(dm, _core.QuantileDMatrix)
            codes = sm.codes(dm._shira_x, clamp=qdm)
            w = dm.get_weight()
            stock = None
            if not qdm and sm.below_min_split(dm._shira_x):
                stock = _core.DMatrix(dm._shira_x, base_margin=np.zeros((n, C) if C > 1 else n, F32))
                if self._full is None:
                    self._full = _core.Booster(model_file=bytearray(export.to_json(sm)))
            rm = None
            m0 = None if start is None else start.evals[len(self._sets)]
            if stock is None and C == 1 and isinstance(sm.trees, export.LazyTrees) and m0 is None:
                rm = export.round_margins(sm, codes, len(sm.trees))
            self._sets.append(SimpleNamespace(codes=codes, y=dm.get_label(), w=w if w.size else None,
                                              zeros=np.zeros((n, 1), F32), stock=stock, rm=rm,
                                              m=np.full((n, C), F32(sm.base_margin), dtype=F32) if m0 is None
                                              else np.array(m0, dtype=F32).reshape(n, C)))

    def num_boosted_rounds(self) -> int:
        return self._r0

    def advance(self, i: int) -> None:
        C = self._sm.num_class
        for st in self._sets:
            if st.stock is not None:
                assert self._full is not None
                leaves = self._full.predict(st.stock, output_margin=True, iteration_range=(i, i + 1))
                st.m = (st.m + np.asarray(leaves, F32).reshape(st.m.shape)).astype(F32)
                continue
            if st.rm is not None:
                st.m[:, 0] = st.rm[i]
                continue
            for c in range(C):
                leaf = _trees.predict_margin([self._sm.trees[i * C + c]], st.codes, 0.0)
                st.m[:, c] = (st.m[:, c] + leaf).astype(F32)

    def eval_set(self, evals: Any, iteration: int = 0, feval: Any = None, output_margin: bool = True) -> str:
        assert feval is None and len(evals) == len(self._sets)
        mats = []
        for st, (_, name) in zip(self._sets, evals):
            bm = st.m[:, 0] if st.m.shape[1] == 1 else st.m
            mats.append((_core.DMatrix(st.zeros, label=st.y, weight=st.w, base_margin=bm, nthread=self._nt), name))
        return super().eval_set(mats, iteration, None, output_margin)


_METRIC_NAMES: dict[str, str] = {}


def _metric_name(params: Any, sm: export.ShiraModel) -> str:
    key = json.dumps([sm.objective, sm.objective_json], sort_keys=True, default=str)
    if key not in _METRIC_NAMES:
        raw = export.zero_tree_json(sm.objective, sm.base_score, 1, sm.num_class, sm.objective_json)
        b = _core.Booster(params, None, bytearray(raw))
        bm = np.zeros((1, sm.num_class) if sm.num_class > 1 else 1, F32)
        one = _core.DMatrix(np.zeros((1, 1), F32), label=np.ones(1, F32), base_margin=bm)
        s = b.eval_set([(one, "d")], 0)
        assert s.startswith("[0]\td-") and s.count(":") == 1, s
        _METRIC_NAMES[key] = s[len("[0]\td-"):s.index(":")]
    return _METRIC_NAMES[key]


class _HistBooster(_core.Booster):

    def __init__(self, params: Any, sm: export.ShiraModel, evals: list, hist: np.ndarray) -> None:
        super().__init__()
        assert len(evals) == 1 and hist.ndim == 1 and hist.size * sm.num_class == len(sm.trees)
        self._hist, self._metric = hist, _metric_name(params, sm)

    def num_boosted_rounds(self) -> int:
        return 0

    def advance(self, i: int) -> None:
        assert 0 <= i < self._hist.size

    def eval_set(self, evals: Any, iteration: int = 0, feval: Any = None, output_margin: bool = True) -> str:
        assert feval is None and len(evals) == 1 and 0 <= iteration < self._hist.size
        return f"[{iteration}]\t{evals[0][1]}-{self._metric}:{float(self._hist[iteration]):.17f}"


def _engine_eval_exact(sm: export.ShiraModel, dm: Any) -> bool:
    if isinstance(dm, _core.QuantileDMatrix):
        return True
    x = dm._shira_x
    assert x is not None and x.shape[1] == sm.num_feature
    f, cut = sm.internal()
    last = np.unique(f[cut == sm.ip[f + 1] - sm.ip[f] - 1]).tolist()
    first = np.unique(f[cut == 0]).tolist()
    above = any(bool(np.any(x[:, f] >= sm.cuts[sm.ip[f + 1] - 1])) for f in last)
    below = any(bool(np.any(x[:, f] < sm.cuts[sm.ip[f]])) for f in first)
    return not (above or below)


def _stops_at(params: Any, sm: export.ShiraModel, dtrain: Any, evals: list, esr: int, maximize: Any) -> int:
    box = _cb.CallbackContainer([_cb.EarlyStopping(rounds=esr, maximize=maximize)], metric=None,
                                output_margin=False)
    rep = box.before_training(_ReplayBooster(params, sm, evals))
    rounds = len(sm.trees) // sm.num_class
    assert rounds >= 1 and esr >= 1
    for i in range(rounds):
        rep.advance(i)
        if box.after_iteration(rep, i, dtrain, evals):
            return i + 1
    return -1


def _replay(params: Any, sm: export.ShiraModel, dtrain: Any, evals: list, opts: dict[str, Any],
            hist: np.ndarray | None = None, start: SimpleNamespace | None = None) -> Booster:
    assert start is None or hist is None
    callbacks = [] if opts.get("callbacks") is None else copy.copy(list(opts["callbacks"]))
    verbose = opts.get("verbose_eval", True)
    if verbose:
        callbacks.append(_cb.EvaluationMonitor(period=1 if verbose is True else verbose))
    if opts.get("early_stopping_rounds"):
        callbacks.append(_cb.EarlyStopping(rounds=opts["early_stopping_rounds"], maximize=opts.get("maximize")))
    box = _cb.CallbackContainer(callbacks, metric=None, output_margin=False)
    first = _HistBooster(params, sm, evals, hist) if hist is not None else _ReplayBooster(params, sm, evals, start)
    rep = box.before_training(first)
    rounds = len(sm.trees) // sm.num_class
    assert isinstance(rep, (_ReplayBooster, _HistBooster)) and rounds * sm.num_class == len(sm.trees)
    n_keep = rounds
    for i in range(rounds):
        if box.before_iteration(rep, i, dtrain, evals):
            n_keep = i
            break
        if evals:
            rep.advance(i)
        if box.after_iteration(rep, i, dtrain, evals):
            n_keep = i + 1
            break
    old = None if start is None else start.old
    final = _final_booster(params, sm.head(n_keep), dtrain, evals, old)
    attrs = {k: v for k, v in rep.attributes().items() if v is not None}
    if attrs:
        final.set_attr(**attrs)
    final = box.after_training(final)
    if opts.get("evals_result") is not None:
        opts["evals_result"].update(box.history)
    assert final.num_boosted_rounds() <= n_keep + (0 if old is None else old.num_boosted_rounds())
    return final.reset()


def _continued_json(old: Any, sm: export.ShiraModel) -> bytes:
    a = json.loads(old.save_raw("json"))
    b = export.to_dict(sm)
    ga, gb = a["learner"]["gradient_booster"]["model"], b["learner"]["gradient_booster"]["model"]
    k0 = len(ga["trees"])
    assert a["learner"]["gradient_booster"]["name"] == "gbtree" and int(ga["gbtree_model_param"]["num_trees"]) == k0
    for t in gb["trees"]:
        ga["trees"].append(dict(t, id=k0 + int(t["id"])))
    ga["tree_info"] = list(ga["tree_info"]) + list(gb["tree_info"])
    ip = [int(v) for v in ga["iteration_indptr"]]
    ga["iteration_indptr"] = ip + [ip[-1] + int(v) for v in gb["iteration_indptr"][1:]]
    ga["gbtree_model_param"]["num_trees"] = str(len(ga["trees"]))
    assert ga["iteration_indptr"][-1] == len(ga["trees"]) == len(ga["tree_info"])
    return json.dumps(a, separators=(",", ":")).encode()


def _final_booster(params: Any, sm: export.ShiraModel, dtrain: Any, evals: list, old: Any = None) -> Booster:
    if old is not None:
        bst = Booster(params, [dtrain] + [d for d, _ in evals], bytearray(_continued_json(old, sm)))
        assert bst.num_boosted_rounds() == old.num_boosted_rounds() + len(sm.trees) // sm.num_class
        return bst
    raw = export.to_ubj(sm)
    raw = export.to_json(sm) if raw is None else raw
    cache = [dtrain] + [d for d, _ in evals]
    if not any(isinstance(d, QuantileDMatrix) and d._lazy() for d in cache):
        bst = Booster(params, cache, bytearray(raw), shira_model=sm)
    else:
        bst = Booster(params, [], bytearray(raw), shira_model=sm)
        for d in cache:
            bst._assign_dmatrix_features(d)
    names, types = getattr(dtrain, "feature_names", None), getattr(dtrain, "feature_types", None)
    if names is not None and bst.feature_names is None:
        bst.feature_names, bst.feature_types = names, types
    assert bst.num_boosted_rounds() * sm.num_class == len(sm.trees)
    return bst


def _fallback(reason: str, args: tuple, kwargs: dict[str, Any]) -> Any:
    global last_backend
    LOG.warning("shira.xgboost: running stock XGBoost: %s", reason)
    last_backend = "xgboost"
    backend_history.append(last_backend)
    return _STOCK["train"](*args, **kwargs)


def _old_booster(xgb_model: Any) -> Any:
    if xgb_model is None:
        return None
    if isinstance(xgb_model, _core.Booster):
        return xgb_model
    return _core.Booster(model_file=xgb_model)


def _start_margins(dm: Any, old: Any, C: int) -> np.ndarray | None:
    if old is not None:
        m = np.asarray(old.predict(dm, output_margin=True), dtype=F32)
    else:
        m = np.asarray(dm.get_float_info("base_margin"), dtype=F32)
        if m.size == 0:
            return None
    assert m.size == dm.num_row() * C
    return np.ascontiguousarray(m.reshape(dm.num_row(), C))


def _start_reason(params: Any, dtrain: Any, rounds: int, ev: list, kw: dict[str, Any]) -> str | None:
    has_bm = isinstance(dtrain, _core.DMatrix) and dtrain.get_float_info("base_margin").size > 0
    if kw.get("xgb_model") is None and not has_bm:
        return "no per-row start margin"
    if kw.get("obj") is not None or kw.get("custom_metric") is not None:
        return "a custom objective or metric with a per-row start margin"
    p = normalize_params(params)[0]
    reason = why_not(params)
    if reason:
        return reason
    obj = str(p.get("objective", "reg:squarederror"))
    if obj.startswith("rank:") or obj in ADAPTIVE or p.get("multi_strategy", "one_output_per_tree") != "one_output_per_tree":
        return f"objective {obj} / multi_strategy with a per-row start margin"
    for cb in kw.get("callbacks") or []:
        if type(cb) not in (_cb.EarlyStopping, _cb.EvaluationMonitor):
            return f"callback {type(cb).__name__}"
    if not 1 <= int(rounds) <= MAX_ROUNDS:
        return f"num_boost_round {rounds}"
    reason = _data_reason(dtrain, "dtrain", margin_ok=True)
    for dm, name in ev:
        reason = reason or _data_reason(dm, f"eval set {name!r}", margin_ok=True)
        if not reason and (dm.num_col() != dtrain.num_col() or isinstance(dm, _core.QuantileDMatrix)):
            reason = f"eval set {name!r}: another feature count, or a QuantileDMatrix"
    if reason or kw.get("xgb_model") is None:
        return reason
    old = _old_booster(kw["xgb_model"])
    cfg = json.loads(old.save_config())["learner"]
    if cfg["gradient_booster"]["name"] != "gbtree" or int(cfg["learner_model_param"]["num_class"]) > 1:
        return "xgb_model is not a single-output gbtree model"
    if str(p.get("objective", cfg["objective"]["name"])) != cfg["objective"]["name"]:
        return "xgb_model's objective differs from the call's"
    if int(cfg["learner_model_param"]["num_feature"]) != dtrain.num_col():
        return "xgb_model's feature count differs"
    return None


def _start_fit(params: Any, dtrain: Any, rounds: int, ev: list, kw: dict[str, Any]) -> tuple[Any, str | None]:
    p = dict(normalize_params(params)[0])
    old = _old_booster(kw.get("xgb_model"))
    if old is not None:
        cfg = json.loads(old.save_config())["learner"]
        p.setdefault("objective", cfg["objective"]["name"])
        bsv = json.loads(cfg["learner_model_param"]["base_score"])
        p["base_score"] = float(bsv[0] if isinstance(bsv, list) else bsv)
    C = int(p.get("num_class", 1))
    m0 = _start_margins(dtrain, old, C)
    assert m0 is not None
    p["_shira_start_margin"] = m0
    sm, _, reason, _ = _train_shira(p, dtrain, int(rounds), [], {}, allow_fr=False)
    if reason or sm is None:
        return None, reason
    start = SimpleNamespace(old=old, r0=0 if old is None else old.num_boosted_rounds(),
                            evals=[_start_margins(dm, old, C) for dm, _ in ev])
    LOG.info("shira.xgboost: per-row start margins (%s) on SHIRA's exact engine (stock's own model)",
             "a continued model" if old is not None else "base_margin")
    if ev or kw.get("early_stopping_rounds") or kw.get("callbacks") or old is not None:
        return _replay(params, sm, dtrain, ev, kw, None, start), None
    return _final_booster(params, sm, dtrain, ev).reset(), None


def _shira_model(params: Any, dtrain: Any, rounds: int, ev: list,
                 kw: dict[str, Any]) -> tuple[export.ShiraModel | None, str | None]:
    p, metrics, _ = normalize_params(params)
    if metrics:
        kw = dict(kw, early_stopping_rounds=None)
    n_log = len(first_read_log)
    sm, esr, reason, hist = _train_shira(p, dtrain, rounds, ev, kw)
    if reason or sm is None:
        return None, reason, None
    fr_es = bool(esr) and len(first_read_log) > n_log and first_read_log[-1].get("kept") is not None
    if fr_es and not _engine_eval_exact(sm, ev[-1][0]):
        LOG.info("shira.xgboost: the first-read fit does not apply to this call, the ordinary fit runs: early stopping "
                 "where the engine's eval binning does not route every eval row as XGBoost does (a value above a "
                 "feature's last cut or below its first), so the sample fit's history would not be stock's")
        sm, esr, reason, hist = _train_shira(p, dtrain, rounds, ev, kw, allow_fr=False)
        fr_es = False
    trained = len(sm.trees) // sm.num_class
    agree = _engine_eval_exact(sm, ev[-1][0]) if esr else True
    if (esr and not fr_es and trained < rounds and not agree
            and _stops_at(params, sm, dtrain, ev, esr, kw.get("maximize")) != trained):
        LOG.info("shira.xgboost: engine early stop at %d disagrees with the replay; training all rounds", trained)
        sm, _, reason, hist = _train_shira(p, dtrain, rounds, ev, dict(kw, early_stopping_rounds=None))
    assert reason is not None or sm is not None
    fast = (hist is not None and agree and len(ev) == 1 and (sm.num_class == 1 or fr_es) and not kw.get("callbacks")
            and kw.get("custom_metric") is None and kw.get("obj") is None and not kw.get("maximize"))
    return sm, reason, (hist if fast else None)


def _stock_intercept(p: dict[str, Any], y: np.ndarray, w: np.ndarray | None, th: int) -> np.ndarray:
    C = int(p["num_class"])
    dm = _core.DMatrix(np.zeros((y.size, 1), dtype=F32), label=y, weight=w, nthread=th)
    b = _STOCK["train"]({"objective": p.get("objective"), "num_class": C, "max_depth": 1, "nthread": th}, dm, 1)
    v = np.array(json.loads(json.loads(b.save_config())["learner"]["learner_model_param"]["base_score"]), dtype=F32)
    assert v.shape == (C,) and np.all(np.isfinite(v))
    return v


def _intercept_reason(p: dict[str, Any], dtrain: Any, kw: dict[str, Any]) -> str | None:
    if not first_read():
        return "SHIRA_XGB_MODE=ordinary"
    if any(kw.get(k) for k in ("callbacks", "obj", "custom_metric", "xgb_model")):
        return "callbacks or a custom objective / metric with the per-class intercept"
    if kw.get("evals") and any(getattr(d, "_shira_x", None) is None for d, _ in kw["evals"]):
        return "an eval set that is not a plain numpy array, with the per-class intercept"
    if getattr(dtrain, "_shira_x", None) is None or not isinstance(dtrain._shira_x, np.ndarray):
        return "the training matrix is not a plain numpy array"
    if _is_cat(dtrain):
        return "categorical features (the general route takes them)"
    v = os.environ.get("SHIRA_XGB_MULTI_INTERCEPT_MIN_ROWS", "")
    n, f = int(dtrain._shira_x.shape[0]), int(dtrain._shira_x.shape[1])
    if n < (int(v) if v.isdigit() else _machine.value("xgb_route_multi_intercept_min_rows", f)):
        return ("the per-class intercept route is not measured faster than stock on this machine at this size (switch "
                "xgb_route_multi_intercept_min_rows; SHIRA_XGB_MULTI_INTERCEPT_MIN_ROWS=0 forces it)")
    return None


def _multi_intercept_fit(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any]) -> tuple[Any, str | None]:
    from shira_cpp import product_cpp

    p, _, _ = normalize_params(params)
    why = _intercept_reason(p, dtrain, kw)
    if why:
        return None, why
    x = np.ascontiguousarray(dtrain._shira_x, dtype=F32)
    y = np.ascontiguousarray(dtrain.get_label(), dtype=F32)
    w0 = dtrain.get_weight()
    w = np.ascontiguousarray(w0, dtype=F32) if w0.size else None
    n, nf = x.shape
    m = first_read_rows(nf)
    if m is None or n <= m:
        return None, f"{n} rows <= the shape sample ({m}): the sample would be every row"
    th = _call_threads(p)
    base = _stock_intercept(p, y, w, th)
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    ds = _core.DMatrix(np.ascontiguousarray(x[rows]), label=y[rows], weight=None if w is None else w[rows], nthread=th)
    evs = [(_core.DMatrix(np.ascontiguousarray(d._shira_x, dtype=F32), label=d.get_label(),
                          weight=d.get_weight() if d.get_weight().size else None, nthread=th), nm)
           for d, nm in (kw.get("evals") or [])]
    hist: dict = {}
    sp = dict(params if isinstance(params, dict) else dict(params), base_score=base.tolist())
    cth = _route_fit_threads(p, nf, int(rows.size), bool(evs))
    if cth:
        sp["nthread"] = cth
    bst = _STOCK["train"](sp, ds, rounds,
                          evals=evs, early_stopping_rounds=kw.get("early_stopping_rounds"), evals_result=hist,
                          verbose_eval=kw.get("verbose_eval", True) if evs else False, maximize=kw.get("maximize"))
    attrs = {k: v for k, v in bst.attributes().items() if v is not None}
    if kw.get("evals_result") is not None:
        kw["evals_result"].update(hist)
    doc = json.loads(bytes(bst.save_raw("json")))
    trees = doc["learner"]["gradient_booster"]["model"]["trees"]
    C, T = int(p["num_class"]), len(trees)
    assert T % C == 0 and 1 <= T // C <= rounds
    leaf_nodes = [np.flatnonzero(np.asarray(t["left_children"]) == -1) for t in trees]
    nl = np.array([ln.size for ln in leaf_nodes], dtype=np.int32)
    pos = [np.full(len(t["left_children"]), -1, dtype=np.int64) for t in trees]
    for t in range(T):
        pos[t][leaf_nodes[t]] = np.arange(nl[t])
    node = np.asarray(bst.predict(_core.DMatrix(x, nthread=th), pred_leaf=True), dtype=np.int64).reshape(n, T)
    ids = np.empty((T, n), dtype=np.uint16 if int(nl.max()) > 255 else np.uint8)
    for t in range(T):
        ids[t] = pos[t][node[:, t]]
    assert int(ids.max()) < int(nl.max())
    mono = _mono_vector(p, nf)
    tab, cnt = None, None
    if mono is not None:
        maxn = max(len(t["left_children"]) for t in trees)
        tab = np.zeros((T, maxn, 4), dtype=np.int32)
        cnt = np.array([len(t["left_children"]) for t in trees], dtype=np.int32)
        for t in range(T):
            lc, rc = np.asarray(trees[t]["left_children"]), np.asarray(trees[t]["right_children"])
            inner = np.flatnonzero(lc != -1)
            tab[t, :cnt[t], 0] = pos[t]
            tab[t, inner, 1], tab[t, inner, 2] = lc[inner], rc[inner]
            tab[t, inner, 3] = mono[np.asarray(trees[t]["split_indices"])[inner]]
            assert cnt[t] == 2 * nl[t] - 1 and np.all(tab[t, inner, 1] > inner) and np.all(tab[t, inner, 2] > inner)
    leaf, _ = product_cpp.product_softmax_reg(ids, nl, C, y, base, tree=tab, count=cnt,
                                              eta=_pget(p, ("eta", "learning_rate"), 0.3),
                                              reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
                                              reg_alpha=_pget(p, ("alpha", "reg_alpha"), 0.0),
                                              min_child_weight=_pget(p, ("min_child_weight",), 1.0),
                                              max_delta_step=_pget(p, ("max_delta_step",), 0.0), weight=w, nthread=th)
    for t in range(T):
        sc = trees[t]["split_conditions"]
        bw = trees[t]["base_weights"]
        for j, nid in enumerate(leaf_nodes[t].tolist()):
            sc[nid] = bw[nid] = float(leaf[t, j])
    raw = bytearray(json.dumps(doc).encode())
    out = Booster(params, [dtrain], raw)
    _route_names(out, dtrain)
    if attrs:
        out.set_attr(**attrs)
    LOG.info("shira.xgboost: fitted the first-read way (SHIRA's default): multi-class without base_score (XGBoost 3's "
             "per-class intercept, stock's own for all %d rows); tree shapes by stock XGBoost on %d stride-sample rows, "
             "leaf values on all %d rows by the softmax product; prediction is stock XGBoost's; SHIRA_XGB_MODE=ordinary "
             "runs stock", n, m, n)
    return out, None


SHAPE_ONLY_METHODS = ("approx", "exact")
DART_KEYS = ("rate_drop", "skip_drop", "one_drop", "sample_type", "normalize_type")
ADAPTIVE = ("reg:absoluteerror", "reg:quantileerror")
REFRESH_OBJ = ("rank:ndcg", "rank:pairwise", "rank:map", "reg:squaredlogerror", "binary:hinge")
RANK_KEYS = ("lambdarank_pair_method", "lambdarank_num_pair_per_sample", "lambdarank_unbiased", "lambdarank_bias_norm",
             "ndcg_exp_gain")


def _shape_free(p: dict[str, Any]) -> dict[str, Any]:
    q = dict(p)
    if q.get("tree_method") in SHAPE_ONLY_METHODS:
        q.pop("tree_method")
    lossguide = q.get("grow_policy", "depthwise") == "lossguide"
    depth = int(q.get("max_depth", 6))
    if not lossguide and (depth == 0 or depth > 12 or int(q.get("max_leaves", 0)) != 0):
        q.update(max_depth=6, max_leaves=0)
    if lossguide and (depth > 12 or (depth == 0 and int(q.get("max_leaves", 0)) == 0)):
        q.update(max_depth=0, max_leaves=31)
    if float(q.get("subsample", 1.0)) < 1.0 and "nthread" not in q:
        q["nthread"] = 1
    if int(q.get("num_parallel_tree", 1)) > 1:
        q["num_parallel_tree"] = 1
    if q.get("booster") == "dart":
        q = {k: v for k, v in q.items() if k not in DART_KEYS and k != "booster"}
    assert len(q) <= len(p) + 2
    return q


def _is_cat(dm: Any) -> bool:
    ft = getattr(dm, "feature_types", None)
    return ft is not None and any(t == "c" for t in ft)


_SRC_KEYS = ("weight", "missing", "nthread", "feature_names", "feature_types", "enable_categorical")


def _cat_matrix(dm: Any) -> Any:
    if not isinstance(dm, _core.QuantileDMatrix):
        return dm
    src = dm.__dict__.get("_sx_src")
    if src is None:
        return None
    data, label, kw = src
    return _core.DMatrix(data, label, **{k: kw[k] for k in _SRC_KEYS if kw.get(k) is not None})


def _data_reason_cat(dm: Any, name: str) -> str | None:
    if not isinstance(dm, (DMatrix, QuantileDMatrix)):
        return f"{name}: categorical features in a matrix other than shira.xgboost.DMatrix / QuantileDMatrix"
    if isinstance(dm, _core.QuantileDMatrix) and (dm.__dict__.get("_sx_src") is None
                                                  or any(dm.__dict__["_sx_src"][2].get(k) is not None
                                                         for k in ("base_margin", "group", "qid", "label_lower_bound",
                                                                   "feature_weights"))):
        return f"{name}: a categorical QuantileDMatrix whose input SHIRA did not keep"
    for key, what in (("base_margin", "base_margin"), ("label_lower_bound", "label bounds"),
                      ("feature_weights", "feature_weights")):
        if dm.get_float_info(key).size > 0:
            return f"{name}: {what}"
    if dm.get_uint_info("group_ptr").size > 0:
        return f"{name}: query groups"
    y, w = dm.get_label(), dm.get_weight()
    if y.size != dm.num_row() or not np.isfinite(y).all():
        return f"{name}: labels missing, non-finite or multi-target"
    if w.size and (w.size != dm.num_row() or not np.isfinite(w).all()):
        return f"{name}: weights not one finite value per row"
    return None


def _cat_call_reason(q: dict[str, Any], dtrain: Any, rounds: int, ev: list, kw: dict[str, Any]) -> str | None:
    reason = why_not(q)
    if reason and reason != MULTI_NO_BASE:
        return reason
    for key in ("obj", "custom_metric", "xgb_model"):
        if kw.get(key) is not None:
            return f"{key} is set"
    if not 1 <= int(rounds) <= MAX_ROUNDS:
        return f"num_boost_round {rounds}"
    for dm, name in [(dtrain, "dtrain")] + list(ev):
        r = _data_reason_cat(dm, name if dm is dtrain else f"eval set {name!r}")
        if r or dm.num_col() != dtrain.num_col():
            return r or f"eval set {name!r} feature count differs"
    return reason


FIRST_READ_ERRORS = (AssertionError, ValueError, RuntimeError, AttributeError, TypeError, ArithmeticError)


def _route_names(out: Any, dtrain: Any) -> Any:
    names, types = getattr(dtrain, "feature_names", None), getattr(dtrain, "feature_types", None)
    if names is not None and out.feature_names is None:
        out.feature_names, out.feature_types = names, types
    return out


MOT_OBJ = ("reg:squarederror", "binary:logistic", "reg:logistic")


def _mot_reason(params: Any, dtrain: Any, rounds: int, ev: list, kw: dict[str, Any]) -> str | None:
    p = dict(params) if isinstance(params, dict) else dict(params or {})
    if p.get("multi_strategy") != "multi_output_tree" or not first_read():
        return "not multi_output_tree (or SHIRA_XGB_MODE=ordinary)"
    if str(p.get("objective") or "reg:squarederror") not in MOT_OBJ or str(p.get("booster") or "gbtree") != "gbtree":
        return f"multi_output_tree with objective {p.get('objective')!r} or booster {p.get('booster')!r} (the route takes {', '.join(MOT_OBJ)})"
    if ev or any(kw.get(k) for k in ("obj", "custom_metric", "xgb_model", "callbacks", "early_stopping_rounds")):
        return "multi_output_tree with evaluation sets, callbacks, a custom objective / metric or a starting model"
    if p.get("num_parallel_tree") not in (None, 1, 1.0, "1") or float(p.get("subsample") or 1.0) < 1.0:
        return "multi_output_tree with num_parallel_tree or subsample"
    if not isinstance(getattr(dtrain, "_shira_x", None), np.ndarray) or _is_cat(dtrain):
        return "multi_output_tree on a matrix that is not a plain numpy array"
    n = int(dtrain.num_row())
    if not 1 <= int(rounds) <= MAX_ROUNDS or dtrain.get_label().size < 2 * n or dtrain.get_label().size % n:
        return "multi_output_tree with one target or a bad round count"
    return None


def _mot_fit(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any]) -> tuple:
    why = _mot_reason(params, dtrain, rounds, [], kw)
    if why:
        return None, why
    p = dict(params)
    x = dtrain._shira_x
    n, nf = int(x.shape[0]), int(x.shape[1])
    Y = np.ascontiguousarray(dtrain.get_label(), dtype=np.float64).reshape(n, -1)
    K = int(Y.shape[1])
    w0 = dtrain.get_weight()
    w = np.asarray(w0, dtype=np.float64) if w0.size else None
    th = _call_threads(normalize_params(p)[0])
    m = min(n, int(_small_rows(n, first_read_rows(nf), nf)))
    own = _machine.value("xgb_mot_rows", nf)
    if own > 0 and not os.environ.get("SHIRA_XGB_FIRST_READ_ROWS", "").strip():
        m = min(m, max(own, min(n, _machine.value("xgb_small_floor_rows", nf))))
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    if p.get("base_score") is not None:
        base = np.full(K, float(p["base_score"]), dtype=np.float64)
    else:
        dz = _core.DMatrix(np.zeros((n, 1), dtype=F32), label=Y.astype(F32), weight=w, nthread=th)
        bz = _STOCK["train"]({"objective": p.get("objective") or "reg:squarederror", "multi_strategy": "multi_output_tree",
                              "tree_method": "hist", "max_depth": 1, "nthread": th,
                              "scale_pos_weight": _spw(p)}, dz, 1)
        base = np.asarray(json.loads(json.loads(bz.save_config())["learner"]["learner_model_param"]["base_score"]),
                          dtype=np.float64).reshape(-1)
    assert base.size == K
    ds = _core.DMatrix(np.ascontiguousarray(x[rows]), label=Y[rows].astype(F32), weight=None if w is None else w[rows], nthread=th)
    cth = _route_fit_threads(p, nf, int(rows.size), False)
    bst = _STOCK["train"](dict(p, base_score=base.tolist(), **({"nthread": cth} if cth else {})), ds, int(rounds),
                          verbose_eval=False)
    doc = json.loads(bytes(bst.save_raw("json")))
    trees = doc["learner"]["gradient_booster"]["model"]["trees"]
    T = len(trees)
    logistic = str(p.get("objective") or "reg:squarederror") != "reg:squarederror"
    eta = _pget(p, ("eta", "learning_rate"), 0.3)
    lam, alpha = _pget(p, ("lambda", "reg_lambda"), 1.0), _pget(p, ("alpha", "reg_alpha"), 0.0)
    mds = _pget(p, ("max_delta_step",), 0.0)
    mcw = _pget(p, ("min_child_weight",), 1.0)
    bm = np.log(base / (1.0 - base)) if logistic else base.copy()
    from shira_cpp import product_cpp

    leaf_nodes, nl, _, _, _ = _json_tables(trees, None)
    ids = _route_leaf_ids(trees, leaf_nodes, nl, np.ascontiguousarray(x, dtype=F32), th)
    vals, _ = product_cpp.xgb_mot(ids, nl, Y, bm, logistic, weight=None if w is None else w.astype(F32), eta=eta,
                                  reg_lambda=lam, reg_alpha=alpha, min_child_weight=mcw, max_delta_step=mds, nthread=th,
                                  scale_pos_weight=_spw(p))
    for t in range(T):
        tr = trees[t]
        rc = np.asarray(tr["right_children"], dtype=np.int64)
        lw, bw = tr["leaf_weights"], tr["base_weights"]
        for j, nid in enumerate(leaf_nodes[t].tolist()):
            o = int(rc[nid])
            for k in range(K):
                lw[o * K + k] = bw[nid * K + k] = float(vals[t, j, k])
    out = _RouteBooster(params, [dtrain], bytearray(json.dumps(doc).encode()))
    _route_names(out, dtrain)
    first_read_log.append({"n": n, "m": m, "kept": m, "route": "multi_output_tree", "hist": None})
    LOG.info("shira.xgboost: fitted the first-read way (SHIRA's default) by the vector-leaf route: multi_output_tree, %d "
             "targets; tree shapes by stock XGBoost with the caller's settings on %d stride-sample rows, every leaf's %d values "
             "on all %d rows (not stock's trees); prediction is stock XGBoost's; SHIRA_XGB_MODE=ordinary runs stock", K, m, K, n)
    return out, None


def _guarded_route(fn: Callable[..., tuple], *args: Any) -> tuple:
    try:
        return fn(*args)
    except FIRST_READ_ERRORS as e:
        LOG.warning("shira.xgboost: the route failed (%s: %s); the call falls back", type(e).__name__, e)
        return None, f"the route failed ({type(e).__name__}: {e})"


def _sample_route_reason(params: Any, dtrain: Any, rounds: int, ev: list, kw: dict[str, Any]) -> str | None:
    from shira_cpp import product_cpp

    p, metrics, bad = normalize_params(params)
    if bad or not first_read():
        return bad or "SHIRA_XGB_MODE=ordinary"
    cat = _is_cat(dtrain)
    q = _shape_free(p)
    custom = kw.get("obj") is not None
    if custom:
        if (cat or int(p.get("num_parallel_tree", 1)) > 1 or str(p.get("booster", "gbtree")) != "gbtree"
                or _mono_vector(p, int(dtrain.num_col())) is not None or str(p.get("objective", "")).startswith("multi:")):
            return ("a custom objective with categorical features, num_parallel_tree > 1, a booster other than gbtree, "
                    "monotone constraints or a multi-class objective")
        kw = {**kw, "obj": None}
    refresh = q.get("objective") in REFRESH_OBJ or custom
    if refresh:
        q = {k: v for k, v in q.items() if k not in RANK_KEYS}
        q["objective"] = "reg:squarederror"
        if int(p.get("num_parallel_tree", 1)) > 1 or str(p.get("booster", "gbtree")) != "gbtree":
            return "rank / squaredlogerror / hinge with num_parallel_tree > 1 or a booster other than gbtree"
        if cat:
            return "rank / squaredlogerror / hinge with categorical features"
    r2 = (_cat_call_reason if cat else _call_reason)(q, dtrain, rounds, ev, kw)
    rank = str(p.get("objective", "")).startswith("rank:")
    if rank and r2 is not None and r2.endswith(": query groups"):
        r2 = _rank_groups_reason(dtrain, ev)
    if rank and r2 is None and dtrain.get_uint_info("group_ptr").size < 5:
        r2 = "rank:* without query groups (or fewer than 4)"
    if r2 not in (None, MULTI_NO_BASE):
        return r2
    if metrics or kw.get("callbacks"):
        return "eval_metric or callbacks with a shape setting stock grows"
    obj = p.get("objective", "reg:squarederror")
    if obj not in product_cpp.OBJ and obj not in ("multi:softprob", "multi:softmax") + ADAPTIVE + REFRESH_OBJ:
        return f"objective {obj!r} has no product rule"
    if obj in ADAPTIVE and _mono_vector(p, int(dtrain.num_col())) is not None:
        return "reg:absoluteerror / reg:quantileerror with monotone constraints (the leaf quantile refit takes no bounds)"
    if obj in product_cpp.OBJ2 and (_pget(p, ("alpha", "reg_alpha"), 0.0) != 0.0
                                    or _mono_vector(p, int(dtrain.num_col())) is not None):
        return "squaredlogerror / hinge with reg_alpha or monotone constraints"
    if p.get("booster") == "dart" and (obj not in product_cpp.OBJ or int(p.get("num_parallel_tree", 1)) > 1
                                       or _pget(p, ("alpha", "reg_alpha"), 0.0) != 0.0
                                       or _mono_vector(p, int(dtrain.num_col())) is not None):
        return "booster dart with multi-class, absoluteerror / quantileerror, num_parallel_tree, reg_alpha or monotone"
    if int(p.get("num_parallel_tree", 1)) > 1 and (obj not in product_cpp.OBJ or _pget(p, ("alpha", "reg_alpha"), 0.0) != 0.0
                                                    or _mono_vector(p, int(dtrain.num_col())) is not None):
        return "num_parallel_tree > 1 with multi-class, absoluteerror / quantileerror, reg_alpha or monotone constraints"
    if not cat and not isinstance(getattr(dtrain, "_shira_x", None), np.ndarray):
        return "the training matrix is not a plain numpy array"
    if not cat and any(not isinstance(getattr(d, "_shira_x", None), np.ndarray) for d, _ in ev):
        return "an eval set that is not a plain numpy array"
    m = _small_rows(int(dtrain.num_row()), first_read_rows(int(dtrain.num_col())), int(dtrain.num_col()))
    if m is None:
        return "the automatic sample choice (SHIRA_XGB_MODE=first_read_auto) has no fixed sample for the general route"
    if os.environ.get("SHIRA_XGB_FIRST_READ_ROWS", "").strip() and int(m) > int(dtrain.num_row()):
        LOG.info("shira.xgboost: the general route refused: the forced sample exceeds the rows (SHIRA_XGB_FIRST_READ_ROWS > the training rows)")
        return "the forced sample exceeds the rows (SHIRA_XGB_FIRST_READ_ROWS > the training rows)"
    return None


def _rank_groups_reason(dtrain: Any, ev: list) -> str | None:
    for dm, name in [(dtrain, "dtrain")] + list(ev):
        if not isinstance(getattr(dm, "_shira_x", None), np.ndarray):
            return f"{name}: not a plain numpy matrix"
        y, w = dm.get_label(), dm.get_weight()
        if y.size != dm.num_row() or not np.isfinite(y).all():
            return f"{name}: labels missing, non-finite or multi-target"
        if w.size:
            return f"{name}: ranking weights (one per query group)"
        if dm.get_float_info("base_margin").size or dm.get_float_info("label_lower_bound").size:
            return f"{name}: base_margin or label bounds"
        if dm.feature_types is not None and any(t == "c" for t in dm.feature_types):
            return f"{name}: categorical features with ranking"
    return None


RANK_ENGINE_OBJ = ("rank:ndcg", "rank:pairwise", "rank:map")


def _rank_engine_reason(params: Any, dtrain: Any, rounds: int, evals: list, kw: dict[str, Any],
                        force: bool = False) -> str | None:
    p, _, _ = normalize_params(params)
    obj = str(p.get("objective", "reg:squarederror"))
    if obj not in RANK_ENGINE_OBJ:
        return "not a ranking objective"
    force = "1" if force else os.environ.get("SHIRA_XGB_RANK_ENGINE", "").strip()
    if force == "0":
        return "SHIRA_XGB_RANK_ENGINE=0"
    if evals or any(kw.get(k) for k in ("early_stopping_rounds", "callbacks", "maximize")) or any(
            kw.get(k) is not None for k in ("xgb_model", "custom_metric", "obj")):
        return "eval sets, early stopping, callbacks, a starting model, a custom objective or metric with ranking"
    if p.get("lambdarank_unbiased", False) or str(p.get("lambdarank_pair_method", "topk")) not in ("topk", "mean"):
        return "lambdarank_unbiased or a pair method the engine does not reproduce"
    if str(p.get("lambdarank_pair_method", "topk")) == "mean" and sys.platform != "darwin":
        return "lambdarank_pair_method mean off macOS (stock's libstdc++ pair draws are not reproduced)"
    if int(dtrain.get_uint_info("group_ptr").size) < 2:
        return "ranking without query groups"
    why = _rank_groups_reason(dtrain, []) or (dtrain.get_float_info("feature_weights").size > 0 and "feature_weights")
    if why:
        return str(why)
    if not 1 <= int(rounds) <= MAX_ROUNDS:
        return f"num_boost_round {rounds}"
    n, nf = int(dtrain.num_row()), int(dtrain.num_col())
    key = "xgb_rank_ndcg_engine_max_rows" if obj == "rank:ndcg" else "xgb_rank_pair_map_engine_max_rows"
    if force != "1" and n >= _machine.value(key, nf):
        return (f"{obj} on {n} rows: today's route on this machine (switch {key}; SHIRA_XGB_RANK_ENGINE=1 forces the "
                "engine)")
    rest = {k: v for k, v in p.items() if k not in RANK_KEYS and k not in PROBE_KEYS[9:]}
    return why_not(dict(rest, objective="reg:squarederror"))


def _rank_engine_fit(params: Any, dtrain: Any, rounds: int) -> tuple[export.ShiraModel | None, str | None]:
    p, _, _ = normalize_params(params)
    run, why = _prepare(p, dtrain)
    if why:
        return None, why
    assert run is not None and rounds >= 1
    gp = np.ascontiguousarray(dtrain.get_uint_info("group_ptr"), dtype=np.int64)
    assert gp.size >= 2 and int(gp[-1]) == int(run.codes.shape[0])
    q = _engine_params(p, run.sm.base_score)
    trees, n_rounds, _ = _fit.engine_es(q, run.codes, run.sm.ip, run.y, int(rounds), None, None, 0, group_ptr=gp)
    assert n_rounds == rounds
    run.sm.trees = trees
    LOG.info("shira.xgboost: %s on SHIRA's XGBoost engine, every row, %d query groups (stock's own "
             "model)", p.get("objective"), gp.size - 1)
    return run.sm, None


def _refresh_values(params: Any, dtrain: Any, bst: Any, obj: Any = None) -> Any:
    rounds = int(bst.num_boosted_rounds())
    assert rounds >= 1
    q = dict(params if isinstance(params, dict) else dict(params), process_type="update", updater="refresh",
             refresh_leaf=True)
    base = json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]
    q["base_score"] = json.loads(base)[0] if base.startswith("[") else float(base)
    full = _core.DMatrix(dtrain._shira_x, label=dtrain.get_label(), nthread=_call_threads(normalize_params(params)[0]))
    gp = dtrain.get_uint_info("group_ptr")
    if gp.size:
        full.set_group(np.diff(gp.astype(np.int64)))
    w = dtrain.get_weight()
    if w.size:
        full.set_weight(w)
    return _STOCK["train"](q, full, rounds, xgb_model=bst, verbose_eval=False, obj=obj)


def _route_leaf_ids(trees: list, leaf_nodes: list, nl: np.ndarray, x: np.ndarray, th: int) -> np.ndarray:
    from shira_cpp import outputs_cpp

    raw = []
    for t, tr in enumerate(trees):
        lc = np.asarray(tr["left_children"], dtype=np.int32)
        rc = np.asarray(tr["right_children"], dtype=np.int32)
        leaf = lc < 0
        cond = np.asarray(tr["split_conditions"], dtype=np.float32).astype(np.float64)
        lid = np.full(lc.size, -1, dtype=np.int32)
        lid[leaf_nodes[t]] = np.arange(nl[t], dtype=np.int32)
        raw.append(outputs_cpp.RawTree(np.where(leaf, -1, np.asarray(tr["split_indices"], dtype=np.int32)),
                                       np.where(leaf, 0.0, cond), np.asarray(tr["default_left"], dtype=np.int32),
                                       np.maximum(lc, 0), np.maximum(rc, 0), np.zeros(lc.size), leaf_id=lid))
    fr = outputs_cpp.build(raw, int(x.shape[1]), 0, 0, 0)
    fr.in_f32 = True
    out = fr.leaf(np.ascontiguousarray(x, dtype=F32), threads=th)
    assert out.shape == (x.shape[0], len(trees)) and int(out.min()) >= 0
    from shira_cpp import product_cpp

    return product_cpp.leaf_transpose(out, int(nl.max()) > 255, th)


KCAT_CODES = 288


def _cat_raw(dtrain: Any) -> np.ndarray | None:
    x = getattr(dtrain, "_shira_x", None)
    if isinstance(x, np.ndarray):
        return x
    src = dtrain.__dict__.get("_sx_src")
    if src is None or type(src[0]).__name__ != "DataFrame":
        return None
    df = src[0]
    out = np.empty(df.shape, dtype=F32)
    for j, c in enumerate(df.columns):
        col = df[c]
        if str(col.dtype) == "category":
            codes = col.cat.codes.to_numpy().astype(F32)
            codes[codes < 0] = np.nan
            out[:, j] = codes
        elif getattr(col.dtype, "kind", "?") in "fiub":
            out[:, j] = col.to_numpy(dtype=F32)
        else:
            return None
    return out


def _route_leaf_ids_cat(trees: list, leaf_nodes: list, nl: np.ndarray, x: np.ndarray, is_cat: np.ndarray,
                        th: int) -> np.ndarray | None:
    from shira_cpp import product_cpp

    n, nf = x.shape
    thr = [set() for _ in range(nf)]
    for tr in trees:
        st, lc = np.asarray(tr["split_type"]), np.asarray(tr["left_children"])
        for i in np.flatnonzero((lc != -1) & (st == 0)).tolist():
            thr[int(tr["split_indices"][i])].add(float(np.float32(tr["split_conditions"][i])))
    tabs = [np.array(sorted(t), dtype=F32) for t in thr]
    from shira_cpp import outputs_cpp

    xc = np.ascontiguousarray(x, dtype=F32)
    off = np.zeros(nf + 1, dtype=np.int32)
    off[1:] = np.cumsum([0 if is_cat[f] else tabs[f].size for f in range(nf)])
    flat = np.ascontiguousarray(np.concatenate([tabs[f].astype(np.float64) for f in range(nf) if not is_cat[f]]
                                               or [np.zeros(0)]), dtype=np.float64)
    codes = np.empty((n, nf), dtype=np.uint16)
    rc = outputs_cpp._lib().shira_bin_raw_f32(xc.ctypes.data, n, nf, off.ctypes.data, flat.ctypes.data, 0, th,
                                              codes.ctypes.data)
    if rc != 0:
        return None
    for f in np.flatnonzero(is_cat).tolist():
        col = xc[:, f]
        nan = np.isnan(col)
        v = np.where(nan, 0.0, col)
        if np.any(v < 0) or np.any(v > KCAT_CODES - 2) or np.any(v != np.floor(v)):
            return None
        codes[:, f] = np.where(nan, 0, v.astype(np.int64) + 1)
    maxn = max(len(tr["left_children"]) for tr in trees)
    T = len(trees)
    nodes = np.zeros((T, maxn, 5), dtype=np.int32)
    cat_words: list[np.ndarray] = [np.zeros(9, dtype=np.uint32)]
    for t, tr in enumerate(trees):
        lc, rc = np.asarray(tr["left_children"]), np.asarray(tr["right_children"])
        cat_of = {int(nid): (int(a), int(b)) for nid, a, b in zip(tr["categories_nodes"], tr["categories_segments"],
                                                                tr["categories_sizes"])}
        for i in range(lc.size):
            if lc[i] == -1:
                nodes[t, i] = (-1, 0, 0, 0, 0)
                continue
            f, dl = int(tr["split_indices"][i]), int(tr["default_left"][i])
            if int(tr["split_type"][i]) == 1:
                a, b = cat_of[i]
                inset = np.zeros(KCAT_CODES, dtype=bool)
                cats = np.asarray(tr["categories"][a:a + b], dtype=np.int64)
                inset[cats[cats < KCAT_CODES - 1] + 1] = True
                left = ~inset
                left[0] = bool(dl)
                bits = np.packbits(left, bitorder="little").view(np.uint32)
                nodes[t, i] = (f, 9 * len(cat_words), 2, lc[i], rc[i])
                cat_words.append(np.ascontiguousarray(bits[:9]))
            else:
                k = int(np.searchsorted(tabs[f], np.float32(tr["split_conditions"][i]), side="left")) + 1
                nodes[t, i] = (f, k, dl, lc[i], rc[i])
    count = np.array([len(tr["left_children"]) for tr in trees], dtype=np.int32)
    ids, nl2 = product_cpp.leaf_ids_lanes(codes, nodes, count, nthread=th, cat=np.concatenate(cat_words))
    assert np.array_equal(nl2, nl)
    want = np.uint16 if int(nl.max()) > 255 else np.uint8
    return ids if ids.dtype == want else ids.astype(want)


def _stock_base(p: dict[str, Any], y: np.ndarray, w: np.ndarray | None, th: int) -> Any:
    if "base_score" in p:
        return float(p["base_score"])
    if str(p.get("objective", "")).startswith("multi:"):
        return _stock_intercept(p, y, w, th)
    probe = {k: p[k] for k in PROBE_KEYS if k in p and k not in ("max_bin", "base_score", "num_class")}
    dm = _core.DMatrix(np.zeros((y.size, 1), dtype=F32), label=y, weight=w, nthread=th)
    b = _STOCK["train"](dict(probe, max_depth=1, nthread=th), dm, 1)
    v = json.loads(json.loads(b.save_config())["learner"]["learner_model_param"]["base_score"])
    v = float(v[0] if isinstance(v, list) else v)
    assert math.isfinite(v)
    return v


def _json_tables(trees: list, mono: np.ndarray | None) -> tuple:
    T = len(trees)
    leaf_nodes = [np.flatnonzero(np.asarray(t["left_children"]) == -1) for t in trees]
    nl = np.array([ln.size for ln in leaf_nodes], dtype=np.int32)
    pos = [np.full(len(t["left_children"]), -1, dtype=np.int64) for t in trees]
    for t in range(T):
        pos[t][leaf_nodes[t]] = np.arange(nl[t])
    if mono is None:
        return leaf_nodes, nl, pos, None, None
    maxn = max(len(t["left_children"]) for t in trees)
    tab = np.zeros((T, maxn, 4), dtype=np.int32)
    cnt = np.array([len(t["left_children"]) for t in trees], dtype=np.int32)
    for t in range(T):
        lc, rc = np.asarray(trees[t]["left_children"]), np.asarray(trees[t]["right_children"])
        inner = np.flatnonzero(lc != -1)
        tab[t, :cnt[t], 0] = pos[t]
        tab[t, inner, 1], tab[t, inner, 2] = lc[inner], rc[inner]
        tab[t, inner, 3] = mono[np.asarray(trees[t]["split_indices"])[inner]]
        assert cnt[t] == 2 * nl[t] - 1 and np.all(tab[t, inner, 1] > inner) and np.all(tab[t, inner, 2] > inner)
    return leaf_nodes, nl, pos, tab, cnt


class _DartWatch(_cb.TrainingCallback):

    def __init__(self) -> None:
        super().__init__()
        self.w: list[np.ndarray] = []

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        gb = json.loads(bytes(model.save_raw("json")))["learner"]["gradient_booster"]
        self.w.append(np.asarray(gb.get("weight_drop", []), dtype=np.float64))
        assert self.w[-1].size == len(self.w)
        return False


def _dart_values(p: dict[str, Any], ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None,
                 margin: float, hist_w: list, th: int) -> np.ndarray:
    from shira_cpp import product_cpp

    obj = p.get("objective", "reg:squarederror")
    T, n = ids.shape
    assert len(hist_w) == T and T >= 1
    kw = dict(eta=_pget(p, ("eta", "learning_rate"), 0.3), reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
              min_child_weight=_pget(p, ("min_child_weight",), 1.0),
              max_delta_step=_pget(p, ("max_delta_step",), 0.7 if obj == "count:poisson" else 0.0), weight=w,
              obj_a=(_pget(p, ("huber_slope",), 1.0) if obj == "reg:pseudohubererror"
                     else _pget(p, ("tweedie_variance_power",), 1.5) if obj == "reg:tweedie" else 1.0),
              obj_mds=_pget(p, ("max_delta_step",), 0.7), nthread=th)
    leaf = np.zeros((T, int(nl.max())), dtype=F32)
    S = np.full(n, float(F32(margin)), dtype=np.float64)
    contrib = lambda i: leaf[i, : nl[i]][ids[i]].astype(np.float64)
    prev = np.zeros(0, dtype=np.float64)
    for r in range(T):
        now = hist_w[r]
        drop = np.flatnonzero(now[:r] != prev)
        m_r = S.copy()
        for i in drop.tolist():
            m_r -= prev[i] * contrib(i)
        lf, _ = product_cpp.product(ids[r:r + 1], nl[r:r + 1], y, margin, obj, pred=m_r.astype(F32), **kw)
        leaf[r, : nl[r]] = lf[0, : nl[r]]
        for i in drop.tolist():
            S += (now[i] - prev[i]) * contrib(i)
        S += now[r] * contrib(r)
        prev = now
    return leaf


def _parallel_values(p: dict[str, Any], ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None,
                     margin: float, K: int, th: int) -> np.ndarray:
    from shira_cpp import product_cpp

    obj = p.get("objective", "reg:squarederror")
    T, n = ids.shape
    assert T % K == 0 and K > 1
    eta = _pget(p, ("eta", "learning_rate"), 0.3) / K
    kw = dict(reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0), min_child_weight=_pget(p, ("min_child_weight",), 1.0),
              max_delta_step=_pget(p, ("max_delta_step",), 0.7 if obj == "count:poisson" else 0.0), weight=w,
              obj_a=(_pget(p, ("huber_slope",), 1.0) if obj == "reg:pseudohubererror"
                     else _pget(p, ("tweedie_variance_power",), 1.5) if obj == "reg:tweedie" else 1.0),
              obj_mds=_pget(p, ("max_delta_step",), 0.7), nthread=th)
    if os.environ.get("SHIRA_XGB_PARALLEL_PER_TREE", "") != "1":
        return product_cpp.product_parallel(ids, nl, y, margin, obj, K, eta=eta, **kw)[0]
    leaf = np.zeros((T, int(nl.max())), dtype=F32)
    pred = np.full(n, F32(margin), dtype=F32)
    for r in range(T // K):
        add = np.zeros(n, dtype=F32)
        for k in range(K):
            t = r * K + k
            lf, pr = product_cpp.product(ids[t:t + 1], nl[t:t + 1], y, margin, obj, eta=eta, pred=pred, **kw)
            leaf[t, : nl[t]] = lf[0, : nl[t]]
            add += pr - pred
        pred = pred + add
    return leaf


def _flag(p: dict[str, Any], key: str, default: bool) -> bool:
    v = p.get(key)
    return default if v is None else str(v).strip().lower() not in ("0", "false", "no")


def _xrn_on(p: dict[str, Any], dtrain: Any) -> bool:
    if str(p.get("objective", "")) != "rank:ndcg" or _is_cat(dtrain):
        return False
    if str(p.get("lambdarank_pair_method", "topk")) != "topk" or _flag(p, "lambdarank_unbiased", False):
        return False
    if str(p.get("booster", "gbtree")) != "gbtree" or int(p.get("num_parallel_tree", 1)) > 1:
        return False
    if _mono_vector(p, int(dtrain.num_col())) is not None:
        return False
    nq = int(dtrain.get_uint_info("group_ptr").size) - 1
    w = dtrain.get_weight()
    return nq >= 1 and (w.size == 0 or w.size == nq)


def _xrn_values(p: dict[str, Any], ids: np.ndarray, nl: np.ndarray, y: np.ndarray, dtrain: Any, base: float,
                th: int) -> np.ndarray:
    from shira_cpp import product_cpp

    gp = dtrain.get_uint_info("group_ptr").astype(np.int64)
    w = dtrain.get_weight()
    npair = p.get("lambdarank_num_pair_per_sample")
    topk = 32 if npair in (None, "", 4294967295, "4294967295") else int(npair)
    return product_cpp.xgb_rank_ndcg(ids, nl, y, gp, base, None if w.size == 0 else np.asarray(w, dtype=F32), topk=topk,
                                     exp_gain=_flag(p, "ndcg_exp_gain", True),
                                     norm_by_diff=_flag(p, "lambdarank_score_normalization", True),
                                     normalization=_flag(p, "lambdarank_normalization", True),
                                     eta=_pget(p, ("eta", "learning_rate"), 0.3), reg_lambda=_pget(p, ("lambda", "reg_lambda"), 1.0),
                                     reg_alpha=_pget(p, ("alpha", "reg_alpha"), 0.0),
                                     min_child_weight=_pget(p, ("min_child_weight",), 1.0),
                                     max_delta_step=_pget(p, ("max_delta_step",), 0.0), nthread=th)[0]


def _custom_route_values(p: dict[str, Any], ids: np.ndarray, nl: np.ndarray, dtrain: Any, margin: float,
                         obj: Callable[..., Any]) -> np.ndarray:
    T, n = int(ids.shape[0]), int(ids.shape[1])
    assert T >= 1 and n >= 1
    eta, lam = _pget(p, ("eta", "learning_rate"), 0.3), _pget(p, ("lambda", "reg_lambda"), 1.0)
    alpha, mcw = _pget(p, ("alpha", "reg_alpha"), 0.0), _pget(p, ("min_child_weight",), 1.0)
    mds = _pget(p, ("max_delta_step",), 0.0)
    pred = np.full(n, margin, dtype=F32)
    leaf = np.zeros((T, int(nl.max())), dtype=F32)
    for t in range(T):
        g, h = obj(pred.copy(), dtrain)
        g, h = np.asarray(g, dtype=F32).ravel(), np.asarray(h, dtype=F32).ravel()
        if g.shape != (n,) or h.shape != (n,):
            raise ValueError("a custom objective with more than one output per row (the value pass takes one)")
        k, idx = int(nl[t]), ids[t].astype(np.int64)
        G, H = np.bincount(idx, g.astype(np.float64), k), np.bincount(idx, h.astype(np.float64), k)
        v = -np.sign(G) * np.maximum(np.abs(G) - alpha, 0.0) / (H + lam)
        if mds > 0.0:
            v = np.clip(v, -mds, mds)
        v = np.where((H < mcw) | (H <= 0.0), 0.0, v)
        leaf[t, :k] = (v * eta).astype(F32)
        pred += leaf[t, :k][idx]
    return leaf


def _sample_route_values(p: dict[str, Any], ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None,
                         margin: Any, tab: Any, cnt: Any, th: int, own: np.ndarray | None = None,
                         dart_w: list | None = None) -> np.ndarray:
    from shira_cpp import product_cpp

    obj = p.get("objective", "reg:squarederror")
    eta, lam = _pget(p, ("eta", "learning_rate"), 0.3), _pget(p, ("lambda", "reg_lambda"), 1.0)
    if p.get("booster") == "dart" and dart_w is not None:
        return _dart_values(p, ids, nl, y, w, float(margin), dart_w, th)
    K = int(p.get("num_parallel_tree", 1))
    if K > 1:
        return _parallel_values(p, ids, nl, y, w, float(margin), K, th)
    if obj in product_cpp.OBJ2:
        return product_cpp.product_obj2(ids, nl, y, margin, obj, eta=eta, reg_lambda=lam,
                                        min_child_weight=_pget(p, ("min_child_weight",), 1.0),
                                        max_delta_step=_pget(p, ("max_delta_step",), 0.0), weight=w, nthread=th)[0]
    if obj in ADAPTIVE:
        alpha = 0.5 if obj == "reg:absoluteerror" else _pget(p, ("quantile_alpha",), 0.5)
        return product_cpp.product_renew_xgb(ids, nl, y, margin, alpha, eta=eta, weight=w, fallback=own, nthread=th)[0]
    alpha, mcw = _pget(p, ("alpha", "reg_alpha"), 0.0), _pget(p, ("min_child_weight",), 1.0)
    mds = _pget(p, ("max_delta_step",), 0.7 if obj == "count:poisson" else 0.0)
    if obj.startswith("multi:"):
        return product_cpp.product_softmax_reg(ids, nl, int(p["num_class"]), y, margin, tree=tab, count=cnt, eta=eta,
                                               reg_lambda=lam, reg_alpha=alpha, min_child_weight=mcw,
                                               max_delta_step=_pget(p, ("max_delta_step",), 0.0), weight=w,
                                               nthread=th)[0]
    obj_a = (_pget(p, ("huber_slope",), 1.0) if obj == "reg:pseudohubererror"
             else _pget(p, ("tweedie_variance_power",), 1.5) if obj == "reg:tweedie" else 1.0)
    obj_mds = _pget(p, ("max_delta_step",), 0.7)
    if tab is not None:
        return product_cpp.product_bounded(ids, nl, tab, cnt, y, margin, obj, eta=eta, reg_lambda=lam, min_child_weight=mcw,
                                           weight=w, nthread=th, obj_mds=obj_mds, reg_alpha=alpha, max_delta_step=mds,
                                           obj_a=obj_a)[0]
    if alpha != 0.0:
        return product_cpp.product_l1(ids, nl, y, margin, obj, eta=eta, reg_lambda=lam, reg_alpha=alpha,
                                      min_child_weight=mcw, max_delta_step=mds, weight=w, obj_a=obj_a, obj_mds=obj_mds,
                                      nthread=th)[0]
    return product_cpp.product(ids, nl, y, margin, obj, eta=eta, reg_lambda=lam, min_child_weight=mcw, max_delta_step=mds,
                               weight=w, obj_a=obj_a, obj_mds=obj_mds, nthread=th)[0]


ROUTE_SHAPES = ("deep", "subsample", "method", "categorical", "parallel", "rank", "refresh", "adaptive", "hinge",
                "dart")


def _route_shapes(p: dict[str, Any], dtrain: Any) -> list[str]:
    obj = str(p.get("objective", "reg:squarederror"))
    lossguide = p.get("grow_policy", "depthwise") == "lossguide"
    depth = int(p.get("max_depth", 6))
    out = []
    if (not lossguide and (depth == 0 or depth > 12 or int(p.get("max_leaves", 0)) != 0)) or (
            lossguide and (depth > 12 or (depth == 0 and int(p.get("max_leaves", 0)) == 0))):
        out.append("deep")
    if float(p.get("subsample", 1.0)) < 1.0 and "nthread" not in p:
        out.append("subsample")
    if p.get("tree_method") in SHAPE_ONLY_METHODS:
        out.append("method")
    if _is_cat(dtrain):
        out.append("categorical")
    if int(p.get("num_parallel_tree", 1)) > 1:
        out.append("parallel")
    if obj.startswith("rank:"):
        out.append("rank")
    elif obj in REFRESH_OBJ:
        out.append("refresh")
        if obj == "binary:hinge":
            out.append("hinge")
    if obj in ADAPTIVE:
        out.append("adaptive")
    if p.get("booster") == "dart":
        out.append("dart")
    assert all(s in ROUTE_SHAPES for s in out)
    return out


def _route_guard(p: dict[str, Any], dtrain: Any) -> str | None:
    forced = {s.strip() for s in os.environ.get("SHIRA_XGB_GENERAL_ROUTE", "").split(",") if s.strip()}
    n, nf = int(dtrain.num_row()), int(dtrain.num_col())
    off = [s for s in _route_shapes(p, dtrain)
           if "all" not in forced and s not in forced and n < _machine.value(f"xgb_route_{s}_min_rows", nf)]
    if not off:
        return None
    return (f"the general route for {' / '.join(off)} is not measured faster than stock on this machine at {n} rows x "
            f"{nf} (switch xgb_route_<shape>_min_rows in _machine.py; SHIRA_XGB_GENERAL_ROUTE=<shape> forces it)")


def _route_rows(p: dict[str, Any], dtrain: Any) -> int:
    nf = int(dtrain.num_col())
    base = int(first_read_rows(nf))
    if os.environ.get("SHIRA_XGB_FIRST_READ_ROWS", "").strip():
        return base
    own = [_machine.value(f"xgb_route_{s}_rows", nf) for s in _route_shapes(p, dtrain)]
    own = [v for v in own if v > 0]
    m = min(int(dtrain.num_row()), int(_small_rows(int(dtrain.num_row()), min(own) if own else base, nf)))
    cap = int(_machine.value("xgb_route_rank_cap_rows", nf))
    assert cap >= 0 and m >= 1
    return min(m, cap) if cap and _xrn_on(p, dtrain) else m


def _route_sample_threads(p: dict[str, Any], features: int) -> int:
    if float(p.get("subsample", 1.0)) >= 1.0 or "nthread" in p:
        return 0
    v = int(_machine.value("xgb_route_subsample_threads", features))
    assert v >= 0
    return min(v, _threads.xgb_default_nthread()) if v else 0


def _route_fit_threads(p: dict[str, Any], features: int, rows: int, evals: bool) -> int:
    p, _, _ = normalize_params(p)
    if evals or str(p.get("objective", "")).startswith("rank:") or float(p.get("subsample", 1.0)) < 1.0:
        return 0
    cap = int(_machine.value("xgb_route_sample_threads", features))
    top = int(_machine.value("xgb_route_sample_threads_rows", features))
    assert cap >= 0 and top >= 0 and rows >= 1
    if cap == 0 or rows > top:
        return 0
    return cap if cap < _call_threads(p) else 0


def _stock_sample_fit(params: Any, dtrain: Any, rounds: int, ev: list, kw: dict[str, Any], reason: str) -> tuple:
    why = _sample_route_reason(params, dtrain, rounds, ev, kw)
    if why:
        return None, why
    p, _, _ = normalize_params(params)
    why = _route_guard(p, dtrain)
    if why:
        return None, why
    cat = _is_cat(dtrain)
    x = None if cat else dtrain._shira_x
    y = np.ascontiguousarray(dtrain.get_label(), dtype=F32)
    w0 = dtrain.get_weight()
    w = np.ascontiguousarray(w0, dtype=F32) if w0.size else None
    n, nf = int(dtrain.num_row()), int(dtrain.num_col())
    m = _route_rows(p, dtrain)
    th = _call_threads(p)
    custom = kw.get("obj") is not None
    xrn = _xrn_on(p, dtrain)
    refresh = str(p.get("objective", "")).startswith("rank:") and not xrn
    base = None if (refresh or custom or xrn) else _stock_base(p, y, w, th)
    multi = str(p.get("objective", "")).startswith("multi:")
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    gp = dtrain.get_uint_info("group_ptr").astype(np.int64)
    gs = None
    if gp.size:
        Q = gp.size - 1
        k = max(2, min(Q, int(np.ceil(Q * m / n))))
        qsel = np.unique((Q - 1) * np.arange(k, dtype=np.int64) // max(1, k - 1))
        rows = np.concatenate([np.arange(gp[g], gp[g + 1], dtype=np.int64) for g in qsel])
        gs = np.diff(gp)[qsel]
    full = _cat_matrix(dtrain) if cat else None
    if cat:
        ds = full.slice(rows)
        evs = [(_cat_matrix(d), nm) for d, nm in ev]
    else:
        ds = _core.DMatrix(np.ascontiguousarray(x[rows]), label=y[rows], weight=None if w is None else w[rows], nthread=th)
        if gs is not None:
            ds.set_group(gs)
        evs = [(_core.DMatrix(np.ascontiguousarray(d._shira_x, dtype=F32), label=d.get_label(),
                              weight=d.get_weight() if d.get_weight().size else None, nthread=th), nm) for d, nm in ev]
        for (de, _), (d, _) in zip(evs, ev):
            if d.get_uint_info("group_ptr").size:
                de.set_group(np.diff(d.get_uint_info("group_ptr").astype(np.int64)))
    hist: dict = {}
    watch = _DartWatch() if p.get("booster") == "dart" else None
    sp = dict(params if isinstance(params, dict) else dict(params))
    if not refresh and not custom and not xrn:
        sp["base_score"] = base.tolist() if multi else base
    sth = _route_sample_threads(p, nf)
    if sth:
        sp["nthread"] = sth
    cth = _route_fit_threads(p, nf, int(rows.size), bool(evs))
    if cth:
        sp["nthread"] = cth
    bst = _STOCK["train"](sp, ds, rounds, evals=evs, early_stopping_rounds=kw.get("early_stopping_rounds"), obj=kw.get("obj"),
                          evals_result=hist, verbose_eval=kw.get("verbose_eval", True) if evs else False,
                          maximize=kw.get("maximize"), callbacks=None if watch is None else [watch])
    attrs = {k: v for k, v in bst.attributes().items() if v is not None}
    if refresh:
        if attrs.get("best_iteration") is not None:
            bst = bst[: int(attrs["best_iteration"]) + 1]
        out = Booster(params, [dtrain], bytearray(bytes(_refresh_values(params, dtrain, bst, kw.get("obj")).save_raw("json"))))
        _route_names(out, dtrain)
        if attrs:
            out.set_attr(**attrs)
        if kw.get("evals_result") is not None:
            kw["evals_result"].update(hist)
        first_read_log.append({"n": n, "m": int(rows.size), "kept": int(rows.size), "route": "stock sample + refresh",
                               "hist": None})
        LOG.info("shira.xgboost: fitted the first-read way (SHIRA's default) by the general route: %s; tree shapes by "
                 "stock XGBoost on %d stride-sample rows%s, leaf values on all %d rows by stock's refresh updater"
                 + (" with the caller's objective" if kw.get("obj") is not None else "") + " "
                 "(not stock's trees); prediction is stock XGBoost's; SHIRA_XGB_MODE=ordinary runs stock", reason, rows.size,
                 "" if gs is None else f" ({gs.size} whole queries)", n)
        return out, None
    doc = json.loads(bytes(bst.save_raw("json")))
    gbm = doc["learner"]["gradient_booster"]
    trees = gbm.get("gbtree", gbm)["model"]["trees"]
    leaf_nodes, nl, pos, tab, cnt = _json_tables(trees, _mono_vector(p, nf))
    if int(nl.max()) > 65535:
        return None, f"a sample tree has {int(nl.max())} leaves (> 65535, the product's 16-bit leaf ids)"
    T = len(trees)
    xr = _cat_raw(dtrain) if cat else None
    is_cat = np.array([t == "c" for t in (dtrain.feature_types or [])], dtype=bool) if cat else None
    ids = _route_leaf_ids_cat(trees, leaf_nodes, nl, xr, is_cat, th) if xr is not None else None
    if ids is not None:
        pass
    elif cat:
        node = np.asarray(bst.predict(full, pred_leaf=True), dtype=np.int64).reshape(n, T)
        ids = np.empty((T, n), dtype=np.uint16 if int(nl.max()) > 255 else np.uint8)
        for t in range(T):
            ids[t] = pos[t][node[:, t]]
    else:
        ids = _route_leaf_ids(trees, leaf_nodes, nl, x, th)
    if custom or xrn:
        base = float(str(json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]).strip("[]").split(",")[0])
    margin = base if multi else _objective_and_margin(p, nf, p.get("objective", "reg:squarederror"), base, 1)[1]
    own = np.zeros((T, int(nl.max())), dtype=F32)
    for t in range(T):
        own[t, : nl[t]] = [trees[t]["split_conditions"][j] for j in leaf_nodes[t].tolist()]
    if xrn:
        leaf = _xrn_values(p, ids, nl, y, dtrain, float(base), th)
    else:
        leaf = (_custom_route_values(p, ids, nl, dtrain, float(margin), kw["obj"]) if custom
                else _sample_route_values(p, ids, nl, y, w, margin, tab, cnt, th, own, None if watch is None else watch.w))
    for t in range(T):
        sc, bw = trees[t]["split_conditions"], trees[t]["base_weights"]
        for j, nid in enumerate(leaf_nodes[t].tolist()):
            sc[nid] = bw[nid] = float(leaf[t, j])
    lazy = isinstance(dtrain, QuantileDMatrix) and dtrain._lazy()
    out = _RouteBooster(params, [] if lazy else [dtrain], bytearray(json.dumps(doc).encode()))
    _route_names(out, dtrain)
    if watch is not None:
        out.__dict__["_shira_dart_w"] = list(watch.w)
    if attrs:
        out.set_attr(**attrs)
    if kw.get("evals_result") is not None:
        kw["evals_result"].update(hist)
    first_read_log.append({"n": n, "m": m, "kept": m, "route": "stock sample", "hist": None})
    LOG.info("shira.xgboost: fitted the first-read way (SHIRA's default) by the general route: %s; tree shapes by stock "
             "XGBoost with the caller's settings on %d stride-sample rows, leaf values on all %d rows by the product "
             "(not stock's trees); prediction is stock XGBoost's; SHIRA_XGB_MODE=ordinary runs stock", reason, m, n)
    return out, None


AFT_EPS, AFT_GMIN, AFT_GMAX, AFT_HMIN, AFT_HMAX = 1e-12, -15.0, 15.0, 1e-16, 15.0


def _aft_dist(name: str, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(over="ignore", invalid="ignore"):
        if name == "normal":
            pdf = np.exp(-z * z / 2.0) / np.sqrt(2.0 * np.pi)
            from scipy.special import erf

            cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
            return pdf, cdf, -z * pdf, (z * z - 1.0) * pdf
        w = np.exp(z)
        big = ~np.isfinite(w) | ~np.isfinite(w * w)
        if name == "logistic":
            pdf = np.where(big, 0.0, w / ((1.0 + w) * (1.0 + w)))
            cdf = np.where(np.isfinite(w), w / (1.0 + w), 1.0)
            g = np.where(np.isfinite(w), pdf * (1.0 - w) / (1.0 + w), 0.0)
            h = np.where(big, 0.0, pdf * (w * w - 4.0 * w + 1.0) / ((1.0 + w) * (1.0 + w)))
            return pdf, cdf, g, h
        assert name == "extreme"
        pdf = np.where(np.isfinite(w), w * np.exp(-w), 0.0)
        cdf = 1.0 - np.exp(-w)
        g = np.where(np.isfinite(w), (1.0 - w) * pdf, 0.0)
        h = np.where(big, 0.0, (w * w - 3.0 * w + 1.0) * pdf)
        return pdf, cdf, g, h


def _aft_limits(name: str, kind: np.ndarray, sign: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    s1, s2 = 1.0 / sigma, 1.0 / (sigma * sigma)
    if name == "normal":
        g = np.select([kind == 1, kind == 2], [np.where(sign, AFT_GMIN, 0.0), np.where(sign, 0.0, AFT_GMAX)],
                      np.where(sign, AFT_GMIN, AFT_GMAX))
        h = np.select([kind == 1, kind == 2], [np.where(sign, s2, AFT_HMIN), np.where(sign, AFT_HMIN, s2)], s2)
        return g, h
    if name == "logistic":
        g = np.select([kind == 1, kind == 2], [np.where(sign, -s1, 0.0), np.where(sign, 0.0, s1)], np.where(sign, -s1, s1))
        return g, np.full(kind.shape, AFT_HMIN)
    g = np.select([kind == 1, kind == 2], [np.where(sign, AFT_GMIN, 0.0), np.where(sign, 0.0, s1)],
                  np.where(sign, AFT_GMIN, s1))
    h = np.where(kind == 2, AFT_HMIN, np.where(sign, AFT_HMAX, AFT_HMIN))
    return g, h


def _aft_objective(p: dict[str, Any], lo: np.ndarray, hi: np.ndarray, w: np.ndarray | None) -> Callable[..., Any]:
    name, sigma = str(p.get("aft_loss_distribution", "normal")), float(p.get("aft_loss_distribution_scale", 1.0))
    assert name in ("normal", "logistic", "extreme") and sigma > 0.0
    lo64, hi64 = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    unc = lo64 == hi64
    right, left = np.isinf(hi64) & ~unc, (lo64 <= 0.0) & ~unc
    kind = np.where(unc, 0, np.where(right, 1, np.where(left, 2, 3)))
    with np.errstate(divide="ignore", invalid="ignore"):
        llo, lhi = np.log(lo64), np.log(hi64)

    idx = [np.flatnonzero(kind == k) for k in range(4)]
    w64 = None if w is None else np.asarray(w, dtype=np.float64)

    def fn(pred: np.ndarray, _: Any) -> tuple[np.ndarray, np.ndarray]:
        mu = np.asarray(pred, dtype=np.float64)
        g, h = np.empty(mu.size), np.empty(mu.size)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for k in range(4):
                ix = idx[k]
                if ix.size == 0:
                    continue
                m_k = mu[ix]
                zl = (llo[ix] - m_k) / sigma if k != 2 else None
                zu = (lhi[ix] - m_k) / sigma if k >= 2 else None
                if k == 0 and name == "normal":
                    gk, hk = -zl / sigma, np.full(ix.size, 1.0 / (sigma * sigma))
                else:
                    if k == 0:
                        p0, _c, g0, h0 = _aft_dist(name, zl)
                        num_g, den_g = g0, sigma * p0
                        num_h, den_h = -(p0 * h0 - g0 * g0), sigma * sigma * p0 * p0
                        sign = zl > 0
                    else:
                        pl, cl, gl, _h = _aft_dist(name, zl) if zl is not None else (0.0, 0.0, 0.0, 0.0)
                        pu, cu, gu, _h = _aft_dist(name, zu) if zu is not None else (0.0, 1.0, 0.0, 0.0)
                        num_g, den_g = pu - pl, sigma * (cu - cl)
                        num_h, den_h = -((cu - cl) * (gu - gl) - (pu - pl) * (pu - pl)), (sigma * (cu - cl)) ** 2
                        sign = ((zu > 0) if zu is not None else False) | ((zl > 0) if zl is not None else False)
                    gk, hk = num_g / den_g, num_h / den_h
                    bad_g, bad_h = (den_g < AFT_EPS) & ~np.isfinite(gk), (den_h < AFT_EPS) & ~np.isfinite(hk)
                    if bool(bad_g.any()) or bool(bad_h.any()):
                        lg, lh = _aft_limits(name, np.full(ix.size, k), np.broadcast_to(sign, (ix.size,)), sigma)
                        gk, hk = np.where(bad_g, lg, gk), np.where(bad_h, lh, hk)
                g[ix], h[ix] = gk, hk
        g, h = np.clip(g, AFT_GMIN, AFT_GMAX), np.clip(h, AFT_HMIN, AFT_HMAX)
        if w64 is not None:
            g, h = g * w64, h * w64
        assert bool(np.all(np.isfinite(g))) and bool(np.all(np.isfinite(h)))
        return g.astype(F32), h.astype(F32)

    return fn


def _aft_reason(params: Any, dtrain: Any, rounds: int, ev: list, kw: dict[str, Any]) -> str | None:
    p, metrics, bad = normalize_params(params)
    if bad or p.get("objective") != "survival:aft" or not first_read():
        return bad or "not survival:aft (or SHIRA_XGB_MODE=ordinary: no exact engine takes it)"
    if ev or metrics or any(kw.get(k) for k in ("obj", "custom_metric", "xgb_model", "callbacks", "early_stopping_rounds")):
        return "survival:aft with evaluation sets, a metric, callbacks, a custom objective / metric or a starting model"
    if str(p.get("booster", "gbtree")) != "gbtree" or int(p.get("num_parallel_tree", 1)) > 1 or "base_score" in p:
        return "survival:aft with another booster, num_parallel_tree or a given base_score"
    if str(p.get("aft_loss_distribution", "normal")) not in ("normal", "logistic", "extreme"):
        return "survival:aft with an unknown aft_loss_distribution"
    if not isinstance(dtrain, DMatrix) or not isinstance(getattr(dtrain, "_shira_x", None), np.ndarray) or _is_cat(dtrain):
        return "survival:aft on a matrix that is not a plain numpy array"
    n = int(dtrain.num_row())
    lo, hi = dtrain.get_float_info("label_lower_bound"), dtrain.get_float_info("label_upper_bound")
    if lo.size != n or hi.size != n or not bool(np.all(lo >= 0.0)) or not bool(np.all(hi >= lo)) or bool(np.any(np.isnan(lo))):
        return "survival:aft without both label bounds per row (or with bounds stock rejects)"
    w = dtrain.get_weight()
    if dtrain.get_float_info("base_margin").size or dtrain.get_uint_info("group_ptr").size or (
            w.size and (w.size != n or not bool(np.all(np.isfinite(w))))):
        return "survival:aft with a base margin, query groups or bad weights"
    if _mono_vector(p, int(dtrain.num_col())) is not None or not 1 <= int(rounds) <= MAX_ROUNDS:
        return "survival:aft with monotone constraints or a bad round count"
    return None


def _aft_fit(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any]) -> tuple:
    why = _aft_reason(params, dtrain, rounds, [], kw)
    if why:
        return None, why
    p, _, _ = normalize_params(params)
    x = dtrain._shira_x
    n, nf, th = int(x.shape[0]), int(x.shape[1]), _call_threads(p)
    lo = np.ascontiguousarray(dtrain.get_float_info("label_lower_bound"), dtype=F32)
    hi = np.ascontiguousarray(dtrain.get_float_info("label_upper_bound"), dtype=F32)
    w0 = dtrain.get_weight()
    w = np.ascontiguousarray(w0, dtype=F32) if w0.size else None
    m = _small_rows(n, first_read_rows(nf), nf)
    if m is None:
        return None, "no fixed sample for the survival route (SHIRA_XGB_MODE=first_read_auto)"
    m = min(n, int(m))
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    ds = _core.DMatrix(np.ascontiguousarray(x[rows]), weight=None if w is None else w[rows], nthread=th,
                       label_lower_bound=lo[rows], label_upper_bound=hi[rows])
    bst = _STOCK["train"](dict(params if isinstance(params, dict) else dict(params)), ds, int(rounds), verbose_eval=False)
    doc = json.loads(bytes(bst.save_raw("json")))
    trees = doc["learner"]["gradient_booster"]["model"]["trees"]
    leaf_nodes, nl, _, _, _ = _json_tables(trees, None)
    if int(nl.max()) > 65535:
        return None, f"a sample tree has {int(nl.max())} leaves (> 65535, the product's 16-bit leaf ids)"
    ids = _route_leaf_ids(trees, leaf_nodes, nl, x, th)
    base = float(str(json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]).strip("[]").split(",")[0])
    assert base > 0.0, "survival:aft's base_score is a time"
    leaf = _custom_route_values(p, ids, nl, dtrain, math.log(base), _aft_objective(p, lo, hi, w))
    for t in range(len(trees)):
        sc, bw = trees[t]["split_conditions"], trees[t]["base_weights"]
        for j, nid in enumerate(leaf_nodes[t].tolist()):
            sc[nid] = bw[nid] = float(leaf[t, j])
    out = _RouteBooster(params, [dtrain], bytearray(json.dumps(doc).encode()))
    _route_names(out, dtrain)
    first_read_log.append({"n": n, "m": m, "kept": m, "route": "stock sample (survival:aft)", "hist": None})
    LOG.info("shira.xgboost: fitted the first-read way (SHIRA's default) by the general route: survival:aft; tree shapes by "
             "stock XGBoost on %d stride-sample rows, leaf values on all %d rows from XGBoost's AFT gradients (not stock's trees); "
             "prediction is stock XGBoost's; SHIRA_XGB_MODE=ordinary runs stock", m, n)
    return out, None


def train(params: Any, dtrain: Any, num_boost_round: int = 10, *, evals: Any = None, obj: Any = None,
          maximize: Any = None, early_stopping_rounds: Any = None, evals_result: Any = None,
          verbose_eval: Any = True, xgb_model: Any = None, callbacks: Any = None,
          custom_metric: Any = None) -> Any:
    global last_backend
    assert dtrain is not None and params is not None
    kw = {"evals": evals, "obj": obj, "maximize": maximize, "early_stopping_rounds": early_stopping_rounds,
          "evals_result": evals_result, "verbose_eval": verbose_eval, "xgb_model": xgb_model,
          "callbacks": callbacks, "custom_metric": custom_metric}
    ev = list(evals) if evals else []
    if version_reason() is not None:
        return _fallback(str(version_reason()), (params, dtrain, num_boost_round), kw)
    if obj is not None and _custom_engine_reason(params, dtrain, int(num_boost_round), ev, kw) is None:
        sm_c, why_c = _custom_engine_fit(params, dtrain, int(num_boost_round), obj)
        if sm_c is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return _final_booster(params, sm_c, dtrain, ev).reset()
        LOG.info("shira.xgboost: the custom objective does not go to the engine: %s", why_c)
    if _rank_engine_reason(params, dtrain, int(num_boost_round), ev, kw) is None:
        try:
            sm_r, why_r = _rank_engine_fit(params, dtrain, int(num_boost_round))
        except (RuntimeError, AssertionError, ValueError) as e:
            sm_r, why_r = None, f"engine: {type(e).__name__}: {e}"
        if sm_r is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return _final_booster(params, sm_r, dtrain, ev).reset()
        LOG.info("shira.xgboost: the ranking call does not go to the engine: %s", why_r)
    if _start_reason(params, dtrain, int(num_boost_round), ev, kw) is None:
        bst, why = _guarded_route(_start_fit, params, dtrain, int(num_boost_round), ev, kw)
        if bst is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return bst
        LOG.info("shira.xgboost: the per-row start margin route does not apply: %s", why)
    if _aft_reason(params, dtrain, int(num_boost_round), ev, kw) is None:
        bst, why = _guarded_route(_aft_fit, params, dtrain, int(num_boost_round), kw)
        if bst is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return bst
        LOG.info("shira.xgboost: the survival:aft route does not apply: %s", why)
    reason = _call_reason(params, dtrain, num_boost_round, ev, kw)
    if reason and _mot_reason(params, dtrain, int(num_boost_round), ev, kw) is None:
        bst, why = _guarded_route(_mot_fit, params, dtrain, int(num_boost_round), kw)
        if bst is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return bst
        reason = f"{reason} (the first read not applied: {why})"
    if reason == MULTI_NO_BASE:
        bst, why = _guarded_route(_multi_intercept_fit, params, dtrain, int(num_boost_round), kw)
        if bst is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return bst
        reason = f"{reason} (the first read not applied: {why})"
    if reason and reason != MULTI_NO_BASE and _sample_route_reason(params, dtrain, int(num_boost_round), ev, kw) is None:
        bst, why = _guarded_route(_stock_sample_fit, params, dtrain, int(num_boost_round), ev, kw, reason)
        if bst is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return bst
        reason = f"{reason} (the first read not applied: {why})"
    adaptive = normalize_params(params)[0].get("objective") in ADAPTIVE
    adaptive = adaptive and not _adaptive_engine_on(normalize_params(params)[0], int(dtrain.num_row()), int(dtrain.num_col()))
    if not reason and adaptive and _sample_route_reason(params, dtrain, int(num_boost_round), ev, kw) is None:
        bst, why = _guarded_route(_stock_sample_fit, params, dtrain, int(num_boost_round), ev, kw,
                                  "reg:absoluteerror / reg:quantileerror (XGBoost's leaf quantile)")
        if bst is not None:
            last_backend = "shira"
            backend_history.append(last_backend)
            return bst
        LOG.info("shira.xgboost: the general route does not apply to this call: %s", why)
    sm = None
    if not reason:
        sm, reason, hist = _shira_model(params, dtrain, int(num_boost_round), ev, kw)
    if reason:
        return _fallback(reason, (params, dtrain, num_boost_round), kw)
    assert sm is not None
    if ev or kw.get("early_stopping_rounds") or kw.get("callbacks"):
        bst = _replay(params, sm, dtrain, ev, kw, hist)
    else:
        bst = _final_booster(params, sm, dtrain, ev).reset()
    last_backend = "shira"
    backend_history.append(last_backend)
    return bst


class _ShiraFold:

    def __init__(self, rep: _ReplayBooster, watch: list) -> None:
        self.bst, self.watchlist = rep, watch
        rep.__dict__["_shira_cv_locked"] = True

    def update(self, iteration: int, fobj: Any) -> None:
        assert fobj is None and iteration >= 0
        self.bst.advance(iteration)

    def eval(self, iteration: int, feval: Any, output_margin: bool) -> str:
        return self.bst.eval_set(self.watchlist, iteration, feval, output_margin)

    def num_boosted_rounds(self) -> int:
        return 0


def _cv_categorical(dtrain: Any) -> bool:
    return isinstance(dtrain, DMatrix) and dtrain.__dict__.get("_sx_src") is not None and any(
        t == "c" for t in (dtrain.feature_types or []))


def _cv_booster_route(p: dict[str, Any], dtrain: Any) -> bool:
    return (_cv_categorical(dtrain) or int(p.get("num_parallel_tree", 1)) > 1
            or _cv_pred_route(p))


def _cv_pred_route(p: dict[str, Any]) -> bool:
    return (p.get("multi_strategy") == "multi_output_tree" or p.get("booster") == "dart"
            or p.get("objective") == "survival:aft")


def _cv_booster_reason(p: dict[str, Any], dtrain: Any, rounds: int, kw: dict[str, Any]) -> str | None:
    if not first_read():
        return ("categorical feature types / num_parallel_tree > 1 / multi_output_tree / booster dart under "
                "SHIRA_XGB_MODE=ordinary: no exact engine takes them; the default runs them on the first read per fold (not stock's trees)")
    if any(kw.get(k) is not None for k in ("fpreproc", "obj")) or int(p.get("num_class", 1)) > 1:
        return ("categorical / parallel-tree / multi-output / dart cv with fpreproc, a custom objective or more than "
                "one class")
    if any(type(cb) is _cb.LearningRateScheduler for cb in kw.get("callbacks") or []):
        return "categorical / parallel-tree / multi-output / dart cv with a LearningRateScheduler"
    if dtrain.get_float_info("base_margin").size > 0 or len(dtrain.get_uint_info("group_ptr")) > 1:
        return "categorical / parallel-tree / multi-output / dart cv with a base margin or query groups"
    if _cv_pred_route(p) and (not isinstance(dtrain, DMatrix) or getattr(dtrain, "_shira_x", None) is None):
        return "multi-output / dart cv on a DMatrix that is not SHIRA's with its raw matrix"
    nfold = int(kw.get("nfold", 3))
    if not 2 <= nfold <= dtrain.num_row() or not 1 <= int(rounds) <= MAX_ROUNDS:
        return f"cv nfold {nfold} / rounds {rounds}"
    return None


def _cv_booster_data(dtrain: Any, rows: np.ndarray, nthread: int) -> Any:
    if not _cv_categorical(dtrain):
        return _cv_fold_data(dtrain, rows, nthread)
    data, _, kwargs = dtrain.__dict__["_sx_src"]
    sub = data.iloc[rows] if hasattr(data, "iloc") else np.asarray(data)[rows]
    keep = {k: v for k, v in kwargs.items() if k in ("enable_categorical", "feature_types", "feature_names", "missing",
                                                     "max_cat_to_onehot")}
    y, w = dtrain.get_label(), dtrain.get_weight()
    return DMatrix(sub, label=y[rows], weight=w[rows] if w.size else None, nthread=nthread or None, **keep)


def _cv_booster_fold(plst: list, bst: Any, rounds: int, dtr: Any, dte: Any) -> Any:
    raw = json.loads(bytes(bst.save_raw("json")))
    trees = raw["learner"]["gradient_booster"]["model"]["trees"]
    T = len(trees)
    assert T % int(rounds) == 0, "whole rounds"
    width = max(len(t["split_conditions"]) for t in trees)
    leaf = np.zeros((T, width), dtype=F32)
    for t, tr in enumerate(trees):
        leaf[t, :len(tr["split_conditions"])] = np.asarray(tr["split_conditions"], dtype=F32)
    cfg = json.loads(bst.save_config())
    obj_json = cfg["learner"]["objective"]
    base = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    sm = SimpleNamespace(objective=obj_json["name"], base_score=base, num_class=1, objective_json=obj_json)
    margin = (F32(math.log(base)) if obj_json["name"] == "survival:aft"
              else export.xgb_base_margin(obj_json["name"], base, obj_json, 1))
    watch, sets = [(dtr, "train"), (dte, "test")], []
    own_pass = obj_json["name"] == "survival:aft" and all(isinstance(getattr(d, "_shira_x", None), np.ndarray) for d, _ in watch)
    if own_pass:
        leaf_nodes, nl, _, _, _ = _json_tables(trees, None)
        leaf = np.zeros((T, int(nl.max())), dtype=F32)
        for t in range(T):
            leaf[t, : nl[t]] = np.asarray(trees[t]["split_conditions"], dtype=F32)[leaf_nodes[t]]
        th = _call_threads(normalize_params(dict(plst))[0])
    for d, _ in watch:
        ids = (_route_leaf_ids(trees, leaf_nodes, nl, d._shira_x, th) if own_pass else
               np.asarray(bst.predict(d, pred_leaf=True)).reshape(d.num_row(), -1).astype(np.int32).T.copy())
        assert ids.shape[0] == T
        w = d.get_weight()
        lab = d.get_label()
        sets.append(SimpleNamespace(ids=ids, rows=None, leaf=leaf, base=margin, y=lab if lab.size else None,
                                    w=w if w.size else None, n=int(d.num_row()),
                                    lo=d.get_float_info("label_lower_bound"), hi=d.get_float_info("label_upper_bound")))
    return _ShiraFold(_TreeRoundReplay(plst, sm, sets, T // int(rounds)), watch)


def _cv_booster_folds(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any]) -> tuple[list | None, str | None]:
    plst, p = _cv_params(params, kw)
    nt, folds = int(p.get("nthread", 0)), []
    tp = {k: v for k, v in (params.items() if isinstance(params, dict) else params) if k != "eval_metric"}
    for tr, te in _cv_folds(dtrain, int(kw.get("nfold", 3)), int(kw.get("seed", 0)), kw.get("shuffle", True),
                            bool(kw.get("stratified")), kw.get("folds")):
        dtr, dte = _cv_booster_data(dtrain, tr, nt), _cv_booster_data(dtrain, te, nt)
        bst = train(dict(tp), dtr, int(rounds), verbose_eval=False)
        if last_backend != "shira":
            return None, "cv fold: xgb.train sent the fold to stock (its reason is in the log)"
        if _cv_pred_route(p) and p.get("objective") != "survival:aft":
            watch = [(dtr, "train"), (dte, "test")]
            folds.append(_ShiraFold(_PredReplay(plst, bst, watch, int(rounds), bst.__dict__.get("_shira_dart_w")), watch))
            continue
        folds.append(_cv_booster_fold(plst, bst, int(rounds), dtr, dte))
    return folds, None


def _cv_reason(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any]) -> str | None:
    rank = str(normalize_params(params)[0].get("objective", "")) in RANK_ENGINE_OBJ
    pn0 = normalize_params(params)[0]
    if pn0.get("booster") == "gblinear":
        for cb in kw.get("callbacks") or []:
            if not isinstance(cb, _cb.TrainingCallback):
                return f"cv callback {type(cb).__name__} is not an xgboost.callback.TrainingCallback (stock raises too)"
        return _gblinear_reason(pn0, dtrain, rounds, kw)
    reason = (_rank_engine_reason(params, dtrain, rounds, [], {}, force=True) if rank
              else None if _cv_booster_route(pn0, dtrain) else why_not(params))
    if reason == MULTI_NO_BASE:
        reason = None
    if reason:
        return reason
    for cb in kw.get("callbacks") or []:
        if not isinstance(cb, _cb.TrainingCallback):
            return f"cv callback {type(cb).__name__} is not an xgboost.callback.TrainingCallback (stock raises too)"
    pn = normalize_params(params)[0]
    if _cv_booster_route(pn, dtrain):
        return _cv_booster_reason(pn, dtrain, rounds, kw)
    if not isinstance(dtrain, DMatrix) or isinstance(dtrain, _core.QuantileDMatrix) or dtrain._shira_x is None:
        return "cv dtrain is not a SHIRA DMatrix with its raw matrix"
    if dtrain.get_float_info("base_margin").size > 0 and (kw.get("obj") is not None or pn.get("objective") in ADAPTIVE):
        return "cv with a base margin and a custom objective or an adaptive-leaf objective"
    if len(dtrain.get_uint_info("group_ptr")) > 1 and (kw.get("stratified") or kw.get("folds") is not None):
        return "cv with query groups and stratified or given folds (stock raises too: its fold slices drop the groups)"
    if any(t == "c" for t in (dtrain.feature_types or [])):
        return "cv with categorical feature types"
    nfold = int(kw.get("nfold", 3))
    if not 2 <= nfold <= dtrain.num_row() or not 1 <= int(rounds) <= MAX_ROUNDS:
        return f"cv nfold {nfold} / rounds {rounds}"
    return None


_SAMPLING_KEYS = ("subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode")


def cv_form() -> str:
    v = os.environ.get("SHIRA_CV_FORM", "").strip().lower()
    assert v in ("", "matrix_scored", "perfold"), f"SHIRA_CV_FORM={v!r}: '', matrix_scored or perfold"
    return v


def _cv_sampling_reason(params: Any) -> str | None:
    pn = normalize_params(params)[0]
    if any(float(pn.get(k, 1.0)) < 1.0 for k in _SAMPLING_KEYS):
        return ("row / column sampling on exact folds (stock's folds draw from one process-wide RNG, interleaved across "
                "the folds each round: not reproducible fold by fold; the first read's matrix form takes sampling from its size)")
    return None


def _cv_folds(dtrain: Any, nfold: int, seed: int, shuffle: bool, stratified: bool = False,
              folds: Any = None) -> list[tuple[np.ndarray, np.ndarray]]:
    np.random.seed(seed)
    n = dtrain.num_row()
    gp = np.asarray(dtrain.get_uint_info("group_ptr"), dtype=np.int64)
    if gp.size > 1 and not stratified and folds is None:
        gidx = np.random.permutation(gp.size - 1) if shuffle is True else np.arange(gp.size - 1)
        out_g = np.array_split(gidx, nfold)
        in_g = [np.concatenate([out_g[i] for i in range(nfold) if k != i]) for k in range(nfold)]
        rows = lambda gs: np.concatenate([np.arange(gp[g], gp[g + 1]) for g in gs])
        pairs = [(rows(a), rows(b)) for a, b in zip(in_g, out_g)]
        assert all(a.size + b.size == n for a, b in pairs)
        return pairs
    if folds is not None or stratified:
        lab = dtrain.get_label()
        if folds is not None:
            try:
                pairs = [(np.asarray(x[0]), np.asarray(x[1])) for x in folds]
            except TypeError:
                pairs = [(np.asarray(a), np.asarray(b)) for a, b in folds.split(X=lab, y=lab)]
        else:
            from xgboost.compat import XGBStratifiedKFold
            if XGBStratifiedKFold is None:
                raise NotImplementedError("shira.xgboost.cv: stratified cv needs scikit-learn (as stock)")
            pairs = [(np.asarray(a), np.asarray(b)) for a, b in
                     XGBStratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed).split(X=lab, y=lab)]
        assert len(pairs) >= 1
        return pairs
    idx = np.random.permutation(n) if shuffle is True else np.arange(n)
    out_idset = np.array_split(idx, nfold)
    in_idset = [np.concatenate([out_idset[i] for i in range(nfold) if k != i]) for k in range(nfold)]
    assert len(in_idset) == nfold and all(a.size + b.size == n for a, b in zip(in_idset, out_idset))
    return list(zip(in_idset, out_idset))


def _cv_ungrouped(d: Any, nthread: int) -> Any:
    w = d.get_weight()
    return DMatrix(d._shira_x, label=d.get_label(), weight=w if w.size else None, nthread=nthread or None)


def _cv_fold_data(dtrain: Any, rows: np.ndarray, nthread: int) -> Any:
    y, w = dtrain.get_label(), dtrain.get_weight()
    if y.size != dtrain.num_row() and y.size % max(1, dtrain.num_row()) == 0:
        y = y.reshape(dtrain.num_row(), -1)
    bm = dtrain.get_float_info("base_margin")
    C = bm.size // max(1, dtrain.num_row())
    surv = {k: dtrain.get_float_info(k)[rows] for k in ("label_lower_bound", "label_upper_bound")
            if dtrain.get_float_info(k).size}
    d = DMatrix(dtrain._shira_x[rows], label=y[rows] if y.size else None, weight=w[rows] if w.size else None,
                nthread=nthread or None, base_margin=bm.reshape(-1, C)[rows].reshape(-1) if bm.size else None, **surv)
    gp = np.asarray(dtrain.get_uint_info("group_ptr"), dtype=np.int64)
    if gp.size > 1:
        gid = np.searchsorted(gp, np.asarray(rows, dtype=np.int64), side="right") - 1
        cut = np.flatnonzero(np.diff(gid)) + 1
        d.set_group(np.diff(np.concatenate([[0], cut, [gid.size]])))
    return d


cv_scored_log: list[dict] = []
last_cv_form: str | None = None


class _LeafReplay(Booster):

    def __init__(self, params: Any, sm: export.ShiraModel, sets: list) -> None:
        raw = export.zero_tree_json(sm.objective, sm.base_score, 1, sm.num_class, sm.objective_json)
        Booster.__init__(self, params, None, bytearray(raw))
        self._raw, self._params = bytes(raw), params
        nt = int(normalize_params(params)[0].get("nthread", 0)) or None
        self._sets = []
        for s in sets:
            n = int(s.y.size) if s.y is not None else int(s.n)
            if np.ndim(s.base) == 2:
                m = np.array(s.base, dtype=F32)
                assert m.shape == (n, int(sm.num_class))
            else:
                m = (np.full(n, F32(s.base), dtype=F32) if np.ndim(s.base) == 0
                     else np.array(s.base, dtype=F32).reshape(n))
            ev = _core.DMatrix(np.zeros((n, 1), F32), label=s.y, weight=s.w, base_margin=m, nthread=nt,
                               group=getattr(s, "g", None))
            for key, v in (("label_lower_bound", getattr(s, "lo", None)), ("label_upper_bound", getattr(s, "hi", None))):
                if v is not None and np.size(v):
                    ev.set_float_info(key, v)
            self._sets.append(SimpleNamespace(ids=s.ids, rows=s.rows, leaf=s.leaf, m=m, ev=ev))
            assert s.leaf.dtype == F32 and s.ids.shape[0] == s.leaf.shape[0]

    def num_boosted_rounds(self) -> int:
        return 0

    def advance(self, i: int) -> None:
        for st in self._sets:
            if st.m.ndim == 2:
                C = st.m.shape[1]
                for c in range(C):
                    t = i * C + c
                    idx = st.ids[t] if st.rows is None else st.ids[t][st.rows]
                    st.m[:, c] += st.leaf[t][idx]
                continue
            idx = st.ids[i] if st.rows is None else st.ids[i][st.rows]
            st.m = st.m + st.leaf[i][idx]

    def eval_set(self, evals: Any, iteration: int = 0, feval: Any = None, output_margin: bool = True) -> str:
        assert len(evals) == len(self._sets)
        mats = []
        for st, (_, name) in zip(self._sets, evals):
            st.ev.set_base_margin(st.m)
            mats.append((st.ev, name))
        return _core.Booster(self._params, None, bytearray(self._raw)).eval_set(mats, iteration, feval, output_margin)


class _TreeRoundReplay(_LeafReplay):

    def __init__(self, params: Any, sm: Any, sets: list, per_round: int) -> None:
        super().__init__(params, sm, sets)
        self._k = int(per_round)
        assert self._k >= 1

    def advance(self, i: int) -> None:
        for t in range(i * self._k, (i + 1) * self._k):
            for st in self._sets:
                st.m = st.m + st.leaf[t][st.ids[t]]


class _PredReplay(Booster):

    def __init__(self, params: Any, bst: Any, watch: list, rounds: int, dart_w: list | None = None) -> None:
        doc = json.loads(bytes(bst.save_raw("json")))
        gb = doc["learner"]["gradient_booster"]
        model = gb.get("gbtree", gb)["model"]
        assert len(model["trees"]) >= 1 and len(model["trees"]) % int(rounds) == 0, "whole rounds"
        model["trees"], model["tree_info"] = [], []
        model["gbtree_model_param"]["num_trees"] = "0"
        if "iteration_indptr" in model:
            model["iteration_indptr"] = [0]
        if "weight_drop" in gb:
            gb["weight_drop"] = []
        self._raw = json.dumps(doc).encode()
        Booster.__init__(self, params, None, bytearray(self._raw))
        self._params, self._bst = params, bst
        self._dart = None if dart_w is None else [np.asarray(v, dtype=np.float64) for v in dart_w]
        assert self._dart is None or len(self._dart) == int(rounds)
        nt = int(normalize_params(params)[0].get("nthread", 0)) or None
        self._rounds = int(rounds)
        self._sets = []
        for d, _ in watch:
            n = int(d.num_row())
            base = self._base_margin(d, n)
            lab, w = d.get_label(), d.get_weight()
            lab = lab.reshape(n, -1) if (lab.size != n and lab.size) else lab
            ev = _core.DMatrix(np.zeros((n, 1), F32), label=lab if lab.size else None, weight=w if w.size else None,
                               nthread=nt)
            for key in ("label_lower_bound", "label_upper_bound"):
                v = d.get_float_info(key)
                if v.size:
                    ev.set_float_info(key, v)
            gp = np.asarray(d.get_uint_info("group_ptr"), dtype=np.int64)
            if gp.size > 1:
                ev.set_group(np.diff(gp))
            assert base.ndim == 2 and base.shape[0] == 1
            self._sets.append(SimpleNamespace(d=d, base=base, m=np.tile(base, (n, 1)), ev=ev))

    def num_boosted_rounds(self) -> int:
        return 0

    def _base_margin(self, d: Any, n: int) -> np.ndarray:
        few = d.slice(list(range(min(n, 64))))
        mg = lambda a, b: np.asarray(_core.Booster.predict(self._bst, few, output_margin=True, iteration_range=(a, b),
                                                           strict_shape=True), dtype=np.float64)
        if self._rounds < 2:
            return np.zeros((1, mg(0, 1).shape[1]))
        base = np.median(mg(0, 1) + mg(1, 2) - mg(0, 2), axis=0, keepdims=True)
        assert base.ndim == 2 and bool(np.all(np.isfinite(base)))
        return base

    def _term(self, st: Any, t: int) -> np.ndarray:
        pr = _core.Booster.predict(self._bst, st.d, output_margin=True, iteration_range=(t, t + 1), strict_shape=True)
        out = np.asarray(pr, dtype=np.float64) - st.base
        assert out.shape == st.m.shape
        return out

    def advance(self, i: int) -> None:
        for st in self._sets:
            if self._rounds < 2:
                st.m = self._term(st, i)
                continue
            if self._dart is None:
                st.m = st.m + self._term(st, i)
                continue
            now, fin = self._dart[i], self._dart[-1]
            prev = self._dart[i - 1] if i else np.zeros(0)
            for t in range(i):
                if now[t] != prev[t]:
                    st.m = st.m + (now[t] - prev[t]) / fin[t] * self._term(st, t)
            st.m = st.m + now[i] / fin[i] * self._term(st, i)

    def eval_set(self, evals: Any, iteration: int = 0, feval: Any = None, output_margin: bool = True) -> str:
        assert len(evals) == len(self._sets)
        mats = []
        for st, (_, name) in zip(self._sets, evals):
            st.ev.set_base_margin(st.m.astype(F32))
            mats.append((st.ev, name))
        return _core.Booster(self._params, None, bytearray(self._raw)).eval_set(mats, iteration, feval, output_margin)


def _leaf_values_of(trees: Any) -> np.ndarray:
    T = len(trees)
    counts = [int(trees.count[t]) for t in range(T)]
    isl = [np.flatnonzero(trees.nodes[t, :counts[t], 0] < 0) for t in range(T)]
    out = np.zeros((T, max(1, max(a.size for a in isl))), dtype=F32)
    for t in range(T):
        out[t, :isl[t].size] = trees.leaf[t, isl[t]]
    return out


def _cv_params(params: Any, kw: dict[str, Any], raw: bool = False) -> tuple[list, Any]:
    prm = dict(params)
    if isinstance(params, list):
        mets = [v for k, v in params if k == "eval_metric"]
        if "eval_metric" in prm:
            prm["eval_metric"] = mets
    metrics = kw.get("metrics") or ()
    metrics = [metrics] if isinstance(metrics, str) else list(metrics)
    if not metrics and "eval_metric" in prm:
        metrics = prm["eval_metric"] if isinstance(prm["eval_metric"], list) else [prm["eval_metric"]]
    prm.pop("eval_metric", None)
    if raw:
        return metrics, prm
    return list(prm.items()) + [("eval_metric", m) for m in metrics], normalize_params(prm)[0]


def _cv_fold_sets(params: Any, dtrain: Any, kw: dict[str, Any]) -> tuple[list, str | None]:
    metrics, prm = _cv_params(params, kw, raw=True)
    nt, fp, out = int(normalize_params(prm)[0].get("nthread", 0)), kw.get("fpreproc"), []
    for tr, te in _cv_folds(dtrain, int(kw.get("nfold", 3)), int(kw.get("seed", 0)), kw.get("shuffle", True),
                            bool(kw.get("stratified")), kw.get("folds")):
        dtr, dte, tp = _cv_fold_data(dtrain, tr, nt), _cv_fold_data(dtrain, te, nt), prm
        if fp is not None:
            dtr, dte, tp = fp(dtr, dte, prm.copy())
            (dtr, why_tr), (dte, why_te) = _cv_fold_own(dtr, nt), _cv_fold_own(dte, nt)
            for d, why in ((dtr, why_tr), (dte, why_te)):
                why = why or _cv_fold_reason(d)
                if why:
                    return [], f"cv fpreproc returned a fold {why}"
            if not isinstance(tp, dict):
                return [], f"cv fpreproc returned params of type {type(tp).__name__}, not a dict"
            why = why_not(tp)
            if why:
                return [], f"cv fpreproc returned fold {len(out)}'s params with {why}"
        out.append((list(tp.items()) + [("eval_metric", m) for m in metrics], normalize_params(tp)[0], dtr, dte))
    return out, None


def _cv_fold_own(d: Any, nthread: int) -> tuple[Any, str | None]:
    if isinstance(d, DMatrix) and not isinstance(d, _core.QuantileDMatrix) and d._shira_x is not None \
            and d.feature_names is None and d.feature_types is None:
        return d, None
    if not isinstance(d, _core.DMatrix) or isinstance(d, _core.QuantileDMatrix):
        return d, "that is not a DMatrix that keeps its entries (a QuantileDMatrix, or another type)"
    if d.feature_types is not None and "c" in d.feature_types:
        return d, "DMatrix with categorical feature types"
    gp = np.asarray(d.get_uint_info("group_ptr"), dtype=np.int64)
    csr = d.get_data()
    n, f = int(d.num_row()), int(d.num_col())
    assert csr.shape == (n, f) and csr.indptr.size == n + 1
    x = np.full((n, f), np.nan, dtype=F32)
    x[np.repeat(np.arange(n, dtype=np.int64), np.diff(csr.indptr)), csr.indices] = csr.data.astype(F32)
    lab, w = d.get_label(), d.get_weight()
    lo, hi = d.get_float_info("label_lower_bound"), d.get_float_info("label_upper_bound")
    extra = {"label_lower_bound": lo, "label_upper_bound": hi} if lo.size or hi.size else {}
    bm = d.get_float_info("base_margin")
    if bm.size:
        extra["base_margin"] = bm
    out = DMatrix(x, label=lab if lab.size else None, weight=w if w.size else None, nthread=nthread or None, **extra)
    if gp.size > 1:
        out.set_group(np.diff(gp))
    assert out._shira_x is not None and out.num_row() == n
    return out, None


def _cv_fold_reason(d: Any) -> str | None:
    if not isinstance(d, DMatrix) or isinstance(d, _core.QuantileDMatrix) or d._shira_x is None:
        return "DMatrix that is not a SHIRA DMatrix with its raw matrix (build it with shira.xgboost.DMatrix from numpy)"
    if len(d.get_uint_info("group_ptr")) > 1:
        return "DMatrix with query groups"
    if d.feature_names is not None or d.feature_types is not None:
        return "DMatrix with feature names / types"
    return None


def _cv_matrix(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any], score_sample: bool = False) -> list | None:
    plst, p = _cv_params(params, kw)
    if len(dtrain.get_uint_info("group_ptr")) > 1:
        return None
    if _spw(p) != 1.0 and p.get("objective", "reg:squarederror") in SPW_OBJ:
        return None
    if str(p.get("objective", "")).startswith("multi:"):
        return None
    run, why = _prepare(p, dtrain)
    if why or run is None or not first_read() or run.sm.num_class != 1 or run.sm.objective in ADAPTIVE:
        return None
    if _first_read_reason(p, run, 0) is not None or _mono_vector(p, int(run.codes.shape[1])) is not None:
        return None
    n, th = int(run.codes.shape[0]), _call_threads(p)
    fixed = _engine_rows(p, int(run.codes.shape[1]), n)
    assert fixed is not None and fixed < n
    rows = (n - 1) * np.arange(fixed, dtype=np.int64) // max(1, fixed - 1)
    ws = None if run.w is None else np.ascontiguousarray(run.w[rows])
    q = _engine_params(p, run.sm.base_score)
    trees, n_rounds, _ = _fit.engine_es(q, np.ascontiguousarray(run.codes[rows]), run.sm.ip,
                                        np.ascontiguousarray(run.y[rows]), int(rounds), ws, None, 0)
    assert n_rounds == int(rounds) and isinstance(trees, export.LazyTrees)
    ids, nl = _leaf_pass(p, run.codes, trees, th)
    folds, obj = [], run.sm.objective
    w_all = np.ones(n, dtype=F32) if run.w is None else np.asarray(run.w, dtype=F32)
    for tr, te in _cv_folds(dtrain, int(kw.get("nfold", 3)), int(kw.get("seed", 0)), kw.get("shuffle", True),
                            bool(kw.get("stratified")), kw.get("folds")):
        base = _fit.intercept(obj, run.y[tr], None if run.w is None else run.w[tr], th, float(p.get("huber_slope", 1.0)))
        if base is None:
            return None
        margin = float(_fit.engine_base_margin(obj, float(base)))
        wk = np.zeros(n, dtype=F32)
        wk[tr] = w_all[tr]
        runk = SimpleNamespace(sm=SimpleNamespace(objective=obj, num_class=1, base_margin=margin), y=run.y, w=wk)
        leaf = np.ascontiguousarray(_first_read_values(p, runk, ids, nl, None), dtype=F32)
        tes = te if score_sample else np.setdiff1d(np.asarray(te, dtype=np.int64), rows, assume_unique=False)
        if tes.size == 0:
            return None
        cv_scored_log.append({"fold": len(folds), "scored": int(tes.size), "held_out": int(np.asarray(te).size)})
        sets = [SimpleNamespace(ids=ids, rows=r, leaf=leaf, base=margin, y=run.y[r],
                                w=None if run.w is None else run.w[r]) for r in (tr, tes)]
        folds.append(_ShiraFold(_LeafReplay(plst, run.sm, sets), [(None, "train"), (None, "test")]))
    return folds


def _cv_exact_fold(plst: list, sm: export.ShiraModel, dtr: Any, dte: Any, nt: int, starts0: list | None = None) -> Any:
    from shira_cpp import product_cpp

    watch = [(dtr, "train"), (dte, "test")]
    lt = sm.trees
    starts = starts0 or [_start_margins(d, None, sm.num_class) for d in (dtr, dte)]
    if not isinstance(lt, export.LazyTrees) or any(sm.below_min_split(d._shira_x) for d in (dtr, dte)):
        st = None if all(v is None for v in starts) else SimpleNamespace(old=None, r0=0, evals=starts)
        return _ShiraFold(_ReplayBooster(plst, sm, watch, st), watch)
    C = int(sm.num_class)
    leaf = _leaf_values_of(lt)
    sets = []
    for d in (dtr, dte):
        ids, _ = product_cpp.leaf_ids(sm.codes(d._shira_x, clamp=False), lt.nodes, lt.count, nthread=max(1, nt))
        w = d.get_weight()
        gq = np.asarray(d.get_uint_info("group_ptr"), dtype=np.int64)
        b0 = starts[len(sets)]
        base = (F32(sm.base_margin) if b0 is None else b0[:, 0].copy()) if C == 1 else (
            np.full((d.num_row(), C), F32(sm.base_margin), dtype=F32) if b0 is None else np.array(b0, dtype=F32))
        sets.append(SimpleNamespace(ids=ids, rows=None, leaf=leaf, y=d.get_label(), base=base,
                                    w=w if w.size else None, g=np.diff(gq) if gq.size > 1 else None))
    return _ShiraFold(_LeafReplay(plst, sm, sets), watch)


def _cv_obj(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any],
            etas: list | None = None) -> tuple[list | None, str | None]:
    sets, why = _cv_fold_sets(params, dtrain, kw)
    if why:
        return None, why
    obj, folds = kw["obj"], []
    for plst, p, dtr, dte in sets:
        nt = int(p.get("nthread", 0))
        if etas is not None:
            p = dict(p, _shira_etas=etas)
        why = _custom_engine_reason(p, dtr, int(rounds), [], {"obj": obj})
        sm, why = (None, why) if why else _custom_engine_fit(p, dtr, int(rounds), obj, all_rows=etas is not None)
        if why or sm is None:
            return None, f"cv fold with a custom objective: {why}"
        folds.append(_cv_exact_fold(plst, sm, dtr, dte, nt))
    return folds, None


def _cv_etas(p: dict[str, Any], kw: dict[str, Any], rounds: int) -> tuple[list | None, str | None]:
    sched = [cb for cb in kw.get("callbacks") or [] if type(cb) is _cb.LearningRateScheduler]
    if not sched:
        return None, None
    calls = [[cb.learning_rates(e) for cb in sched] for e in range(int(rounds))]
    rates = [c[-1] for c in calls]
    etas = [F32(str(_pget(p, ("eta", "learning_rate"), 0.3)))] + [F32(str(v)) for v in rates[:-1]]
    assert len(etas) == int(rounds) and all(e > 0 for e in etas)
    return etas, None


def _cv_shira(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any],
              etas: list | None = None) -> tuple[list | None, str | None]:
    sets, why = _cv_fold_sets(params, dtrain, kw)
    if why:
        return None, why
    folds = []
    for plst, p, dtr, dte in sets:
        nt, starts = int(p.get("nthread", 0)), None
        why = None if (first_read() and etas is None) else _cv_sampling_reason(p)
        if why:
            return None, why
        if etas is not None:
            p = dict(p, _shira_etas=etas)
        if str(p.get("objective", "")) in RANK_ENGINE_OBJ:
            reason = _rank_engine_reason(p, dtr, int(rounds), [], {}, force=True)
            sm, reason = (None, reason) if reason else _rank_engine_fit(p, dtr, int(rounds))
        else:
            dfit = dtr if len(dtr.get_uint_info("group_ptr")) <= 1 else _cv_ungrouped(dtr, nt)
            m0 = _start_margins(dfit, None, int(p.get("num_class", 1)))
            if m0 is None and str(p.get("objective", "")).startswith("multi:") and "base_score" not in p:
                starts = _cv_multi_starts(p, dtr, dte, nt)
                m0 = starts[0]
                p = dict(p, base_score=0.5)
            if m0 is not None:
                p = dict(p, _shira_start_margin=m0)
            sm, _, reason, _ = _train_shira(p, dfit, int(rounds), [], {}, allow_fr=etas is None and m0 is None)
        if reason or sm is None:
            return None, f"cv fold: {reason}"
        folds.append(_cv_exact_fold(plst, sm, dtr, dte, nt, starts))
    return folds, None


def _cv_multi_starts(p: dict[str, Any], dtr: Any, dte: Any, nt: int) -> list:
    C, th = int(p["num_class"]), max(1, nt or _call_threads(p))
    w = dtr.get_weight()
    base = _stock_intercept(p, np.asarray(dtr.get_label(), dtype=F32), np.asarray(w, dtype=F32) if w.size else None, th)
    one = _core.DMatrix(np.zeros((1, 1), F32), label=np.zeros(1, F32))
    z = _STOCK["train"]({"objective": p["objective"], "num_class": C, "base_score": base.tolist(), "nthread": th}, one, 0)
    m = np.asarray(z.predict(one, output_margin=True), dtype=F32).reshape(C)
    assert m.shape == (C,) and bool(np.all(np.isfinite(m)))
    return [np.ascontiguousarray(np.tile(m, (d.num_row(), 1))) for d in (dtr, dte)]


GBL_OBJ = ("reg:squarederror", "reg:logistic", "binary:logistic")
_GBL_ONLY = ("booster", "updater", "feature_selector", "top_k", "tolerance")


def _gblinear_reason(p: dict[str, Any], dtrain: Any, rounds: int, kw: dict[str, Any]) -> str | None:
    obj = str(p.get("objective", "reg:squarederror"))
    if obj not in GBL_OBJ:
        return f"booster gblinear with objective {obj!r} (the linear folds take {', '.join(GBL_OBJ)})"
    if str(p.get("feature_selector", "cyclic")) != "cyclic" or str(p.get("updater", "shotgun")) not in ("shotgun", "coord_descent"):
        return "booster gblinear with a feature_selector other than cyclic or another updater"
    if float(p.get("tolerance", 0.0)) != 0.0 or int(p.get("num_class", 1)) > 1 or _spw(p) != 1.0:
        return "booster gblinear with tolerance, more than one class or scale_pos_weight"
    if any(kw.get(k) is not None for k in ("fpreproc", "obj")):
        return "booster gblinear cv with fpreproc or a custom objective"
    if any(type(cb) is _cb.LearningRateScheduler for cb in kw.get("callbacks") or []):
        return "booster gblinear cv with a LearningRateScheduler"
    if not isinstance(dtrain, DMatrix) or isinstance(dtrain, _core.QuantileDMatrix) or dtrain._shira_x is None:
        return "booster gblinear cv on a DMatrix that is not SHIRA's with its raw matrix"
    why = _data_reason(dtrain, "dtrain")
    if why:
        return f"booster gblinear cv: {why}"
    nfold = int(kw.get("nfold", 3))
    if not 2 <= nfold <= dtrain.num_row() or not 1 <= int(rounds) <= MAX_ROUNDS:
        return f"cv nfold {nfold} / rounds {rounds}"
    return None


def _seq_sum(a: np.ndarray, exact: bool) -> float:
    return float(np.cumsum(a, dtype=np.float64)[-1]) if exact else float(np.sum(a, dtype=np.float64))


class _LinearReplay(Booster):

    def __init__(self, params: Any, p: dict[str, Any], dtr: Any, dte: Any, exact: bool) -> None:
        obj = str(p.get("objective", "reg:squarederror"))
        y = np.ascontiguousarray(dtr.get_label(), dtype=F32)
        w0 = dtr.get_weight()
        self._w8 = np.ascontiguousarray(w0, dtype=F32) if w0.size else None
        th = _call_threads(p)
        base = F32(p["base_score"]) if "base_score" in p else _fit.intercept(obj, y, self._w8, th, 1.0)
        assert base is not None, "the linear folds' objectives have a closed-form intercept"
        nf = int(dtr.num_col())
        obj_json, margin = _objective_and_margin({k: v for k, v in p.items() if k not in _GBL_ONLY}, nf, obj, float(base), 1)
        self._raw = bytes(export.zero_tree_json(obj, float(base), 1, 1, obj_json))
        self._params = [(k, v) for k, v in (params.items() if isinstance(params, dict) else params) if k not in _GBL_ONLY]
        Booster.__init__(self, self._params, None, bytearray(self._raw))
        self._obj, self._y, self._exact = obj, y, bool(exact)
        self._eta = float(_pget(p, ("eta", "learning_rate"), 0.5))
        sw = float(y.size) if self._w8 is None else float(np.sum(self._w8, dtype=np.float64))
        self._lam = _pget(p, ("lambda", "reg_lambda"), 0.0) * sw
        self._alpha = _pget(p, ("alpha", "reg_alpha"), 0.0) * sw
        self._margin0, self._bias = F32(margin), F32(0.0)
        self._coef = np.zeros(nf, dtype=F32)
        nt = int(p.get("nthread", 0)) or None
        self._sets = []
        for d in (dtr, dte):
            x = np.nan_to_num(np.asarray(d._shira_x, dtype=F32), nan=0.0)
            ww = d.get_weight()
            ev = _core.DMatrix(np.zeros((x.shape[0], 1), F32), label=d.get_label(), weight=ww if ww.size else None, nthread=nt)
            self._sets.append(SimpleNamespace(x=np.asfortranarray(x), ev=ev))
        assert self._sets[0].x.shape[0] == y.size and self._eta > 0.0

    def num_boosted_rounds(self) -> int:
        return 0

    def _margin(self, x: np.ndarray) -> np.ndarray:
        m = np.full(x.shape[0], F32(self._bias + self._margin0), dtype=F32)
        for j in range(x.shape[1]):
            if self._coef[j] != 0.0:
                m += x[:, j] * self._coef[j]
        return m

    def _gradients(self, m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._obj == "reg:squarederror":
            g, h = m - self._y, np.ones(m.size, dtype=F32)
        else:
            pr = (F32(1.0) / (F32(1.0) + np.exp(-m, dtype=F32))).astype(F32)
            g, h = pr - self._y, np.maximum(pr * (F32(1.0) - pr), F32(1e-16))
        if self._w8 is not None:
            g, h = g * self._w8, h * self._w8
        return g.astype(F32), h.astype(F32)

    def _delta(self, sg: float, sh: float, wj: float) -> float:
        if sh < 1e-5:
            return 0.0
        g2, h2 = sg + self._lam * wj, sh + self._lam
        tmp = wj - g2 / h2
        return max(-(g2 + self._alpha) / h2, -wj) if tmp >= 0 else min(-(g2 - self._alpha) / h2, -wj)

    def advance(self, i: int) -> None:
        x = self._sets[0].x
        g, h = self._gradients(self._margin(x))
        dbias = F32(self._eta * (-_seq_sum(g, self._exact) / _seq_sum(h, self._exact)))
        self._bias = F32(self._bias + dbias)
        g = g + h * dbias
        for j in range(x.shape[1]):
            v = x[:, j]
            hv = h * v
            dw = F32(self._eta * self._delta(_seq_sum(g * v, self._exact), _seq_sum(hv * v, self._exact), float(self._coef[j])))
            if dw == 0.0:
                continue
            self._coef[j] = F32(self._coef[j] + dw)
            g = g + hv * dw
        assert bool(np.all(np.isfinite(self._coef))) and i >= 0

    def eval_set(self, evals: Any, iteration: int = 0, feval: Any = None, output_margin: bool = True) -> str:
        assert len(evals) == len(self._sets)
        mats = []
        for st, (_, name) in zip(self._sets, evals):
            st.ev.set_base_margin(self._margin(st.x))
            mats.append((st.ev, name))
        return _core.Booster(self._params, None, bytearray(self._raw)).eval_set(mats, iteration, feval, output_margin)


def _linear_sums(x64: np.ndarray, y: np.ndarray, w: np.ndarray | None, rows: np.ndarray) -> SimpleNamespace:
    xs = x64[rows]
    ys = np.asarray(y[rows], dtype=np.float64)
    ws = None if w is None else np.asarray(w[rows], dtype=np.float64)
    xw = xs if ws is None else xs * ws[:, None]
    yw = ys if ws is None else ys * ws
    out = SimpleNamespace(G=xw.T @ xs, s=xw.sum(axis=0), c=xs.T @ yw, sw=float(ys.size if ws is None else ws.sum()),
                          sy=float(yw.sum()), yy=float(ys @ yw))
    assert out.G.shape == (x64.shape[1], x64.shape[1]) and out.sw > 0.0
    return out


class _LinearGram(Booster):

    def __init__(self, params: Any, raw: bytes, p: dict[str, Any], tr: Any, te: Any, margin0: float) -> None:
        self._params, self._raw = params, raw
        Booster.__init__(self, params, None, bytearray(raw))
        self._tr, self._te = tr, te
        self._eta = float(_pget(p, ("eta", "learning_rate"), 0.5))
        self._lam = _pget(p, ("lambda", "reg_lambda"), 0.0) * tr.sw
        self._alpha = _pget(p, ("alpha", "reg_alpha"), 0.0) * tr.sw
        nf = int(tr.s.size)
        self._a0, self._b, self._w, self._gw = float(margin0), 0.0, np.zeros(nf), np.zeros(nf)
        assert self._eta > 0.0 and nf >= 1

    def num_boosted_rounds(self) -> int:
        return 0

    def advance(self, i: int) -> None:
        t = self._tr
        sg = t.sw * (self._a0 + self._b) + float(t.s @ self._w) - t.sy
        self._b = float(F32(self._b + F32(self._eta * (-sg / t.sw))))
        a = self._a0 + self._b
        for j in range(self._w.size):
            sgj, shj, wj = a * t.s[j] + self._gw[j] - t.c[j], float(t.G[j, j]), float(self._w[j])
            if shj < 1e-5:
                continue
            g2, h2 = sgj + self._lam * wj, shj + self._lam
            d = max(-(g2 + self._alpha) / h2, -wj) if wj - g2 / h2 >= 0 else min(-(g2 - self._alpha) / h2, -wj)
            dw = float(F32(self._eta * d))
            if dw == 0.0:
                continue
            new = float(F32(wj + dw))
            self._gw += t.G[:, j] * (new - wj)
            self._w[j] = new
        assert bool(np.all(np.isfinite(self._w))) and i >= 0

    def _rmse(self, st: Any) -> float:
        a, w = self._a0 + self._b, self._w
        sse = a * a * st.sw + float(w @ st.G @ w) + st.yy + 2.0 * a * float(st.s @ w) - 2.0 * a * st.sy - 2.0 * float(w @ st.c)
        return math.sqrt(max(sse, 0.0) / st.sw)

    def eval_set(self, evals: Any, iteration: int = 0, feval: Any = None, output_margin: bool = True) -> str:
        assert len(evals) == 2 and feval is None
        return f"[{iteration}]" + "".join(f"\t{name}-rmse:{self._rmse(st):.17g}"
                                          for st, (_, name) in zip((self._tr, self._te), evals))


def _cv_linear_folds(params: Any, dtrain: Any, rounds: int, kw: dict[str, Any]) -> tuple[list | None, str | None]:
    plst, p = _cv_params(params, kw)
    nt, folds = int(p.get("nthread", 0)), []
    pairs = _cv_folds(dtrain, int(kw.get("nfold", 3)), int(kw.get("seed", 0)), kw.get("shuffle", True),
                      bool(kw.get("stratified")), kw.get("folds"))
    mets = [v for k, v in plst if k == "eval_metric"]
    obj = str(p.get("objective", "reg:squarederror"))
    if (first_read() and obj == "reg:squarederror" and all(m == "rmse" for m in mets) and kw.get("custom_metric") is None
            and not any(cb for cb in kw.get("callbacks") or [] if type(cb) not in (_cb.EarlyStopping, _cb.EvaluationMonitor))):
        x64 = np.nan_to_num(np.asarray(dtrain._shira_x, dtype=np.float64), nan=0.0)
        y = np.ascontiguousarray(dtrain.get_label(), dtype=F32)
        w0 = dtrain.get_weight()
        w = np.ascontiguousarray(w0, dtype=F32) if w0.size else None
        th, nf = _call_threads(p), int(dtrain.num_col())
        scorer = [(k, v) for k, v in plst if k not in _GBL_ONLY]
        for tr, te in pairs:
            tr, te = np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)
            base = F32(p["base_score"]) if "base_score" in p else _fit.intercept(
                obj, np.ascontiguousarray(y[tr]), None if w is None else np.ascontiguousarray(w[tr]), th, 1.0)
            assert base is not None
            obj_json, margin = _objective_and_margin({k: v for k, v in p.items() if k not in _GBL_ONLY}, nf, obj,
                                                     float(base), 1)
            raw = bytes(export.zero_tree_json(obj, float(base), 1, 1, obj_json))
            rep = _LinearGram(scorer, raw, p, _linear_sums(x64, y, w, tr), _linear_sums(x64, y, w, te), float(margin))
            folds.append(_ShiraFold(rep, [(None, "train"), (None, "test")]))
        return folds, None
    for tr, te in pairs:
        dtr, dte = _cv_fold_data(dtrain, tr, nt), _cv_fold_data(dtrain, te, nt)
        folds.append(_ShiraFold(_LinearReplay(plst, p, dtr, dte, not first_read()), [(dtr, "train"), (dte, "test")]))
    assert folds
    return folds, None


def _cv_loop(cvfolds: list, dtrain: Any, rounds: int, kw: dict[str, Any]) -> Any:
    cbs = [] if kw.get("callbacks") is None else copy.copy(list(kw["callbacks"]))
    if kw["verbose_eval"]:
        cbs.append(_cb.EvaluationMonitor(period=1 if kw["verbose_eval"] is True else kw["verbose_eval"],
                                         show_stdv=kw["show_stdv"]))
    if kw["early_stopping_rounds"]:
        cbs.append(_cb.EarlyStopping(rounds=kw["early_stopping_rounds"], maximize=kw["maximize"]))
    box = _cb.CallbackContainer(cbs, metric=kw.get("custom_metric"), is_cv=True, output_margin=callable(kw.get("obj")))
    booster = _training._PackedBooster(cvfolds)
    box.before_training(booster)
    results: dict[str, list[float]] = {}
    for i in range(int(rounds)):
        if box.before_iteration(booster, i, dtrain, None):
            break
        booster.update(i, None)
        should_break = box.after_iteration(booster, i, dtrain, None)
        for key, mean, std in box.aggregated_cv:
            results.setdefault(key + "-mean", []).append(mean)
            results.setdefault(key + "-std", []).append(std)
        if should_break:
            for k in results:
                results[k] = results[k][: (booster.best_iteration + 1)]
            break
    out: Any = results
    if kw["as_pandas"]:
        try:
            import pandas as pd
            out = pd.DataFrame.from_dict(results)
        except ImportError:
            out = results
    box.after_training(booster)
    return out


def cv(params: Any, dtrain: Any, num_boost_round: int = 10, *, nfold: int = 3, stratified: bool = False,
       folds: Any = None, metrics: Any = (), obj: Any = None, maximize: Any = None, early_stopping_rounds: Any = None,
       fpreproc: Any = None, as_pandas: bool = True, verbose_eval: Any = None, show_stdv: bool = True,
       seed: int = 0, callbacks: Any = None, shuffle: bool = True, custom_metric: Any = None) -> Any:
    global last_backend
    kw = {"nfold": nfold, "stratified": stratified, "folds": folds, "metrics": metrics, "obj": obj,
          "maximize": maximize, "early_stopping_rounds": early_stopping_rounds, "fpreproc": fpreproc,
          "as_pandas": as_pandas, "verbose_eval": verbose_eval, "show_stdv": show_stdv, "seed": seed,
          "callbacks": callbacks, "shuffle": shuffle, "custom_metric": custom_metric}
    if (isinstance(dtrain, _core.DMatrix) and not isinstance(dtrain, _core.QuantileDMatrix)
            and getattr(dtrain, "_shira_x", None) is None and not _is_cat(dtrain)):
        d2, why0 = _cv_fold_own(dtrain, int(normalize_params(params)[0].get("nthread", 0)))
        if why0 is None:
            dtrain = d2
    reason = _cv_reason(params, dtrain, num_boost_round, kw)
    etas, why = _cv_etas(normalize_params(params)[0], kw, num_boost_round) if not reason else (None, None)
    reason = reason or why
    cvfolds = None
    if not reason:
        form = cv_form()
        cv_scored_log.clear()
        linear = normalize_params(params)[0].get("booster") == "gblinear"
        cvfolds = (None if (form == "perfold" or obj is not None or fpreproc is not None or etas is not None or linear
                            or dtrain.get_float_info("base_margin").size > 0
                            or _cv_booster_route(normalize_params(params)[0], dtrain))
                   else _cv_matrix(params, dtrain, num_boost_round, kw, score_sample=form == "matrix_scored"))
        if cvfolds is not None:
            sc = sum(d["scored"] for d in cv_scored_log)
            ho = sum(d["held_out"] for d in cv_scored_log)
            LOG.info("shira.xgboost: cv the matrix form: tree shapes grown once on the first read's sample of "
                     "all rows, leaf values per fold on its training rows; the held-out metric scored on %d of %d "
                     "held-out rows (%s)", sc, ho, "the sample's rows scored too: SHIRA_CV_FORM=matrix_scored"
                     if form == "matrix_scored" else "the shape sample's rows are never scored")
    global last_cv_form
    last_cv_form = ((form if form == "matrix_scored" else "matrix") if cvfolds is not None
                    else ("perfold" if first_read() and etas is None and dtrain.get_float_info("base_margin").size == 0
                          and not (str(normalize_params(params)[0].get("objective", "")).startswith("multi:")
                                   and "base_score" not in normalize_params(params)[0])
                          and str(normalize_params(params)[0].get("objective", "")) not in RANK_ENGINE_OBJ else "exact"))
    booster = not reason and _cv_booster_route(normalize_params(params)[0], dtrain)
    if not reason and normalize_params(params)[0].get("booster") == "gblinear":
        cvfolds, reason = _cv_linear_folds(params, dtrain, num_boost_round, kw)
        last_cv_form = "linear"
    elif booster:
        cvfolds, reason = _cv_booster_folds(params, dtrain, num_boost_round, kw)
        last_cv_form = "perfold"
    elif not reason and cvfolds is None and obj is not None:
        cvfolds, reason = _cv_obj(params, dtrain, num_boost_round, kw, etas)
    elif not reason and cvfolds is None:
        cvfolds, reason = _cv_shira(params, dtrain, num_boost_round, kw, etas)
    if not reason and custom_metric is not None and any(
            not isinstance(f.bst, (_LeafReplay, _PredReplay, _LinearReplay, _LinearGram)) for f in cvfolds or []):
        reason = "a custom metric with a fold outside the leaf-id scorer (multi-output, or rows below the lowest cut)"
    if reason and not first_read():
        LOG.warning("shira.xgboost: SHIRA_XGB_MODE=ordinary and no exact SHIRA engine takes this cv call (%s): running "
                    "stock XGBoost's cv (the exact answer); the default mode runs it on SHIRA", reason)
        out = _training.cv(params, dtrain, num_boost_round, nfold=nfold, stratified=stratified, folds=folds,
                           metrics=metrics, obj=obj, maximize=maximize, early_stopping_rounds=early_stopping_rounds,
                           fpreproc=fpreproc, as_pandas=as_pandas, verbose_eval=verbose_eval, show_stdv=show_stdv,
                           seed=seed, callbacks=callbacks, shuffle=shuffle, custom_metric=custom_metric)
        last_backend, last_cv_form = "xgboost", "stock"
        backend_history.append(last_backend)
        return out
    if reason:
        raise NotImplementedError(f"shira.xgboost.cv: no SHIRA route for this call: {reason}")
    assert cvfolds is not None and len(cvfolds) >= 1
    if etas is not None:
        kw = dict(kw, callbacks=[cb for cb in kw["callbacks"] if type(cb) is not _cb.LearningRateScheduler])
    out = _cv_loop(cvfolds, dtrain, num_boost_round, kw)
    last_backend = "shira"
    backend_history.append(last_backend)
    return out


class _SklearnRoutes:

    NAMES = ("train", "DMatrix", "QuantileDMatrix")

    def __enter__(self) -> None:
        self.saved = {k: getattr(_sk, k) for k in self.NAMES}
        mine = {"train": train, "DMatrix": DMatrix, "QuantileDMatrix": QuantileDMatrix}
        for k in self.NAMES:
            setattr(_sk, k, mine[k])

    def __exit__(self, *exc: Any) -> None:
        for k, v in self.saved.items():
            setattr(_sk, k, v)


class XGBRegressor(_sk.XGBRegressor):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> XGBRegressor:
        with _SklearnRoutes():
            super().fit(X, y, **kwargs)
        return self


class XGBClassifier(_sk.XGBClassifier):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> XGBClassifier:
        with _SklearnRoutes():
            super().fit(X, y, **kwargs)
        return self


class XGBRFRegressor(_sk.XGBRFRegressor):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> XGBRFRegressor:
        with _SklearnRoutes():
            super().fit(X, y, **kwargs)
        return self


class XGBRFClassifier(_sk.XGBRFClassifier):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> XGBRFClassifier:
        with _SklearnRoutes():
            super().fit(X, y, **kwargs)
        return self


class XGBRanker(_sk.XGBRanker):

    def fit(self, X: Any, y: Any, **kwargs: Any) -> XGBRanker:
        with _SklearnRoutes():
            super().fit(X, y, **kwargs)
        return self


_PATCHED: dict[str, Any] = {}


def patch() -> None:
    mine = {"train": train, "DMatrix": DMatrix, "QuantileDMatrix": QuantileDMatrix,
            "XGBRegressor": XGBRegressor, "XGBClassifier": XGBClassifier,
            "XGBRFRegressor": XGBRFRegressor, "XGBRFClassifier": XGBRFClassifier,
            "XGBRanker": XGBRanker,
            "cv": cv}
    if not _PATCHED:
        _PATCHED.update({k: getattr(_xgb, k) for k in mine})
    for k, v in mine.items():
        setattr(_xgb, k, v)
    assert _xgb.train is train


def unpatch() -> None:
    for k, v in _PATCHED.items():
        setattr(_xgb, k, v)
    _PATCHED.clear()
    assert _xgb.train is _STOCK["train"]


def __getattr__(name: str) -> Any:
    return getattr(_xgb, name)
