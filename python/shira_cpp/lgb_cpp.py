from __future__ import annotations

import ctypes
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from shira_cpp import binning_cpp
except ImportError:
    import binning_cpp

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
LIB_PATH = (Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB")
            else SHIRA / "lib" / ("libshira.dylib" if sys.platform == "darwin" else "libshira.so"))
_P = ctypes.c_void_p
_I64 = ctypes.c_int64
_I32 = ctypes.c_int32
_LIB: ctypes.CDLL | None = None
MAX_THREADS = 64
K_ZERO = float(np.float32(1e-35))
K_SPARSE = 0.7
MISS_NONE, MISS_ZERO, MISS_NAN = 0, 1, 2
_ERRORS = {-1: "rejected its arguments", -2: "a bag came out empty",
           -3: "an internal check failed (line on stderr); no result", -4: "out of memory or thread start failed"}
_ERRORS[-5] = ("stock LightGBM's default-mode reduction here gives different bits in different thread arrival orders "
               "(an OpenMP race in stock itself) and SHIRA_LGB_OMP_AMBIGUOUS=stock asks for stock's own run")
OBJECTIVES = {"regression": 0, "l2": 0, "regression_l2": 0, "mean_squared_error": 0, "mse": 0,
              "rmse": 0, "l2_root": 0, "root_mean_squared_error": 0,
              "binary": 1, "multiclass": 2, "softmax": 2, "multiclassova": 3, "multiclass_ova": 3,
              "ova": 3, "ovr": 3, "regression_l1": 4, "l1": 4, "mean_absolute_error": 4, "mae": 4,
              "huber": 5, "quantile": 6, "poisson": 7, "gamma": 8, "tweedie": 9, "cross_entropy": 10,
              "xentropy": 10, "lambdarank": 11}

SUPPORTED = {"objective", "boost_from_average", "num_leaves", "max_depth", "learning_rate", "lambda_l2",
             "lambda_l1", "min_data_in_leaf", "min_sum_hessian_in_leaf", "min_gain_to_split", "max_bin",
             "min_data_in_bin", "feature_pre_filter", "deterministic", "force_col_wise", "force_row_wise",
             "num_threads",
             "verbose", "sigmoid", "bin_construct_sample_cnt", "data_random_seed", "seed",
             "use_missing", "zero_as_missing", "bagging_fraction", "bagging_freq", "bagging_seed",
             "feature_fraction", "feature_fraction_bynode", "feature_fraction_seed", "path_smooth",
             "num_class", "alpha", "tweedie_variance_power", "poisson_max_delta_step",
             "data_sample_strategy", "top_rate", "other_rate", "boosting", "drop_rate", "skip_drop",
             "max_drop", "drop_seed", "xgboost_dart_mode", "uniform_drop", "categorical_feature", "cat_l2",
             "cat_smooth", "max_cat_threshold", "max_cat_to_onehot", "min_data_per_group",
             "lambdarank_truncation_level", "lambdarank_norm", "label_gain",
             "extra_trees", "extra_seed", "monotone_constraints", "max_delta_step",
             "scale_pos_weight",
             "learning_rates",
             "is_unbalance",
             "_shira_init_score", "_shira_continued", "_shira_iter0", "_shira_col_extra"}
VERIFIED: set = {"base", "missing", "weights", "bagging", "feature_fraction", "max_depth", "lambda_l1",
                 "boost_from_average", "path_smooth", "multiclass", "obj_l1", "obj_huber", "obj_quantile",
                 "obj_poisson", "obj_gamma", "obj_tweedie", "obj_cross_entropy", "goss", "dart", "categorical",
                 "lambdarank",
                 "row_wise", "default_mode",
                 "max_delta_step",
                 "scale_pos_weight",
                 "learning_rates",
                 "is_unbalance",
                 "init_score",
                 "iter0"}


class ShiraLgbOpts(ctypes.Structure):
    _fields_ = [("weight", _P), ("default_bin", _P), ("missing_type", _P), ("num_class", _I32),
                ("max_depth", _I32), ("lambda_l1", ctypes.c_double), ("path_smooth", ctypes.c_double),
                ("boost_from_average", _I32), ("bagging_freq", _I32), ("bagging_fraction", ctypes.c_double),
                ("bagging_seed", _I32), ("feature_fraction_seed", _I32), ("feature_fraction", ctypes.c_double),
                ("feature_fraction_bynode", ctypes.c_double), ("out_init", _P), ("out_ntrees", _P),
                ("alpha", ctypes.c_double), ("tweedie_variance_power", ctypes.c_double),
                ("poisson_max_delta_step", ctypes.c_double), ("goss", _I32), ("lgb_threads", _I32),
                ("top_rate", ctypes.c_double), ("other_rate", ctypes.c_double), ("dart", _I32),
                ("max_drop", _I32), ("drop_seed", _I32), ("xgboost_dart_mode", _I32), ("uniform_drop", _I32),
                ("drop_rate", ctypes.c_double), ("skip_drop", ctypes.c_double), ("categorical", _P),
                ("cat_l2", ctypes.c_double), ("cat_smooth", ctypes.c_double), ("max_cat_threshold", _I32),
                ("max_cat_to_onehot", _I32), ("min_data_per_group", _I32), ("out_cat", _P),
                ("query_bounds", _P), ("num_queries", _I32), ("truncation_level", _I32),
                ("lambdarank_norm", _I32), ("label_gain", _P), ("num_label_gain", _I32),
                ("omp_reduce", _I32), ("hist_rowwise", _P), ("dense_rate", _P),
                ("out_stats", _P), ("lut16", _P), ("extra_trees", _I32), ("extra_seed", _I32), ("monotone", _P),
                ("max_delta_step", ctypes.c_double),
                ("order_mode", _I32), ("out_order", _P),
                ("scale_pos_weight", ctypes.c_double),
                ("learning_rates", _P),
                ("out_leaf0", _P), ("out_drop_n", _P), ("out_drop_idx", _P), ("drop_cap", _I32),
                ("is_unbalance", _I32),
                ("init_score", _P), ("continued", _I32),
                ("iter0", _I32), ("col_extra", _P)]


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        assert LIB_PATH.exists(), f"build first: {SHIRA / 'build.sh'}"
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.shira_train_lgb.restype = ctypes.c_int
        lib.shira_train_lgb.argtypes = [_P, _I64, _I32, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P,
                                        _I32, _I32, _P, _P, _P, _P, _P]
        lib.shira_train_lgb_opts.restype = ctypes.c_int
        lib.shira_train_lgb_opts.argtypes = [_P, _I64, _I32, _I32, _P, _P, _P, _I32, _P, _P, _P, _P, _P,
                                             _I32, _I32, _P, _P, _P, _P, _P, ctypes.POINTER(ShiraLgbOpts)]
        lib.shira_lgb_tree_text.restype = ctypes.c_int
        lib.shira_lgb_tree_text.argtypes = [_I32, _I32, _I32, _I32, _P, _P, _P, _P, _P, _P, _P, _I32, _P, _P, _P, _P, _I64,
                                            _P, ctypes.POINTER(ctypes.c_int64)]
        lib.shira_lgb_atof.restype = ctypes.c_int
        lib.shira_lgb_atof.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
        lib.shira_lgb_check_labels.restype = ctypes.c_int
        lib.shira_lgb_check_labels.argtypes = [_P, _I64, _I32, _P, _P, _I32, _I32, _I32]
        lib.shira_lgb_rank_values.restype = ctypes.c_int
        lib.shira_lgb_rank_values.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _I64, _P, _I32, _I32, _I32,
                                              ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _P, _P]
        lib.shira_lgb_rank_values_w.restype = ctypes.c_int
        lib.shira_lgb_rank_values_w.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _P, _I64, _P, _I32, _I32, _I32,
                                                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _P, _P]
        lib.shira_lgb_rank_values_t.restype = ctypes.c_int
        lib.shira_lgb_rank_values_t.argtypes = [_P, _I32, _I64, _P, _I32, _I32, _P, _P, _P, _I64, _P, _I32, _I32, _I32,
                                                ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, _I32,
                                                _P, _P]
        _LIB = lib
    return _LIB


def tree_text(nleaves: np.ndarray, nodes: np.ndarray, default_left: np.ndarray, leaf: np.ndarray, gain: np.ndarray,
              stats: np.ndarray | None, shrink: np.ndarray, upper: list, missing_type: np.ndarray) -> tuple[str, np.ndarray]:
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    nd = np.ascontiguousarray(nodes[:nl.size], dtype=np.int32)
    dl = np.ascontiguousarray(default_left[:nl.size], dtype=np.int32)
    lf = np.ascontiguousarray(leaf[:nl.size], dtype=np.float64)
    gn = np.ascontiguousarray(gain[:nl.size], dtype=np.float64)
    st = None if stats is None else np.ascontiguousarray(stats[:nl.size], dtype=np.float64)
    sh = np.ascontiguousarray(shrink, dtype=np.float64)
    assert nd.ndim == 3 and nd.shape[2] == 4 and dl.shape == gn.shape == nd.shape[:2] and sh.size == nl.size
    assert st is None or (st.ndim == 2 and (st.shape[1] + 2) % 5 == 0)
    ptr = np.concatenate([[0], np.cumsum([len(u) for u in upper])]).astype(np.int32)
    up = np.ascontiguousarray(np.concatenate([np.asarray(u, dtype=np.float64) for u in upper]))
    mt = np.ascontiguousarray(missing_type, dtype=np.int32)
    sizes, used = np.zeros(nl.size, dtype=np.int64), ctypes.c_int64(0)
    cap = int(nl.sum()) * 200 + 400 * nl.size
    for _ in range(2):
        buf = ctypes.create_string_buffer(cap)
        rc = _lib().shira_lgb_tree_text(nl.size, nd.shape[1], lf.shape[1], 0 if st is None else (st.shape[1] + 2) // 5,
                                        _ptr(nl), _ptr(nd), _ptr(dl), _ptr(lf), _ptr(gn), _ptr(st), _ptr(sh), len(upper),
                                        _ptr(ptr), _ptr(up), _ptr(mt), buf, cap, _ptr(sizes), ctypes.byref(used))
        if rc != 0:
            raise RuntimeError(f"shira_lgb_tree_text failed ({rc})")
        if used.value <= cap:
            break
        cap = int(used.value)
    assert used.value <= cap and int(sizes.sum()) == used.value
    return buf.raw[:used.value].decode("ascii"), sizes


def _ptr(a: np.ndarray | None) -> int | None:
    return None if a is None else a.ctypes.data


def lgb_atof(s: str) -> float:

    out = ctypes.c_double(0.0)
    rc = _lib().shira_lgb_atof(ctypes.c_char_p(str(s).strip().encode()), ctypes.byref(out))
    if rc != 0:
        raise ValueError(f"not a LightGBM number: {s!r}")
    return out.value


def _pval(params: dict, key: str, default):
    return params[key] if key in params else default


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).lower() in ("true", "1", "+")


def require(option: str) -> None:
    if option not in VERIFIED and os.environ.get("SHIRA_LGB_UNVERIFIED") != "1":
        raise NotImplementedError(f"LightGBM option '{option}' is not verified bit-identical in SHIRA "
                                  "(use LightGBM itself)")


def _options_used(d: dict, params: dict) -> list[str]:
    used = []
    names = {4: "obj_l1", 5: "obj_huber", 6: "obj_quantile", 7: "obj_poisson", 8: "obj_gamma", 11: "lambdarank",
             9: "obj_tweedie", 10: "obj_cross_entropy"}
    if d["objective"] >= 2:
        used.append(names.get(d["objective"], "multiclass"))
    if d["objective"] in (5, 6) and "alpha" in params:
        used.append(names[d["objective"]])
    flags = (("bagging", d["bagging_freq"] > 0 and d["bagging_fraction"] < 1.0),
             ("feature_fraction", d["feature_fraction"] < 1.0 or d["feature_fraction_bynode"] < 1.0),
             ("max_depth", d["max_depth"] > 0), ("lambda_l1", d["lambda_l1"] > 0.0),
             ("boost_from_average", d["boost_from_average"]), ("path_smooth", d["path_smooth"] > 0.0),
             ("goss", d["goss"] == 1), ("dart", d["dart"] == 1), ("max_delta_step", d["max_delta_step"] > 0.0), ("categorical", bool(d["categorical_feature"])),
             ("scale_pos_weight", d.get("scale_pos_weight", 1.0) != 1.0),
             ("learning_rates", d.get("learning_rates") is not None),
             ("is_unbalance", d.get("is_unbalance", 0) == 1),
             ("init_score", d.get("init_score") is not None),
             ("iter0", d.get("iter0", 0) > 0),
             ("missing", "use_missing" in params or "zero_as_missing" in params),
             ("default_mode", d.get("omp_reduce", 0) >= 1 and d.get("omp_threads", 1) > 1),
             ("row_wise", d.get("hist") == "row" and d.get("omp_threads", 1) > 1))
    return used + [name for name, on in flags if on]


_OMP_KIND: list[int] = []


def _omp_kind_of_stock() -> tuple[int, str]:
    try:
        import lightgbm.basic as lgb_basic

        lib = lgb_basic._LIB
    except (ImportError, AttributeError) as e:
        return 1, f"stock lib_lightgbm handle not readable ({type(e).__name__}): libomp order kept"
    if hasattr(lib, "__kmpc_fork_call"):
        return 1, "stock lib_lightgbm loads LLVM libomp"
    if hasattr(lib, "GOMP_parallel"):
        return 2, "stock lib_lightgbm loads GNU libgomp"
    return 1, "stock lib_lightgbm exports no OpenMP entry point: libomp order kept"


def omp_kind() -> int:
    env = os.environ.get("SHIRA_LGB_OMP_KIND", "")
    if env in ("1", "2"):
        return int(env)
    if not _OMP_KIND:
        kind, why = _omp_kind_of_stock()
        logging.getLogger("shira").info("LightGBM default-mode reductions: kind %d (%s)", kind, why)
        _OMP_KIND.append(kind)
    assert _OMP_KIND[0] in (1, 2)
    return _OMP_KIND[0]


def omp_default_threads() -> int:
    env = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    n = int(env) if env.isdigit() and int(env) > 0 else (os.cpu_count() or 1)
    assert n >= 1
    return n


def check_params(params: dict) -> dict:
    bad = sorted(set(params) - SUPPORTED)
    assert not bad, f"unsupported LightGBM parameters: {bad}"
    obj = str(_pval(params, "objective", "regression"))
    assert obj in OBJECTIVES, f"objective {obj} not supported"
    assert "seed" not in params, "seed changes every sub-seed; not supported"
    assert _pval(params, "feature_pre_filter", True) is False, "set feature_pre_filter False"
    assert str(_pval(params, "data_sample_strategy", "bagging")) in ("bagging", "goss")
    assert str(_pval(params, "boosting", "gbdt")) in ("gbdt", "dart"), "boosting gbdt or dart only"
    d = {"objective": OBJECTIVES[obj], "min_data": int(_pval(params, "min_data_in_leaf", 20)),
         "max_depth": int(_pval(params, "max_depth", -1)), "boost_from_average": _bool(_pval(params, "boost_from_average", True)),
         "goss": 1 if str(_pval(params, "data_sample_strategy", "bagging")) == "goss" else 0,
         "dart": 1 if str(_pval(params, "boosting", "gbdt")) == "dart" else 0,
         "num_threads": int(_pval(params, "num_threads", 0)),
         "xgboost_dart_mode": 1 if _bool(_pval(params, "xgboost_dart_mode", False)) else 0,
         "uniform_drop": 1 if _bool(_pval(params, "uniform_drop", False)) else 0,
         "max_drop": int(_pval(params, "max_drop", 50)), "drop_seed": int(_pval(params, "drop_seed", 4))}
    for key, default in (("alpha", 0.9), ("tweedie_variance_power", 1.5), ("poisson_max_delta_step", 0.7),
                         ("top_rate", 0.2), ("other_rate", 0.1), ("drop_rate", 0.1), ("skip_drop", 0.5),
                         ("cat_l2", 10.0), ("cat_smooth", 10.0)):
        d[key] = lgb_atof(str(_pval(params, key, default)))
    for key, default in (("max_cat_threshold", 32), ("max_cat_to_onehot", 4), ("min_data_per_group", 100)):
        d[key] = int(_pval(params, key, default))
    d["extra_trees"] = 1 if _bool(_pval(params, "extra_trees", False)) else 0
    d["is_unbalance"] = 1 if _bool(_pval(params, "is_unbalance", False)) else 0
    if params.get("learning_rates") is not None:
        lr = np.ascontiguousarray([lgb_atof(str(v)) for v in params["learning_rates"]], dtype=np.float64)
        assert lr.ndim == 1 and lr.size >= 1 and bool(np.all(lr > 0)), "learning_rates: positive, one per round"
        d["learning_rates"] = lr
    if params.get("_shira_init_score") is not None:
        isc = np.ascontiguousarray(params["_shira_init_score"], dtype=np.float64)
        assert isc.ndim == 2 and isc.size >= 1 and bool(np.all(np.isfinite(isc))), "init score: finite (n, num_class)"
        d["init_score"] = isc
    d["continued"] = 1 if params.get("_shira_continued") else 0
    d["iter0"] = int(params.get("_shira_iter0", 0) or 0)
    if params.get("_shira_col_extra") is not None:
        ce = np.ascontiguousarray(params["_shira_col_extra"], dtype=np.int32)
        assert ce.ndim == 1 and ce.size == d["iter0"] + 1 and bool(np.all(ce >= 0)), "col_extra: iter0 + 1 counts"
        d["col_extra"] = ce
    d["monotone"] = [int(v) for v in (_pval(params, "monotone_constraints", None) or [])]
    assert all(v in (-1, 0, 1) for v in d["monotone"]), "monotone_constraints: -1 / 0 / 1 per feature"
    d["extra_seed"] = int(_pval(params, "extra_seed", 6))
    cf = _pval(params, "categorical_feature", [])
    assert all(isinstance(j, (int, np.integer)) for j in cf), "categorical_feature: column indices"
    d["categorical_feature"] = sorted(int(j) for j in cf)
    d["truncation_level"] = int(_pval(params, "lambdarank_truncation_level", 30))
    d["lambdarank_norm"] = 1 if _bool(_pval(params, "lambdarank_norm", True)) else 0
    lg = _pval(params, "label_gain", None)
    d["label_gain"] = (np.array([0.0] + [float((1 << i) - 1) for i in range(1, 31)]) if lg is None
                       else np.array([lgb_atof(str(v)) for v in lg], dtype=np.float64))
    nl = int(_pval(params, "num_leaves", 31))
    if d["max_depth"] > 0 and "num_leaves" not in params:
        nl = min(nl, int(2 ** d["max_depth"]))
    d["num_leaves"] = nl
    for key, default in (("min_sum_hessian_in_leaf", 1e-3), ("lambda_l2", 0.0), ("learning_rate", 0.1),
                         ("min_gain_to_split", 0.0), ("sigmoid", 1.0), ("lambda_l1", 0.0), ("path_smooth", 0.0),
                         ("max_delta_step", 0.0),
                         ("scale_pos_weight", 1.0),
                         ("bagging_fraction", 1.0), ("feature_fraction", 1.0), ("feature_fraction_bynode", 1.0)):
        d[key] = lgb_atof(str(_pval(params, key, default)))
    for key, default in (("bagging_freq", 0), ("bagging_seed", 3), ("feature_fraction_seed", 2), ("num_class", 1)):
        d[key] = int(_pval(params, key, default))
    if d["objective"] in (2, 3):
        assert d["num_class"] >= 2, "multiclass needs num_class >= 2"
    else:
        assert d["num_class"] == 1, "num_class > 1 needs a multiclass objective"
    d["use_missing"] = _bool(_pval(params, "use_missing", True))
    d["zero_as_missing"] = _bool(_pval(params, "zero_as_missing", False))
    d["max_bin"] = int(_pval(params, "max_bin", 255))
    d["min_data_in_bin"] = int(_pval(params, "min_data_in_bin", 3))
    d["sample_cnt"] = int(_pval(params, "bin_construct_sample_cnt", 200000))
    d["data_random_seed"] = int(_pval(params, "data_random_seed", 1))
    if d["goss"]:
        assert not (d["bagging_freq"] > 0 and d["bagging_fraction"] != 1.0), "Cannot use bagging in GOSS"
    d["deterministic"] = _bool(_pval(params, "deterministic", False))
    d["omp_reduce"] = 0 if d["deterministic"] else omp_kind()
    d["omp_threads"] = d["num_threads"] if d["num_threads"] > 0 else omp_default_threads()
    assert d["omp_threads"] >= 1
    d["force_col_wise"] = _bool(_pval(params, "force_col_wise", False))
    d["force_row_wise"] = _bool(_pval(params, "force_row_wise", False))
    assert not (d["force_col_wise"] and d["force_row_wise"]), "force_col_wise and force_row_wise both set"
    d["hist"] = "row" if d["force_row_wise"] else ("col" if d["force_col_wise"] else "auto")
    for option in _options_used(d, params):
        require(option)
    return d


def find_bin_cat(sample_values: np.ndarray, total: int, max_bin: int, min_data_in_bin: int,
                 use_missing: bool = True, zero_as_missing: bool = False) -> dict:
    b2c, info = binning_cpp.lgb_cat_mapper(sample_values, int(total), int(max_bin), int(min_data_in_bin),
                                           use_missing, zero_as_missing)
    bin2cat = [int(c) for c in b2c.tolist()]
    nb = len(bin2cat)
    return {"categorical": True, "cat2bin": {c: b for b, c in enumerate(bin2cat)}, "bin2cat": bin2cat,
            "b2c": b2c, "num_bin": nb, "trivial": nb <= 1, "missing_type": int(info[1]),
            "default_bin": int(info[2]), "mfb": int(info[3]), "upper": np.zeros(nb)}


def value_codes(x: np.ndarray) -> tuple[np.ndarray, list]:
    n, nf = x.shape
    codes = np.zeros((n, nf), dtype=np.uint16)
    uniq = []
    for j in range(nf):
        u, inv = np.unique(x[:, j], return_inverse=True)
        assert u.size < 65536, "too many distinct values for 16-bit codes"
        codes[:, j] = inv.reshape(-1).astype(np.uint16)
        uniq.append(u.astype(np.float64))
    return codes, uniq


def fold_mappers_cpp(x: np.ndarray, lo: int, hi: int, cfg: dict) -> list:
    xs = np.ascontiguousarray(x[lo:hi], dtype=np.float64)
    nth = max(1, min(int(cfg.get("omp_threads", 1)), 64))
    ptrs, upper, info = binning_cpp.lgb_mappers(xs, cfg["max_bin"], cfg["min_data_in_bin"], cfg["sample_cnt"],
                                                cfg["data_random_seed"], cfg["use_missing"], cfg["zero_as_missing"], nth)
    out = []
    cat_rows = None
    for j in range(x.shape[1]):
        if j in cfg["categorical_feature"]:
            if cat_rows is None:
                cat_rows = binning_cpp.lgb_sample_indices(hi - lo, cfg["sample_cnt"], cfg["data_random_seed"]) + lo
            out.append(find_bin_cat(x[cat_rows, j], cat_rows.size, cfg["max_bin"], cfg["min_data_in_bin"],
                                    cfg["use_missing"], cfg["zero_as_missing"]))
            continue
        nb, mt, db, mf = (int(v) for v in info[j])
        out.append({"upper": upper[ptrs[j]: ptrs[j + 1]].copy(), "num_bin": nb, "trivial": nb <= 1,
                    "default_bin": db, "mfb": mf, "missing_type": mt, "info": info[j].copy()})
    return out


def _lut_column(mp: dict, u: np.ndarray) -> np.ndarray:
    if mp.get("categorical"):
        return binning_cpp.lgb_cat_bin(u, mp["b2c"]).astype(np.int64)
    ptrs = np.array([0, mp["upper"].size], dtype=np.int32)
    info = np.ascontiguousarray(mp["info"].reshape(1, 4), dtype=np.int32)
    if int(mp["num_bin"]) > 256:
        return binning_cpp.lgb_bin2(np.ascontiguousarray(u.reshape(-1, 1)), ptrs, mp["upper"], info, code_bytes=2)[0][:, 0]
    return binning_cpp.lgb_bin(np.ascontiguousarray(u.reshape(-1, 1)), ptrs, mp["upper"], info)[:, 0]


def build_luts(x: np.ndarray, uniq: list, folds, cfg: dict):
    nf, k_n = x.shape[1], len(folds)
    c_max = max(u.size for u in uniq)
    mappers = [fold_mappers_cpp(x, int(lo), int(hi), cfg) for lo, hi in folds]
    wide = max(m["num_bin"] for mp in mappers for m in mp) > 256
    lut = np.zeros((k_n, nf, c_max), dtype=np.uint16 if wide else np.uint8)
    nbin, mfb, dbin = (np.zeros((k_n, nf), dtype=np.int32) for _ in range(3))
    mtype = np.zeros((k_n, nf), dtype=np.int8)
    for k, mp in enumerate(mappers):
        for j in range(nf):
            b = _lut_column(mp[j], uniq[j])
            assert mp[j]["num_bin"] <= (65535 if wide else 256) and (b < mp[j]["num_bin"]).all()
            lut[k, j, : b.size] = b
            nbin[k, j], mfb[k, j] = mp[j]["num_bin"], mp[j]["mfb"]
            dbin[k, j], mtype[k, j] = mp[j]["default_bin"], mp[j]["missing_type"]
    return lut, nbin, mfb, dbin, mtype, mappers


def prepare(x: np.ndarray, folds, params: dict) -> dict:
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 2 and not np.isinf(x).any()
    cfg = check_params(params)
    if np.isnan(x).any():
        require("missing")
    codes, uniq = value_codes(x)
    lut, nbin, mfb, dbin, mtype, mappers = build_luts(x, uniq, folds, cfg)
    iscat = None
    if cfg["categorical_feature"]:
        iscat = np.zeros(x.shape[1], dtype=np.int8)
        iscat[cfg["categorical_feature"]] = 1
    return {"cfg": cfg, "codes": codes, "uniq": uniq, "lut": lut, "nbin": nbin, "mfb": mfb, "dbin": dbin,
            "mtype": mtype, "mappers": mappers, "folds": [(int(a), int(b)) for a, b in folds], "iscat": iscat}


def dense_rates(prep: dict) -> np.ndarray:
    cfg, codes, lut = prep["cfg"], prep["codes"], prep["lut"]
    k_n, nf = len(prep["folds"]), codes.shape[1]
    out = np.zeros((k_n, nf), dtype=np.float64)
    for k, (lo, hi) in enumerate(prep["folds"]):
        rows = binning_cpp.lgb_sample_indices(hi - lo, cfg["sample_cnt"], cfg["data_random_seed"]) + lo
        assert rows.size >= 1
        cs = np.ascontiguousarray(codes[rows].T)
        hit = np.count_nonzero(lut[k][np.arange(nf)[:, None], cs] == np.asarray(prep["mfb"][k])[:, None], axis=1)
        live = np.asarray(prep["nbin"][k]) > 1
        out[k, live] = 1.0 - hit[live] / rows.size
    return np.ascontiguousarray(out)


def fold_rowwise(prep: dict, cfg: dict) -> np.ndarray | None:
    k_n = len(prep["folds"])
    if cfg["hist"] == "col":
        return None
    if cfg["hist"] == "row":
        rw = np.ones(k_n, dtype=np.uint8)
    elif prep.get("rowwise") is not None:
        rw = np.ascontiguousarray(prep["rowwise"], dtype=np.uint8)
        assert rw.shape == (k_n,) and bool(np.isin(rw, (0, 1)).all())
    else:
        assert cfg["omp_threads"] == 1, ("histogram method not resolved: LightGBM picks col-wise or row-wise by "
                                         "timing; set force_col_wise or force_row_wise")
        return None
    if rw.any() and cfg["omp_threads"] > 1:
        require("row_wise")
    return rw if rw.any() else None


def _opts(cfg: dict, prep: dict, w32, init, ntrees) -> ShiraLgbOpts:
    o = ShiraLgbOpts()
    o.weight, o.default_bin, o.missing_type = _ptr(w32), _ptr(prep["dbin"]), _ptr(prep["mtype"])
    o.num_class, o.max_depth = cfg["num_class"], cfg["max_depth"]
    o.lambda_l1, o.path_smooth = cfg["lambda_l1"], cfg["path_smooth"]
    o.boost_from_average = 1 if cfg["boost_from_average"] else 0
    o.bagging_freq, o.bagging_fraction, o.bagging_seed = cfg["bagging_freq"], cfg["bagging_fraction"], cfg["bagging_seed"]
    o.feature_fraction_seed, o.feature_fraction = cfg["feature_fraction_seed"], cfg["feature_fraction"]
    o.feature_fraction_bynode = cfg["feature_fraction_bynode"]
    o.out_init, o.out_ntrees = _ptr(init), _ptr(ntrees)
    o.alpha, o.tweedie_variance_power = cfg["alpha"], cfg["tweedie_variance_power"]
    o.poisson_max_delta_step = cfg["poisson_max_delta_step"]
    o.max_delta_step = cfg.get("max_delta_step", 0.0)
    o.scale_pos_weight = cfg.get("scale_pos_weight", 1.0)
    o.is_unbalance = int(cfg.get("is_unbalance", 0))
    if cfg.get("learning_rates") is not None:
        o.learning_rates = _ptr(cfg["learning_rates"])
    if cfg.get("init_score") is not None:
        o.init_score = _ptr(cfg["init_score"])
    o.continued = int(cfg.get("continued", 0))
    o.iter0 = int(cfg.get("iter0", 0))
    if cfg.get("col_extra") is not None:
        o.col_extra = _ptr(cfg["col_extra"])
    o.goss, o.top_rate, o.other_rate = cfg["goss"], cfg["top_rate"], cfg["other_rate"]
    o.lgb_threads = cfg["num_threads"] if cfg["num_threads"] > 0 or not cfg["goss"] else cfg["omp_threads"]
    o.omp_reduce = cfg.get("omp_reduce", 0)
    o.extra_trees, o.extra_seed = int(cfg.get("extra_trees", 0)), int(cfg.get("extra_seed", 6))
    if cfg.get("monotone"):
        assert len(cfg["monotone"]) == prep["codes"].shape[1], "monotone_constraints: one sign per feature"
        assert not cfg["path_smooth"] and prep.get("iscat") is None, "monotone with smoothing / categorical"
        assert not cfg.get("max_delta_step", 0.0) > 0.0, "monotone with max_delta_step"
        prep["_monotone"] = np.ascontiguousarray(cfg["monotone"], dtype=np.int8)
        o.monotone = _ptr(prep["_monotone"])
    if o.omp_reduce:
        o.lgb_threads = cfg["omp_threads"]
    o.dart, o.max_drop, o.drop_seed = cfg["dart"], cfg["max_drop"], cfg["drop_seed"]
    o.xgboost_dart_mode, o.uniform_drop = cfg["xgboost_dart_mode"], cfg["uniform_drop"]
    o.drop_rate, o.skip_drop = cfg["drop_rate"], cfg["skip_drop"]
    o.categorical = _ptr(prep.get("iscat"))
    o.cat_l2, o.cat_smooth = cfg["cat_l2"], cfg["cat_smooth"]
    o.max_cat_threshold, o.max_cat_to_onehot = cfg["max_cat_threshold"], cfg["max_cat_to_onehot"]
    o.min_data_per_group = cfg["min_data_per_group"]
    return o


_LABEL_ERRORS = {1: "binary labels must be 0 / 1", 2: "labels >= 0, sum > 0", 3: "labels >= 0, sum > 0",
                 4: "cross_entropy labels in [0, 1]", 5: "rank labels", 6: "labels in [0, num_class)",
                 7: "every class must appear in every fold"}


def _check_labels(cfg: dict, y: np.ndarray, folds) -> None:
    yy = np.ascontiguousarray(y, dtype=np.float32)
    lo = np.ascontiguousarray([f[0] for f in folds], dtype=np.int64)
    hi = np.ascontiguousarray([f[1] for f in folds], dtype=np.int64)
    rc = _lib().shira_lgb_check_labels(_ptr(yy), yy.size, int(cfg["objective"]), _ptr(lo), _ptr(hi), len(folds),
                                       int(cfg["num_class"]), int(cfg["label_gain"].size))
    assert rc != -1, "shira_lgb_check_labels: bad arguments"
    assert rc == 0, _LABEL_ERRORS.get(rc, f"label check {rc}")


def fold_groups(group, lo: int, hi: int) -> list[int]:
    bounds = np.concatenate([[0], np.cumsum(np.asarray(group, dtype=np.int64))])
    inner = bounds[(bounds > lo) & (bounds < hi)]
    edges = np.concatenate([[lo], inner, [hi]])
    return np.diff(edges).astype(int).tolist()


def _query_bounds(cfg: dict, folds, group) -> np.ndarray | None:
    if cfg["objective"] != 11:
        assert group is None, "group is for lambdarank"
        return None
    assert group is not None, "lambdarank needs group (query sizes)"
    blocks = []
    for lo, hi in folds:
        sizes = fold_groups(group, lo, hi)
        assert max(sizes) <= 10000, "queries above DCGCalculator::kMaxPosition rows"
        blocks.append(np.concatenate([[0], np.cumsum(sizes)]))
    return np.ascontiguousarray(np.concatenate(blocks), dtype=np.int32)


def train(prep: dict, y: np.ndarray, rounds: int, threads: int = 6, trees_out: bool = False,
          weight: np.ndarray | None = None, group=None) -> dict:
    cfg, folds = prep["cfg"], prep["folds"]
    assert 1 <= threads <= MAX_THREADS and rounds >= 1
    codes = np.ascontiguousarray(prep["codes"])
    n, nf = codes.shape
    yy = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
    assert yy.size == n
    _check_labels(cfg, yy, folds)
    if weight is not None:
        require("weights")
    w32 = None if weight is None else np.ascontiguousarray(np.asarray(weight, dtype=np.float32))
    assert w32 is None or (w32.size == n and (w32 >= 0).all())
    lo = np.ascontiguousarray([f[0] for f in folds], dtype=np.int64)
    hi = np.ascontiguousarray([f[1] for f in folds], dtype=np.int64)
    k_n, n_l, n_c = len(folds), cfg["num_leaves"], cfg["num_class"]
    n_t = rounds * n_c
    rec_d = np.array([cfg["min_sum_hessian_in_leaf"], cfg["lambda_l2"], cfg["learning_rate"],
                      cfg["min_gain_to_split"], cfg["sigmoid"]], dtype=np.float64)
    rec_i = np.array([n_l, cfg["min_data"], cfg["objective"]], dtype=np.int64)
    out_pred = np.zeros(int((hi - lo).sum()) * n_c, dtype=np.float64)
    nodes5 = np.zeros((k_n, n_t, n_l - 1, 5), dtype=np.int32) if trees_out else None
    leaf = np.zeros((k_n, n_t, n_l), dtype=np.float64) if trees_out else None
    gain = np.zeros((k_n, n_t, n_l - 1), dtype=np.float64) if trees_out else None
    nleaves = np.zeros((k_n, n_t), dtype=np.int32)
    init = np.zeros((k_n, n_c), dtype=np.float64)
    ntrees = np.zeros(k_n, dtype=np.int32)
    lut = np.ascontiguousarray(prep["lut"])
    assert lut.dtype in (np.uint8, np.uint16), lut.dtype
    opts = _opts(cfg, prep, w32, init, ntrees)
    amb = os.environ.get("SHIRA_LGB_OMP_AMBIGUOUS", "keep").strip().lower()
    assert amb in ("keep", "stock"), f"SHIRA_LGB_OMP_AMBIGUOUS={amb!r}: keep or stock"
    opts.order_mode = 1 if prep.get("order_exact") and amb == "stock" else 0
    prep["_order_counts"] = np.zeros((k_n, 2), dtype=np.int32)
    opts.out_order = _ptr(prep["_order_counts"])
    if lut.dtype == np.uint16:
        opts.lut16 = _ptr(lut)
    rw = fold_rowwise(prep, cfg)
    drate = dense_rates(prep) if rw is not None else None
    if rw is not None and prep.get("drate") is not None:
        drate = np.ascontiguousarray(prep["drate"], dtype=np.float64)
    opts.hist_rowwise, opts.dense_rate = _ptr(rw), _ptr(drate)
    if rw is not None:
        opts.lgb_threads = cfg["omp_threads"]
    cat = np.zeros((k_n, n_t, n_l - 1, 8), dtype=np.uint32) if (trees_out and prep.get("iscat") is not None) else None
    opts.out_cat = _ptr(cat)
    stats = np.zeros((k_n, n_t, 5 * n_l - 2), dtype=np.float64) if trees_out else None
    opts.out_stats = _ptr(stats)
    leaf0 = drop_n = drop_idx = None
    if prep.get("dart_trace") and cfg["dart"]:
        cap = int(cfg["max_drop"]) if int(cfg["max_drop"]) > 0 else rounds
        leaf0 = np.zeros((k_n, n_t, n_l), dtype=np.float64)
        drop_n = np.zeros((k_n, rounds), dtype=np.int32)
        drop_idx = np.full((k_n, rounds, cap), -1, dtype=np.int32)
        opts.out_leaf0, opts.out_drop_n, opts.out_drop_idx, opts.drop_cap = _ptr(leaf0), _ptr(drop_n), _ptr(drop_idx), cap
    qb = _query_bounds(cfg, folds, group)
    assert qb is None or w32 is None, "weights with lambdarank are not verified"
    lgain = np.ascontiguousarray(cfg["label_gain"], dtype=np.float64)
    opts.query_bounds, opts.num_queries = _ptr(qb), 0 if qb is None else int(qb.size)
    opts.truncation_level, opts.lambdarank_norm = cfg["truncation_level"], cfg["lambdarank_norm"]
    opts.label_gain, opts.num_label_gain = _ptr(lgain), int(lgain.size)
    t0 = time.perf_counter()
    rc = _lib().shira_train_lgb_opts(_ptr(codes), n, nf, lut.shape[2], _ptr(yy), _ptr(lo), _ptr(hi), k_n, _ptr(lut),
                                     _ptr(prep["nbin"]), _ptr(prep["mfb"]), _ptr(rec_d), _ptr(rec_i), rounds, threads,
                                     _ptr(out_pred), _ptr(nodes5), _ptr(leaf), _ptr(gain), _ptr(nleaves),
                                     ctypes.byref(opts))
    secs = time.perf_counter() - t0
    if rc < 0:
        raise RuntimeError(f"shira_train_lgb_opts failed ({rc}): " + _ERRORS.get(rc, "unknown error"))
    if rc == 1:
        raise ValueError("a fold's first tree had no split: LightGBM would start from the label average")
    offs = np.concatenate([[0], np.cumsum(hi - lo)]) * n_c
    preds = [out_pred[offs[k]: offs[k + 1]] for k in range(k_n)]
    if n_c > 1:
        preds = [p.reshape(n_c, -1) for p in preds]
    return {"preds": preds, "seconds": secs, "status": rc, "nodes": None if nodes5 is None else nodes5[..., :4],
            "default_left": None if nodes5 is None else nodes5[..., 4], "leaf": leaf, "gain": gain,
            "nleaves": nleaves, "init": init, "ntrees": ntrees, "cat": cat, "stats": stats,
            "leaf0": leaf0, "drop_n": drop_n, "drop_idx": drop_idx}


def cat_bins(res: dict, k: int, t: int, v: int) -> list[int]:
    words = res["cat"][k, t, v]
    return [b for b in range(256) if (int(words[b >> 5]) >> (b & 31)) & 1]


def fold_bins(prep: dict, k: int) -> np.ndarray:
    lo, hi = prep["folds"][k]
    c = prep["codes"][lo:hi].astype(np.int64)
    return np.stack([prep["lut"][k, j][c[:, j]] for j in range(c.shape[1])], axis=1)


def tree_output(res: dict, k: int, r: int, bins: np.ndarray, prep: dict | None = None) -> np.ndarray:
    nl = int(res["nleaves"][k, r])
    if nl <= 1:
        return np.full(bins.shape[0], float(res["leaf"][k, r, 0]))
    nodes, leaf = res["nodes"][k, r], res["leaf"][k, r]
    dl = res["default_left"][k, r] if res.get("default_left") is not None else np.zeros(nl - 1, np.int32)
    cur = np.zeros(bins.shape[0], dtype=np.int64)
    rows = np.arange(bins.shape[0])
    for _ in range(nl):
        act = cur >= 0
        if not act.any():
            break
        nd = cur[act]
        f = nodes[nd, 0]
        b = bins[rows[act], f]
        go_left = b <= nodes[nd, 1]
        if prep is not None:
            mt, db, nb = prep["mtype"][k][f], prep["dbin"][k][f], prep["nbin"][k][f]
            miss = ((mt == MISS_ZERO) & (b == db)) | ((mt == MISS_NAN) & (b == nb - 1))
            go_left = np.where(miss, dl[nd] != 0, go_left)
            if prep.get("iscat") is not None:
                words = res["cat"][k, r][nd, b >> 5].astype(np.int64)
                in_set = ((words >> (b & 31)) & 1) == 1
                go_left = np.where(prep["iscat"][f] == 1, in_set, go_left)
        cur[act] = np.where(go_left, nodes[nd, 2], nodes[nd, 3])
    assert (cur < 0).all()
    return leaf[~cur]


def check_bins_against_dataset(prep: dict, k: int, ds, x: np.ndarray, scratch: Path) -> dict:
    import lightgbm as lgb
    nf = x.shape[1]
    nb_lgb = [ds.feature_num_bin(j) for j in range(nf)]
    n_u = max(u.size for u in prep["uniq"])
    probe = np.stack([np.resize(u, n_u) for u in prep["uniq"]], axis=1)
    pds = lgb.Dataset(probe, reference=ds, params=ds.params).construct()
    path = scratch / f"probe_{k}.txt"
    pds._dump_text(str(path))
    lines = path.read_text().splitlines()
    groups = int(next(ln for ln in lines if ln.startswith("num_groups:")).split()[1])
    feats = int(next(ln for ln in lines if ln.startswith("num_features:")).split()[1])
    rows = [ln for ln in lines if ln and ln[0].isdigit() and ln.endswith(", ")][-n_u:]
    got = np.array([[int(v) for v in ln.rstrip(", ").split(", ")] for ln in rows])
    want = np.stack([np.resize(prep["lut"][k, j, : prep["uniq"][j].size], n_u) for j in range(nf)], axis=1)
    ok_nb = nb_lgb == prep["nbin"][k].tolist()
    return {"fold": k, "num_bin_ok": ok_nb, "groups": groups, "features": feats, "no_bundling": groups == feats,
            "value_bins_ok": bool(got.shape == want.shape and (got == want).all())}


def rank_values(ids: np.ndarray, nleaves: np.ndarray, y: np.ndarray, group: np.ndarray, label_gain: np.ndarray,
                truncation_level: int, lambdarank_norm: int, sigmoid: float, eta: float, lambda_l2: float,
                lambda_l1: float, weight: np.ndarray | None = None, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    ids = np.ascontiguousarray(ids)
    assert ids.ndim == 2 and ids.dtype in (np.uint8, np.uint16)
    T, n = ids.shape
    nl = np.ascontiguousarray(nleaves, dtype=np.int32)
    lmax = int(nl.max())
    yy = np.ascontiguousarray(y, dtype=np.float32)
    qb = np.ascontiguousarray(np.concatenate([[0], np.cumsum(np.asarray(group, dtype=np.int64))]), dtype=np.int32)
    lg = np.ascontiguousarray(label_gain, dtype=np.float64)
    assert yy.shape == (n,) and int(qb[-1]) == n and nl.shape == (T,)
    pr = np.zeros(n, dtype=np.float64)
    leaf = np.zeros((T, lmax), dtype=np.float64)
    if nthread > 1:
        w = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
        assert w is None or w.shape == (n,)
        rc = _lib().shira_lgb_rank_values_t(ids.ctypes.data, ids.dtype.itemsize, n, nl.ctypes.data, T, lmax,
                                            yy.ctypes.data, None if w is None else w.ctypes.data, qb.ctypes.data, qb.size,
                                            lg.ctypes.data, lg.size, int(truncation_level), int(lambdarank_norm),
                                            float(sigmoid), float(eta), float(lambda_l2), float(lambda_l1), int(nthread),
                                            pr.ctypes.data, leaf.ctypes.data)
    elif weight is not None:
        w = np.ascontiguousarray(weight, dtype=np.float32)
        assert w.shape == (n,)
        rc = _lib().shira_lgb_rank_values_w(ids.ctypes.data, ids.dtype.itemsize, n, nl.ctypes.data, T, lmax,
                                            yy.ctypes.data, w.ctypes.data, qb.ctypes.data, qb.size, lg.ctypes.data,
                                            lg.size, int(truncation_level), int(lambdarank_norm), float(sigmoid),
                                            float(eta), float(lambda_l2), float(lambda_l1), pr.ctypes.data,
                                            leaf.ctypes.data)
    else:
        rc = _lib().shira_lgb_rank_values(ids.ctypes.data, ids.dtype.itemsize, n, nl.ctypes.data, T, lmax, yy.ctypes.data,
                                          qb.ctypes.data, qb.size, lg.ctypes.data, lg.size, int(truncation_level),
                                          int(lambdarank_norm), float(sigmoid), float(eta), float(lambda_l2),
                                          float(lambda_l1), pr.ctypes.data, leaf.ctypes.data)
    if rc != 0:
        raise RuntimeError(f"shira_lgb_rank_values failed rc={rc}")
    return leaf, pr
