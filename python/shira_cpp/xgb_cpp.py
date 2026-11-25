from __future__ import annotations

import ctypes
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
LIB_PATH = (Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB")
            else SHIRA / "lib" / ("libshira.dylib" if sys.platform == "darwin" else "libshira.so"))
_P, _I32, _I64, _F = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.c_float
_LIB: ctypes.CDLL | None = None
F32 = np.float32
K_RT_EPS = F32(1e-6)

OBJ = {"reg:pseudohubererror": 0, "binary:logistic": 1, "reg:squarederror": 2, "reg:logistic": 3,
       "binary:logitraw": 4, "count:poisson": 5, "reg:tweedie": 6, "reg:gamma": 7,
       "reg:absoluteerror": 8, "reg:quantileerror": 9, "survival:cox": 10, "multi:softprob": 11,
       "multi:softmax": 11, "rank:pairwise": 12, "rank:ndcg": 13, "rank:map": 14}
LOG_LINK = {"count:poisson", "reg:tweedie", "reg:gamma", "survival:cox"}
KEYS = {"_shira_etas",
        "_shira_start_margin", "_shira_eval_start_margin",
        "_shira_base_margin",
        "objective", "max_depth", "eta", "learning_rate", "lambda", "reg_lambda", "alpha", "reg_alpha",
        "gamma", "min_split_loss", "min_child_weight", "max_delta_step", "base_score", "max_bin",
        "tree_method", "nthread", "seed", "subsample", "colsample_bytree", "colsample_bylevel",
        "colsample_bynode", "huber_slope", "quantile_alpha", "tweedie_variance_power", "num_class",
        "grow_policy", "max_leaves", "monotone_constraints", "interaction_constraints", "shira_lowcost",
        "sampling_method", "verbosity", "disable_default_eval_metric", "lambdarank_pair_method",
        "lambdarank_num_pair_per_sample", "lambdarank_normalization", "lambdarank_score_normalization",
        "ndcg_exp_gain", "lambdarank_unbiased", "enable_categorical", "max_cat_to_onehot",
        "max_cat_threshold"}


GRADFN = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64, ctypes.POINTER(ctypes.c_float),
                          ctypes.c_int64, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float))
OBJ_CUSTOM = 15


class Opts(ctypes.Structure):
    _fields_ = [("weight", _P), ("num_class", _I32), ("grow_policy", _I32), ("max_leaves", _I32),
                ("reg_alpha", _F), ("max_delta_step", _F), ("obj_param", _F),
                ("poisson_max_delta_step", _F), ("monotone", _P), ("inter_feat", _P), ("inter_off", _P),
                ("n_inter", _I32), ("early_stopping_rounds", _I32), ("eval_codes", _P), ("eval_y", _P),
                ("eval_w", _P), ("n_eval", _I64), ("eval_lo", _P), ("eval_hi", _P), ("out_eval", _P),
                ("out_best", _P), ("out_rounds", _P), ("group_ptr", _P), ("n_groups", _I32),
                ("rank_pair_method", _I32), ("rank_num_pair", _I32), ("rank_flags", _I32), ("is_cat", _P),
                ("max_cat_to_onehot", _I32), ("max_cat_threshold", _I32), ("out_cat", _P), ("cat_words", _I32),
                ("out_cover", _P), ("out_gain", _P), ("out_bw", _P),
                ("grad_cb", GRADFN), ("grad_ctx", _P),
                ("etas", _P),
                ("start_margin", _P), ("eval_start_margin", _P)]


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        assert LIB_PATH.exists(), f"build first: {SHIRA / 'build.sh'}"
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.shira_train_xgb_opts.restype = ctypes.c_int
        lib.shira_train_xgb_opts.argtypes = [_P, _I64, _I32, _P, _P, _P, _P, _I32, _P, _P, _I32, _I32,
                                             _P, _P, _P, _P, ctypes.POINTER(Opts)]
        lib.shira_train_xgb_opts_c8.restype = ctypes.c_int
        lib.shira_train_xgb_opts_c8.argtypes = [_P, _I64, _I32, _P, _P, _P, _P, _I32, _P, _P, _I32, _I32,
                                                _P, _P, _P, _P, ctypes.POINTER(Opts)]
        lib.shira_train_xgb_raw.restype = ctypes.c_int
        lib.shira_train_xgb_raw.argtypes = [_P, _I64, _I32, _I32, _P, _P, _P, _I32, _P, _P, _I32, _I32,
                                            _P, _P, _P, _P, ctypes.POINTER(Opts), _P, _P, _P]
        lib.shira_predict_margin.restype = ctypes.c_int
        lib.shira_predict_margin.argtypes = [_P, _I64, _I32, _I32, _P, _P, _P, _F, _P]
        lib.shira_predict_margin_cat.restype = ctypes.c_int
        lib.shira_predict_margin_cat.argtypes = [_P, _I64, _I32, _I32, _P, _P, _P, _P, _I32, _F, _P]
        _LIB = lib
    return _LIB


_LIBM: list = []


def _logf(v: np.float32) -> np.float32:
    if not _LIBM:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("m") or None)
        lib.logf.restype, lib.logf.argtypes = ctypes.c_float, [ctypes.c_float]
        _LIBM.append(lib)
    out = F32(_LIBM[0].logf(float(v)))
    assert np.isfinite(out) or not np.isfinite(v)
    return out


def base_margin(objective: str, base_score: float) -> float:
    b = F32(base_score)
    if objective in ("binary:logistic", "reg:logistic"):
        b = min(max(b, K_RT_EPS), F32(1) - K_RT_EPS)
        return float(-_logf(F32(1) / b - F32(1)))
    if objective in LOG_LINK:
        assert b > 0, f"base_score must be > 0 for {objective}"
        return float(F32(math.log(float(b))))
    return float(b)


def _get(p: dict, *names, default=None):
    for nm in names:
        if nm in p:
            return p[nm]
    return default


def _check(p: dict) -> str:
    bad = set(p) - KEYS
    assert not bad, f"unsupported XGBoost params in shira xgb_cpp: {sorted(bad)}"
    obj = p.get("objective", "reg:squarederror")
    assert obj in OBJ, f"objective {obj} not supported"
    assert p.get("tree_method", "hist") == "hist", "tree_method must be hist"
    assert p.get("sampling_method", "uniform") == "uniform", "only uniform sampling"
    assert "base_score" in p, "base_score is required (intercept estimation is not reproduced)"
    if obj == "reg:quantileerror":
        qa = p.get("quantile_alpha", 0.5)
        assert np.ndim(qa) == 0, "one quantile_alpha only (multi-quantile is refused)"
    assert not p.get("lambdarank_unbiased", False), "lambdarank_unbiased is refused"
    assert p.get("lambdarank_pair_method", "topk") in ("topk", "mean")
    if obj.startswith("multi:"):
        assert int(p.get("num_class", 0)) >= 2, "num_class >= 2 required"
    else:
        assert int(p.get("num_class", 1)) == 1, "num_class only with multi:*"
    return obj


_SHUFFLE: list[int] = []


def shuffle_kind() -> int:
    if not _SHUFFLE:
        import xgboost
        gcc = xgboost.build_info().get("GCC_VERSION")
        _SHUFFLE.append(0 if not gcc else (1 if int(gcc[0]) < 11 else 2))
    return _SHUFFLE[0]


def _rec(p: dict, obj: str, threads: int) -> tuple[np.ndarray, np.ndarray]:
    rec_d = np.array([float(_get(p, "lambda", "reg_lambda", default=1.0)),
                      float(p.get("min_child_weight", 1.0)), float(_get(p, "gamma", "min_split_loss", default=0.0)),
                      float(_get(p, "eta", "learning_rate", default=0.3)),
                      float(p["_shira_base_margin"]) if p.get("_shira_base_margin") is not None
                      else base_margin(obj, p["base_score"]),
                      float(p.get("subsample", 1.0)), float(p.get("colsample_bytree", 1.0)),
                      float(p.get("colsample_bylevel", 1.0)), float(p.get("colsample_bynode", 1.0))],
                     dtype=np.float64)
    md = int(p.get("max_depth", 6))
    rec_i = np.array([md, OBJ[obj], int(p.get("seed", 0)) & 0xFFFFFFFF, int(p.get("nthread", threads)),
                      int(p.get("shira_lowcost", 0)) | (shuffle_kind() << 8)], dtype=np.int64)
    return rec_d, rec_i


def _constraints(p: dict, nf: int, keep: list) -> tuple:
    mono_p, inter_p, n_inter = None, None, 0
    mc = p.get("monotone_constraints")
    if mc is not None:
        if isinstance(mc, str):
            mc = [int(v) for v in mc.strip("()").split(",") if v.strip()]
        mono = np.zeros(nf, dtype=np.int8)
        mono[:len(mc)] = np.asarray(mc, dtype=np.int8)
        assert len(mc) <= nf and np.all(np.abs(mono) <= 1)
        keep.append(mono)
        mono_p = mono.ctypes.data
    ic = p.get("interaction_constraints")
    if ic is not None:
        if isinstance(ic, str):
            ic = json.loads(ic)
        feat = np.array([int(f) for s in ic for f in s], dtype=np.int32)
        off = np.concatenate([[0], np.cumsum([len(s) for s in ic])]).astype(np.int32)
        keep += [feat, off]
        inter_p, n_inter = (feat.ctypes.data, off.ctypes.data), len(ic)
    return mono_p, inter_p, n_inter


def _opts(p: dict, obj: str, nf: int, weight, keep: list) -> Opts:
    o = Opts()
    if weight is not None:
        w = np.ascontiguousarray(weight, dtype=F32)
        keep.append(w)
        o.weight = w.ctypes.data
    o.num_class = int(p.get("num_class", 1))
    o.grow_policy = 1 if p.get("grow_policy", "depthwise") == "lossguide" else 0
    assert p.get("grow_policy", "depthwise") in ("depthwise", "lossguide")
    o.max_leaves = int(p.get("max_leaves", 0))
    o.reg_alpha = float(_get(p, "alpha", "reg_alpha", default=0.0))
    o.max_delta_step = float(p.get("max_delta_step", 0.0))
    o.obj_param = float({"reg:pseudohubererror": p.get("huber_slope", 1.0),
                         "reg:quantileerror": p.get("quantile_alpha", 0.5),
                         "reg:tweedie": p.get("tweedie_variance_power", 1.5)}.get(obj, 1.0))
    o.poisson_max_delta_step = float(p.get("max_delta_step", 0.7))
    mono_p, inter_p, n_inter = _constraints(p, nf, keep)
    o.monotone = mono_p
    if inter_p is not None:
        o.inter_feat, o.inter_off = inter_p
    o.n_inter = n_inter
    if p.get("_shira_etas") is not None:
        et = np.ascontiguousarray(p["_shira_etas"], dtype=F32)
        assert et.ndim == 1 and et.size >= 1 and bool(np.all(et > 0))
        keep.append(et)
        o.etas = et.ctypes.data
    for key, field in (("_shira_start_margin", "start_margin"), ("_shira_eval_start_margin", "eval_start_margin")):
        if p.get(key) is not None:
            sm = np.ascontiguousarray(p[key], dtype=F32).reshape(-1)
            assert sm.size >= 1 and bool(np.all(np.isfinite(sm)))
            keep.append(sm)
            setattr(o, field, sm.ctypes.data)
    return o


def _rank_opts(o: Opts, p: dict, obj: str, group_ptr, n: int, keep: list) -> None:
    if not obj.startswith("rank:"):
        assert group_ptr is None, "group_ptr only with rank:* objectives"
        return
    assert group_ptr is not None, "rank:* needs group_ptr (query group row offsets)"
    gp = np.ascontiguousarray(group_ptr, dtype=np.int64)
    assert gp[0] == 0 and gp[-1] <= n and np.all(np.diff(gp) > 0)
    keep.append(gp)
    o.group_ptr, o.n_groups = gp.ctypes.data, gp.size - 1
    o.rank_pair_method = 1 if p.get("lambdarank_pair_method", "topk") == "mean" else 0
    o.rank_num_pair = int(p.get("lambdarank_num_pair_per_sample", 0))
    o.rank_flags = (int(bool(p.get("lambdarank_normalization", True))) | 2 * int(bool(p.get("lambdarank_score_normalization", True)))
                    | 4 * int(bool(p.get("ndcg_exp_gain", True))))


def _eval_opts(o: Opts, ev, esr: int, k: int, rounds: int, keep: list, out: dict) -> None:
    if ev is None:
        assert esr == 0, "early stopping needs an eval set"
        return
    ecodes, ey, ew, eranges = ev
    ec = np.ascontiguousarray(ecodes.view(np.uint16) if ecodes.dtype == np.int16 else ecodes, dtype=np.uint16)
    eyy = np.ascontiguousarray(ey, dtype=F32)
    elo = np.array([a for a, _ in eranges], dtype=np.int64)
    ehi = np.array([b for _, b in eranges], dtype=np.int64)
    assert len(eranges) == k and ec.shape[0] == eyy.shape[0]
    out["eval"] = np.zeros((k, rounds), dtype=np.float64)
    out["best"] = np.zeros(k, dtype=np.int32)
    out["rounds"] = np.zeros(k, dtype=np.int32)
    keep += [ec, eyy, elo, ehi]
    o.eval_codes, o.eval_y, o.n_eval = ec.ctypes.data, eyy.ctypes.data, ec.shape[0]
    if ew is not None:
        eww = np.ascontiguousarray(ew, dtype=F32)
        keep.append(eww)
        o.eval_w = eww.ctypes.data
    o.eval_lo, o.eval_hi = elo.ctypes.data, ehi.ctypes.data
    o.out_eval, o.out_best, o.out_rounds = (out["eval"].ctypes.data, out["best"].ctypes.data,
                                            out["rounds"].ctypes.data)
    o.early_stopping_rounds = int(esr)


def bin_codes(ip: np.ndarray, cuts: np.ndarray, x: np.ndarray) -> np.ndarray:
    codes = np.zeros(x.shape, dtype=np.uint16)
    for j in range(x.shape[1]):
        cj = cuts[ip[j]:ip[j + 1]].astype(F32)
        c = np.clip(np.searchsorted(cj, x[:, j], side="right"), 1, cj.size - 1)
        codes[:, j] = np.where(np.isnan(x[:, j]), 0, c)
    return codes


def cat_codes(cuts_f: np.ndarray, col: np.ndarray) -> np.ndarray:
    cj = cuts_f.astype(F32)
    pos = np.searchsorted(cj, col, side="left")
    ok = np.isnan(col) | ((pos < cj.size) & (cj[np.minimum(pos, cj.size - 1)] == col))
    assert np.all(ok), "category value not in the cuts"
    return np.where(np.isnan(col), 0, pos + 1).astype(np.uint16)


def _cat_opts(o: Opts, p: dict, cat_features, nbf: np.ndarray, shape: tuple, keep: list, out: dict) -> None:
    if not cat_features:
        assert "max_cat_to_onehot" not in p and "max_cat_threshold" not in p, "categorical params without cat_features"
        return
    is_cat = np.zeros(nbf.size, dtype=np.uint8)
    is_cat[list(cat_features)] = 1
    words = int(max(1, (int(nbf[is_cat == 1].max()) + 31) // 32))
    out["cat"] = np.zeros(shape + (words,), dtype=np.uint32)
    keep.append(is_cat)
    o.is_cat, o.cat_words, o.out_cat = is_cat.ctypes.data, words, out["cat"].ctypes.data
    o.max_cat_to_onehot = int(p.get("max_cat_to_onehot", 4))
    o.max_cat_threshold = int(p.get("max_cat_threshold", 64))


def maxn_of(p: dict) -> int:
    if p.get("grow_policy", "depthwise") == "lossguide":
        ml = int(p.get("max_leaves", 0))
        leaves = ml if ml > 0 else (1 << min(int(p.get("max_depth", 6)), 12))
        return 2 * leaves + 1
    return 1 << (int(p.get("max_depth", 6)) + 1)


def train(params: dict, codes: np.ndarray, nbf: np.ndarray, y: np.ndarray, folds: list[tuple[int, int]],
          rounds: int, threads: int = 4, weight: np.ndarray | None = None, evals=None,
          early_stopping_rounds: int = 0, group_ptr: np.ndarray | None = None,
          cat_features: list[int] | None = None, grad_fn=None) -> dict:
    obj = _check(params)
    c8 = codes.dtype == np.uint8
    c = np.ascontiguousarray(codes) if c8 else np.ascontiguousarray(codes.view(np.uint16) if codes.dtype == np.int16
                                                                    else codes, dtype=np.uint16)
    n, nf = c.shape
    nbf32 = np.ascontiguousarray(nbf, dtype=np.int32)
    assert not c8 or (int(nbf32.max()) <= 256 and cat_features is None), "uint8 codes: <= 256 bins, numeric features"
    yy = np.ascontiguousarray(y, dtype=F32)
    assert nbf32.size == nf and yy.size == n and folds and 0 < rounds <= 100000 and 1 <= threads <= 64
    keep: list = []
    rec_d, rec_i = _rec(params, obj, threads)
    o = _opts(params, obj, nf, weight, keep)
    out: dict = {}
    _eval_opts(o, evals, early_stopping_rounds, len(folds), rounds, keep, out)
    _rank_opts(o, params, obj, group_ptr, n, keep)
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    k, C, maxn = len(folds), int(o.num_class), maxn_of(params)
    out["nodes"] = np.zeros((k, rounds * C, maxn, 5), dtype=np.int32)
    out["leaf"] = np.zeros((k, rounds * C, maxn), dtype=F32)
    out["count"] = np.zeros((k, rounds * C), dtype=np.int32)
    out["pred"] = np.zeros((k, C, n), dtype=F32)
    out["cover"] = np.zeros((k, rounds * C, maxn), dtype=F32)
    out["gain"] = np.zeros((k, rounds * C, maxn), dtype=F32)
    o.out_cover, o.out_gain = out["cover"].ctypes.data, out["gain"].ctypes.data
    out["bw"] = np.zeros((k, rounds * C, maxn), dtype=F32)
    o.out_bw = out["bw"].ctypes.data
    _cat_opts(o, params, cat_features, nbf32, (k, rounds * C, maxn), keep, out)
    err: list = []
    if grad_fn is not None:
        assert k == 1 and C == 1 and evals is None and group_ptr is None, "custom objective: one fold, one output"
        rec_i[1] = OBJ_CUSTOM
        o.weight = None
        keep.append(_custom_cb(grad_fn, err))
        o.grad_cb = keep[-1]
    entry = _lib().shira_train_xgb_opts_c8 if c8 else _lib().shira_train_xgb_opts
    rc = entry(c.ctypes.data, n, nf, nbf32.ctypes.data, yy.ctypes.data, lo.ctypes.data,
               hi.ctypes.data, k, rec_d.ctypes.data, rec_i.ctypes.data, rounds, threads,
               out["nodes"].ctypes.data, out["leaf"].ctypes.data, out["count"].ctypes.data,
               out["pred"].ctypes.data, ctypes.byref(o))
    if rc == -3 and err:
        raise err[0]
    if rc != 0:
        raise RuntimeError(f"shira_train_xgb_opts failed rc={rc} (-1 refused input, -2 engine check)")
    out["num_class"] = C
    return out


def _custom_cb(grad_fn, err: list):
    def cb(_ctx, _rnd, _lo, margins, m, g, h):
        try:
            pr = np.ctypeslib.as_array(margins, shape=(int(m),)).copy()
            gg, hh = grad_fn(pr)
            ga = np.ascontiguousarray(gg, dtype=F32).reshape(-1)
            ha = np.ascontiguousarray(hh, dtype=F32).reshape(-1)
            if ga.size != m or ha.size != m:
                raise ValueError(f"the custom objective returned {ga.size} / {ha.size} values for {int(m)} rows")
            np.ctypeslib.as_array(g, shape=(int(m),))[:] = ga
            np.ctypeslib.as_array(h, shape=(int(m),))[:] = ha
            return 0
        except BaseException as e:
            err.append(e)
            return 1
    return GRADFN(cb)


def predict(out: dict, k: int, codes: np.ndarray, base: float, n_rounds: int | None = None) -> np.ndarray:
    C = int(out["num_class"])
    nr = int(out["rounds"][k]) if n_rounds is None and "rounds" in out else (n_rounds or out["count"].shape[1] // C)
    c = np.ascontiguousarray(codes.view(np.uint16) if codes.dtype == np.int16 else codes, dtype=np.uint16)
    n, nf = c.shape
    res = np.empty((n, C), dtype=F32)
    for cls in range(C):
        ts = [r * C + cls for r in range(nr)]
        cnt = out["count"][k, ts].astype(np.int64)
        assert np.all(cnt > 0)
        off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int32)
        nodes = np.ascontiguousarray(np.concatenate([out["nodes"][k, t, :m] for t, m in zip(ts, cnt)]), dtype=np.int32)
        leaf = np.ascontiguousarray(np.concatenate([out["leaf"][k, t, :m] for t, m in zip(ts, cnt)]), dtype=F32)
        col = np.empty(n, dtype=F32)
        if "cat" in out:
            bits = np.ascontiguousarray(np.concatenate([out["cat"][k, t, :m] for t, m in zip(ts, cnt)]), dtype=np.uint32)
            rc = _lib().shira_predict_margin_cat(c.ctypes.data, n, nf, len(ts), off.ctypes.data, nodes.ctypes.data,
                                                 leaf.ctypes.data, bits.ctypes.data, bits.shape[1], F32(base),
                                                 col.ctypes.data)
        else:
            rc = _lib().shira_predict_margin(c.ctypes.data, n, nf, len(ts), off.ctypes.data, nodes.ctypes.data,
                                             leaf.ctypes.data, F32(base), col.ctypes.data)
        if rc != 0:
            raise RuntimeError(f"shira predict failed rc={rc}")
        res[:, cls] = col
    return res


def train_raw(params: dict, x: np.ndarray, y: np.ndarray, folds: list[tuple[int, int]], rounds: int,
              threads: int = 4, weight: np.ndarray | None = None, group_ptr: np.ndarray | None = None) -> dict:
    obj = _check(params)
    xx = np.ascontiguousarray(x, dtype=F32)
    n, nf = xx.shape
    max_bin = int(params.get("max_bin", 256))
    yy = np.ascontiguousarray(y, dtype=F32)
    assert yy.size == n and folds and 0 < rounds <= 100000 and 1 <= threads <= 64
    keep: list = []
    rec_d, rec_i = _rec(params, obj, threads)
    o = _opts(params, obj, nf, weight, keep)
    _rank_opts(o, params, obj, group_ptr, n, keep)
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    k, C, maxn = len(folds), int(o.num_class), maxn_of(params)
    out = {"nodes": np.zeros((k, rounds * C, maxn, 5), dtype=np.int32), "leaf": np.zeros((k, rounds * C, maxn), dtype=F32),
           "count": np.zeros((k, rounds * C), dtype=np.int32), "pred": np.zeros((k, C, n), dtype=F32),
           "ptrs": np.zeros(nf + 1, dtype=np.uint32), "values": np.zeros(nf * max_bin, dtype=F32),
           "min": np.zeros(nf, dtype=F32), "num_class": C}
    rc = _lib().shira_train_xgb_raw(xx.ctypes.data, n, nf, max_bin, yy.ctypes.data, lo.ctypes.data, hi.ctypes.data, k,
                                    rec_d.ctypes.data, rec_i.ctypes.data, rounds, threads, out["nodes"].ctypes.data,
                                    out["leaf"].ctypes.data, out["count"].ctypes.data, out["pred"].ctypes.data,
                                    ctypes.byref(o), out["ptrs"].ctypes.data, out["values"].ctypes.data,
                                    out["min"].ctypes.data)
    if rc != 0:
        raise RuntimeError(f"shira_train_xgb_raw failed rc={rc} (-1 refused input, -2 engine check)")
    return out
