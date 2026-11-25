from __future__ import annotations

import functools
import logging
import math
import os
import time
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import numpy as np

from shira_cpp import binning_cpp, product_cpp

from . import _lgb_fit as _fit
from . import _machine

LOG = logging.getLogger("shira.lightgbm")
FIRST_READ_M0 = 100_000
FIRST_READ_GAIN = 0.005
FIRST_READ_HOLD = 20
FIRST_READ_HOLD_ROWS = 100_000
FIRST_READ_AUTO_MIN = 3_000_000
FIRST_READ_SCORE = 2
FIRST_READ_RICH = 0.015
RF_ETA = 1e-200
K_EPS = 1e-15
OBJ = {0: "reg:squarederror", 1: "binary:logistic"}
OBJ_L3 = {7: "count:poisson", 8: "reg:gamma", 9: "reg:tweedie", 10: "binary:logistic"}
first_read_log: list[dict] = []


FIRST_READ_DEFAULT_ROWS = 200_000


def _mode() -> str:
    v = os.environ.get("SHIRA_LGB_MODE", "").strip().lower()
    return v if v in ("first_read", "first_read_auto") else ("ordinary" if v else "first_read")


def mode() -> bool:
    return _mode() != "ordinary"


def _sample_threads(option: str, th: int, features: int) -> int:
    assert th >= 1 and option in ("linear",)
    v = os.environ.get("SHIRA_LGB_FIRST_READ_SAMPLE_THREADS", "").strip()
    n = int(v) if v.isdigit() else _machine.value("lgb_first_read_sample_threads_" + option, features)
    return th if n <= 0 else min(th, n)


def fixed_rows(features: int, option: str = "", rows: int = 0) -> int | None:
    assert option in ("", "mono", "linear")
    v = os.environ.get("SHIRA_LGB_FIRST_READ_ROWS", "").strip()
    if not v:
        if _mode() == "first_read_auto":
            return None
        m = _machine.value("lgb_first_read_rows" + ("_" + option if option else ""), features)
        mid = _machine.value("lgb_first_read_rows_mid", features) if (rows and not option) else 0
        lo = _machine.value("lgb_first_read_mid_min_rows", features)
        if mid and lo <= rows < _machine.value("lgb_first_read_mid_below_rows", features):
            m = mid
        small = _machine.value("lgb_first_read_rows_small", features) if (rows and not option) else 0
        if small and _machine.value("lgb_first_read_small_min_rows", features) <= rows < _machine.value(
                "lgb_first_read_small_below_rows", features):
            m = small
        return m
    if not v.isdigit() or int(v) < 1000:
        raise ValueError(f"SHIRA_LGB_FIRST_READ_ROWS={v!r}: a whole number of rows, at least 1000")
    return int(v)


def reason(plan: SimpleNamespace, cfg: dict) -> str | None:
    checks = [
        (_early(plan) is not None and (not plan.valid or bool(plan.tr.cat) or bool(getattr(plan, "rf", False))
                                         or getattr(plan, "mono", None) is not None or bool(getattr(plan, "linear", False))),
         "early stopping without a validation set, or with categorical features, rf, monotone constraints or linear_tree"),
        (plan.tr.g is not None and (cfg["objective"] != 11
                                     or bool(getattr(plan, "rf", False)) or getattr(plan, "mono", None) is not None
                                     or bool(cfg.get("dart")) or bool(cfg.get("goss")) or float(cfg["path_smooth"]) != 0.0),
         "a query group (lambdarank takes it, but not with rf, monotone constraints, dart, goss or path_smooth)"),
        (plan.tr.g is not None and int(plan.tr.x.shape[0]) < int(os.environ.get("SHIRA_LGB_FIRST_READ_RANK_MIN_ROWS", "")
                                                                 or _machine.value("lgb_first_read_rank_min_rows", int(plan.tr.x.shape[1]))),
         "lambdarank: the first read is not measured faster than stock on this machine at this size (switch "
         "lgb_first_read_rank_min_rows; SHIRA_LGB_FIRST_READ_RANK_MIN_ROWS=0 forces it)"),
        (cfg["objective"] not in OBJ and cfg["objective"] not in OBJ_L3 and cfg["objective"] not in (2, 3, 4, 5, 6, 11),
         "objective other than regression (L2 / L1), binary, multiclass(ova), huber, quantile, poisson, gamma, tweedie or "
         "cross_entropy"),
        (cfg["objective"] in (4, 6) and (bool(getattr(plan, "rf", False)) or getattr(plan, "mono", None) is not None),
         "regression_l1 / quantile with rf or monotone constraints"),
        (_lgbg_kind(cfg) is not None and bool(getattr(plan, "rf", False)),
         "huber / binary with sigmoid != 1 with rf"),
        (cfg["objective"] == 2 and bool(getattr(plan, "rf", False)),
         "multiclass with rf"),
        (cfg["objective"] in OBJ_L3 and bool(getattr(plan, "rf", False)),
         "poisson / gamma / tweedie / cross_entropy with rf"),
        (cfg["objective"] == 3 and (bool(getattr(plan, "rf", False)) or getattr(plan, "mono", None) is not None
                                    or bool(cfg.get("dart")) or float(cfg["path_smooth"]) != 0.0
                                    or bool(getattr(plan, "linear", False))),
         "multiclassova with rf, monotone constraints, dart, path_smooth or linear_tree"),
        ((float(cfg.get("scale_pos_weight", 1.0)) != 1.0 or bool(int(cfg.get("is_unbalance", 0))))
         and (int(cfg["objective"]) != 1 or _lgbg_kind(cfg) is not None
                                                            or getattr(plan, "mono", None) is not None
                                                            or bool(cfg.get("dart")) or float(cfg["path_smooth"]) != 0.0
                                                            or getattr(plan, "custom_obj", None) is not None
                                                            or getattr(plan, "builtin_obj", None) is not None),
         "scale_pos_weight / is_unbalance off the plain binary value pass (sigmoid 1)"),
        (bool(cfg.get("dart")) and (cfg["objective"] not in OBJ or _lgbg_kind(cfg) is not None
                                    or bool(int(cfg.get("xgboost_dart_mode", 0))) or bool(int(cfg.get("uniform_drop", 0)))
                                    or float(cfg["path_smooth"]) != 0.0 or bool(getattr(plan, "rf", False))
                                    or getattr(plan, "mono", None) is not None),
         "dart with an objective other than L2 / binary (sigmoid 1), xgboost_dart_mode, uniform_drop, path_smooth, rf or "
         "monotone constraints"),
        (bool(cfg.get("goss")) and (bool(getattr(plan, "rf", False)) or getattr(plan, "mono", None) is not None),
         "goss with rf or monotone constraints"),
        (float(cfg["path_smooth"]) != 0.0 and (cfg["objective"] not in OBJ or _lgbg_kind(cfg) is not None
                                                 or bool(getattr(plan, "rf", False)) or getattr(plan, "mono", None) is not None),
         "path_smooth with an objective other than L2 / binary (sigmoid 1), rf or monotone constraints"),
        (float(cfg["lambda_l1"]) != 0.0 and bool(getattr(plan, "rf", False)), "lambda_l1 with rf"),
        (float(cfg.get("max_delta_step", 0.0)) > 0.0 and (cfg["objective"] in (2, 11) or _lgbg_kind(cfg) is not None
                                                          or bool(cfg.get("dart")) or bool(getattr(plan, "linear", False))
                                                          or getattr(plan, "mono", None) is not None),
         "max_delta_step with multiclass, lambdarank, huber / sigmoid != 1, dart, linear_tree or monotone constraints"),
    ]
    for bad, why in checks:
        if bad:
            return why
    return None


def _fobj_dataset(plan: SimpleNamespace) -> Any:
    from . import lightgbm as slg

    ds = plan.tr.ds
    if getattr(ds, "_handle", None) is not None:
        return ds
    tr = plan.tr
    out = slg._basic.Dataset(np.zeros((tr.x.shape[0], 1), dtype=np.float32), label=tr.y, weight=tr.w, group=tr.g,
                             params={"verbose": -1}, free_raw_data=False)
    return out.construct()


def _values_custom(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict, th: int,
                   *, fobj: Any, ds: Any) -> tuple[np.ndarray, np.ndarray, float]:
    assert th >= 1 and ids.shape[0] == m.leaf.shape[0]
    T, n = int(ids.shape[0]), int(ids.shape[1])
    l1, l2 = float(cfg["lambda_l1"]), float(cfg["lambda_l2"])
    eta, mds = float(cfg["learning_rate"]), float(cfg.get("max_delta_step", 0.0))
    score = np.zeros(n, dtype=np.float64)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    for t in range(T):
        g, h = fobj(score.copy(), ds)
        g = np.asarray(g, dtype=np.float32).astype(np.float64).ravel()
        h = np.asarray(h, dtype=np.float32).astype(np.float64).ravel()
        assert g.shape == (n,) and h.shape == (n,), "a custom objective must give one gradient per row"
        k, idx = int(nl[t]), ids[t].astype(np.int64)
        G, H = np.bincount(idx, g, k), np.bincount(idx, h, k)
        out = -np.sign(G) * np.maximum(np.abs(G) - l1, 0.0) / (H + l2)
        if mds > 0.0:
            out = np.clip(out, -mds, mds)
        leaf[t, :k] = out * eta
        score += leaf[t, idx]
    return leaf, score, 0.0


def _values_fair(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict, th: int,
                 *, c: float) -> tuple[np.ndarray, np.ndarray, float]:
    assert th >= 1 and c > 0.0
    init = _init(0, y, w, bool(cfg["boost_from_average"]))
    leaf64, scores = product_cpp.product_lgb(ids, nl, y.astype(np.float32), init, "fair", c, eta=float(cfg["learning_rate"]),
                                             reg_lambda=float(cfg["lambda_l2"]), reg_alpha=float(cfg["lambda_l1"]),
                                             min_child_weight=float(cfg.get("min_sum_hessian_in_leaf", 1e-3)),
                                             weight=None if w is None else w.astype(np.float32), nthread=th)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, scores, init


def _values_mape(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
                 th: int) -> tuple[np.ndarray, np.ndarray, float]:
    lw = (1.0 / np.maximum(1.0, np.abs(y.astype(np.float64)))).astype(np.float32)
    if w is not None:
        lw = (lw * w.astype(np.float32)).astype(np.float32)
    init = float(np.float32(_lgb_wpercentile(y.astype(np.float32).astype(np.float64), lw, 0.5))) \
        if bool(cfg["boost_from_average"]) else 0.0
    leaf64, scores = product_cpp.product_renew_w(ids, nl, y.astype(np.float32), lw, init, 0.5,
                                                 eta=float(cfg["learning_rate"]), nthread=th)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, scores, init


def _init(obj: int, y: np.ndarray, w: np.ndarray | None, boost: bool) -> float:
    if not boost:
        return 0.0
    yd = y.astype(np.float64)
    avg = float(yd.mean()) if w is None else float((yd * w).sum() / w.astype(np.float64).sum())
    if obj == 0:
        return avg
    p = min(max(avg, K_EPS), 1.0 - K_EPS)
    return float(np.log(p / (1.0 - p)))


def _nodes(m: Any) -> tuple[np.ndarray, np.ndarray]:
    T, L1 = m.nodes.shape[0], m.nodes.shape[1]
    maxn = 2 * L1 + 1
    out = np.zeros((T, maxn, 5), dtype=np.int32)
    count = np.zeros(T, dtype=np.int32)
    for t in range(T):
        nl = int(m.nleaves[t])
        ni = nl - 1
        count[t] = 2 * nl - 1
        nd = m.nodes[t, :ni].astype(np.int64)
        kid = lambda c: np.where(c >= 0, c, ni + ~c)
        out[t, :ni, 0] = nd[:, 0]
        out[t, :ni, 1] = nd[:, 1] + 1
        out[t, :ni, 2] = m.default_left[t, :ni].astype(np.int32)
        out[t, :ni, 3] = kid(nd[:, 2])
        out[t, :ni, 4] = kid(nd[:, 3])
        out[t, ni:ni + nl, 0] = -1
    return out, count


CAT_WORDS = 9


def _cat_feats(m: Any) -> list[int]:
    b2c = getattr(m, "bin2cat", None)
    return [] if b2c is None else [j for j, b in enumerate(b2c) if b is not None]


def _cat_table(m: Any, nodes: np.ndarray, cats: list[int]) -> np.ndarray | None:
    iscat = np.zeros(len(m.upper), dtype=bool)
    iscat[cats] = True
    words: list[np.ndarray] = []
    for t in range(nodes.shape[0]):
        ni = int(m.nleaves[t]) - 1
        for i in np.flatnonzero(iscat[nodes[t, :ni, 0]]).tolist():
            bits = int.from_bytes(np.ascontiguousarray(m.cat[t, i], dtype="<u4").tobytes(), "little") << 1
            nodes[t, i, 1], nodes[t, i, 2] = CAT_WORDS * len(words), 2
            words.append(np.frombuffer(bits.to_bytes(4 * CAT_WORDS, "little"), dtype="<u4"))
    assert all(w.size == CAT_WORDS for w in words)
    return np.concatenate(words) if words else None


def _leaf_ids_cat(m: Any, x: np.ndarray, th: int, cats: list[int]) -> tuple[np.ndarray, np.ndarray]:
    cs = set(cats)
    plain = [np.array([np.inf]) if j in cs else np.asarray(u, dtype=np.float64) for j, u in enumerate(m.upper)]
    info = np.ascontiguousarray(m.info, dtype=np.int32).copy()
    info[cats] = (1, 0, 0, 0)
    ptrs = np.concatenate([[0], np.cumsum([u.size for u in plain])]).astype(np.int32)
    bins, has_inf = binning_cpp.lgb_bin2(x, ptrs, np.concatenate(plain), info, nthread=th, code_bytes=2)
    if has_inf:
        raise _fit.Unsupported("infinite feature values")
    for j in cats:
        bins[:, j] = binning_cpp.lgb_cat_bin(x[:, j], np.asarray(m.bin2cat[j], dtype=np.int64))
    codes = bins
    codes += 1
    for j in range(codes.shape[1]):
        nb, mt, db = int(m.info[j, 0]), int(m.info[j, 1]), int(m.info[j, 2])
        if j in cs or mt == 0:
            continue
        if mt == 2:
            codes[codes[:, j] == nb, j] = 0
        else:
            codes[codes[:, j] == db + 1, j] = 0
    nodes, count = _nodes(m)
    table = _cat_table(m, nodes, cats)
    if table is None:
        return product_cpp.leaf_ids(codes, nodes, count, nthread=th)
    return product_cpp.leaf_ids_cat(codes, nodes, count, table, nthread=th)


def _leaf_ids(m: Any, x: np.ndarray, th: int) -> tuple[np.ndarray, np.ndarray]:
    if getattr(m, "grower", "shira") == "stock":
        return _leaf_ids_stock(m, x, th)
    cats = _cat_feats(m)
    if cats:
        return _leaf_ids_cat(m, x, th, cats)
    ptrs = np.concatenate([[0], np.cumsum([u.size for u in m.upper])]).astype(np.int32)
    if not bool((m.info[:, 1] != 0).any()) and int(m.info[:, 0].max()) <= 256:
        bins8, has_inf = binning_cpp.lgb_bin2(x, ptrs, np.concatenate(m.upper), m.info, nthread=th, code_bytes=1)
        if has_inf:
            raise _fit.Unsupported("infinite feature values")
        nodes, count = _nodes(m)
        return product_cpp.leaf_ids(bins8, nodes, count, nthread=th)
    bins, has_inf = binning_cpp.lgb_bin2(x, ptrs, np.concatenate(m.upper), m.info, nthread=th, code_bytes=2)
    if has_inf:
        raise _fit.Unsupported("infinite feature values")
    codes = bins
    codes += 1
    for j in range(codes.shape[1]):
        nb, mt, db = int(m.info[j, 0]), int(m.info[j, 1]), int(m.info[j, 2])
        if mt == 2:
            codes[codes[:, j] == nb, j] = 0
        elif mt == 1:
            codes[codes[:, j] == db + 1, j] = 0
    nodes, count = _nodes(m)
    return product_cpp.leaf_ids(codes, nodes, count, nthread=th)


CORR_SWEEPS = 10
CORR_HOLD = 10
corr_log: list[dict] = []


def _corr_switch(name: str, default: int, lo: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v.isdigit() and int(v) >= lo else default


def _label_weights(y: np.ndarray, w: np.ndarray | None, unb: bool) -> tuple[float, float]:
    if not unb:
        return 1.0, 1.0
    rows = np.asarray(y) if w is None else np.asarray(y)[np.asarray(w) > 0]
    pos = int(np.sum(rows > 0))
    neg = int(rows.size) - pos
    if pos == 0 or neg == 0:
        return 1.0, 1.0
    return (1.0, pos / neg) if pos > neg else (neg / pos, 1.0)


def _values(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
            th: int, shape_rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    obj = int(cfg["objective"])
    init = _init(obj, y, w, bool(cfg["boost_from_average"]))
    spw = float(cfg.get("scale_pos_weight", 1.0))
    unb = bool(int(cfg.get("is_unbalance", 0)))
    if (spw != 1.0 or unb) and obj == 1:
        base = np.ones(y.size) if w is None else np.asarray(w, dtype=np.float64)
        lw_pos, lw_neg = _label_weights(y, w, unb)
        w = np.where(np.asarray(y) > 0, base * (lw_pos * spw), base * lw_neg)
    l1 = float(cfg["lambda_l1"])
    kw = dict(eta=float(cfg["learning_rate"]), reg_lambda=float(cfg["lambda_l2"]),
              min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
              max_delta_step=float(cfg.get("max_delta_step", 0.0)),
              weight=None if w is None else w.astype(np.float32), nthread=th)
    if l1 == 0.0:
        leaf32, pred = product_cpp.product(ids, nl, y.astype(np.float32), init, OBJ[obj], **kw)
    else:
        leaf32, pred = product_cpp.product_l1(ids, nl, y.astype(np.float32), init, OBJ[obj], reg_alpha=l1, **kw)
    sweeps = _corr_switch("SHIRA_LGB_CORR_SWEEPS", CORR_SWEEPS, 0)
    why = ("no shape rows given" if shape_rows is None else "SHIRA_LGB_CORR_SWEEPS=0" if sweeps == 0
           else "lambda_l1" if l1 != 0.0 else "max_delta_step" if kw["max_delta_step"] != 0.0 else None)
    if why is None:
        lv, mg, log = product_cpp.corr_drive(ids, nl, y, kw["weight"], OBJ[obj], kw["reg_lambda"], kw["min_child_weight"],
                                             init, leaf32, shape_rows, sweeps, _corr_switch("SHIRA_LGB_CORR_HOLD",
                                                                                            CORR_HOLD, 2), nthread=th,
                                             margin0=pred)
        corr_log.append(log)
        LOG.info("shira.lightgbm: amortised corrections: %d of at most %d sweeps kept by the held-back loss "
                 "(%s rows outside the shape sample); SHIRA_LGB_CORR_SWEEPS=0 turns them off", log["sweeps"], sweeps,
                 log.get("held_back", 0))
        if lv is not None:
            leaf32, pred = lv, mg
    elif shape_rows is not None:
        corr_log.append({"sweeps": 0, "reason": why})
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf32.shape[1]] = leaf32
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, pred.astype(np.float64), init


def mono_signs(cfg: dict, nf: int) -> np.ndarray | None:
    txt = str(cfg.get("monotone_constraints", "")).strip()
    if not txt:
        return None
    v = np.array([int(float(t)) for t in txt.split(",")], dtype=np.int32)
    if v.size != nf or not np.isin(v, (-1, 0, 1)).all():
        raise _fit.Unsupported("monotone_constraints: not one of -1 / 0 / 1 per feature (stock decides)")
    return v if bool(v.any()) else None


def mono_reason(cfg: dict, tr: Any) -> str | None:
    checks = [
        (cfg["objective"] not in ("regression", "binary", "poisson", "gamma", "tweedie", "cross_entropy", "huber",
                                  "multiclass"),
         "monotone constraints with an objective other than regression (L2), binary, huber, multiclass or the poisson "
         "family"),
        (int(cfg["num_class"]) != 1 and cfg["objective"] != "multiclass", "monotone constraints with more than one class"),
        (bool(tr.cat), "monotone constraints with categorical features"),
        (tr.g is not None, "monotone constraints with a query group"),
        (cfg["boosting"] != "gbdt", "monotone constraints with boosting other than gbdt"),
    ]
    for bad, why in checks:
        if bad:
            return why
    return _mono_switch(cfg, tr)


def _mono_switch(cfg: dict, tr: Any) -> str | None:
    obj = cfg["objective"]
    key = ("huber" if obj == "huber" else "multiclass" if obj == "multiclass"
           else "binary_sigmoid" if obj == "binary" and float(cfg.get("sigmoid", 1.0)) != 1.0 else None)
    if key is None:
        return None
    env = os.environ.get("SHIRA_LGB_FIRST_READ_MONO_MIN_ROWS", "")
    lo = int(env) if env else _machine.value(f"lgb_first_read_mono_{key}_min_rows", int(tr.x.shape[1]))
    if int(tr.x.shape[0]) >= lo:
        return None
    return (f"monotone constraints with {key.replace('_', ' ')}: not measured faster than stock on this machine at this "
            f"size (switch lgb_first_read_mono_{key}_min_rows; SHIRA_LGB_FIRST_READ_MONO_MIN_ROWS=0 forces it)")


def _all_row_stats(sm: Any, ids: np.ndarray, nl: np.ndarray, leaf: np.ndarray, th: int = 1) -> np.ndarray:
    T, L = int(ids.shape[0]), int(leaf.shape[1])
    assert nl.shape == (T,) and int(nl.max()) <= L
    old = None if sm.stats is None else np.asarray(sm.stats, dtype=np.float64)
    L0 = None if old is None else (old.shape[1] + 2) // 5
    st = np.zeros((T, 5 * L - 2), dtype=np.float64)
    cnt_all, icnt_all, isum_all = product_cpp.node_counts(ids, nl, sm.nodes, leaf, th)
    for t in range(T):
        k = int(nl[t])
        ni = k - 1
        cnt = cnt_all[t, :k]
        icnt, isum = icnt_all[t, :max(ni, 0)], isum_all[t, :max(ni, 0)]
        st[t, L:L + k] = cnt
        st[t, 4 * L - 2:4 * L - 2 + ni] = icnt
        st[t, 2 * L:2 * L + ni] = np.where(icnt > 0, isum / np.maximum(icnt, 1.0), 0.0)
        if old is not None and t < old.shape[0]:
            st[t, :k] = old[t, :k]
            st[t, 3 * L - 1:3 * L - 1 + ni] = old[t, 3 * L0 - 1:3 * L0 - 1 + ni]
        else:
            st[t, :k] = cnt
            st[t, 3 * L - 1:3 * L - 1 + ni] = icnt
        biased = t < sm.num_class and abs(float(np.atleast_1d(sm.init)[t % sm.num_class])) > K_EPS and k > 1
        st[t, 5 * L - 3] = 1.0 if (biased or bool(getattr(sm, "average", False))) else float(sm.learning_rate)
    return st


def _values_mono(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict, th: int,
                 mono: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    obj = int(cfg["objective"])
    l3 = obj in OBJ_L3
    nodes, count = _nodes(m)
    T, maxn = nodes.shape[0], nodes.shape[1]
    tree = np.zeros((T, maxn, 4), dtype=np.int32)
    for t in range(T):
        k = int(m.nleaves[t])
        ni = k - 1
        tree[t, :ni, 0] = -1
        tree[t, ni:ni + k, 0] = np.arange(k)
        tree[t, :ni, 1], tree[t, :ni, 2] = nodes[t, :ni, 3], nodes[t, :ni, 4]
        tree[t, :ni, 3] = mono[nodes[t, :ni, 0]]
    assert np.array_equal(nl, m.nleaves) and np.array_equal(count, 2 * m.nleaves - 1)
    if obj == 2 or _lgbg_kind(cfg) is not None:
        vf = _values_mc if obj == 2 else _values_lgbg
        return vf(m, ids, nl, y, w, cfg, th, tree=tree, count=count)
    init = _init_l3(obj, y, w, bool(cfg["boost_from_average"])) if l3 else _init(obj, y, w, bool(cfg["boost_from_average"]))
    leaf32, pred = product_cpp.product_bounded(ids, nl, tree, count, y.astype(np.float32), init,
                                               OBJ_L3[obj] if l3 else OBJ[obj],
                                               eta=float(cfg["learning_rate"]), reg_lambda=float(cfg["lambda_l2"]),
                                               min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
                                               weight=None if w is None else w.astype(np.float32), nthread=th,
                                               reg_alpha=float(cfg["lambda_l1"]),
                                               obj_a=float(cfg.get("tweedie_variance_power", 1.5)) if obj == 9 else 1.0,
                                               obj_mds=float(cfg.get("poisson_max_delta_step", 0.7)))
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf32.shape[1]] = leaf32
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, pred.astype(np.float64), init


def linear_reason(cfg: dict, tr: Any, kw: dict, mono: Any) -> str | None:
    checks = [
        (cfg["objective"] != "regression", "linear_tree with an objective other than regression (L2)"),
        (int(cfg["num_class"]) != 1, "linear_tree with more than one class"),
        (bool(tr.cat), "linear_tree with categorical features"),
        (tr.g is not None, "linear_tree with a query group"),
        (cfg["boosting"] != "gbdt", "linear_tree with boosting other than gbdt"),
        (mono is not None, "linear_tree with monotone constraints"),
        (int(cfg["bagging_freq"]) > 0 and float(cfg["bagging_fraction"]) < 1.0, "linear_tree with bagging"),
        (bool(kw.get("valid_sets")) or "early_stopping_round" in kw or cfg.get("early_stopping_round", "0") not in ("0", ""),
         "linear_tree with evaluation sets or early stopping"),
    ]
    for bad, why in checks:
        if bad:
            return why
    return None


def _linear_lists(text: str, T: int, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    nums = [ln[len("num_features="):].split() for ln in text.split("\n") if ln.startswith("num_features=")]
    lists = [ln[len("leaf_features="):].split() for ln in text.split("\n") if ln.startswith("leaf_features=")]
    if len(nums) != T or len(lists) != T:
        raise _fit.Unsupported("linear_tree: the sample model's text has no leaf feature lists")
    fptr = np.zeros((T, lmax + 1), dtype=np.int32)
    feats: list[int] = []
    for t in range(T):
        ks = [int(v) for v in nums[t]]
        assert len(ks) <= lmax and sum(ks) == len(lists[t])
        fptr[t, 0] = len(feats)
        feats.extend(int(v) for v in lists[t])
        fptr[t, 1:len(ks) + 1] = fptr[t, 0] + np.cumsum(ks)
        fptr[t, len(ks) + 1:] = fptr[t, len(ks)]
    return fptr, np.array(feats, dtype=np.int32)


def _linear_text(text: str, nl: np.ndarray, leaf: np.ndarray, const: np.ndarray, coeff: np.ndarray, fptr: np.ndarray,
                 init: float) -> str:
    g = lambda v: "%.17g" % float(v)
    out, t = [], -1
    for ln in text.split("\n"):
        if ln.startswith("Tree="):
            t += 1
        k = 0 if t < 0 else int(nl[t])
        bias = init if t == 0 else 0.0
        if ln.startswith("leaf_value="):
            ln = "leaf_value=" + " ".join(g(float(v) + bias) for v in leaf[t, :k])
        elif ln.startswith("leaf_const="):
            ln = "leaf_const=" + " ".join(g(float(v) + bias) for v in const[t, :k])
        elif ln.startswith("leaf_coeff="):
            ln = "leaf_coeff=" + "".join("".join(g(c) + " " for c in coeff[fptr[t, j]:fptr[t, j + 1]]) + " " for j in range(k))
        out.append(ln)
    assert t + 1 == nl.size
    new = "\n".join(out)
    marks, pos = [], new.find("\nTree=")
    for _ in range(nl.size + 1):
        if pos < 0:
            break
        marks.append(pos + 1)
        pos = new.find("\nTree=", pos + 1)
    marks.append(new.find("\nend of trees") + 1)
    sizes = [len(new[a:b].encode()) for a, b in zip(marks[:-1], marks[1:])]
    head = [ln for ln in new.split("\n") if ln.startswith("tree_sizes=")]
    assert len(head) == 1 and len(sizes) == nl.size
    return new.replace(head[0], "tree_sizes=" + " ".join(str(v) for v in sizes), 1)


LIN_MAX_FEATURES = 63


def _path_features(sm: Any, t: int) -> list:
    k = int(sm.nleaves[t])
    out: list = [[] for _ in range(k)]
    stack = [(0, ())]
    for _ in range(4 * k + 4):
        if not stack:
            break
        v, fs = stack.pop()
        if v < 0:
            out[~v] = sorted(set(fs))
            continue
        f, lc, rc = int(sm.nodes[t, v, 0]), int(sm.nodes[t, v, 2]), int(sm.nodes[t, v, 3])
        stack.append((lc, fs + (f,)))
        stack.append((rc, fs + (f,)))
    assert not stack
    return out


def _fit_linear_own(plan: SimpleNamespace, cfg: dict, rows: np.ndarray, th_s: int, th: int, t0: float) -> tuple[Any, str]:
    tr, n = plan.tr, plan.tr.x.shape[0]
    sm = _shapes(plan, rows, th_s, None)
    if int(np.min(sm.nleaves)) < 2:
        raise _fit.Unsupported("a tree of the sample fit has one leaf")
    t1 = time.perf_counter()
    ids, nl = _leaf_ids(sm, tr.x, th)
    t2 = time.perf_counter()
    T, lmax = int(nl.size), int(nl.max())
    fptr = np.zeros((T, lmax + 1), dtype=np.int32)
    feats: list[int] = []
    for t in range(T):
        fptr[t, 0] = len(feats)
        for j, fl in enumerate(_path_features(sm, t)):
            if len(fl) > LIN_MAX_FEATURES:
                raise _fit.Unsupported(f"linear_tree: a leaf has more than {LIN_MAX_FEATURES} path features")
            feats.extend(fl)
            fptr[t, j + 1] = len(feats)
        fptr[t, int(nl[t]) + 1:] = len(feats)
    lfeat = np.array(feats if feats else [0], dtype=np.int32)
    init = _init(0, tr.y, tr.w, bool(cfg["boost_from_average"]))
    leaf32, const, coeff, pred = product_cpp.linear_values(
        ids, nl, tr.x, fptr, lfeat, tr.y.astype(np.float32), init, OBJ[0], eta=float(cfg["learning_rate"]),
        reg_lambda=float(cfg["lambda_l2"]), min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
        linear_lambda=float(plan.rs.cfg.get("linear_lambda", "0")), weight=None if tr.w is None else tr.w.astype(np.float32),
        nthread=th)
    leaf = np.zeros(sm.leaf.shape, dtype=np.float64)
    leaf[:, :leaf32.shape[1]] = leaf32
    cst = np.zeros(sm.leaf.shape, dtype=np.float64)
    cst[:, :const.shape[1]] = const
    if abs(init) > K_EPS:
        leaf[0, :int(nl[0])] += init
        cst[0, :int(nl[0])] += init
    sm.init = np.array([init], dtype=np.float64)
    sm.leaf, sm.scores = leaf, pred.astype(np.float64)
    sm.stats = _all_row_stats(sm, ids, nl, leaf, th)
    sm.linear = (cst, fptr, lfeat, coeff)
    first_read_log.append({"n": n, "m": int(rows.size), "grower": "shira", "linear": True, "threads": th,
                           "t_shapes": t1 - t0, "t_ids": t2 - t1, "t_values": time.perf_counter() - t2})
    line = (f"fitted the first-read way (SHIRA's default): linear_tree: tree shapes grown by SHIRA's engine at {th_s} threads on "
            f"a stride sample of {rows.size} of {n} rows (not stock's trees; accuracy depends on the data), every leaf's linear model "
            f"(on its path's features) solved on all {n} rows ({time.perf_counter() - t0:.2f} s); SHIRA_LGB_MODE=ordinary turns "
            f"it off")
    return sm, line


def fit_linear(plan: SimpleNamespace) -> tuple[Any, str] | None:
    from . import lightgbm as slg

    cfg = _fit.check_engine(plan.q)
    tr, n = plan.tr, plan.tr.x.shape[0]
    m = fixed_rows(int(plan.tr.x.shape[1]), "linear")
    if m is None or m >= n:
        return None
    th = _fit.engine_threads(plan.rs, n)
    t0 = time.perf_counter()
    rows = (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    if not _stock_grower(th, int(rows.size), int(tr.x.shape[1])):
        th_s = _sample_threads("linear", th, int(tr.x.shape[1]))
        return _fit_linear_own(plan, cfg, rows, th_s, th, t0)
    ds = slg._basic.Dataset(np.ascontiguousarray(tr.x[rows]), label=tr.y[rows], weight=None if tr.w is None else tr.w[rows],
                            params={"verbose": -1, "linear_tree": True}, free_raw_data=False)
    bst = slg._STOCK["train"](dict(plan.eff, num_threads=int(th), verbose=-1), ds, plan.rounds)
    sm = _stock_model(bst, tr.x.shape[1], cfg)
    if int(np.min(sm.nleaves)) < 2:
        raise _fit.Unsupported("a tree of the sample fit has one leaf")
    t1 = time.perf_counter()
    ids, nl = _leaf_ids_stock(sm, tr.x, th)
    t2 = time.perf_counter()
    text = bst.model_to_string()
    fptr, feats = _linear_lists(text, int(nl.size), int(nl.max()))
    init = _init(0, tr.y, tr.w, bool(cfg["boost_from_average"]))
    leaf, const, coeff, _ = product_cpp.linear_values(
        ids, nl, tr.x, fptr, feats, tr.y.astype(np.float32), init, OBJ[0], eta=float(cfg["learning_rate"]),
        reg_lambda=float(cfg["lambda_l2"]), min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
        linear_lambda=float(plan.rs.cfg.get("linear_lambda", "0")), weight=None if tr.w is None else tr.w.astype(np.float32),
        nthread=th)
    out = _linear_text(text, nl, leaf, const, coeff, fptr, init)
    first_read_log.append({"n": n, "m": int(rows.size), "grower": "stock", "linear": True, "threads": th,
                           "t_shapes": t1 - t0, "t_ids": t2 - t1, "t_values": time.perf_counter() - t2})
    line = (f"fitted the first-read way (SHIRA's default): linear_tree shapes and leaf feature lists grown by stock LightGBM "
            f"at {th} threads on a stride sample of {rows.size} of {n} rows (not stock's trees; accuracy depends on the data), every "
            f"leaf's linear model solved on all {n} rows ({time.perf_counter() - t0:.2f} s); SHIRA_LGB_MODE=ordinary turns it off")
    return out, line


def rf_reason(cfg: dict, tr: Any, kw: dict) -> str | None:
    checks = [
        (cfg["objective"] != "regression", "boosting=rf with an objective other than regression (L2)"),
        (int(cfg["num_class"]) != 1, "boosting=rf with more than one class"),
        (bool(tr.cat), "boosting=rf with categorical features"),
        (tr.g is not None, "boosting=rf with a query group"),
        (not ((int(cfg["bagging_freq"]) > 0 and 0.0 < float(cfg["bagging_fraction"]) < 1.0)
              or 0.0 < float(cfg["feature_fraction"]) < 1.0), "boosting=rf without bagging or feature_fraction (stock raises)"),
        (bool(kw.get("valid_sets")), "boosting=rf with evaluation sets (the replay adds trees up; rf averages them)"),
    ]
    for bad, why in checks:
        if bad:
            return why
    return None


def _values_rf(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
               th: int) -> tuple[np.ndarray, np.ndarray, float]:
    obj = int(cfg["objective"])
    init = _init(obj, y, w, bool(cfg["boost_from_average"]))
    y32 = y.astype(np.float32)
    w32 = None if w is None else w.astype(np.float32)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    acc = np.zeros(y.shape[0], dtype=np.float64)
    for t in range(ids.shape[0]):
        l32, _ = product_cpp.product(np.ascontiguousarray(ids[t:t + 1]), nl[t:t + 1], y32, init, OBJ[obj], eta=1.0,
                                     reg_lambda=float(cfg["lambda_l2"]),
                                     min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
              max_delta_step=float(cfg.get("max_delta_step", 0.0)),
                                     weight=w32, nthread=th)
        k = int(nl[t])
        leaf[t, :k] = l32[0, :k].astype(np.float64) + init
        acc += leaf[t, ids[t]]
    assert ids.shape[0] >= 1
    return leaf, acc / ids.shape[0], init


def _holdback_loss(obj: int, margin: np.ndarray, y: np.ndarray) -> float:
    if obj == 0:
        return float(np.mean((margin.astype(np.float64) - y) ** 2))
    p = 1.0 / (1.0 + np.exp(-margin.astype(np.float64)))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _shapes(plan: SimpleNamespace, rows: np.ndarray, th: int, th_call: int | None = None) -> Any:
    if getattr(plan, "shape_route", None) or (th_call is not None and _stock_grower(th_call, int(rows.size), int(plan.tr.x.shape[1]))):
        sm = _shapes_stock(plan, rows, th if th_call is None else th_call)
        sm.grower = "stock"
        return sm
    tr = plan.tr
    w = None if tr.w is None else tr.w[rows]
    sm = _fit.fit_folds(plan.q, plan.rs, np.ascontiguousarray(tr.x[rows]), tr.y[rows], [(0, rows.size)], plan.rounds,
                        w, None, stock_params=plan.stock_params, threads=th, bundle_rule=False)[0]
    sm.grower = "shira"
    return sm


K_ZERO = 1e-35
_MT = {"None": 0, "Zero": 1, "NaN": 2}


def _walk_dump(tree: dict, ni: int) -> list:
    kid = lambda c: int(c["split_index"]) if "split_index" in c else ~int(c["leaf_index"])
    out, stack = [], [tree["tree_structure"]]
    for _ in range(2 * ni + 1):
        if not stack:
            break
        nd = stack.pop()
        if "split_index" not in nd:
            continue
        assert nd.get("decision_type", "<=") == "<=", "a categorical split (decision_type ==)"
        out.append((int(nd["split_index"]), nd, kid(nd["left_child"]), kid(nd["right_child"])))
        stack.extend([nd["left_child"], nd["right_child"]])
    assert len(out) == ni and not stack
    return out


def _stock_model(bst: Any, F: int, cfg: dict) -> Any:
    info = bst.dump_model()["tree_info"]
    T, L = len(info), max(int(t["num_leaves"]) for t in info)
    walks = [_walk_dump(t, int(t["num_leaves"]) - 1) for t in info]
    if min(int(t["num_leaves"]) for t in info) < 2:
        raise _fit.Unsupported("a tree of the sample fit has one leaf")
    thr = [set() for _ in range(F)]
    mt = np.zeros(F, dtype=np.int8)
    for w in walks:
        for _, nd, _, _ in w:
            thr[int(nd["split_feature"])].add(float(nd["threshold"]))
            mt[int(nd["split_feature"])] = _MT[nd.get("missing_type", "None")]
    upper = [np.array(sorted(u), dtype=np.float64) if u else np.array([np.inf]) for u in thr]
    nodes = np.zeros((T, max(L - 1, 1), 4), dtype=np.int32)
    dl = np.zeros((T, max(L - 1, 1)), dtype=np.int32)
    gain = np.zeros((T, max(L - 1, 1)), dtype=np.float64)
    for t, w in enumerate(walks):
        for i, nd, lc, rc in w:
            f = int(nd["split_feature"])
            k = int(np.searchsorted(upper[f], float(nd["threshold"])))
            assert upper[f][k] == float(nd["threshold"])
            nodes[t, i] = (f, k, lc, rc)
            dl[t, i] = 1 if nd.get("default_left", False) else 0
            gain[t, i] = float(nd.get("split_gain", 0.0))
    fi_line = [ln for ln in bst.model_to_string().splitlines() if ln.startswith("feature_infos=")][0]
    nb = np.array([u.size + 1 for u in upper], dtype=np.int32)
    inf4 = np.stack([nb, mt.astype(np.int32), np.zeros(F, np.int32), np.zeros(F, np.int32)], axis=1)
    return _fit.FoldModel(nodes, dl, np.zeros((T, L)), gain, np.array([int(t["num_leaves"]) for t in info], np.int32),
                          upper, mt, int(cfg.get("num_class", 1)), float(cfg["learning_rate"]), np.zeros(1), None,
                          fi_line[len("feature_infos="):].split(" "), np.ascontiguousarray(inf4))


def _leaf_ids_stock(m: Any, x: np.ndarray, th: int) -> tuple[np.ndarray, np.ndarray]:
    F = x.shape[1]
    up = [np.append(u, np.inf) if np.isfinite(u[-1]) else u for u in m.upper]
    ptrs = np.concatenate([[0], np.cumsum([u.size for u in up])]).astype(np.int32)
    info = np.stack([np.array([u.size for u in up], np.int32), np.zeros(F, np.int32), np.zeros(F, np.int32),
                     np.zeros(F, np.int32)], axis=1)
    miss = [j for j in range(F) if int(m.missing_type[j]) != 0]
    wide = int(info[:, 0].max()) > 256
    bins, has_inf = binning_cpp.lgb_bin2(x, ptrs, np.concatenate(up), np.ascontiguousarray(info), nthread=th,
                                         code_bytes=2 if (miss or wide) else 1)
    if has_inf:
        raise _fit.Unsupported("infinite feature values")
    nodes, count = _nodes(m)
    if not miss and not wide:
        return product_cpp.leaf_ids(bins, nodes, count, nthread=th)
    codes = bins
    codes += 1
    for j in miss:
        col = x[:, j]
        hit = np.isnan(col) if int(m.missing_type[j]) == 2 else (np.isnan(col) | (np.abs(col) <= K_ZERO))
        codes[hit, j] = 0
    return product_cpp.leaf_ids(codes, nodes, count, nthread=th)


def _stock_grower(th: int, rows: int, features: int) -> bool:
    g = os.environ.get("SHIRA_LGB_FIRST_READ_GROWER", "")
    if g in ("stock", "shira"):
        return g == "stock"
    assert th >= 1 and rows >= 1
    return _fit.x86_guard(th, rows, features) is not None


def _route_threads(plan: SimpleNamespace, rows: int, th: int) -> int:
    if not getattr(plan, "shape_route", None):
        return th
    env = os.environ.get("SHIRA_LGB_ROUTE_SAMPLE_ROWS_PER_THREAD", "").strip()
    per = int(env) if env.isdigit() else _machine.value("lgb_route_sample_rows_per_thread", int(plan.tr.x.shape[1]))
    return th if per <= 0 else max(1, min(th, rows // per))


def _shapes_stock(plan: SimpleNamespace, rows: np.ndarray, th: int) -> Any:
    from . import lightgbm as slg
    th = _route_threads(plan, int(rows.size), th)

    tr = plan.tr
    ds = slg._basic.Dataset(np.ascontiguousarray(tr.x[rows]), label=tr.y[rows],
                            weight=None if tr.w is None else tr.w[rows], params={"verbose": -1}, free_raw_data=False)
    p = dict(plan.eff, num_threads=int(th), verbose=-1)
    bst = slg._STOCK["train"](p, ds, plan.rounds)
    return _stock_model(bst, tr.x.shape[1], _fit.check_engine(plan.q))


def _early(plan: SimpleNamespace) -> Any | None:
    from . import lightgbm as slg

    cb = next((c for c in (plan.callbacks or []) if "EarlyStopping" in type(c).__name__), None)
    if cb is not None:
        return slg._cb.early_stopping(stopping_rounds=int(cb.stopping_rounds), first_metric_only=bool(cb.first_metric_only),
                                      verbose=False, min_delta=cb.min_delta)
    r = plan.p.get("early_stopping_round", 0)
    if not r or int(r) <= 0:
        return None
    return slg._cb.early_stopping(stopping_rounds=int(r), first_metric_only=bool(plan.p.get("first_metric_only", False)),
                                  verbose=False, min_delta=plan.p.get("early_stopping_min_delta", 0.0))


def _shapes_early(plan: SimpleNamespace, rows: np.ndarray, th: int, stop: Any) -> Any:
    from . import lightgbm as slg

    tr = plan.tr
    ds = slg._basic.Dataset(np.ascontiguousarray(tr.x[rows]), label=tr.y[rows], weight=None if tr.w is None else tr.w[rows],
                            params={"verbose": -1}, free_raw_data=False)
    vs = [slg._basic.Dataset(a.x, label=a.y, weight=a.w, reference=ds) for _, a in plan.valid]
    drop = ("early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change")
    th = _route_threads(plan, int(rows.size), th)
    p = {k: v for k, v in dict(plan.eff, num_threads=int(th), verbose=-1).items() if k not in drop}
    bst = slg._STOCK["train"](p, ds, plan.rounds, valid_sets=vs, valid_names=[nm for nm, _ in plan.valid], callbacks=[stop])
    sm = _stock_model(bst, tr.x.shape[1], _fit.check_engine(plan.q))
    sm.grower = "stock"
    assert 1 <= sm.n_trees <= plan.rounds * sm.num_class and sm.n_trees % sm.num_class == 0
    plan.rounds = int(sm.n_trees) // sm.num_class
    return sm


def _choose(plan: SimpleNamespace, cfg: dict, th_s: int, th: int) -> int | None:
    tr, n = plan.tr, plan.tr.x.shape[0]
    step = max(FIRST_READ_HOLD, n // FIRST_READ_HOLD_ROWS)
    hold = np.arange(0, n, step)
    rest = np.setdiff1d(np.arange(n), hold, assume_unique=True)
    obj = int(cfg["objective"])
    tried, prev, m = [], None, FIRST_READ_M0
    while 2 * m < rest.size:
        srows = rest[(rest.size - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)]
        sm = _shapes(plan, srows, th_s, th)
        k2 = min(rest.size, FIRST_READ_SCORE * m)
        vrows = rest[(rest.size - 1) * np.arange(k2, dtype=np.int64) // max(1, k2 - 1)]
        both = np.concatenate([vrows, hold])
        ids, nl = _leaf_ids(sm, np.ascontiguousarray(tr.x[both]), th)
        k = vrows.size
        leaf32, _ = product_cpp.product(np.ascontiguousarray(ids[:, :k]), nl, tr.y[vrows].astype(np.float32),
                                        _init(obj, tr.y[vrows], None, bool(cfg["boost_from_average"])), OBJ[obj],
                                        eta=float(cfg["learning_rate"]), reg_lambda=float(cfg["lambda_l2"]),
                                        min_child_weight=float(cfg["min_sum_hessian_in_leaf"]), nthread=th)
        hid = ids[:, k:]
        margin = _init(obj, tr.y[vrows], None, bool(cfg["boost_from_average"])) + leaf32[np.arange(hid.shape[0])[:, None], hid].astype(np.float64).sum(axis=0)
        loss = _holdback_loss(obj, margin, tr.y[hold].astype(np.float64))
        tried.append({"m": int(m), "loss": loss})
        if prev is not None:
            gain = (prev - loss) / max(abs(prev), 1e-300)
            if len(tried) == 2 and gain > FIRST_READ_RICH:
                first_read_log.append({"n": n, "tried": tried, "kept": None, "exit": "signal-rich"})
                return None
            if gain <= FIRST_READ_GAIN:
                first_read_log.append({"n": n, "tried": tried, "kept": int(m // 2)})
                return int(m // 2)
        prev, m = loss, 2 * m
    first_read_log.append({"n": n, "tried": tried, "kept": None})
    return None


def _init_l3(obj: int, y: np.ndarray, w: np.ndarray | None, boost: bool) -> float:
    assert obj in OBJ_L3
    if not boost:
        return 0.0
    yd = y.astype(np.float64)
    avg = float(yd.mean()) if w is None else float((yd * w).sum() / w.astype(np.float64).sum())
    if obj == 10:
        p = min(max(avg, K_EPS), 1.0 - K_EPS)
        return float(np.log(p / (1.0 - p)))
    if not avg > 0.0:
        raise _fit.Unsupported("poisson / gamma / tweedie with a label mean <= 0 (LightGBM's init is -inf)")
    return float(np.log(avg))


def _values_l3(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
               th: int) -> tuple[np.ndarray, np.ndarray, float]:
    obj = int(cfg["objective"])
    init = _init_l3(obj, y, w, bool(cfg["boost_from_average"]))
    kw = dict(eta=float(cfg["learning_rate"]), reg_lambda=float(cfg["lambda_l2"]),
              min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
              max_delta_step=float(cfg.get("max_delta_step", 0.0)),
              weight=None if w is None else w.astype(np.float32), nthread=th,
              obj_a=float(cfg.get("tweedie_variance_power", 1.5)) if obj == 9 else 1.0,
              obj_mds=float(cfg.get("poisson_max_delta_step", 0.7)))
    l1 = float(cfg["lambda_l1"])
    if l1 == 0.0:
        leaf32, pred = product_cpp.product(ids, nl, y.astype(np.float32), init, OBJ_L3[obj], **kw)
    else:
        leaf32, pred = product_cpp.product_l1(ids, nl, y.astype(np.float32), init, OBJ_L3[obj], reg_alpha=l1, **kw)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf32.shape[1]] = leaf32
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, pred.astype(np.float64), init


def _values_mc(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
               th: int, tree: np.ndarray | None = None, count: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = int(cfg["num_class"])
    lab = y.astype(np.int64)
    assert lab.min() >= 0 and lab.max() < C
    share = np.bincount(lab, weights=None if w is None else w.astype(np.float64), minlength=C).astype(np.float64)
    share /= share.sum()
    if np.any(share <= K_EPS) or np.any(share >= 1.0 - K_EPS):
        raise _fit.Unsupported("multiclass with a class LightGBM does not train (share 0 or 1)")
    init = np.log(np.maximum(K_EPS, share)) if bool(cfg["boost_from_average"]) else np.zeros(C)
    leaf64, scores = product_cpp.product_softmax_lgb(ids, nl, C, y.astype(np.float32), init,
                                                     eta=float(cfg["learning_rate"]), reg_lambda=float(cfg["lambda_l2"]),
                                                     reg_alpha=float(cfg["lambda_l1"]),
                                                     min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
                                                     weight=None if w is None else w.astype(np.float32), nthread=th,
                                                     tree=tree, count=count)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    for c in range(C):
        if abs(float(init[c])) > K_EPS and int(m.nleaves[c]) > 1:
            leaf[c, :int(m.nleaves[c])] += init[c]
    return leaf, scores, init


def _values_ova(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
                th: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    C = int(cfg["num_class"])
    T = int(ids.shape[0])
    lab = y.astype(np.int64)
    assert T % C == 0 and lab.min() >= 0 and lab.max() < C
    cb = dict(cfg, objective=1, num_class=1)
    vf = _values_lgbg if _lgbg_kind(cb) else _values
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    scores = np.zeros((C, y.size), dtype=np.float64)
    init = np.zeros(C, dtype=np.float64)
    for c in range(C):
        yc = (lab == c).astype(np.float64)
        share = float(yc.mean()) if w is None else float((yc * w).sum() / w.astype(np.float64).sum())
        if share <= K_EPS or share >= 1.0 - K_EPS:
            raise _fit.Unsupported("multiclassova with a class LightGBM does not train (share 0 or 1)")
        sub = SimpleNamespace(leaf=np.zeros((T // C, m.leaf.shape[1]), dtype=np.float64), nleaves=nl[c::C])
        lc, sc, ic = vf(sub, np.ascontiguousarray(ids[c::C]), np.ascontiguousarray(nl[c::C]), yc, w, cb, th)
        leaf[c::C] = lc
        scores[c] = sc
        init[c] = ic
    return leaf, scores, init


def _lgbg_kind(cfg: dict) -> str | None:
    if int(cfg["objective"]) == 5:
        return "huber"
    if int(cfg["objective"]) == 1 and float(cfg.get("sigmoid", 1.0)) != 1.0:
        return "binary"
    return None


def _values_lgbg(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
                 th: int, tree: np.ndarray | None = None, count: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    kind = _lgbg_kind(cfg)
    assert kind is not None
    a = float(cfg["alpha"]) if kind == "huber" else float(cfg["sigmoid"])
    init = 0.0
    if bool(cfg["boost_from_average"]):
        yd = (y > 0).astype(np.float64) if kind == "binary" else y.astype(np.float64)
        avg = float(yd.mean()) if w is None else float((yd * w).sum() / w.astype(np.float64).sum())
        if kind == "binary":
            pv = min(max(avg, K_EPS), 1.0 - K_EPS)
            init = float(np.log(pv / (1.0 - pv)) / a)
        else:
            init = avg
    leaf64, scores = product_cpp.product_lgb(ids, nl, y.astype(np.float32), init, kind, a, eta=float(cfg["learning_rate"]),
                                             reg_lambda=float(cfg["lambda_l2"]), reg_alpha=float(cfg["lambda_l1"]),
                                             min_child_weight=float(cfg["min_sum_hessian_in_leaf"]),
                                             weight=None if w is None else w.astype(np.float32), nthread=th,
                                             tree=tree, count=count)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, scores, init


def _lgb_wpercentile(v: np.ndarray, w: np.ndarray, alpha: float) -> float:
    cnt = int(v.size)
    assert cnt == w.size and cnt >= 1
    if cnt <= 1:
        return float(v[0])
    si = np.argsort(v, kind="stable")
    cdf = np.cumsum(w[si].astype(np.float64))
    threshold = float(cdf[-1]) * alpha
    pos = min(int(np.searchsorted(cdf, threshold, side="right")), cnt - 1)
    if pos in (0, cnt - 1):
        return float(v[si[pos]])
    v1, v2 = float(v[si[pos - 1]]), float(v[si[pos]])
    if cdf[pos + 1] - cdf[pos] >= 1.0:
        return (threshold - float(cdf[pos])) / float(cdf[pos + 1] - cdf[pos]) * (v2 - v1) + v1
    return v2


def _lgb_percentile(v: np.ndarray, alpha: float) -> float:
    v = np.asarray(v, dtype=np.float32)
    cnt = int(v.size)
    assert cnt >= 1 and 0.0 < alpha < 1.0
    if cnt <= 1:
        return float(v[0])
    float_pos = (1.0 - alpha) * (cnt - 1)
    pos = int(float_pos)
    if pos >= cnt - 1:
        return float(v.min())
    d = np.sort(v)[::-1]
    diff = float(np.float32(d[pos]) - np.float32(d[pos + 1]))
    return float(np.float32(float(d[pos]) - diff * (float_pos - pos)))


def _values_renew(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
                  th: int) -> tuple[np.ndarray, np.ndarray, float]:
    alpha = 0.5 if int(cfg["objective"]) == 4 else float(cfg["alpha"])
    if w is not None:
        init = float(np.float32(_lgb_wpercentile(y.astype(np.float32).astype(np.float64), w.astype(np.float32), alpha))) \
            if bool(cfg["boost_from_average"]) else 0.0
        leaf64, scores = product_cpp.product_renew_w(ids, nl, y.astype(np.float32), w.astype(np.float32), init, alpha,
                                                     eta=float(cfg["learning_rate"]), nthread=th)
    else:
        init = _lgb_percentile(y, alpha) if bool(cfg["boost_from_average"]) else 0.0
        leaf64, scores = product_cpp.product_renew(ids, nl, y.astype(np.float32), init, alpha,
                                                   eta=float(cfg["learning_rate"]), nthread=th)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, scores, init


def _values_smooth(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
                   th: int) -> tuple[np.ndarray, np.ndarray, float]:
    obj = int(cfg["objective"])
    l3 = obj in OBJ_L3
    init = (_init_l3 if l3 else _init)(obj, y, w, bool(cfg["boost_from_average"]))
    nodes, count = _nodes(m)
    T, maxn = nodes.shape[0], nodes.shape[1]
    tree = np.zeros((T, maxn, 4), dtype=np.int32)
    for t in range(T):
        k = int(m.nleaves[t])
        tree[t, :k - 1, 0] = -1
        tree[t, k - 1:2 * k - 1, 0] = np.arange(k)
        tree[t, :k - 1, 1], tree[t, :k - 1, 2] = nodes[t, :k - 1, 3], nodes[t, :k - 1, 4]
    assert np.array_equal(nl, m.nleaves) and np.array_equal(count, 2 * m.nleaves - 1)
    leaf32, pred = product_cpp.product_smooth(ids, nl, tree, count, y.astype(np.float32), init,
                                              OBJ_L3[obj] if l3 else OBJ[obj], float(cfg["path_smooth"]),
                                              eta=float(cfg["learning_rate"]), reg_lambda=float(cfg["lambda_l2"]),
                                              reg_alpha=float(cfg["lambda_l1"]),
                                              max_delta_step=float(cfg.get("max_delta_step", 0.0)),
                                              weight=None if w is None else w.astype(np.float32), nthread=th,
                                              obj_a=float(cfg.get("tweedie_variance_power", 1.5)) if obj == 9 else 1.0,
                                              obj_mds=float(cfg.get("poisson_max_delta_step", 0.7)))
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf32.shape[1]] = leaf32
    if abs(init) > K_EPS and int(m.nleaves[0]) > 1:
        leaf[0, :int(m.nleaves[0])] += init
    return leaf, pred.astype(np.float64), init


K_ZERO_THR = float(np.float32(1e-35))


def _dart_drops(cfg: dict, rounds: int) -> list:
    import math

    x = int(cfg["drop_seed"]) & 0xFFFFFFFF
    lr, drop_rate, skip_drop, max_drop = (float(cfg["learning_rate"]), float(cfg["drop_rate"]), float(cfg["skip_drop"]),
                                          int(cfg["max_drop"]))
    tw: list = []
    sw = 0.0
    out = []
    for it in range(rounds):
        drop = []
        x = (214013 * x + 2531011) & 0xFFFFFFFF
        skip = float(np.float32(((x >> 16) & 0x7FFF) / 32768.0)) < skip_drop
        if not skip:
            rate = drop_rate
            inv_avg = len(tw) / sw if sw != 0.0 else float("nan")
            if max_drop > 0 and not math.isnan(inv_avg):
                rate = min(rate, max_drop * inv_avg / sw)
            for i in range(it):
                x = (214013 * x + 2531011) & 0xFFFFFFFF
                if float(np.float32(((x >> 16) & 0x7FFF) / 32768.0)) < rate * tw[i] * inv_avg:
                    drop.append(i)
                    if len(drop) >= max_drop:
                        break
        k = float(len(drop))
        shrink = lr / (1.0 + k)
        for i in drop:
            sw = _fma(-tw[i], 1.0 / (k + 1.0), sw)
            tw[i] *= k / (k + 1.0)
        tw.append(shrink)
        sw += shrink
        out.append((drop, k, shrink))
    return out


def _fma_exact(a: float, b: float, c: float) -> float:
    assert math.isfinite(a) and math.isfinite(b) and math.isfinite(c)
    v = Fraction(a) * Fraction(b) + Fraction(c)
    return float(v.numerator / v.denominator) if v != 0 else (0.0 if (a * b) + c >= 0 else -0.0)


_fma = getattr(math, "fma", None) or _fma_exact


def _values_dart(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
                 th: int) -> tuple[np.ndarray, np.ndarray, float]:
    obj = int(cfg["objective"])
    init = _init(obj, y, w, bool(cfg["boost_from_average"]))
    drops = _dart_drops(cfg, int(ids.shape[0]))
    leaf64, scores = product_cpp.product_dart(ids, nl, y, init, obj, drops, lambda_l1=float(cfg["lambda_l1"]),
                                              lambda_l2=float(cfg["lambda_l2"]), weight=w, nthread=th)
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    return leaf, scores, init


def _values_dart_numpy(m: Any, ids: np.ndarray, nl: np.ndarray, y: np.ndarray, w: np.ndarray | None, cfg: dict,
                       th: int) -> tuple[np.ndarray, np.ndarray, float]:
    obj = int(cfg["objective"])
    init = _init(obj, y, w, bool(cfg["boost_from_average"]))
    T = int(ids.shape[0])
    drops = _dart_drops(cfg, T)
    l1, l2 = float(cfg["lambda_l1"]), float(cfg["lambda_l2"])
    yd = y.astype(np.float32).astype(np.float64)
    wd = None if w is None else w.astype(np.float32).astype(np.float64)
    lab = np.where(y > 0, 1.0, -1.0)
    vals = [np.zeros(int(nl[t])) for t in range(T)]
    score = np.full(y.shape[0], init, dtype=np.float64)
    for t, (drop, k, rate) in enumerate(drops):
        for i in drop:
            score -= vals[i][ids[i]]
        if obj == 0:
            g = (score - yd).astype(np.float32).astype(np.float64)
            h = np.ones_like(g)
        else:
            r = -lab / (1.0 + np.exp(lab * score))
            g = r.astype(np.float32).astype(np.float64)
            h = (np.abs(r) * (1.0 - np.abs(r))).astype(np.float32).astype(np.float64)
        if wd is not None:
            g = (g * wd).astype(np.float32).astype(np.float64)
            h = (h * wd).astype(np.float32).astype(np.float64)
        G = np.bincount(ids[t], weights=g, minlength=int(nl[t]))
        H = np.bincount(ids[t], weights=h, minlength=int(nl[t]))
        thr = np.sign(G) * np.maximum(np.abs(G) - l1, 0.0)
        v = np.where(H > 0, -thr / (H + l2), 0.0) * rate
        v[np.abs(v) <= K_ZERO_THR] = 0.0
        score += v[ids[t]]
        if t == 0 and abs(init) > K_EPS and int(nl[0]) > 1:
            v = v + init
        vals[t] = v
        for i in drop:
            a = vals[i] * (1.0 / (k + 1.0))
            a[np.abs(a) <= K_ZERO_THR] = 0.0
            a = a * (-k)
            a[np.abs(a) <= K_ZERO_THR] = 0.0
            vals[i] = -a
            score += vals[i][ids[i]]
    leaf = np.zeros(m.leaf.shape, dtype=np.float64)
    for t in range(T):
        leaf[t, :vals[t].size] = vals[t]
    return leaf, score, init


def _rank_sample_stock(plan: SimpleNamespace, rows: np.ndarray, gs: np.ndarray, th: int, stop: Any) -> Any:
    from . import lightgbm as slg

    tr = plan.tr
    ds = slg._basic.Dataset(np.ascontiguousarray(tr.x[rows]), label=tr.y[rows], weight=None if tr.w is None else tr.w[rows],
                            group=gs, params={"verbose": -1}, free_raw_data=False)
    vs = [slg._basic.Dataset(a.x, label=a.y, weight=a.w, group=a.g, reference=ds) for _, a in plan.valid] if stop else []
    drop = ("early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change")
    th = _route_threads(plan, int(rows.size), th)
    p = {k: v for k, v in dict(plan.eff, num_threads=int(th), verbose=-1).items() if k not in drop}
    bst = slg._STOCK["train"](p, ds, plan.rounds, valid_sets=vs or None,
                              valid_names=[nm for nm, _ in plan.valid] if stop else None,
                              callbacks=[stop] if stop is not None else None)
    sm = _stock_model(bst, tr.x.shape[1], _fit.check_engine(plan.q))
    sm.grower = "stock"
    assert 1 <= sm.n_trees <= plan.rounds
    plan.rounds = int(sm.n_trees)
    return sm


XE_A, XE_C = np.uint32(214013), np.uint32(2531011)


def _values_xendcg(ids: np.ndarray, nl: np.ndarray, y: np.ndarray, g: np.ndarray, cfg: dict, seed: int,
                   T: int, th: int = 1) -> tuple[np.ndarray, np.ndarray]:
    n, Q = int(ids.shape[1]), int(g.size)
    assert T == ids.shape[0] and int(g.sum()) == n and Q >= 1
    if os.environ.get("SHIRA_LGB_XENDCG_NUMPY", "").strip() != "1":
        from shira_cpp import product_cpp
        return product_cpp.lgb_xendcg(ids, nl, y, g, int(seed), float(cfg["learning_rate"]), float(cfg["lambda_l1"]),
                                      float(cfg["lambda_l2"]), float(cfg.get("max_delta_step", 0.0)),
                                      nthread=max(1, int(th)))
    starts = np.concatenate([[0], np.cumsum(g)[:-1]]).astype(np.int64)
    qid = np.repeat(np.arange(Q), g)
    pos = np.arange(n) - starts[qid]
    multi = g[qid] > 1
    state = (np.uint32(seed) + np.arange(Q, dtype=np.uint32)).astype(np.uint32)
    gain = np.power(2.0, np.asarray(y, dtype=np.float64).astype(np.int64)).astype(np.float64)
    l1, l2 = float(cfg["lambda_l1"]), float(cfg["lambda_l2"])
    eta, mds = float(cfg["learning_rate"]), float(cfg.get("max_delta_step", 0.0))
    score = np.zeros(n, dtype=np.float64)
    leaf = np.zeros((T, int(nl.max())), dtype=np.float64)
    gmax = int(g.max())
    Aj = np.ones(gmax + 1, dtype=np.uint32)
    Cj = np.zeros(gmax + 1, dtype=np.uint32)
    with np.errstate(over="ignore"):
        for j in range(1, gmax + 1):
            Aj[j] = Aj[j - 1] * XE_A
            Cj[j] = Cj[j - 1] * XE_A + XE_C
    Ar, Cr = Aj[pos + 1], Cj[pos + 1]
    Aq, Cq = Aj[np.where(g > 1, g, 0)], Cj[np.where(g > 1, g, 0)]
    for t in range(T):
        with np.errstate(over="ignore"):
            xs = Ar * state[qid] + Cr
            state = Aq * state + Cq
        draw = np.where(multi, ((xs >> np.uint32(16)) & np.uint32(0x7FFF)).astype(np.float32) / np.float32(32768.0), 0.0
                        ).astype(np.float64)
        mx = np.maximum.reduceat(score, starts)
        e = np.exp(score - mx[qid])
        rho = e / np.add.reduceat(e, starts)[qid]
        par = gain - draw
        inv = 1.0 / np.maximum(1e-15, np.add.reduceat(par, starts))[qid]
        t1 = -par * inv + rho
        lam = t1.copy()
        p1 = t1 / (1.0 - rho)
        t2 = rho * (np.add.reduceat(p1, starts)[qid] - p1)
        lam += t2
        p2 = t2 / (1.0 - rho)
        lam += rho * (np.add.reduceat(p2, starts)[qid] - p2)
        hes = rho * (1.0 - rho)
        lam = np.where(multi, lam, 0.0).astype(np.float32).astype(np.float64)
        hes = np.where(multi, hes, 0.0).astype(np.float32).astype(np.float64)
        k, idx = int(nl[t]), ids[t].astype(np.int64)
        G, H = np.bincount(idx, lam, k), np.bincount(idx, hes, k)
        out = -np.sign(G) * np.maximum(np.abs(G) - l1, 0.0) / (H + l2)
        if mds > 0.0:
            out = np.clip(out, -mds, mds)
        leaf[t, :k] = out * eta
        score += leaf[t, idx]
    return leaf, score


def _fit_xendcg(plan: SimpleNamespace, cfg: dict, th: int) -> tuple[Any, str] | None:
    t0 = time.perf_counter()
    g = np.asarray(plan.tr.g, dtype=np.int64)
    n, Q, nf = int(plan.tr.x.shape[0]), int(g.size), int(plan.tr.x.shape[1])
    m = fixed_rows(nf, "", n)
    if m is not None and m >= n:
        floor = _machine.value("lgb_small_floor_rows", nf)
        m = max(floor, n * _machine.value("lgb_small_sample_pct", nf) // 100) if n > floor else m
    if m is None or m >= n or Q < 4:
        return None
    k = max(2, min(Q, int(np.ceil(Q * m / n))))
    qsel = np.unique((Q - 1) * np.arange(k, dtype=np.int64) // max(1, k - 1))
    starts = np.concatenate([[0], np.cumsum(g)])[:-1]
    rows = np.concatenate([np.arange(starts[q], starts[q] + g[q], dtype=np.int64) for q in qsel])
    if rows.size >= n:
        return None
    try:
        sm = _rank_sample_stock(plan, rows, g[qsel], th, None)
        if int(np.min(sm.nleaves)) < 2:
            raise _fit.Unsupported("a tree of the sample fit has one leaf")
        ids, nl = _leaf_ids(sm, plan.tr.x, th)
        leaf64, scores = _values_xendcg(ids, nl, plan.tr.y, g, cfg, int(plan.builtin_obj[1]), int(sm.n_trees), th)
    except _fit.Unsupported as e:
        LOG.info("shira.lightgbm: first_read not applied: %s", e)
        return None
    leaf = np.zeros(sm.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    sm.leaf, sm.scores, sm.average = leaf, scores, False
    sm.init = np.zeros(1, dtype=np.float64)
    sm.stats = _all_row_stats(sm, ids, nl, leaf, th)
    first_read_log.append({"n": n, "m": int(rows.size), "queries": int(qsel.size), "grower": "stock", "threads": th})
    line = (f"fitted the first-read way (SHIRA's default): rank_xendcg, trees grown by stock LightGBM with the caller's "
            f"objective (LightGBM's general route) on {qsel.size} of {Q} whole queries ({rows.size} of {n} rows), leaf "
            f"values solved on all {n} rows from LightGBM's XE-NDCG gradients (not stock's trees) ({time.perf_counter() - t0:.2f} s); "
            f"SHIRA_LGB_MODE=ordinary runs stock")
    return sm, line


def _fit_rank(plan: SimpleNamespace, cfg: dict, th: int) -> tuple[Any, str] | None:
    from shira_cpp import lgb_cpp

    t0 = time.perf_counter()
    g = np.asarray(plan.tr.g, dtype=np.int64)
    n, Q = int(plan.tr.x.shape[0]), int(g.size)
    m = fixed_rows(int(plan.tr.x.shape[1]), "", n)
    if m is None or m >= n or Q < 4:
        return None
    k = max(2, min(Q, int(np.ceil(Q * m / n))))
    qsel = (Q - 1) * np.arange(k, dtype=np.int64) // max(1, k - 1)
    qsel = np.unique(qsel)
    starts = np.concatenate([[0], np.cumsum(g)])[:-1]
    rows = np.concatenate([np.arange(starts[q], starts[q] + g[q], dtype=np.int64) for q in qsel])
    if rows.size >= n:
        return None
    stop = _early(plan)
    stock = plan.tr.w is not None or stop is not None
    stock = stock or _stock_grower(th, int(rows.size), int(plan.tr.x.shape[1]))
    try:
        if stock:
            sm = _rank_sample_stock(plan, rows, g[qsel], th, stop)
        else:
            sm = _fit.fit_folds(plan.q, plan.rs, np.ascontiguousarray(plan.tr.x[rows]), plan.tr.y[rows], [(0, rows.size)],
                                plan.rounds, None, g[qsel], stock_params=plan.stock_params, threads=th,
                                bundle_rule=False)[0]
            sm.grower = "shira"
        if int(np.min(sm.nleaves)) < 2:
            raise _fit.Unsupported("a tree of the sample fit has one leaf")
        ids, nl = _leaf_ids(sm, plan.tr.x, th)
        leaf64, scores = lgb_cpp.rank_values(ids, nl, plan.tr.y, g, np.asarray(cfg["label_gain"], dtype=np.float64),
                                             int(cfg["truncation_level"]), int(cfg["lambdarank_norm"]),
                                             float(cfg["sigmoid"]), float(cfg["learning_rate"]),
                                             float(cfg["lambda_l2"]), float(cfg["lambda_l1"]), weight=plan.tr.w,
                                             nthread=th)
    except _fit.Unsupported as e:
        LOG.info("shira.lightgbm: first_read not applied: %s", e)
        return None
    leaf = np.zeros(sm.leaf.shape, dtype=np.float64)
    leaf[:, :leaf64.shape[1]] = leaf64
    sm.leaf, sm.scores, sm.average = leaf, scores, False
    sm.init = np.zeros(1, dtype=np.float64)
    sm.stats = _all_row_stats(sm, ids, nl, leaf, th)
    first_read_log.append({"n": n, "m": int(rows.size), "queries": int(qsel.size), "grower": sm.grower, "threads": th})
    who = "stock LightGBM" if sm.grower == "stock" else "SHIRA's engine"
    line = (f"fitted the first-read way (SHIRA's default): lambdarank, trees grown by {who} on {qsel.size} of "
            f"{Q} whole queries ({rows.size} of {n} rows, a stride over queries), leaf values solved on all {n} rows "
            f"from LightGBM's own NDCG lambdas (not stock's trees) ({time.perf_counter() - t0:.2f} s); SHIRA_LGB_MODE=ordinary "
            f"turns it off" + ("" if stop is None else f"; early stopping: the sample fit stopped by the caller's rule on "
                                                        f"the caller's validation sets and kept {sm.n_trees} trees"))
    return sm, line


def fit(plan: SimpleNamespace) -> tuple[Any, str] | None:
    cfg = _fit.check_engine(plan.q)
    why = reason(plan, cfg)
    n = plan.tr.x.shape[0]
    if why is not None:
        LOG.info("shira.lightgbm: first_read not applied: %s", why)
        return None
    th = _fit.engine_threads(plan.rs, n)
    if getattr(plan, "builtin_obj", None) is not None and plan.builtin_obj[0] == "xendcg":
        return _fit_xendcg(plan, cfg, th)
    if int(cfg["objective"]) == 11:
        return _fit_rank(plan, cfg, th)
    stop = _early(plan)
    th_s = th
    th_s = int(os.environ.get("SHIRA_LGB_FIRST_READ_SAMPLE_THREADS", th_s))
    t0 = time.perf_counter()
    rf = bool(getattr(plan, "rf", False))
    m = fixed_rows(int(plan.tr.x.shape[1]), "mono" if getattr(plan, "mono", None) is not None else "",
                   0 if (rf or plan.tr.cat or stop is not None) else int(n))
    if getattr(plan, "shape_route", None) and m is not None and m >= n:
        floor = _machine.value("lgb_small_floor_rows", int(plan.tr.x.shape[1]))
        m = max(floor, n * _machine.value("lgb_small_sample_pct", int(plan.tr.x.shape[1])) // 100) if n > floor else m
    if rf or getattr(plan, "mono", None) is not None:
        m = n if (m is None or m >= n) else m
    elif m is None:
        if int(cfg["objective"]) in OBJ_L3 or int(cfg["objective"]) in (2, 4, 6) or _lgbg_kind(cfg):
            LOG.info("shira.lightgbm: first_read not applied: poisson / gamma / tweedie / cross_entropy where the sample "
                     "size is probed (SHIRA_LGB_MODE=first_read_auto)")
            return None
        if stop is not None:
            LOG.info("shira.lightgbm: first_read not applied: early stopping where the sample size is probed")
            return None
        if n < FIRST_READ_AUTO_MIN:
            first_read_log.append({"n": n, "tried": [], "kept": None, "exit": "below auto minimum"})
            return None
        try:
            m = _choose(plan, cfg, th_s, th)
        except _fit.Unsupported as e:
            LOG.info("shira.lightgbm: first_read not applied: %s", e)
            return None
        if m is None:
            return None
    elif m >= n:
        return None
    rows = np.arange(n, dtype=np.int64) if m >= n else (n - 1) * np.arange(m, dtype=np.int64) // max(1, m - 1)
    try:
        t1 = time.perf_counter()
        sm = _shapes(plan, rows, th_s, th) if stop is None else _shapes_early(plan, rows, th, stop)
        if int(np.min(sm.nleaves)) < 2:
            raise _fit.Unsupported("a tree of the sample fit has one leaf")
        if stop is None and int(sm.n_trees) != plan.rounds * int(sm.num_class):
            raise _fit.Unsupported(f"the sample fit stopped at {sm.n_trees} trees (stock stops when no tree can split)")
        t2 = time.perf_counter()
        ids, nl = _leaf_ids(sm, plan.tr.x, th)
        t3 = time.perf_counter()
        mono = getattr(plan, "mono", None)
        if mono is not None:
            leaf, scores, init = _values_mono(sm, ids, nl, plan.tr.y, plan.tr.w, cfg, th, mono)
        else:
            vf = (_values_rf if rf else _values_l3 if int(cfg["objective"]) in OBJ_L3
                  else _values_mc if int(cfg["objective"]) == 2 else _values_ova if int(cfg["objective"]) == 3
                  else _values_lgbg if _lgbg_kind(cfg)
                  else _values_renew if int(cfg["objective"]) in (4, 6) else _values)
            vf = functools.partial(_values, shape_rows=rows) if vf is _values else vf
            vf = _values_smooth if float(cfg["path_smooth"]) != 0.0 and not rf else vf
            vf = _values_dart if bool(cfg.get("dart")) and not rf else vf
            if getattr(plan, "custom_obj", None) is not None:
                vf = functools.partial(_values_custom, fobj=plan.custom_obj, ds=_fobj_dataset(plan))
            if getattr(plan, "builtin_obj", None) is not None:
                vf = (functools.partial(_values_fair, c=plan.builtin_obj[1]) if plan.builtin_obj[0] == "fair"
                      else _values_mape)
            leaf, scores, init = vf(sm, ids, nl, plan.tr.y, plan.tr.w, cfg, th)
        first_read_log.append({"n": n, "m": int(rows.size), "grower": sm.grower, "sample_threads": th if sm.grower == "stock" else th_s,
                               "threads": th, "t_choose": t1 - t0,
                               "t_shapes": t2 - t1, "t_ids": t3 - t2, "t_values": time.perf_counter() - t3})
    except _fit.Unsupported as e:
        LOG.info("shira.lightgbm: first_read not applied: %s", e)
        return None
    sm.leaf, sm.scores = leaf, scores
    sm.average = rf
    sm.init = np.atleast_1d(np.asarray(init, dtype=np.float64))
    sm.stats = _all_row_stats(sm, ids, nl, leaf, th)
    grower = "stock LightGBM (the guard's pick here)" if sm.grower == "stock" else "SHIRA's engine"
    if getattr(plan, "shape_route", None):
        grower = f"stock LightGBM with the caller's {' / '.join(plan.shape_route)} (LightGBM's general route)"
    line = (f"fitted the first-read way (SHIRA's default): trees grown by {grower} at {th if sm.grower == 'stock' else th_s} threads on a stride sample of "
            f"{rows.size} of {n} rows (not stock's trees; accuracy depends on the "
            f"data), leaf values solved on all {n} rows; split gains and importances are from the sample fit, "
            f"leaf_count / internal_count are all-row counts, leaf / internal weights the sample fit's (SHAP values are "
            f"not stock's) ({time.perf_counter() - t0:.2f} s); "
            f"SHIRA_LGB_MODE=ordinary turns it off"
            + ("; boosting=rf: every tree's values are solved at the init score on all rows, not on its bag" if rf else "")
            + ("; monotone constraints: kept by the basic-method bounds on every tree's values" if getattr(plan, "mono", None) is not None else "")
            + (f"; early stopping: the sample fit stopped by the caller's rule on the caller's validation sets and kept "
               f"{sm.n_trees} trees; the best iteration is then decided on the all-rows values" if stop is not None else ""))
    return sm, line
