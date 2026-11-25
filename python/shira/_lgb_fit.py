from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shira_cpp import binning_cpp, lgb_cpp

K_ZERO = lgb_cpp.K_ZERO
MAX_FOLDS = 256
MAX_BINS = 65535
INT_KEYS = ("num_leaves", "max_depth", "min_data_in_leaf", "max_bin", "min_data_in_bin",
            "bin_construct_sample_cnt", "data_random_seed", "bagging_freq", "bagging_seed",
            "feature_fraction_seed", "num_class", "max_drop", "drop_seed", "lambdarank_truncation_level",
            "num_threads", "extra_seed")
BOOL_KEYS = ("use_missing", "zero_as_missing", "boost_from_average", "xgboost_dart_mode", "uniform_drop",
             "lambdarank_norm", "deterministic", "force_col_wise", "force_row_wise", "extra_trees",
             "is_unbalance")
DOUBLE_KEYS = ("learning_rate", "lambda_l1", "lambda_l2", "min_sum_hessian_in_leaf", "min_gain_to_split",
               "sigmoid", "path_smooth", "bagging_fraction", "feature_fraction", "feature_fraction_bynode",
               "alpha", "tweedie_variance_power", "poisson_max_delta_step", "top_rate", "other_rate",
               "drop_rate", "skip_drop", "max_delta_step",
               "scale_pos_weight")
_POSITIVE = (7, 8, 9)


class Unsupported(Exception):
    pass


@dataclass
class Resolved:
    cfg: dict[str, str]
    objective_line: str
    params_block: str


@dataclass
class FoldModel:
    nodes: np.ndarray
    default_left: np.ndarray
    leaf: np.ndarray
    gain: np.ndarray
    nleaves: np.ndarray
    upper: list
    missing_type: np.ndarray
    num_class: int
    learning_rate: float
    init: np.ndarray
    scores: np.ndarray
    feature_infos: list
    info: np.ndarray
    bins: Any = None
    stats: Any = None
    dart: bool = False
    cat: Any = None
    bin2cat: Any = None
    average: bool = False
    sched: bool = False
    leaf0: Any = None
    drops: Any = None

    @property
    def n_trees(self) -> int:
        return int(self.nleaves.size)


def alias_to_main() -> dict[str, str]:
    from lightgbm.basic import _ConfigAliases
    table = _ConfigAliases._get_all_param_aliases()
    out = {a: main for main, aliases in table.items() for a in aliases}
    assert out.get("n_estimators") == "num_iterations" and out.get("reg_lambda") == "lambda_l2"
    return out


def user_values(params: dict[str, Any]) -> dict[str, Any]:
    amap = alias_to_main()
    out: dict[str, Any] = {}
    assert len(params) < 10_000
    for k, v in params.items():
        if v is None:
            continue
        main = amap.get(k, k)
        if main in out and str(out[main]) != str(v):
            raise Unsupported(f"param {main} given under two names with different values")
        out[main] = v
    return out


def _dummy_labels(code: int, m: int, num_class: int) -> np.ndarray:
    if code == 1:
        return (np.arange(m) % 2).astype(np.float32)
    if code in (2, 3):
        return (np.arange(m) % num_class).astype(np.float32)
    if code in _POSITIVE:
        return np.full(m, 1.0, dtype=np.float32) + (np.arange(m) % 2).astype(np.float32)
    if code == 10:
        return np.where(np.arange(m) % 2 == 0, 0.25, 0.75).astype(np.float32)
    if code == 11:
        return (np.arange(m) % 2).astype(np.float32)
    return (0.5 + (np.arange(m) % 2)).astype(np.float32)


_RESOLVE_CACHE: dict[tuple[str, tuple[str, ...]], Resolved] = {}
_RESOLVE_CACHE_MAX = 64


def resolve(params: dict[str, Any], feature_names: list[str]) -> Resolved:
    try:
        key = (json.dumps(params, sort_keys=True, default=repr), tuple(str(n) for n in feature_names))
    except (TypeError, ValueError):
        return _resolve_stock(params, feature_names)
    hit = _RESOLVE_CACHE.get(key)
    if hit is None:
        hit = _resolve_stock(params, feature_names)
        if len(_RESOLVE_CACHE) >= _RESOLVE_CACHE_MAX:
            _RESOLVE_CACHE.clear()
        _RESOLVE_CACHE[key] = hit
    assert hit.objective_line and hit.params_block
    return Resolved(dict(hit.cfg), hit.objective_line, hit.params_block)


def _resolve_stock(params: dict[str, Any], feature_names: list[str]) -> Resolved:
    import lightgbm as lgb
    main = user_values(params)
    obj = str(main.get("objective", "regression"))
    if obj not in lgb_cpp.OBJECTIVES:
        raise Unsupported(f"objective {obj!r} is not in SHIRA's table")
    code = lgb_cpp.OBJECTIVES[obj]
    num_class = int(main.get("num_class", 1))
    if not 1 <= num_class <= 4096:
        raise Unsupported(f"num_class {num_class}")
    m = 2 * max(2, num_class)
    quiet = {k: v for k, v in params.items() if alias_to_main().get(k, k) != "verbosity"}
    quiet["verbosity"] = -1
    x0 = np.zeros((m, len(feature_names)), dtype=np.float64)
    try:
        ds = lgb.Dataset(x0, label=_dummy_labels(code, m, num_class), group=[m] if code == 11 else None,
                         feature_name=list(feature_names), params=dict(quiet))
        text = lgb.Booster(params=dict(quiet), train_set=ds).model_to_string()
    except lgb.basic.LightGBMError as e:
        raise Unsupported(f"LightGBM rejects these params: {str(e).splitlines()[0]}") from e
    lines = text.splitlines()
    obj_line = next((ln for ln in lines if ln.startswith("objective=")), "")
    a, b = text.index("parameters:"), text.index("end of parameters")
    block = text[a: b + len("end of parameters")] + "\n"
    cfg = {ln[1:-1].split(": ", 1)[0]: (ln[1:-1].split(": ", 1) + [""])[1] for ln in block.splitlines()
           if ln.startswith("[") and ln.endswith("]")}
    assert obj_line and "objective" in cfg and "num_leaves" in cfg, "unexpected LightGBM model text"
    user_verb = main.get("verbosity")
    if user_verb is None or int(user_verb) != -1:
        v = 1 if user_verb is None else int(user_verb)
        block = block.replace("[verbosity: -1]", f"[verbosity: {v}]")
        cfg["verbosity"] = str(v)
    return Resolved(cfg, obj_line, block)


def engine_params(rs: Resolved, params: dict[str, Any]) -> dict[str, Any]:
    main = user_values(params)
    cfg = rs.cfg
    q: dict[str, Any] = {"objective": cfg["objective"], "boosting": cfg["boosting"],
                         "data_sample_strategy": cfg["data_sample_strategy"], "feature_pre_filter": False}
    for k in INT_KEYS:
        q[k] = int(cfg[k])
    for k in BOOL_KEYS:
        q[k] = cfg[k] == "1"
    for k in DOUBLE_KEYS:
        if k in main:
            q[k] = main[k]
    if "label_gain" in main:
        lg = main["label_gain"]
        q["label_gain"] = [float(v) for v in (lg.split(",") if isinstance(lg, str) else lg)]
    assert "seed" not in q and q["feature_pre_filter"] is False
    return q


def check_engine(q: dict[str, Any]) -> dict:
    try:
        cfg = lgb_cpp.check_params(dict(q))
    except (AssertionError, NotImplementedError, ValueError, KeyError) as e:
        raise Unsupported(f"engine: {e}") from e
    assert isinstance(cfg, dict) and "objective" in cfg
    return cfg


def _plain(m: dict) -> tuple[np.ndarray, np.ndarray]:
    if m.get("categorical"):
        return np.array([np.inf]), np.array([1, 0, 0, 0], dtype=np.int32)
    return m["upper"], m["info"]


def info_row(m: dict) -> np.ndarray:
    if m.get("categorical"):
        return np.array([m["num_bin"], m["missing_type"], m["default_bin"], m["mfb"]], dtype=np.int32)
    return m["info"]


def _need_filter_cat(cnt: np.ndarray, total: int, filter_cnt: int) -> bool:
    if cnt.size > 2:
        return False
    return not any(int(c) >= filter_cnt and total - int(c) >= filter_cnt for c in cnt[:-1])


def _need_filter(cnt: np.ndarray, total: int, filter_cnt: int) -> bool:
    left = np.cumsum(cnt[:-1])
    assert cnt.ndim == 1 and total >= 0
    return not bool(np.any((left >= filter_cnt) & (total - left >= filter_cnt)))


def _sample_stats(x: np.ndarray, lo: int, hi: int, cfg: dict, mappers: list) -> tuple[np.ndarray, list, list]:
    rows = binning_cpp.lgb_sample_indices(hi - lo, cfg["sample_cnt"], cfg["data_random_seed"]) + lo
    plain = [_plain(m) for m in mappers]
    ptrs = np.concatenate([[0], np.cumsum([u.size for u, _ in plain])]).astype(np.int32)
    upper = np.concatenate([u for u, _ in plain])
    info = np.ascontiguousarray(np.stack([i for _, i in plain]), dtype=np.int32)
    nth = max(1, min(int(cfg.get("omp_threads", 1)), 64))
    nnz, mn, mx, cnt = binning_cpp.lgb_sample_stats(x, rows, ptrs, upper, info, nthread=nth)
    allnan = np.isnan(mn)
    counts, infos = [], []
    for j, mp in enumerate(mappers):
        if mp.get("categorical"):
            b = binning_cpp.lgb_cat_bin(x[rows, j], mp["b2c"])
            c = np.bincount(b, minlength=int(mp["num_bin"])).astype(np.int64)
            counts.append(c)
            nnz[j] = int(rows.size - c[int(mp["mfb"])])
            infos.append("none" if mp["trivial"] else ":".join(str(v) for v in mp["bin2cat"]))
            continue
        counts.append(cnt[j, : int(mp["num_bin"])].copy())
        infos.append("none" if mp["trivial"] or allnan[j] else f"[{mn[j]:.17g}:{mx[j]:.17g}]")
    assert len(counts) == x.shape[1]
    return nnz, counts, infos


def _dataset_rules(x: np.ndarray, lo: int, hi: int, cfg: dict, rs: Resolved, mappers: list,
                   bundle_rule: bool = True) -> list:
    nnz, counts, infos = _sample_stats(x, lo, hi, cfg, mappers)
    total = min(hi - lo, cfg["sample_cnt"])
    used = [j for j, mp in enumerate(mappers) if not mp["trivial"]]
    if rs.cfg["feature_pre_filter"] == "1":
        filter_cnt = int(float(cfg["min_data"] * total) / (hi - lo))
        for j in used:
            if (_need_filter_cat if mappers[j].get("categorical") else _need_filter)(counts[j], total, filter_cnt):
                raise Unsupported(f"feature_pre_filter would drop feature {j} (min_data_in_leaf {cfg['min_data']})")
    if bundle_rule and rs.cfg["enable_bundle"] == "1" and len(used) > 1:
        limit = (total + total // 10000) / 2.0
        sparse = [j for j in used if nnz[j] <= limit]
        if sparse:
            raise Unsupported(f"feature {sparse[0]} is mostly zero: LightGBM may bundle features (EFB); "
                              "set enable_bundle=False")
    assert len(infos) == x.shape[1]
    return infos


def _check_bins(mp: list, cfg: dict) -> None:
    big = max(int(m["num_bin"]) for m in mp)
    if big > MAX_BINS:
        raise Unsupported(f"a feature has {big} bins (more than {MAX_BINS})")
    if any(m.get("categorical") and int(m["num_bin"]) > 256 for m in mp):
        raise Unsupported("a categorical feature has more than 256 bins")
    total = sum(int(m["num_bin"]) for m in mp)
    if total >= 65536:
        raise Unsupported(f"{total} bins over all features (the engine's histogram holds at most 65,535)")
    assert big >= 1 and total >= len(mp)


def _codes_one_fold(x: np.ndarray, mappers: list, nthread: int = 1) -> tuple[np.ndarray, np.ndarray]:
    plain = [_plain(m) for m in mappers]
    ptrs = np.concatenate([[0], np.cumsum([u.size for u, _ in plain])]).astype(np.int32)
    upper = np.concatenate([u for u, _ in plain])
    info = np.ascontiguousarray(np.stack([i for _, i in plain]), dtype=np.int32)
    bins, has_inf = binning_cpp.lgb_bin2(x, ptrs, upper, info, nthread=nthread, code_bytes=2)
    if has_inf:
        raise Unsupported("infinite feature values")
    for j, m in enumerate(mappers):
        if m.get("categorical"):
            bins[:, j] = binning_cpp.lgb_cat_bin(x[:, j], m["b2c"])
    c_max = int(max(m["num_bin"] for m in mappers))
    dt = np.uint8 if c_max <= 256 else np.uint16
    lut = np.broadcast_to(np.arange(c_max, dtype=dt), (1, x.shape[1], c_max)).copy()
    for j, m in enumerate(mappers):
        lut[0, j, int(m["num_bin"]):] = 0
    assert bins.shape == x.shape and c_max <= MAX_BINS and bins.dtype == np.uint16
    return bins, lut


def _codes_merged(x: np.ndarray, all_mappers: list) -> tuple[np.ndarray, np.ndarray]:
    n, nf = x.shape
    k_n = len(all_mappers)
    codes = np.zeros((n, nf), dtype=np.uint16)
    reps = []
    for j in range(nf):
        ub = np.concatenate([mp[j]["upper"] for mp in all_mappers])
        merged = np.unique(ub[~np.isnan(ub)])
        col = x[:, j]
        nan = np.isnan(col)
        c = np.searchsorted(merged, np.where(nan, 0.0, col), side="left")
        assert int(c.max(initial=0)) < merged.size, "a value above every bin bound (the last bound is +inf)"
        c[nan] = merged.size
        codes[:, j] = c
        reps.append(np.concatenate([merged, [np.nan]]))
    c_max = max(r.size for r in reps)
    assert c_max < 65536
    wide = max(int(m["num_bin"]) for mp in all_mappers for m in mp) > 256
    lut = np.zeros((k_n, nf, c_max), dtype=np.uint16 if wide else np.uint8)
    for k, mp in enumerate(all_mappers):
        for j in range(nf):
            b = lgb_cpp._lut_column(mp[j], reps[j])
            lut[k, j, : b.size] = b
    return codes, lut


def prepare(x: np.ndarray, folds: list[tuple[int, int]], cfg: dict, rs: Resolved,
            bundle_rule: bool = True) -> tuple[dict, list]:
    assert x.ndim == 2 and x.dtype == np.float64 and 1 <= len(folds) <= MAX_FOLDS
    n, nf = x.shape
    single = len(folds) == 1 and folds[0] == (0, n)
    if not single and np.isinf(x).any():
        raise Unsupported("infinite feature values")
    nth = max(1, min(int(cfg.get("omp_threads", 1)), 64))
    cats = list(cfg.get("categorical_feature") or [])
    if cats and not single:
        raise Unsupported("categorical features with more than one fold")
    if cats and cfg.get("hist") == "row":
        raise Unsupported("categorical features with force_row_wise")
    all_mappers, infos = [], []
    for lo, hi in folds:
        assert 0 <= lo < hi <= n
        mp = lgb_cpp.fold_mappers_cpp(x, int(lo), int(hi), cfg)
        _check_bins(mp, cfg)
        all_mappers.append(mp)
    codes, lut = _codes_one_fold(x, all_mappers[0], nth) if single else _codes_merged(x, all_mappers)
    for (lo, hi), mp in zip(folds, all_mappers):
        infos.append(_dataset_rules(x, int(lo), int(hi), cfg, rs, mp, bundle_rule))
    k_n = len(folds)
    nbin, mfb, dbin = (np.zeros((k_n, nf), dtype=np.int32) for _ in range(3))
    mtype = np.zeros((k_n, nf), dtype=np.int8)
    for k, mp in enumerate(all_mappers):
        for j in range(nf):
            nbin[k, j], mfb[k, j], dbin[k, j] = mp[j]["num_bin"], mp[j]["mfb"], mp[j]["default_bin"]
            mtype[k, j] = mp[j]["missing_type"]
    prep = {"cfg": cfg, "codes": codes, "uniq": None, "lut": lut, "nbin": nbin, "mfb": mfb, "dbin": dbin,
            "mtype": mtype, "mappers": all_mappers, "folds": [(int(a), int(b)) for a, b in folds], "iscat": None}
    if cats:
        prep["iscat"] = np.zeros(nf, dtype=np.int8)
        prep["iscat"][cats] = 1
    return prep, infos


SK_DEFAULT_THREADS: contextvars.ContextVar[bool] = contextvars.ContextVar("shira_lgb_sk_default_threads", default=False)


def x86_guard(threads: int, rows: int = 0, features: int = 0) -> str | None:
    assert threads >= 1 and rows >= 0 and features >= 0
    return None


def auto_rowwise(features: int, rows: int) -> bool:
    from . import _machine

    assert features >= 1 and rows >= 1
    return rows >= _machine.value("lgb_colwise_below_rows", features)


def engine_threads(rs: Resolved, rows: int) -> int:
    nt = int(rs.cfg["num_threads"])
    from . import _threads

    if nt > 0 and not SK_DEFAULT_THREADS.get():
        return max(1, min(lgb_cpp.MAX_THREADS, _threads.explicit_threads(nt)))

    stock_default = nt if nt > 0 else (os.cpu_count() or 1)
    return max(1, min(lgb_cpp.MAX_THREADS, _threads.default_threads("lightgbm", rows, stock_default)))


class _Lines:

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(str(msg))

    def warning(self, msg: str) -> None:
        self.lines.append(str(msg))


def stock_hist_choice(params: dict[str, Any], x: np.ndarray, y: np.ndarray, lo: int, hi: int,
                      group: Any = None) -> int:
    from lightgbm import basic as lb

    assert 0 <= lo < hi <= x.shape[0]
    p = {k: v for k, v in params.items() if k not in lb._ConfigAliases.get("verbosity")}
    p["verbosity"] = 1
    g = None if group is None else lgb_cpp.fold_groups(group, lo, hi)
    cap = _Lines()
    saved = (lb._LOGGER, lb._INFO_METHOD_NAME, lb._WARNING_METHOD_NAME)
    try:
        lb.register_logger(cap)
        ds = lb.Dataset(x[lo:hi], label=y[lo:hi], group=g, params=dict(p))
        lb.Booster(params=dict(p), train_set=ds)
    except lb.LightGBMError as e:
        raise Unsupported(f"LightGBM rejects these params: {str(e).splitlines()[0]}") from e
    finally:
        lb._LOGGER, lb._INFO_METHOD_NAME, lb._WARNING_METHOD_NAME = saved
    text = "\n".join(cap.lines)
    if "Auto-choosing row-wise" in text:
        return 1
    if "Auto-choosing col-wise" in text:
        return 0
    raise Unsupported("LightGBM did not report its histogram method")


def fit_folds(q: dict[str, Any], rs: Resolved, x: np.ndarray, y: np.ndarray, folds: list[tuple[int, int]],
              rounds: int, weight: np.ndarray | None = None, group: np.ndarray | None = None,
              keep_bins: bool = False, stock_params: dict[str, Any] | None = None,
              threads: int | None = None, bundle_rule: bool = True,
              order_exact: bool = False, prep_in: tuple[dict, list] | None = None) -> list[FoldModel]:
    cfg = check_engine(q)
    assert rounds >= 1 and y.shape[0] == x.shape[0]
    if threads is None:
        threads = engine_threads(rs, max(int(b) - int(a) for a, b in folds))
    assert threads >= 1
    guard = x86_guard(threads, max(int(b) - int(a) for a, b in folds), int(x.shape[1]))
    if guard is not None:
        raise Unsupported(guard)
    prep, infos = prepare(x, folds, cfg, rs, bundle_rule) if prep_in is None else prep_in
    assert prep["folds"] == [(int(a), int(b)) for a, b in folds]
    prep["order_exact"] = bool(order_exact)
    if cfg["hist"] == "auto" and cfg["omp_threads"] > 1:
        rw = [1 if auto_rowwise(int(x.shape[1]), int(b) - int(a)) else 0 for a, b in folds]
        prep["rowwise"] = np.asarray(rw, dtype=np.uint8)
    try:
        res = lgb_cpp.train(prep, y, int(rounds), threads=threads,
                            trees_out=True, weight=weight,
                            group=group)
    except (AssertionError, NotImplementedError, ValueError, RuntimeError) as e:
        raise Unsupported(f"engine: {e}") from e
    n_c = cfg["num_class"]
    out = []
    for k in range(len(folds)):
        nt = int(res["ntrees"][k])
        if nt != rounds * n_c:
            raise Unsupported(f"training stopped after {nt // n_c} of {rounds} rounds (no further split)")
        mp = prep["mappers"][k]
        out.append(FoldModel(res["nodes"][k], res["default_left"][k], res["leaf"][k], res["gain"][k],
                             res["nleaves"][k], [m["upper"] for m in mp], prep["mtype"][k].copy(), n_c,
                             float(cfg["learning_rate"]), res["init"][k].copy(), res["preds"][k], infos[k],
                             np.ascontiguousarray(np.stack([info_row(m) for m in mp]), dtype=np.int32),
                             lgb_cpp.fold_bins(prep, k) if keep_bins else None,
                             None if res.get("stats") is None else res["stats"][k], bool(cfg.get("dart")),
                             None if res.get("cat") is None else res["cat"][k],
                             [m.get("bin2cat") for m in mp] if res.get("cat") is not None else None))
    assert len(out) == len(folds)
    if res.get("leaf0") is not None:
        for k, fm in enumerate(out):
            dn = res["drop_n"][k]
            if bool((dn < 0).any()):
                raise Unsupported("a DART round dropped more trees than its record holds")
            fm.leaf0 = res["leaf0"][k]
            fm.drops = [res["drop_idx"][k, r, :int(dn[r])].tolist() for r in range(int(rounds))]
    _order_report(out, prep.get("_order_counts"), rounds * n_c, bool(order_exact))
    return out


def _order_report(out: list[FoldModel], counts: np.ndarray | None, n_trees: int, exact: bool) -> None:
    assert counts is None or counts.shape == (len(out), 2)
    assert n_trees >= 1
    if counts is None:
        return
    for k, fm in enumerate(out):
        fm.order_dependent = (int(counts[k, 0]), int(counts[k, 1]), n_trees)
    trees = int(counts[:, 1].sum())
    if exact and int(counts[:, 0].sum()) > 0:
        logging.getLogger("shira.lightgbm").info(
            "shira.lightgbm: stock LightGBM's own result depends on thread arrival order in %d of %d trees of this fit "
            "(%d reductions); SHIRA's result is one of stock's possible results", trees, n_trees * len(out),
            int(counts[:, 0].sum()))


def subset_groups(group: np.ndarray, rows: np.ndarray, n: int) -> np.ndarray:
    g = np.asarray(group, dtype=np.int64)
    assert int(g.sum()) == n and rows.ndim == 1
    qid = np.repeat(np.arange(g.size), g)
    cnt = np.bincount(qid[rows], minlength=g.size)
    if bool(((cnt != 0) & (cnt != g)).any()):
        raise Unsupported("a cv fold cuts a query group")
    return g[cnt != 0].astype(np.int32)


def fit_subsets(q: dict[str, Any], rs: Resolved, x: np.ndarray, y: np.ndarray, subsets: list, rounds: int,
                weight: np.ndarray | None = None, group: np.ndarray | None = None,
                init_cat: np.ndarray | None = None) -> list[FoldModel]:
    cfg = check_engine(q)
    n = x.shape[0]
    k_n = len(subsets)
    assert rounds >= 1 and y.shape[0] == n and 1 <= k_n <= MAX_FOLDS
    full, infos = prepare(x, [(0, n)], cfg, rs)
    rows = [np.ascontiguousarray(r, dtype=np.int64) for r in subsets]
    for r in rows:
        if r.size < 1 or int(r.min()) < 0 or int(r.max()) >= n or bool((np.diff(r) <= 0).any()):
            raise Unsupported("cv fold rows are not ascending distinct row indices")
    offs = np.concatenate([[0], np.cumsum([r.size for r in rows])]).astype(np.int64)
    cat = np.concatenate(rows)
    rep = lambda a: np.ascontiguousarray(np.repeat(a, k_n, axis=0))
    folds = [(int(offs[k]), int(offs[k + 1])) for k in range(k_n)]
    prep = {"cfg": cfg, "codes": np.ascontiguousarray(full["codes"][cat]), "uniq": None, "lut": rep(full["lut"]),
            "nbin": rep(full["nbin"]), "mfb": rep(full["mfb"]), "dbin": rep(full["dbin"]), "mtype": rep(full["mtype"]),
            "mappers": [full["mappers"][0]] * k_n, "folds": folds, "iscat": full["iscat"],
            "drate": rep(lgb_cpp.dense_rates(full)), "dart_trace": True}
    g_cat = None if group is None else np.concatenate([subset_groups(group, r, n) for r in rows])
    if q.get("_shira_init_score") is not None:
        isc = np.asarray(q["_shira_init_score"])
        assert isc.shape[0] == n
        q = dict(q, _shira_init_score=np.ascontiguousarray(isc[cat]))
        prep["cfg"] = check_engine(q)
    if init_cat is not None:
        assert init_cat.shape[0] == cat.size
        q = dict(q, _shira_init_score=np.ascontiguousarray(init_cat, dtype=np.float64))
        prep["cfg"] = check_engine(q)
    w_cat = None if weight is None else np.ascontiguousarray(weight[cat])
    return fit_folds(q, rs, np.ascontiguousarray(x[cat]), np.ascontiguousarray(y[cat]), folds, rounds, w_cat, g_cat,
                     order_exact=True, prep_in=(prep, [infos[0]] * k_n))
