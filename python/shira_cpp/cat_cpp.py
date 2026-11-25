from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
LIB_PATH = (Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB")
            else SHIRA / "lib" / ("libshira.dylib" if sys.platform == "darwin" else "libshira.so"))
_P = ctypes.c_void_p
_I64 = ctypes.c_int64
_I32 = ctypes.c_int32
_LIB: ctypes.CDLL | None = None
MAX_THREADS = 64
MAX_DEPTH = 16
PACK_BITS = 8
BORDER_SAMPLE_ROWS = 200_000
F32_LOWEST = np.float32(np.finfo(np.float32).min)
F32_MAX = np.float32(np.finfo(np.float32).max)
HEAP_MODE = 0

SUPPORTED = {"iterations", "depth", "learning_rate", "l2_leaf_reg", "loss_function", "score_function",
             "boosting_type", "bootstrap_type", "random_strength", "boost_from_average",
             "leaf_estimation_iterations", "leaf_estimation_method", "border_count", "thread_count",
             "verbose", "random_seed", "grow_policy", "feature_border_type", "allow_writing_files",
             "logging_level", "nan_mode", "random_score_type", "bagging_temperature", "subsample", "mvs_reg",
             "sampling_frequency", "permutation_count", "fold_len_multiplier", "fold_permutation_block",
             "one_hot_max_size", "max_ctr_complexity",
             "shira_fit_quantization"}
REQUIRED = {"score_function": None, "boosting_type": None, "bootstrap_type": None,
            "boost_from_average": None, "leaf_estimation_iterations": 1, "leaf_estimation_method": None}
SCORE_FUNCTIONS = {"L2": 0, "Cosine": 1}
BOOTSTRAPS = {"No": 0, "Bayesian": 1, "Bernoulli": 2, "MVS": 3}
SUBSAMPLE_DEFAULT = {"Bernoulli": 0.66, "MVS": 0.8}
LOSSES = {"RMSE": 0, "Logloss": 1, "MultiClass": 2, "MultiClassOneVsAll": 3, "CrossEntropy": 1, "MAE": 4,
          "Quantile": 4, "Poisson": 5, "Tweedie": 6}
LOSS_PARAMS = {"Quantile": {"alpha", "delta"}, "MAE": set(), "Tweedie": {"variance_power"}}
SCORE_TYPES = {"NormalWithModelSizeDecrease": 1, "Gumbel": 2}
DEFAULTS_OK = {"grow_policy": "SymmetricTree", "feature_border_type": "GreedyLogSum"}


@dataclass
class FoldModel:
    pred: np.ndarray
    borders: list = field(default_factory=list)
    splits: list = field(default_factory=list)
    leaves: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    bundles: list = field(default_factory=list)


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        assert LIB_PATH.exists(), f"build first: {SHIRA / 'build.sh'}"
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.shira_train_cat_ex.restype = ctypes.c_int
        lib.shira_train_cat_ex.argtypes = [_P, _I64, _I32, _I32, _P, _P, _P, _P, _I32, _P, _P, _P, _P, _P, _P, _P,
                                           _I32, _I32, _P, _P, _P, _P, _P]
        lib.shira_cat_code_counts.restype = ctypes.c_int
        lib.shira_cat_code_counts.argtypes = [_P, _I64, _I32, _I32, _P, _P, _I32, _I64, ctypes.c_uint64, _I32,
                                              _P, _P]
        if hasattr(lib, "shira_cat_value_codes"):
            lib.shira_cat_value_codes.restype = ctypes.c_int
            lib.shira_cat_value_codes.argtypes = [_P, _I64, _I32, _I32, _I32, _P, _P, _P]
        lib.shira_cat_border_rows.restype = ctypes.c_int
        lib.shira_cat_border_rows.argtypes = [_I64, _I64, ctypes.c_uint64, _P]
        lib.shira_cat_greedy_logsum.restype = ctypes.c_int
        lib.shira_cat_greedy_logsum.argtypes = [_P, _P, _I64, _I32, _I32, _P, _P]
        for name, args in (("shira_cat_check_labels", [_P, _I64, _I32, _I32, _P, _P, _I32, _P]),
                           ("shira_cat_onehot_bins", [_P, _I64, _I64, _P, _I32, _P, _I32, _P, _P, _P]),
                           ("shira_cat_column_groups", [_P, _I32, _I32, _P, _P, _I32, _P]),
                           ("shira_cat_wide_quantization", [_P, _I64, _I32, _I32, _P, _P, _I32, ctypes.c_uint64,
                                                            _I32, _I32, _I32, _P, _I32, _P, _P, _P, _P, _P])):
            getattr(lib, name).restype = ctypes.c_int
            getattr(lib, name).argtypes = args
        lib.shira_cat_start_rmse.restype = ctypes.c_int32
        lib.shira_cat_start_rmse.argtypes = [_P, _P, _I64, _P]
        lib.shira_cat_start_quantile.restype = ctypes.c_int32
        lib.shira_cat_start_quantile.argtypes = [_P, _P, _I64, ctypes.c_double, ctypes.c_double, _P]
        lib.shira_cat_auto_learning_rate.restype = ctypes.c_int32
        lib.shira_cat_auto_learning_rate.argtypes = [_I32, _I32, _I32, _I64, _I64, _P]
        lib.shira_cat_border_presence.restype = ctypes.c_int
        lib.shira_cat_border_presence.argtypes = [_P, _I64, _I32, _I32, _P, _P, _I32, _I64, ctypes.c_uint64,
                                                  _I32, _P]
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray | None) -> int | None:
    return None if a is None else a.ctypes.data


def check_params(params: dict) -> dict:
    bad = sorted(set(params) - SUPPORTED)
    assert not bad, f"unsupported CatBoost parameters: {bad}"
    for key, want in REQUIRED.items():
        assert key in params and (want is None or params[key] == want), f"set {key}={want!r} (got {params.get(key)!r})"
    for key, want in DEFAULTS_OK.items():
        assert params.get(key, want) == want, f"{key} must be {want}"
    loss, lp = parse_loss(params.get("loss_function", "RMSE"))
    method = "Exact" if LOSSES[loss] == 4 else "Newton"
    assert params["leaf_estimation_method"] == method, f"loss {loss}: set leaf_estimation_method={method!r}"
    depth = int(params.get("depth", 6))
    assert 1 <= depth <= MAX_DEPTH, f"depth {depth} not supported (1..{MAX_DEPTH})"
    border_count = int(params.get("border_count", 254))
    assert 1 <= border_count <= 254, "border_count must be in 1..254"
    nan_mode = params.get("nan_mode", "Min")
    assert nan_mode in ("Min", "Max", "Forbidden"), f"nan_mode {nan_mode} not supported"
    rs = float(np.float32(params.get("random_strength", 1.0)))
    assert np.isfinite(rs) and rs >= 0.0, "random_strength must be >= 0"
    st = params.get("random_score_type", "NormalWithModelSizeDecrease")
    assert st in SCORE_TYPES, f"random_score_type {st} not supported"
    bt = params["bootstrap_type"]
    assert bt in BOOTSTRAPS, f"bootstrap_type {bt} not supported"
    assert params.get("sampling_frequency", "PerTree") == "PerTree", "sampling_frequency must be PerTree"
    assert bt == "Bayesian" or "bagging_temperature" not in params, "bagging_temperature needs Bayesian"
    assert bt in SUBSAMPLE_DEFAULT or "subsample" not in params, "subsample needs Bernoulli or MVS"
    assert bt == "MVS" or "mvs_reg" not in params, "mvs_reg needs MVS"
    sub = float(np.float32(params.get("subsample", SUBSAMPLE_DEFAULT.get(bt, 1.0))))
    assert 0.0 < sub <= 1.0, "subsample must be in (0, 1]"
    bag = float(np.float32(params.get("bagging_temperature", 1.0)))
    assert bag >= 0.0, "bagging_temperature must be >= 0"
    mvs = params.get("mvs_reg")
    boosting = params["boosting_type"]
    assert boosting in ("Plain", "Ordered"), f"boosting_type {boosting} not supported"
    assert boosting == "Ordered" or not {"permutation_count", "fold_len_multiplier",
                                         "fold_permutation_block"} & set(params), "ordered options need Ordered"
    assert boosting == "Plain" or bt == "No", "Ordered boosting with bootstrap is not supported"
    perm_count = int(params.get("permutation_count", 4))
    assert 1 <= perm_count <= 64, "permutation_count must be in 1..64"
    fold_mult = float(params.get("fold_len_multiplier", 2.0))
    assert fold_mult > 1.0, "fold_len_multiplier must be > 1"
    perm_block = int(params.get("fold_permutation_block", 0) or 0)
    assert perm_block >= 0
    assert LOSSES[loss] not in (2, 3) or (boosting == "Plain" and bt == "No"), "multiclass: Plain, bootstrap No only"
    assert LOSSES[loss] < 4 or boosting == "Plain", f"{loss} with Ordered boosting is not supported"
    sf = params["score_function"]
    assert sf in SCORE_FUNCTIONS, f"score_function {sf} not supported"
    bfa = params["boost_from_average"]
    assert bfa in (True, False), "boost_from_average must be True or False"
    assert not bfa or (loss in ("RMSE", "MAE", "Quantile") and boosting == "Plain"), \
        "boost_from_average True: RMSE, MAE or Quantile with Plain boosting"
    assert sf == "L2" or boosting == "Plain", "score_function Cosine: Plain boosting only"
    ohm = int(params.get("one_hot_max_size", 2))
    mcc = params.get("max_ctr_complexity")
    assert 2 <= ohm <= 255, "one_hot_max_size must be in 2..255"
    return {"loss": LOSSES[loss], "depth": depth, "xent": loss == "CrossEntropy", "ohm": ohm, "shuffle": False,
            "mcc": mcc, "ctr": 0, "cosine": sf == "Cosine", "bfa": bool(bfa),
            "plain_basic": boosting == "Plain" and loss in ("RMSE", "Logloss"),
            "ctr_ok": boosting == "Plain" and loss in ("CrossEntropy", "Poisson", "Tweedie", "MAE", "Quantile",
                                                       "MultiClass", "MultiClassOneVsAll"),
            "qa": lp.get("alpha", 0.5), "qd": lp.get("delta", 1e-6), "vp": lp.get("variance_power", 1.5),
            "lr": float(np.float32(params.get("learning_rate", 0.03))),
            "l2": float(np.float32(params.get("l2_leaf_reg", 3.0))),
            "iterations": int(params.get("iterations", 1000)), "border_count": border_count,
            "seed": int(params.get("random_seed", 0) or 0), "nan_mode": nan_mode, "rs": rs,
            "score_type": SCORE_TYPES[st], "boot": BOOTSTRAPS[bt], "subsample": sub, "bag_temp": bag,
            "mvs_reg": float("nan") if mvs is None else float(np.float32(mvs)),
            "ordered": boosting == "Ordered", "perm_count": perm_count, "fold_mult": fold_mult,
            "perm_block": perm_block, "raw_quant": params.get("shira_fit_quantization", False) is True}


def start_value(y32: np.ndarray, w32: np.ndarray | None, loss_spec: str = "RMSE") -> float:
    y32 = np.ascontiguousarray(y32, dtype=np.float32)
    w32 = None if w32 is None else np.ascontiguousarray(w32, dtype=np.float32)
    assert y32.ndim == 1 and y32.size >= 1 and (w32 is None or w32.shape == y32.shape)
    out = ctypes.c_double(0.0)
    name, lp = parse_loss(loss_spec)
    assert name in ("RMSE", "MAE", "Quantile"), f"no starting value rule for {name}"
    if name == "RMSE":
        rc = _lib().shira_cat_start_rmse(_ptr(y32), _ptr(w32), y32.size, ctypes.addressof(out))
    else:
        rc = _lib().shira_cat_start_quantile(_ptr(y32), _ptr(w32), y32.size, float(lp.get("alpha", 0.5)),
                                             float(lp.get("delta", 1e-6)), ctypes.addressof(out))
    assert rc == 0, f"starting value: engine returned {rc}"
    return out.value


def auto_learning_rate(loss: str, use_best_model: bool, bfa: bool, iterations: int, n_rows: int) -> float | None:
    ids = {"RMSE": 0, "Logloss": 1, "MultiClass": 2}
    assert iterations >= 1 and n_rows >= 1
    if loss not in ids:
        return None
    out = ctypes.c_double(0.0)
    rc = _lib().shira_cat_auto_learning_rate(ids[loss], int(bool(use_best_model)), int(bool(bfa)), iterations,
                                             n_rows, ctypes.addressof(out))
    assert rc in (0, 1), f"shira_cat_auto_learning_rate returned {rc}"
    return out.value if rc == 0 else None


def parse_loss(spec: str) -> tuple[str, dict]:
    name, _, rest = str(spec).partition(":")
    assert name in LOSSES, f"loss_function {name} not supported"
    lp = {}
    for item in filter(None, rest.split(";")):
        key, _, val = item.partition("=")
        assert key in LOSS_PARAMS.get(name, set()), f"loss parameter {key!r} not supported for {name}"
        lp[key] = float(val)
    assert name != "Tweedie" or "variance_power" in lp, "Tweedie needs variance_power"
    assert 1.0 < lp.get("variance_power", 1.5) < 2.0, "variance_power must be in (1, 2)"
    assert 0.0 <= lp.get("alpha", 0.5) <= 1.0 and 0.0 <= lp.get("delta", 1e-6) <= 1e-2
    return name, lp


def option_vectors(cfg: dict):
    if (cfg["rs"] == 0.0 and cfg["boot"] == 0 and not cfg["ordered"] and cfg.get("nclass", 1) == 1
            and cfg["loss"] < 4 and not cfg["shuffle"] and not cfg.get("cosine") and not cfg.get("bfa")):
        return None, None
    oi = np.array([16, 8, cfg["seed"] & (2**63 - 1), cfg["score_type"], cfg["boot"], 0, int(cfg["ordered"]),
                   cfg["perm_count"], cfg["perm_block"], cfg.get("nclass", 1), int(cfg["shuffle"]),
                   int(cfg.get("ctr", 0)), int(cfg["mcc"] or 1),
                   int(bool(cfg.get("cosine"))), int(bool(cfg.get("bfa"))),
                   int(bool(cfg.get("raw_quant")))], dtype=np.int64)
    od = np.array([cfg["rs"], cfg["bag_temp"], cfg["subsample"], cfg["mvs_reg"], cfg["fold_mult"], cfg["qa"],
                   cfg["qd"], cfg["vp"]], dtype=np.float64)
    return oi, od


def value_codes(x: np.ndarray, threads: int = 6) -> tuple[np.ndarray, list]:
    xf = np.ascontiguousarray(x, dtype=np.float32)
    assert xf.ndim == 2 and not np.isinf(xf).any(), "2-D numerical features without +-inf only"
    assert not (xf == F32_LOWEST).any() and not (xf == F32_MAX).any(), "float32 lowest/max values not supported"
    n, F = xf.shape
    if not hasattr(_lib(), "shira_cat_value_codes"):
        return _value_codes_numpy(xf, threads)
    codes = np.zeros(xf.shape, dtype=np.uint16)
    uq = np.zeros((F, 65535), dtype=np.float32)
    nu = np.zeros(F, dtype=np.int32)
    rc = _lib().shira_cat_value_codes(xf.ctypes.data, n, F, 65535, max(1, min(threads, MAX_THREADS)),
                                      codes.ctypes.data, uq.ctypes.data, nu.ctypes.data)
    assert rc == 0, f"shira_cat_value_codes returned {rc}"
    return codes, [uq[f, : nu[f]].copy() if nu[f] >= 0 else None for f in range(F)]


def _value_codes_numpy(xf: np.ndarray, threads: int) -> tuple[np.ndarray, list]:
    from concurrent.futures import ThreadPoolExecutor
    codes = np.zeros(xf.shape, dtype=np.uint16)

    def one(f: int) -> np.ndarray | None:
        col = np.ascontiguousarray(xf[:, f])
        u = np.unique(col)
        if u.size > 65535:
            return None
        codes[:, f] = np.searchsorted(u, col).astype(np.uint16)
        return u

    with ThreadPoolExecutor(max_workers=max(1, min(threads, xf.shape[1]))) as ex:
        uniq = list(ex.map(one, range(xf.shape[1])))
    return codes, uniq


_M64 = (1 << 64) - 1
_K0, _K1, _K2, _K3 = 0xc3a5c85c97cb3127, 0xb492b66fbe98f273, 0x9ae16a3b2f90404f, 0xc949d7c7509e6557


def _rot(v: int, s: int) -> int:
    return v if s == 0 else ((v >> s) | (v << (64 - s))) & _M64


def _smix(v: int) -> int:
    return v ^ (v >> 47)


def _h16(u: int, v: int) -> int:
    k = 0x9ddfea08eb382d69
    a = ((u ^ v) * k) & _M64
    a ^= a >> 47
    b = ((v ^ a) * k) & _M64
    b ^= b >> 47
    return (b * k) & _M64


def _weak32(w: int, x: int, y: int, z: int, a: int, b: int) -> tuple[int, int]:
    a = (a + w) & _M64
    b = _rot((b + a + z) & _M64, 21)
    c = a
    a = (a + x + y) & _M64
    b = (b + _rot(a, 44)) & _M64
    return (a + z) & _M64, (b + c) & _M64


def _city_long(s: bytes) -> int:
    n = len(s)
    ld = lambda o: int.from_bytes(s[o:o + 8], "little")
    wk = lambda o, a, b: _weak32(ld(o), ld(o + 8), ld(o + 16), ld(o + 24), a, b)
    x, y, z = ld(0), ld(n - 16) ^ _K1, ld(n - 56) ^ _K0
    v = wk(n - 64, n, y)
    w = wk(n - 32, (n * _K1) & _M64, _K0)
    z = (z + _smix(v[1]) * _K1) & _M64
    x = (_rot((z + x) & _M64, 39) * _K1) & _M64
    y = (_rot(y, 33) * _K1) & _M64
    off = 0
    for _ in range(((n - 1) & ~63) // 64):
        x = (_rot((x + y + v[0] + ld(off + 16)) & _M64, 37) * _K1) & _M64
        y = (_rot((y + v[1] + ld(off + 48)) & _M64, 42) * _K1) & _M64
        x ^= w[1]
        y ^= v[0]
        z = _rot(z ^ w[0], 33)
        v = wk(off, (v[1] * _K1) & _M64, (x + w[0]) & _M64)
        w = wk(off + 32, (z + w[1]) & _M64, y)
        z, x = x, z
        off += 64
    return _h16((_h16(v[0], w[0]) + _smix(y) * _K1 + z) & _M64, (_h16(v[1], w[1]) + x) & _M64)


def cityhash64(s: bytes) -> int:
    n = len(s)
    ld = lambda o: int.from_bytes(s[o:o + 8], "little")
    l4 = lambda o: int.from_bytes(s[o:o + 4], "little")
    if n > 64:
        return _city_long(s)
    if n == 0:
        return _K2
    if n < 4:
        y = (s[0] + (s[n >> 1] << 8)) & 0xffffffff
        z = (n + (s[n - 1] << 2)) & 0xffffffff
        return (_smix(((y * _K2) & _M64) ^ ((z * _K3) & _M64)) * _K2) & _M64
    if n <= 8:
        return _h16((n + (l4(0) << 3)) & _M64, l4(n - 4))
    if n <= 16:
        a, b = ld(0), ld(n - 8)
        return _h16(a, _rot((b + n) & _M64, n)) ^ b
    if n <= 32:
        a, b, c, d = (ld(0) * _K1) & _M64, ld(8), (ld(n - 8) * _K2) & _M64, (ld(n - 16) * _K0) & _M64
        return _h16((_rot((a - b) & _M64, 43) + _rot(c, 30) + d) & _M64, (a + _rot(b ^ _K3, 20) - c + n) & _M64)
    z = ld(24)
    a = (ld(0) + (n + ld(n - 16)) * _K0) & _M64
    b, c = _rot((a + z) & _M64, 52), _rot(a, 37)
    a = (a + ld(8)) & _M64
    c = (c + _rot(a, 7)) & _M64
    a = (a + ld(16)) & _M64
    vf, vs = (a + z) & _M64, (b + _rot(a, 31) + c) & _M64
    a = (ld(16) + ld(n - 32)) & _M64
    z = ld(n - 8)
    b, c = _rot((a + z) & _M64, 52), _rot(a, 37)
    a = (a + ld(n - 24)) & _M64
    c = (c + _rot(a, 7)) & _M64
    a = (a + ld(n - 16)) & _M64
    wf, ws = (a + z) & _M64, (b + _rot(a, 31) + c) & _M64
    r = _smix(((vf + ws) * _K2 + (wf + vs) * _K0) & _M64)
    return (_smix((r * _K0 + vs) & _M64) * _K2) & _M64


def cat_hash(value) -> int:
    return cityhash64(str(value).encode("utf-8")) & 0xffffffff


@dataclass
class CatCodes:
    hashes: np.ndarray

    @property
    def size(self) -> int:
        return int(self.hashes.size)


def _ascii_unique(col: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if col.dtype.kind != "U" or col.dtype.itemsize // 4 > 9 or col.size == 0:
        return None
    u = np.ascontiguousarray(col).view(np.uint32).reshape(col.size, col.dtype.itemsize // 4)
    if int(u.max()) >= 128:
        return None
    key = np.zeros(col.size, dtype=np.int64)
    for k in range(u.shape[1]):
        key = (key << 7) | u[:, k].astype(np.int64)
    head, at = np.unique(key[:4096], return_index=True)
    if head.size > 256:
        return None
    inv = np.minimum(np.searchsorted(head, key), head.size - 1)
    if not np.array_equal(head[inv], key):
        return None
    assert at.size == head.size
    return col[at], inv


def cat_codes(cat: np.ndarray) -> tuple[np.ndarray, list]:
    cat = np.asarray(cat)
    assert cat.ndim == 2, "cat must be a 2-D array (rows x categorical features)"
    codes = np.zeros(cat.shape, dtype=np.uint16)
    info = []
    for j in range(cat.shape[1]):
        col = cat[:, j]
        assert col.dtype.kind in "OUSiu", "categorical values must be strings or integers"
        fast = _ascii_unique(col)
        vals, inv = fast if fast is not None else np.unique(col.astype(str), return_inverse=True)
        assert vals.size <= 65535, f"categorical feature {j}: too many distinct values"
        codes[:, j] = inv.astype(np.uint16)
        info.append(CatCodes(np.array([cat_hash(v) for v in vals], dtype=np.uint32)))
    return codes, info


def cat_bins(codes_col: np.ndarray, counts: np.ndarray, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert codes_col.dtype == np.uint16 and codes_col.ndim == 1 and codes_col.strides[0] % 2 == 0
    cnt = np.ascontiguousarray(counts, dtype=np.int64)
    hs = np.ascontiguousarray(hashes, dtype=np.uint32)
    bin_of = np.zeros(cnt.size, dtype=np.int64)
    hb = np.zeros(cnt.size, dtype=np.int32)
    nbin = np.zeros(1, dtype=np.int32)
    rc = _lib().shira_cat_onehot_bins(codes_col.ctypes.data, codes_col.size, codes_col.strides[0] // 2,
                                      cnt.ctypes.data, cnt.size, hs.ctypes.data, hs.size, bin_of.ctypes.data,
                                      hb.ctypes.data, nbin.ctypes.data)
    assert rc == 0, f"shira_cat_onehot_bins returned {rc}"
    return bin_of, hb[: int(nbin[0])].copy()


def border_rows(m: int, seed: int) -> np.ndarray:
    out = np.zeros(m, dtype=np.uint8)
    rc = _lib().shira_cat_border_rows(m, BORDER_SAMPLE_ROWS, seed & (2**64 - 1), out.ctypes.data)
    assert rc == 0, f"shira_cat_border_rows returned {rc}"
    return out.astype(bool)


def wide_quantization(xf: np.ndarray, f: int, lo: np.ndarray, hi: np.ndarray, cfg: dict, codes: np.ndarray):
    assert xf.dtype == np.float32 and xf.flags.c_contiguous and codes.flags.c_contiguous
    K = lo.size
    bd = np.zeros((K, 256), dtype=np.float32)
    nbd = np.zeros(K, dtype=np.int32)
    luts = np.zeros((K, 65536), dtype=np.int64)
    nco = np.zeros(1, dtype=np.int32)
    bad = np.full(1, -1, dtype=np.int32)
    nm = {"Min": 0, "Max": 1, "Forbidden": 2}[cfg["nan_mode"]]
    rc = _lib().shira_cat_wide_quantization(xf.ctypes.data, xf.shape[0], xf.shape[1], f, lo.ctypes.data,
                                            hi.ctypes.data, K, cfg["seed"] & (2**64 - 1), cfg["border_count"], nm,
                                            HEAP_MODE, codes.ctypes.data, codes.shape[1], bd.ctypes.data,
                                            nbd.ctypes.data, luts.ctypes.data, nco.ctypes.data, bad.ctypes.data)
    assert rc != -20, f"feature {f}: NaN with nan_mode=Forbidden (CatBoost errors)"
    assert rc != -21, f"feature {f}: NaN rows but none in the border sample (CatBoost errors)"
    assert rc != -22, f"feature {f}: union of fold borders too large"
    assert rc > -100, f"shira_cat_greedy_logsum returned {rc + 100}"
    assert rc == 0, f"shira_cat_wide_quantization returned {rc}"
    nc = int(nco[0])
    return [(bd[k, : nbd[k]].copy(), luts[k, :nc].copy()) for k in range(K)], nc


def presence(codes: np.ndarray, C: int, lo: np.ndarray, hi: np.ndarray, seed: int, threads: int) -> np.ndarray:
    n, F = codes.shape
    out = np.zeros((lo.size, F, C), dtype=np.uint8)
    rc = _lib().shira_cat_border_presence(codes.ctypes.data, n, F, C, lo.ctypes.data, hi.ctypes.data, lo.size,
                                          BORDER_SAMPLE_ROWS, seed & (2**64 - 1), threads, out.ctypes.data)
    assert rc == 0, f"shira_cat_border_presence returned {rc}"
    return out.astype(bool)


def greedy_logsum(vals: np.ndarray, cnt: np.ndarray, max_borders: int) -> np.ndarray:
    v = np.ascontiguousarray(vals, dtype=np.float32)
    c = np.ascontiguousarray(cnt, dtype=np.int64)
    out = np.zeros(max_borders, dtype=np.float32)
    nout = np.zeros(1, dtype=np.int32)
    rc = _lib().shira_cat_greedy_logsum(v.ctypes.data, c.ctypes.data, v.size, max_borders, HEAP_MODE,
                                        out.ctypes.data, nout.ctypes.data)
    assert rc == 0, f"shira_cat_greedy_logsum returned {rc}"
    return out[: int(nout[0])].copy()


def feature_quantization(scnt: np.ndarray, u: np.ndarray, in_fold: np.ndarray, border_count: int,
                         nan_mode: str, f: int):
    has_nan_code = u.size > 0 and np.isnan(u[-1])
    nan_code = u.size - 1 if has_nan_code else -1
    nan_sampled = has_nan_code and bool(scnt[nan_code] > 0)
    if has_nan_code and bool(in_fold[nan_code]):
        assert nan_mode != "Forbidden", f"feature {f}: NaN with nan_mode=Forbidden (CatBoost errors)"
        assert nan_sampled, f"feature {f}: NaN rows but none in the border sample (CatBoost errors)"
    fin_cnt = scnt[: u.size - 1] if has_nan_code else scnt[: u.size]
    idx = np.flatnonzero(fin_cnt > 0)
    vals = u[idx]
    budget = border_count - 1 if nan_sampled else border_count
    if vals.size - 1 <= budget:
        b = np.unique(((vals[:-1] + vals[1:]) / np.float32(2)).astype(np.float32))
    else:
        b = greedy_logsum(vals, fin_cnt[idx], budget)
    finite = u[: u.size - 1] if has_nan_code else u
    lut = np.searchsorted(b, finite, side="left").astype(np.int64)
    if nan_sampled and nan_mode == "Min":
        b = np.concatenate([np.array([F32_LOWEST], dtype=np.float32), b])
        lut = np.concatenate([lut + 1, [0]])
    elif nan_sampled and nan_mode == "Max":
        b = np.concatenate([b, np.array([F32_MAX], dtype=np.float32)])
        lut = np.concatenate([lut, [b.size]])
    elif has_nan_code:
        lut = np.concatenate([lut, [0]])
    return b, lut


def fold_quantization(scnt_kf: np.ndarray, uniq: list, in_fold_kf: np.ndarray, border_count: int,
                      nan_mode: str, wide: dict | None = None, cats: dict | None = None):
    F, C = scnt_kf.shape
    lut = np.zeros((F, C), dtype=np.uint8)
    nb = np.ones(F, dtype=np.int32)
    pack = np.full(F, -1, dtype=np.int32)
    borders = []
    for f in range(F):
        if cats and f in cats:
            lf, b = cats[f]
            lut[f, : lf.size] = lf.astype(np.uint8)
            nb[f] = b.size
            borders.append(b)
            continue
        if wide and f in wide:
            b, lf = wide[f]
        else:
            b, lf = feature_quantization(scnt_kf[f], uniq[f], in_fold_kf[f], border_count, nan_mode, f)
        lut[f, : lf.size] = lf.astype(np.uint8)
        nb[f] = b.size + 1
        borders.append(b)
    return lut, nb, pack, borders


def column_groups(nb: np.ndarray, bundles: list, n_float: int | None = None) -> np.ndarray:
    nb32 = np.ascontiguousarray(nb, dtype=np.int32)
    bfeat = np.array([f for parts in bundles for f in parts] or [0], dtype=np.int32)
    bstart = np.array([0] + list(np.cumsum([len(parts) for parts in bundles])), dtype=np.int32)
    pack = np.zeros(nb32.size, dtype=np.int32)
    rc = _lib().shira_cat_column_groups(nb32.ctypes.data, nb32.size, -1 if n_float is None else n_float,
                                        bfeat.ctypes.data, bstart.ctypes.data, len(bundles), pack.ctypes.data)
    assert rc == 0, f"shira_cat_column_groups returned {rc}"
    return pack


def code_counts(codes: np.ndarray, C: int, lo: np.ndarray, hi: np.ndarray, seed: int, threads: int):
    n, F = codes.shape
    out = np.zeros((lo.size, F, C), dtype=np.int64)
    smp = np.zeros((lo.size, F, C), dtype=np.int64)
    rc = _lib().shira_cat_code_counts(codes.ctypes.data, n, F, C, lo.ctypes.data, hi.ctypes.data, lo.size,
                                      BORDER_SAMPLE_ROWS, seed & (2**64 - 1), threads, out.ctypes.data,
                                      smp.ctypes.data)
    assert rc == 0, f"shira_cat_code_counts returned {rc}"
    return out, smp


BUNDLE_CODE = 1 << 20
ONE_HOT_CODE = -2
EFB_MAX_BUCKETS = 1 << 10
EFB_MAX_CANDIDATES = 100


class _EfbState:

    def __init__(self, m: int, nb: np.ndarray, cnt: list, mask_of):
        self.m, self.nb, self.cnt, self.mask_of = m, nb, cnt, mask_of
        self.parts, self.used, self.nd, self.objs, self.f2b = [], [], [], [], {}

    def candidate(self, b: int, f: int) -> bool:
        if self.used[b] + int(self.nb[f]) - 1 >= EFB_MAX_BUCKETS:
            return False
        return self.nd[b] + self.cnt[f] <= self.m

    def try_add(self, b: int, f: int) -> bool:
        if np.any(self.objs[b] & self.mask_of(f)):
            return False
        self.add(b, f)
        return True

    def add(self, b: int, f: int) -> None:
        self.parts[b].append(f)
        self.used[b] += int(self.nb[f]) - 1
        self.nd[b] += self.cnt[f]
        self.objs[b] = self.objs[b] | self.mask_of(f)
        self.f2b[f] = b

    def new_bundle(self, f: int) -> None:
        self.parts.append([])
        self.used.append(0)
        self.nd.append(0)
        self.objs.append(np.zeros(self.m, dtype=bool))
        self.add(len(self.parts) - 1, f)


def _efb_place(st: _EfbState, f: int, F: int) -> None:
    checked = set()
    if f > 0 and (f - 1) in st.f2b:
        b = st.f2b[f - 1]
        if st.candidate(b, f) and st.try_add(b, f):
            return
        checked.add(b)
    if f + 1 < F and (f + 1) in st.f2b:
        b = st.f2b[f + 1]
        if b in checked:
            return
        if st.candidate(b, f) and st.try_add(b, f):
            return
        checked.add(b)
    todo = [b for b in range(len(st.parts)) if b not in checked and st.candidate(b, f)]
    assert len(todo) <= EFB_MAX_CANDIDATES - len(checked), "more than 100 candidate bundles (rng subset not reproduced)"
    for b in todo:
        if st.try_add(b, f):
            return
    st.new_bundle(f)


def _efb_impl(order: list, m: int, nb: np.ndarray, cnt: list, mask_of, F: int) -> list:
    st = _EfbState(m, nb, cnt, mask_of)
    for f in order:
        if int(nb[f]) - 1 >= EFB_MAX_BUCKETS:
            continue
        _efb_place(st, f, F)
    small, keep, small_bins = [], [], 0
    for i, parts in enumerate(st.parts):
        binary_only = st.used[i] == len(parts)
        if binary_only and 2 * len(parts) < 8:
            small_bins += len(parts)
            small.append(i)
        elif len(parts) > 1:
            keep.append(i)
    if -(-small_bins // 8) >= 2 * len(small):
        keep += [i for i in small if len(st.parts[i]) > 1]
    return [st.parts[i] for i in sorted(keep)]


def exclusive_bundles(codes_fold: np.ndarray, counts_kf: np.ndarray, lut: np.ndarray, nb: np.ndarray) -> list:
    m, F = codes_fold.shape
    avail = [f for f in range(F) if nb[f] > 1]
    cnt = [int(counts_kf[f][lut[f] != 0].sum()) if nb[f] > 1 else 0 for f in range(F)]
    cache = {}

    def mask_of(f: int) -> np.ndarray:
        if f not in cache:
            cache[f] = lut[f][codes_fold[:, f]] != 0
        return cache[f]

    by_cnt = sorted(avail, key=lambda f: -cnt[f])
    r0 = _efb_impl(avail, m, nb, cnt, mask_of, F)
    r1 = _efb_impl(by_cnt, m, nb, cnt, mask_of, F)
    red = [sum(len(b) for b in r) - len(r) for r in (r0, r1)]
    return r1 if red[1] > red[0] else r0


LABEL_ERRORS = {-10: "labels must be finite", -11: "Logloss labels must be 0/1",
                -12: "CrossEntropy labels must be in [0, 1]", -13: "Poisson / Tweedie labels must be >= 0",
                -14: "multiclass labels must be 0..K-1", -15: "multiclass labels must use every class 0..K-1",
                -16: "every fold must contain every class"}


def check_labels(y32: np.ndarray, loss: int, xent: bool, lo: np.ndarray | None = None,
                 hi: np.ndarray | None = None) -> int:

    K = 0 if lo is None else lo.size
    nc = np.zeros(1, dtype=np.int32)
    rc = _lib().shira_cat_check_labels(y32.ctypes.data, y32.size, loss, int(xent), _ptr(lo), _ptr(hi), K,
                                       nc.ctypes.data)
    assert rc not in LABEL_ERRORS, LABEL_ERRORS.get(rc)
    assert rc == 0, f"shira_cat_check_labels returned {rc}"
    return int(nc[0])


def labels(y: np.ndarray, loss: int, xent: bool = False) -> np.ndarray:
    y32 = np.ascontiguousarray(y, dtype=np.float32)
    check_labels(y32, loss, xent)
    return y32


def sample_weights(weight, n: int) -> np.ndarray | None:
    if weight is None:
        return None
    w32 = np.ascontiguousarray(weight, dtype=np.float32)
    assert w32.shape == (n,), "weight must have one value per row"
    assert np.isfinite(w32).all() and (w32 > 0).all(), "weights must be finite and > 0"
    return w32


def quantize_folds(codes, uniq, lo, hi, cfg, threads, xf):
    n, F = codes.shape
    nf = xf.shape[1]
    from concurrent.futures import ThreadPoolExecutor

    wide_f = [f for f in range(nf) if uniq[f] is None]
    xf = np.ascontiguousarray(xf, dtype=np.float32) if wide_f else xf
    with ThreadPoolExecutor(max_workers=max(1, min(threads, len(wide_f) or 1))) as ex:
        wide_r = list(ex.map(lambda f: wide_quantization(xf, f, lo, hi, cfg, codes), wide_f))
    wide = dict(zip(wide_f, wide_r))
    C = max([u.size for u in uniq if u is not None] + [w[1] for w in wide.values()])
    counts, scounts = code_counts(codes, C, lo, hi, cfg["seed"], min(threads, lo.size))

    def one_fold(k: int):
        cats = {f: cat_bins(codes[lo[k]:hi[k], f], counts[k][f], uniq[f].hashes) for f in range(nf, F)}
        ctrs = ctr_features(cats, cfg)
        lut, nb, _, bd = fold_quantization(scounts[k], uniq, counts[k] > 0, cfg["border_count"],
                                           cfg["nan_mode"], {f: w[0][k] for f, w in wide.items()}, cats)
        bl = exclusive_bundles(codes[lo[k]:hi[k]], counts[k], lut, nb)
        assert all(f < nf for b in bl for f in b), "exclusive bundle with a categorical feature: not supported yet"
        return lut, nb, ctr_groups(column_groups(nb, bl, nf if F > nf else None), ctrs), bd, bl

    with ThreadPoolExecutor(max_workers=max(1, min(threads, lo.size))) as ex:
        per = list(ex.map(one_fold, range(lo.size)))
    luts = [q[0] for q in per]
    nbs = [q[1] for q in per]
    packs = [q[2] for q in per]
    borders = [q[3] for q in per]
    bundles = [q[4] for q in per]
    return (np.ascontiguousarray(np.stack(luts)), np.ascontiguousarray(np.stack(nbs)),
            np.ascontiguousarray(np.stack(packs)), borders, C, bundles)


def train(x: np.ndarray, y: np.ndarray, folds: list, params: dict, threads: int = 6,
          trees_out: bool = True, weight: np.ndarray | None = None, cat: np.ndarray | None = None) -> list[FoldModel]:
    cfg = check_params(params)
    assert 1 <= threads <= MAX_THREADS
    codes, uniq = value_codes(x, threads)
    if cat is not None:
        ordered_ok = cfg["ordered"] and cfg["loss"] in (0, 1) and not cfg["xent"]
        assert cfg["plain_basic"] or (cfg["ctr_ok"] and cfg["mcc"] is not None) or ordered_ok, (
            "categorical features: Plain boosting with RMSE or Logloss, CTR losses of O11B-A, or Ordered (CTR)")
        ccodes, cinfo = cat_codes(cat)
        assert ccodes.shape[0] == codes.shape[0], "cat must have one row per row of x"
        codes = np.ascontiguousarray(np.hstack([codes, ccodes]))
        uniq = uniq + cinfo
        cfg["shuffle"] = True
    n, F = codes.shape
    y32 = labels(y, cfg["loss"], cfg["xent"])
    w32 = sample_weights(weight, n)
    assert y32.shape == (n,)
    K, R, D = len(folds), cfg["iterations"], cfg["depth"]
    lo = np.array([a for a, _ in folds], dtype=np.int64)
    hi = np.array([b for _, b in folds], dtype=np.int64)
    assert K >= 1 and (lo >= 0).all() and (hi <= n).all() and (lo < hi).all()
    nc = check_labels(y32, cfg["loss"], cfg["xent"], lo, hi)
    cfg["nclass"] = nc
    lut, nbkt, pack, borders, C, bundles = quantize_folds(codes, uniq, lo, hi, cfg, threads,
                                                          np.asarray(x, dtype=np.float32))
    rec_d = np.array([cfg["l2"], cfg["lr"]], dtype=np.float64)
    rec_i = np.array([D, cfg["loss"]], dtype=np.int64)
    ctr_check(cfg, pack, w32)
    ctr_ordered_check(cfg, pack, nbkt, np.asarray(x).shape[1])
    opt_i, opt_d = option_vectors(cfg)
    out_pred = np.zeros(int((hi - lo).sum()) * nc, dtype=np.float64)
    o_split = np.zeros((K, R, D, 2), dtype=np.int32) if trees_out else None
    o_leaf = np.zeros((K, R, nc << D), dtype=np.float64) if trees_out else None
    o_score = np.zeros((K, R, D), dtype=np.float64) if trees_out else None
    o_depth = np.zeros((K, R), dtype=np.int32)
    rc = _lib().shira_train_cat_ex(codes.ctypes.data, n, F, C, y32.ctypes.data, _ptr(w32), lo.ctypes.data,
                                   hi.ctypes.data, K, lut.ctypes.data, nbkt.ctypes.data, pack.ctypes.data,
                                   rec_d.ctypes.data, rec_i.ctypes.data, _ptr(opt_i), _ptr(opt_d), R,
                                   threads,
                                   out_pred.ctypes.data, _ptr(o_split), _ptr(o_leaf), _ptr(o_score),
                                   o_depth.ctypes.data)
    assert rc == 0, f"shira_train_cat_ex returned {rc} (-2 bad weights, -3 histogram memory over the cap)"
    res = _unpack(out_pred, lo, hi, borders, o_split, o_leaf, o_score, o_depth, nc, nbkt, np.asarray(x).shape[1])
    for fm, bp in zip(res, bundles):
        fm.bundles = bp
    ctr_splits(res, pack, F, np.asarray(x).shape[1], nc)
    return res


def _split_value(borders_k: list, nb_k, f: int, s: int, n_float: int) -> float:
    if f >= len(borders_k):
        return float(np.float32(np.float32(s + 1) - np.float32(1e-6)))
    if f < n_float:
        return float(borders_k[f][s])
    return float(borders_k[f][1 if int(nb_k[f]) == 2 else s])


def _unpack(out_pred, lo, hi, borders, o_split, o_leaf, o_score, o_depth, nc=1, nbkt=None,
            n_float: int = 1 << 30) -> list[FoldModel]:
    res, off = [], 0
    for k in range(lo.size):
        m = int(hi[k] - lo[k])
        p = out_pred[off * nc:(off + m) * nc].copy()
        fm = FoldModel(pred=p if nc == 1 else p.reshape(m, nc), borders=borders[k])
        off += m
        if o_split is not None:
            L = o_leaf.shape[2] // nc
            for r in range(o_split.shape[1]):
                d = int(o_depth[k, r])
                fm.splits.append([(int(f), _split_value(borders[k], None if nbkt is None else nbkt[k], int(f),
                                                         int(s), n_float)) for f, s in o_split[k, r, :d]])
                lv = o_leaf[k, r].reshape(nc, L)[:, : 1 << d]
                fm.leaves.append(lv[0].copy() if nc == 1 else lv.T.ravel().copy())
                fm.scores.append(o_score[k, r, :d].copy())
        res.append(fm)
    return res


CTR_CODE = -3
CTR_TYPES = (("Borders", 0.0), ("Borders", 0.5), ("Borders", 1.0), ("Counter", 0.0))
CTR_PRIORS = (0.0, 0.5, 1.0)


def ctr_features(cats: dict, cfg: dict) -> set:
    ctrs = {f for f, (_, hb) in cats.items() if hb.size > cfg["ohm"]}
    for f in ctrs:
        lf, _ = cats[f]
        cats[f] = (np.zeros(lf.size, dtype=np.int64), np.zeros(1, dtype=np.int32))
    if ctrs:
        cfg["ctr"] = 1
    return ctrs


def ctr_groups(pack: np.ndarray, ctrs: set) -> np.ndarray:
    for f in ctrs:
        pack[f] = CTR_CODE
    return pack


def ctr_check(cfg: dict, pack: np.ndarray, w32) -> None:
    if not (pack == CTR_CODE).any():
        assert not cfg["ctr_ok"] or not cfg["shuffle"], "categorical features without CTRs: RMSE or Logloss only"
        return
    assert cfg["mcc"] is not None and 1 <= int(cfg["mcc"]) <= 4, "CTR features: set max_ctr_complexity in 1..4"
    if int(cfg["mcc"]) > 1:
        assert cfg["plain_basic"] and cfg["boot"] == 0, "CTR combinations: Plain boosting, bootstrap No, RMSE / Logloss only"
        assert cfg["rs"] == 0.0, "CTR combinations: random_strength 0 only"
        assert w32 is None, "CTR combinations: sample weights are not supported"
    ordered_ok = cfg["ordered"] and cfg["loss"] in (0, 1) and not cfg["xent"]
    assert cfg["plain_basic"] or cfg["ctr_ok"] or ordered_ok, (
        "CTR features: Plain (RMSE / Logloss / CrossEntropy / Poisson / Tweedie / MAE / Quantile / MultiClass / "
        "MultiClassOneVsAll) or Ordered (RMSE / Logloss) only")
    assert cfg.get("nclass", 1) == 1 or w32 is None, "CTR features with MultiClass: sample weights are not supported"
    if cfg["ordered"]:
        assert cfg["boot"] == 0, "CTR features with Ordered boosting: bootstrap No only"
        assert cfg["rs"] == 0.0, "CTR features with Ordered boosting: random_strength 0 only"
        assert w32 is None, "CTR features with Ordered boosting: sample weights are not supported"


def ctr_ordered_check(cfg: dict, pack: np.ndarray, nbkt: np.ndarray, n_float: int) -> None:
    if cfg["ordered"] and cfg["shuffle"]:
        ok = (pack[:, n_float:] == CTR_CODE) | ((pack[:, n_float:] == -1) & (nbkt[:, n_float:] <= 1))
        assert ok.all(), "Ordered boosting: every categorical feature must have CTR features (one-hot ones are not supported)"


def ctr_registry(k: int) -> list:
    lib = _lib()
    if not hasattr(lib, "_ctr_reg_set"):
        lib.shira_cat_ctr_registry.restype = ctypes.c_int
        lib.shira_cat_ctr_registry.argtypes = [_I32, _P, _I64, _P]
        lib._ctr_reg_set = True
    buf = np.zeros(1 << 20, dtype=np.int32)
    n = np.zeros(1, dtype=np.int64)
    rc = lib.shira_cat_ctr_registry(k, buf.ctypes.data, buf.size, n.ctypes.data)
    assert rc == 0, f"shira_cat_ctr_registry returned {rc}"
    e, out, i = buf[: int(n[0])].tolist(), [], 0
    while i < len(e):
        nc = e[i]; cats = e[i + 1:i + 1 + nc]; i += 1 + nc
        nb = e[i]; bins = [(e[i + 1 + 2 * q], e[i + 2 + 2 * q]) for q in range(nb)]; i += 1 + 2 * nb
        no = e[i]; ohs = [(e[i + 1 + 2 * q], e[i + 2 + 2 * q]) for q in range(no)]; i += 1 + 2 * no
        out.append((cats, bins, ohs))
    return out


def ctr_splits(res: list, pack: np.ndarray, F: int, n_float: int, nclass: int = 1) -> None:
    T = 3 * (nclass - 1) + 1 if nclass > 1 else 4
    for k, fm in enumerate(res):
        if not any(f >= F for tree in fm.splits for f, _ in tree):
            continue
        reg = ctr_registry(k)
        for tree in fm.splits:
            for q, (f, v) in enumerate(tree):
                if f >= F:
                    t = (f - F) % T
                    cats, bins, ohs = reg[(f - F) // T]
                    el = [("cat", c - n_float) for c in cats]
                    el += [("float", b, float(fm.borders[b][s])) for b, s in bins]
                    el += [("onehot", c - n_float, int(fm.borders[c][s])) for c, s in ohs]
                    typ, prior = ("Counter", 0.0) if t == T - 1 else ("Borders", CTR_PRIORS[t % 3])
                    tbd = (t // 3,) if t < T - 1 and t // 3 else ()
                    tree[q] = (("ctr", tuple(sorted(el)), typ, prior) + tbd, v)


def catboost_trees(model, n_float: int = 0) -> tuple[list, list]:
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        model.save_model(str(path), format="json")
        js = json.loads(path.read_text())
    splits, leaves = [], []
    ctr_of = _catboost_ctr_index(js)
    for t in js["oblivious_trees"]:
        splits.append([(int(s["float_feature_index"]), float(np.float32(s["border"])))
                       if s["split_type"] == "FloatFeature" else
                       ((ctr_of[int(s["split_index"])], float(np.float32(s["border"])))
                        if s["split_type"] == "OnlineCtr" else
                        (n_float + int(s["cat_feature_index"]), float(s["value"])))
                       for s in t["splits"]])
        leaves.append(np.array(t["leaf_values"], dtype=np.float64))
    return splits, leaves


def _catboost_ctr_index(js: dict) -> dict:
    fi = js["features_info"]
    idx = sum(len(f.get("borders", []) or []) for f in fi.get("float_features", []))
    idx += sum(len(c.get("values", []) or []) for c in fi.get("categorical_features", []))
    out = {}
    for c in fi.get("ctrs", []):
        el = []
        for e in c["elements"]:
            kind = e["combination_element"]
            if kind == "cat_feature_value":
                el.append(("cat", int(e["cat_feature_index"])))
            elif kind == "float_feature":
                el.append(("float", int(e["float_feature_index"]), float(np.float32(e["border"]))))
            else:
                el.append(("onehot", int(e["cat_feature_index"]), int(e["value"])))
        key = ("ctr", tuple(sorted(el)), c["ctr_type"], float(c["prior_numerator"]))
        tbi = int(c.get("target_border_idx", 0) or 0)
        key = key + ((tbi,) if tbi else ())
        for _ in c["borders"]:
            out[idx] = key
            idx += 1
    return out


def catboost_borders(model) -> list:
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        model.save_model(str(path), format="json")
        js = json.loads(path.read_text())
    return [np.array(f.get("borders", []), dtype=np.float32) for f in js["features_info"]["float_features"]]
