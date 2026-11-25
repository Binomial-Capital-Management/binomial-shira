from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
_LIB_NAME = "libshira.dylib" if sys.platform == "darwin" else "libshira.so"
_EDGE_DOUBLE = sys.platform.startswith("linux")


def _edge(v, sign: int) -> np.float32:
    assert sign in (1, -1)
    if _EDGE_DOUBLE:
        d, e = float(np.float32(v)), float(np.float32(1e-5))
        return np.float32(d + (abs(d) + e)) if sign == 1 else np.float32(d - abs(d) - e)
    f = np.float32(v)
    return np.float32(f + (abs(f) + np.float32(1e-5))) if sign == 1 else np.float32(f - abs(f) - np.float32(1e-5))


LIB_PATH = Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB") else SHIRA / "lib" / _LIB_NAME
_P = ctypes.c_void_p
_I32, _I64, _U32 = ctypes.c_int32, ctypes.c_int64, ctypes.c_uint32
_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        assert LIB_PATH.exists(), f"build first: {SHIRA / 'build.sh'}"
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.shira_xgb_cuts.argtypes = [_P, _I64, _I32, _I32, _I32, _P, _P, _P, _P]
        lib.shira_xgb_bin_rows.argtypes = [_P, _I64, _I32, _P, _P, _I32, _P]
        lib.shira_xgb_sampled_cuts.argtypes = [_P, _I64, _I32, _I32, _I64, _I32, _P, _P]
        lib.shira_xgb_bin_rows_f64.argtypes = [_P, _I64, _I32, _P, _P, _I32, _P, _P, _P]
        lib.shira_xgb_bin_rows_f64_u8.argtypes = [_P, _I64, _I32, _P, _P, _I32, _P, _P, _P, _P]
        lib.shira_xgb_sampled_cuts_s.argtypes = [_P, _I32, _I64, _I32, _I64, _I64, _I32, _I64, _I32, _P, _P]
        lib.shira_xgb_bin_rows_s.argtypes = [_P, _I32, _I64, _I32, _I64, _I64, _P, _P, _I32, _I32, _P, _P, _P, _P]
        lib.shira_hgb_bin_thresholds.argtypes = [_P, _I64, _I32, _I32, _I64, _U32, _P, _I64, _I32, _P, _P, _P]
        lib.shira_hgb_bin_rows.argtypes = [_P, _I64, _I32, _P, _P, _I32, _I32, _P]
        lib.shira_hgb_bin_thresholds_s.argtypes = [_P, _I64, _I32, _I64, _I64, _I32, _I64, _U32, _P, _I64, _I32, _P,
                                                   _P, _P]
        lib.shira_hgb_bin_rows_s.argtypes = [_P, _I64, _I32, _I64, _I64, _P, _P, _I32, _I32, _P]
        lib.shira_hgb_subsample_rows.argtypes = [_I64, _I64, _U32, _P]
        lib.shira_lgb_sample_indices.argtypes = [_I32, _I32, _I32, _P, _P]
        lib.shira_lgb_bin_mappers.argtypes = [_P, _I64, _I32, _I32, _I32, _I32, _I32, _I32, _I32, _I32, _P, _P, _P]
        lib.shira_lgb_bin_rows.argtypes = [_P, _I64, _I32, _P, _P, _P, _I32, _P]
        lib.shira_lgb_bin_rows2.argtypes = [_P, _I64, _I32, _P, _P, _P, _I32, _I32, _P, _P]
        lib.shira_lgb_sample_stats.argtypes = [_P, _I64, _I32, _P, _I64, _P, _P, _P, _I32, _P, _P, _P, _P]
        lib.shira_lgb_sample_stats_w.argtypes = [_P, _I64, _I32, _P, _I64, _P, _P, _P, _I32, _I32, _P, _P, _P, _P]
        lib.shira_lgb_cat_mapper.argtypes = [_P, _I64, _I64, _I32, _I32, _I32, _I32, _P, _P]
        lib.shira_lgb_cat_bin.argtypes = [_P, _I64, _P, _I32, _P]
        for f in (lib.shira_xgb_cuts, lib.shira_xgb_bin_rows, lib.shira_xgb_sampled_cuts, lib.shira_xgb_bin_rows_f64,
                  lib.shira_xgb_bin_rows_f64_u8, lib.shira_xgb_sampled_cuts_s, lib.shira_xgb_bin_rows_s,
                  lib.shira_hgb_bin_thresholds,
                  lib.shira_hgb_bin_rows, lib.shira_hgb_subsample_rows, lib.shira_lgb_sample_indices,
                  lib.shira_lgb_bin_mappers, lib.shira_lgb_bin_rows, lib.shira_lgb_cat_mapper, lib.shira_lgb_bin_rows2,
                  lib.shira_lgb_sample_stats, lib.shira_lgb_sample_stats_w,
                  lib.shira_lgb_cat_bin, lib.shira_hgb_bin_thresholds_s, lib.shira_hgb_bin_rows_s):
            f.restype = ctypes.c_int
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray | None) -> ctypes.c_void_p | None:
    return None if a is None else a.ctypes.data_as(_P)


def _strided(x: np.ndarray) -> tuple[np.ndarray, int, int]:
    x = np.asarray(x)
    if x.dtype == np.float64 and x.ndim == 2 and x.size and all(s > 0 and s % 8 == 0 for s in x.strides):
        return x, x.strides[0] // 8, x.strides[1] // 8
    x = np.ascontiguousarray(x, dtype=np.float64)
    return x, x.shape[1], 1


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise ValueError(f"{what} failed (code {rc}): bad arguments or infinite values")


def xgb_cuts(x: np.ndarray, max_bin: int = 256, nthread: int = 1,
             weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=np.float32)
    assert x.ndim == 2 and x.shape[0] >= 1 and x.shape[1] >= 1
    n, p = x.shape
    w = None if weights is None else np.ascontiguousarray(weights, dtype=np.float32)
    assert w is None or w.shape == (n,)
    ptrs = np.zeros(p + 1, dtype=np.uint32)
    vals = np.zeros(p * int(max_bin), dtype=np.float32)
    mins = np.zeros(p, dtype=np.float32)
    rc = _lib().shira_xgb_cuts(_ptr(x), n, p, int(max_bin), int(nthread), _ptr(w), _ptr(ptrs), _ptr(vals), _ptr(mins))
    _check(rc, "shira_xgb_cuts")
    return ptrs, vals[: int(ptrs[-1])].copy(), mins


def quantile_cut_layout(ptrs: np.ndarray, values: np.ndarray, mins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = mins.size
    assert ptrs.size == p + 1 and values.size == int(ptrs[-1])
    indptr = (ptrs.astype(np.uint64) + np.arange(p + 1, dtype=np.uint64))
    data = np.empty(values.size + p, dtype=np.float32)
    for j in range(p):
        lo = int(indptr[j])
        data[lo] = mins[j]
        data[lo + 1: int(indptr[j + 1])] = values[int(ptrs[j]): int(ptrs[j + 1])]
    return indptr, data


def xgb_codes(x: np.ndarray, ptrs: np.ndarray, values: np.ndarray, nthread: int = 1) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    ptrs = np.ascontiguousarray(ptrs, dtype=np.uint32)
    values = np.ascontiguousarray(values, dtype=np.float32)
    assert x.ndim == 2 and ptrs.size == x.shape[1] + 1
    out = np.empty(x.shape, dtype=np.uint16)
    rc = _lib().shira_xgb_bin_rows(_ptr(x), x.shape[0], x.shape[1], _ptr(ptrs), _ptr(values), int(nthread), _ptr(out))
    _check(rc, "shira_xgb_bin_rows")
    return out


def xgb_view(x: Any) -> tuple[int, int, int] | None:
    if not (isinstance(x, np.ndarray) and x.ndim == 2 and x.dtype in (np.float64, np.float32) and x.size > 0):
        return None
    if x.dtype.byteorder not in ("=", "|") or any(s % x.itemsize for s in x.strides):
        return None
    return (0 if x.dtype == np.float64 else 1), x.strides[0] // x.itemsize, x.strides[1] // x.itemsize


def xgb_sampled_cuts_codes(x: np.ndarray, max_bin: int = 256, sample: int = 200_000,
                           nthread: int = 1, u8: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = xgb_view(x)
    if view is None:
        x = np.ascontiguousarray(x, dtype=np.float64)
        view = xgb_view(x)
    assert view is not None and 2 <= int(max_bin) <= 65534
    dt, rs, cs = view
    n, p = x.shape
    ptrs = np.zeros(p + 1, dtype=np.uint32)
    vals = np.zeros(p * int(max_bin), dtype=np.float32)
    _check(_lib().shira_xgb_sampled_cuts_s(_ptr(x), dt, n, p, rs, cs, int(max_bin), int(sample), int(nthread), _ptr(ptrs),
                                           _ptr(vals)), "shira_xgb_sampled_cuts_s")
    vals = vals[: int(ptrs[-1])].copy()
    mn, mx = np.empty(p, dtype=np.float32), np.empty(p, dtype=np.float32)
    codes = None
    if u8 and int(np.diff(ptrs).max()) <= 255:
        c8, miss = np.empty((n, p), dtype=np.uint8), ctypes.c_int32(0)
        _check(_lib().shira_xgb_bin_rows_s(_ptr(x), dt, n, p, rs, cs, _ptr(ptrs), _ptr(vals), int(nthread), 1, _ptr(c8),
                                           _ptr(mn), _ptr(mx), ctypes.byref(miss)), "shira_xgb_bin_rows_s")
        codes = c8 if miss.value == 0 else None
    if codes is None:
        codes = np.empty((n, p), dtype=np.uint16)
        _check(_lib().shira_xgb_bin_rows_s(_ptr(x), dt, n, p, rs, cs, _ptr(ptrs), _ptr(vals), int(nthread), 2,
                                           _ptr(codes), _ptr(mn), _ptr(mx), None), "shira_xgb_bin_rows_s")
    ip, cv = [0], []
    for j in range(p):
        inner = vals[ptrs[j]:ptrs[j + 1]]
        if not np.isfinite(mn[j]):
            first = np.float32(1e-5)
            cv += [first, _edge(first, 1)]
        else:
            first, last = _edge(mn[j], -1), _edge(mx[j], 1)
            cv += [first, *inner.tolist(), last]
        ip.append(len(cv))
    return np.asarray(ip, dtype=np.int64), np.asarray(cv, dtype=np.float32), codes


def hgb_thresholds(x: np.ndarray, n_bins: int = 256, seed: int = 0, subsample: int | None = 200_000,
                   rows: np.ndarray | None = None, nthread: int = 1) -> tuple[list, np.ndarray]:
    x, rs, cs = _strided(x)
    assert x.ndim == 2 and 3 <= n_bins <= 256 and 0 <= int(seed) <= 0xFFFFFFFF
    n, p = x.shape
    r = None if rows is None else np.ascontiguousarray(rows, dtype=np.int64)
    ptrs = np.zeros(p + 1, dtype=np.int32)
    thr = np.zeros(max(1, p * (n_bins - 2)), dtype=np.float64)
    nbm = np.zeros(p, dtype=np.uint32)
    rc = _lib().shira_hgb_bin_thresholds_s(_ptr(x), n, p, rs, cs, int(n_bins),
                                           -1 if subsample is None else int(subsample), int(seed), _ptr(r),
                                           0 if r is None else r.size, int(nthread), _ptr(ptrs), _ptr(thr), _ptr(nbm))
    _check(rc, "shira_hgb_bin_thresholds")
    return [thr[ptrs[j]: ptrs[j + 1]].copy() for j in range(p)], nbm


def hgb_bin(x: np.ndarray, thresholds: list, n_bins: int = 256, nthread: int = 1) -> np.ndarray:
    x, rs, cs = _strided(x)
    assert x.ndim == 2 and len(thresholds) == x.shape[1]
    ptrs = np.concatenate([[0], np.cumsum([t.size for t in thresholds])]).astype(np.int32)
    flat = np.ascontiguousarray(np.concatenate(thresholds + [np.zeros(1)]), dtype=np.float64)
    out = np.empty(x.shape, dtype=np.uint8, order="F")
    rc = _lib().shira_hgb_bin_rows_s(_ptr(x), x.shape[0], x.shape[1], rs, cs, _ptr(ptrs), _ptr(flat),
                                     int(n_bins) - 1, int(nthread), _ptr(out))
    _check(rc, "shira_hgb_bin_rows")
    return out


def hgb_subsample_rows(n: int, k: int, seed: int) -> np.ndarray:
    out = np.empty(k, dtype=np.int64)
    _check(_lib().shira_hgb_subsample_rows(int(n), int(k), int(seed), _ptr(out)), "shira_hgb_subsample_rows")
    return out


def lgb_sample_indices(n: int, sample_cnt: int = 200_000, seed: int = 1) -> np.ndarray:
    out = np.zeros(max(1, min(n, sample_cnt)), dtype=np.int32)
    ln = ctypes.c_int32(0)
    _check(_lib().shira_lgb_sample_indices(int(n), int(sample_cnt), int(seed), _ptr(out), ctypes.byref(ln)),
           "shira_lgb_sample_indices")
    return out[: ln.value].copy()


def lgb_mappers(x: np.ndarray, max_bin: int = 255, min_data_in_bin: int = 3, sample_cnt: int = 200_000,
                seed: int = 1, use_missing: bool = True, zero_as_missing: bool = False,
                nthread: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    assert x.ndim == 2 and 2 <= max_bin <= 65535
    n, p = x.shape
    ptrs = np.zeros(p + 1, dtype=np.int32)
    upper = np.zeros(p * int(max_bin), dtype=np.float64)
    info = np.zeros((p, 4), dtype=np.int32)
    rc = _lib().shira_lgb_bin_mappers(_ptr(x), n, p, int(max_bin), int(min_data_in_bin), int(sample_cnt), int(seed),
                                      int(bool(use_missing)), int(bool(zero_as_missing)), int(nthread),
                                      _ptr(ptrs), _ptr(upper), _ptr(info))
    _check(rc, "shira_lgb_bin_mappers")
    return ptrs, upper[: int(ptrs[-1])].copy(), info


def lgb_bin(x: np.ndarray, ptrs: np.ndarray, upper: np.ndarray, info: np.ndarray, nthread: int = 1) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float64)
    info = np.ascontiguousarray(info, dtype=np.int32)
    assert x.ndim == 2 and info.shape == (x.shape[1], 4)
    out = np.empty(x.shape, dtype=np.uint8)
    rc = _lib().shira_lgb_bin_rows(_ptr(x), x.shape[0], x.shape[1], _ptr(np.ascontiguousarray(ptrs, dtype=np.int32)),
                                   _ptr(np.ascontiguousarray(upper, dtype=np.float64)), _ptr(info), int(nthread), _ptr(out))
    _check(rc, "shira_lgb_bin_rows")
    return out


def lgb_bin2(x: np.ndarray, ptrs: np.ndarray, upper: np.ndarray, info: np.ndarray, nthread: int = 1,
             code_bytes: int = 2) -> tuple[np.ndarray, bool]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    info = np.ascontiguousarray(info, dtype=np.int32)
    assert x.ndim == 2 and info.shape == (x.shape[1], 4) and code_bytes in (1, 2)
    out = np.empty(x.shape, dtype=np.uint16 if code_bytes == 2 else np.uint8)
    inf = ctypes.c_int32(0)
    rc = _lib().shira_lgb_bin_rows2(_ptr(x), x.shape[0], x.shape[1], _ptr(np.ascontiguousarray(ptrs, dtype=np.int32)),
                                    _ptr(np.ascontiguousarray(upper, dtype=np.float64)), _ptr(info), int(nthread),
                                    int(code_bytes), _ptr(out), ctypes.byref(inf))
    _check(rc, "shira_lgb_bin_rows2")
    return out, inf.value != 0


def lgb_sample_stats(x: np.ndarray, rows: np.ndarray, ptrs: np.ndarray, upper: np.ndarray, info: np.ndarray,
                     nthread: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    r = np.ascontiguousarray(rows, dtype=np.int64)
    p = x.shape[1]
    inf = np.ascontiguousarray(info, dtype=np.int32)
    width = max(256, int(inf.reshape(-1, 4)[:, 0].max()))
    nnz, mn, mx = np.zeros(p, dtype=np.int64), np.empty(p, dtype=np.float64), np.empty(p, dtype=np.float64)
    cnt = np.zeros((p, width), dtype=np.int64)
    args = (_ptr(x), x.shape[0], p, _ptr(r), r.size, _ptr(np.ascontiguousarray(ptrs, dtype=np.int32)),
            _ptr(np.ascontiguousarray(upper, dtype=np.float64)), _ptr(inf), int(nthread))
    if width == 256:
        rc = _lib().shira_lgb_sample_stats(*args, _ptr(nnz), _ptr(mn), _ptr(mx), _ptr(cnt))
    else:
        rc = _lib().shira_lgb_sample_stats_w(*args, width, _ptr(nnz), _ptr(mn), _ptr(mx), _ptr(cnt))
    _check(rc, "shira_lgb_sample_stats")
    return nnz, mn, mx, cnt


def lgb_cat_mapper(vals: np.ndarray, total: int, max_bin: int = 255, min_data_in_bin: int = 3,
                   use_missing: bool = True, zero_as_missing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    v = np.ascontiguousarray(vals, dtype=np.float64).reshape(-1)
    b2c = np.zeros(v.size + 2, dtype=np.int64)
    info = np.zeros(4, dtype=np.int32)
    rc = _lib().shira_lgb_cat_mapper(_ptr(v), v.size, int(total), int(max_bin), int(min_data_in_bin),
                                     int(bool(use_missing)), int(bool(zero_as_missing)), _ptr(b2c), _ptr(info))
    _check(rc, "shira_lgb_cat_mapper (no category, or a value outside int64)")
    return b2c[: int(info[0])].copy(), info


def lgb_cat_bin(values: np.ndarray, bin2cat: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    b2c = np.ascontiguousarray(bin2cat, dtype=np.int64)
    out = np.zeros(v.size, dtype=np.int32)
    _check(_lib().shira_lgb_cat_bin(_ptr(v), v.size, _ptr(b2c), b2c.size, _ptr(out)), "shira_lgb_cat_bin")
    return out
