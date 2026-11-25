from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

_P = ctypes.c_void_p
_I32, _I64, _D = ctypes.c_int32, ctypes.c_int64, ctypes.c_double
_LIB: ctypes.CDLL | None = None
_SIGS = {
    "shira_version": ([], _I32),
    "shira_totals_len": ([_I32], _I64),
    "shira_block_totals": ([_P, _P, _P, _I64, _I32, _P, _D, _P], ctypes.c_int),
    "shira_block_totals_mt": ([_P, _P, _P, _I64, _I32, _P, _D, _P, _I32], ctypes.c_int),
    "shira_block_totals_acc": ([_P, _P, _P, _I64, _I32, _P, _D, _P], ctypes.c_int),
    "shira_block_totals_acc_mt": ([_P, _P, _P, _I64, _I32, _P, _D, _P, _I32], ctypes.c_int),
    "shira_diag_totals": ([_P, _P, _I64, _I32, _P, _P, _P, _P], ctypes.c_int),
    "shira_sum_totals": ([_P, _I32, _I32, _P], ctypes.c_int),
    "shira_sum_totals_ptrs": ([_P, _I32, _I32, _P], ctypes.c_int),
    "shira_wtree_build": ([_P, _I32, _I32, _I32, _P], ctypes.c_int),
    "shira_wtree_window": ([_P, _P, _I32, _I32, _I32, _I32, _I32, _P, _P], ctypes.c_int),
    "shira_fit_ridge": ([_P, _I32, _D, _P, _D, _P, _P], ctypes.c_int),
    "shira_fit_enet": ([_P, _I32, _D, _D, _P, _D, _D, _P, _P, _P, _P], ctypes.c_int),
    "shira_covariance": ([_P, _I32, _I32, _P, _P], ctypes.c_int),
    "shira_pca": ([_P, _I32, _I32, _P, _P, _P, _P, _P, _P], ctypes.c_int),
    "shira_class_diag_totals": ([_P, _P, _I64, _I32, _P, _I32, _P, _P, _P, _P], ctypes.c_int),
    "shira_gaussian_nb": ([_P, _P, _P, _I32, _I32, _P, _D, _P, _P, _P, _P], ctypes.c_int),
    "shira_gaussian_nb_np": ([_P, _P, _I64, _I32, _I32, _D, _P, _P, _P, _P, _P], ctypes.c_int),
    "shira_pca_np": ([_P, _I64, _I32, _I32, _P, _P, _P, _P, _P, _P], ctypes.c_int),
    "shira_gram_np": ([_P, _I64, _I32, _P, _P, _P, _P], ctypes.c_int),
    "shira_linreg_tsqr": ([_P, _P, _I64, _I32, _I32, _P, _P, _P, _P, _P], ctypes.c_int),
    "shira_row_hash": ([_P, _P, _I64, _I32, _I32, _P], ctypes.c_int),
    "shira_row_hash_strided": ([_P, _P, _I64, _I32, _I64, _P], ctypes.c_int),
    "shira_acc_segments": ([_P, _P, _I32, _P, _P, _I32, _P, _D, _I32, _P, _P], ctypes.c_int),
    "shira_acc_merge": ([_P, _I32, _I32, _P], ctypes.c_int),
    "shira_qr_segments": ([_P, _P, _I32, _P, _P, _I32, _P, _D, _I32, _P, _P], ctypes.c_int),
    "shira_linreg_from_parts": ([_P, _I32, _I32, _I64, _P, _D, _P, _P, _P, _P, _P], ctypes.c_int),
    "shira_bayes_ridge_tsqr": ([_P, _P, _I64, _I32, _I32, _P, _P, _P, _P, _P, _P, _P], ctypes.c_int),
    "shira_eq_cond": ([_P, _I32, ctypes.c_double, _P], ctypes.c_int),
    "shira_eq_cond_capped": ([_P, _I32, ctypes.c_double, ctypes.c_double, _P], ctypes.c_int),
    "shira_fit_ridge_capped": ([_P, _I32, ctypes.c_double, _P, ctypes.c_double, ctypes.c_double, _P, _P, _P],
                               ctypes.c_int),
    "shira_qda_tsqr": ([_P, _P, _I64, _I32, _I32, _I32, ctypes.c_double, ctypes.c_double, _P, _P, _P, _P, _P],
                       ctypes.c_int),
    "shira_lda_lsqr": ([_P, _I32, _I32, _P, _P, _P, _P, _P, _P], ctypes.c_int),
    "shira_predict_margin": ([_P, _I64, _I32, _I32, _P, _P, _P, ctypes.c_float, _P], ctypes.c_int),
    "shira_xgb_model_codes": ([_P, _I64, _I32, _P, _P, _I32, _I32, _P], ctypes.c_int),
    "shira_xgb_round_margins": ([_P, _I64, _I32, _I32, _I32, _P, _P, _P, ctypes.c_float, _I32, _P], ctypes.c_int),
    "shira_xgb_trees_ubj": ([_P, _P, _P, _I32, _I32, _P, _P, _I32, _P, _I64], _I64),
    "shira_xgb_trees_ubj_stats": ([_P, _P, _P, _I32, _I32, _P, _P, _I32, _P, _P, _P, _P, _I64], _I64),
    "shira_xgb_mode": ([], ctypes.c_int),
    "shira_xgb_mode_for": ([_I64, _I32], ctypes.c_int),
    "shira_xgb_base_score": ([_I32, _P, _P, _I64, _I32, ctypes.c_float, _P], ctypes.c_int),
}


def lib_path() -> Path:
    env = os.environ.get("SHIRA_LIB")
    if env:
        return Path(env)
    name = "libshira.dylib" if sys.platform == "darwin" else "libshira.so"
    return Path(__file__).resolve().parents[2] / "lib" / name


def lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        path = lib_path()
        assert path.exists(), f"build libshira first: {path.parents[1] / 'build.sh'}"
        handle = ctypes.CDLL(str(path))
        for name, (args, res) in _SIGS.items():
            fn = getattr(handle, name)
            fn.argtypes, fn.restype = args, res
        if sys.platform.startswith("linux"):
            _route_blas(handle)
        _LIB = handle
    return _LIB


blas_route: dict = {}
_NP_CANDS = (("scipy_dgemv_64_", "scipy_ddot_64_", 1), ("dgemv_64_", "ddot_64_", 1), ("scipy_dgemv_", "scipy_ddot_", 0))
_SP_CANDS = (("scipy_ddot_", "scipy_dnrm2_", "scipy_dpotrf_", "scipy_dtrtrs_"), ("ddot_", "dnrm2_", "dpotrf_", "dtrtrs_"))


def _bundled(pkg: str, stem: str) -> list[ctypes.CDLL]:
    import glob
    import importlib
    base = Path(importlib.import_module(pkg).__file__).resolve().parent
    paths = sorted(glob.glob(str(base.parent / f"{pkg}.libs" / f"{stem}*.so*")))[:4]
    return [ctypes.CDLL(p) for p in paths]


def _first(libs: list[ctypes.CDLL], cands: tuple) -> tuple | None:
    for lb in libs:
        for names in cands:
            try:
                return tuple(ctypes.cast(getattr(lb, n), ctypes.c_void_p).value for n in names[:4]
                             if isinstance(n, str)), names
            except AttributeError:
                continue
    return None


def _route_blas(handle: ctypes.CDLL) -> None:
    import numpy
    import scipy.linalg
    np_hit = _first(_bundled("numpy", "libscipy_openblas64"), _NP_CANDS)
    sp_hit = _first(_bundled("scipy", "libscipy_openblas"), _SP_CANDS)
    fn = handle.shira_logistic_set_blas
    fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32] + [ctypes.c_void_p] * 4, ctypes.c_int
    npp = np_hit[0] if np_hit else (None, None)
    ilp = int(np_hit[1][2]) if np_hit else 0
    spp = sp_hit[0] if sp_hit else (None,) * 4
    assert fn(npp[0], npp[1], ilp, *spp) == 0
    blas_route.update(numpy=np_hit[1][:2] if np_hit else None, scipy=sp_hit[1] if sp_hit else None)
    _route_gemm(handle)
    if os.environ.get("SHIRA_BLAS_SHIM", "1") != "0":
        _route_shim(handle)
    elif hasattr(handle, "shira_blas_linked_check"):
        handle.shira_blas_linked_check.restype = ctypes.c_int
        if handle.shira_blas_linked_check() != 0:
            raise RuntimeError("SHIRA_BLAS_SHIM=0 needs a system OpenBLAS/LAPACK (libopenblas.so.0, or liblapack.so.3 "
                               "+ libblas.so.3); none found")
        blas_route["shim_to_scipy"] = "off: system BLAS"


_GEMM_CANDS = ("scipy_cblas_dgemm64_", "cblas_dgemm64_", "scipy_cblas_dgemm64")


def _route_gemm(handle: ctypes.CDLL) -> None:
    if not hasattr(handle, "shira_logistic_set_gemm"):
        return
    for lb in _bundled("numpy", "libscipy_openblas64"):
        for name in _GEMM_CANDS:
            sym = getattr(lb, name, None)
            if sym is not None:
                fn = handle.shira_logistic_set_gemm
                fn.argtypes, fn.restype = [ctypes.c_void_p], ctypes.c_int
                assert fn(ctypes.cast(sym, ctypes.c_void_p).value) == 0
                blas_route["numpy_gemm"] = name
                return
    blas_route["numpy_gemm"] = None


_SHIM = ("ddot_", "dgelsd_", "dgemm_", "dgemv_", "dgesdd_", "dnrm2_", "dpotrf_", "dpotrs_", "dsyevd_", "dsyevr_",
         "dsygvd_", "dsyrk_", "dtrtrs_", "dgeqrf_",
         "openblas_get_num_threads", "openblas_set_num_threads")


def _route_shim(handle: ctypes.CDLL) -> None:
    if not hasattr(handle, "shira_blas_route"):
        return
    fn = handle.shira_blas_route
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_void_p], ctypes.c_int
    libs = _bundled("scipy", "libscipy_openblas")
    routed = []
    for name in _SHIM:
        for lb in libs:
            sym = getattr(lb, "scipy_" + name, None)
            if sym is not None:
                assert fn(name.encode(), ctypes.cast(sym, ctypes.c_void_p).value) == 0
                routed.append(name)
                break
    blas_route["shim_to_scipy"] = f"{len(routed)}/{len(_SHIM)}"


def f64(a, shape=None) -> np.ndarray:
    out = np.ascontiguousarray(a, dtype=np.float64)
    assert shape is None or out.shape == shape, f"shape {out.shape} != {shape}"
    return out


def ptr(a: np.ndarray | None):
    return None if a is None else a.ctypes.data


def check(rc: int, what: str) -> None:
    assert rc == 0, f"{what} failed rc={rc} (-1 bad input, -2 numeric, -3 LAPACK)"
