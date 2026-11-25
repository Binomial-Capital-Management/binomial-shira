from __future__ import annotations

import ctypes
import json
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SHIRA = HERE.parents[1]
LIB_PATH = (Path(os.environ["SHIRA_LIB"]) if os.environ.get("SHIRA_LIB")
            else SHIRA / "lib" / ("libshira.dylib" if sys.platform == "darwin" else "libshira.so"))
_P, _I32, _I64 = ctypes.c_void_p, ctypes.c_int32, ctypes.c_int64
_LIB: ctypes.CDLL | None = None
MAX_THREADS = 64
LGB_LINEAR_FUSED = 1
IMPORTANCE = {"weight": 0, "gain": 1, "cover": 2, "total_gain": 3, "total_cover": 4,
              "split": 5, "lgb_gain": 6, "sk_tree": 7, "sk_forest": 8}


class _CForest(ctypes.Structure):
    _fields_ = [("n_trees", _I32), ("n_groups", _I32), ("nf", _I32), ("reserved", _I32),
                ("tree_off", _P), ("nodes", _P), ("value", _P), ("cover", _P), ("gain", _P),
                ("leaf_id", _P), ("group", _P)]


_SIGS = {
    "shira_bin_raw_f32": [_P, _I64, _I32, _P, _P, _I32, _I32, _P],
    "shira_bin_raw_f64": [_P, _I64, _I32, _P, _P, _I32, _I32, _P],
    "shira_forest_predict": [_P, _P, _I64, _I32, _P, _I32, _P],
    "shira_forest_predict_raw_f32": [_P, _P, _P, _I32, _P, _I64, _I32, _P, _I32, _P],
    "shira_forest_predict_raw_f64": [_P, _P, _P, _I32, _P, _I64, _I32, _P, _I32, _P],
    "shira_forest_leaf": [_P, _P, _I64, _I32, _P],
    "shira_forest_importance": [_P, _I32, _P],
    "shira_forest_shap": [_P, _P, _I64, _I32, _P, _I32, _P],
    "shira_forest_shap_interactions": [_P, _P, _I64, _I32, _P, _I32, _P],
    "shira_forest_node_sums": [_P, _P, _I64, _P, _P, _P, _P, _P],
    "shira_xgb_replay_stats": [_P, _P, _I64, _P, ctypes.c_float, _I32, ctypes.c_double, ctypes.c_double,
                               _P, _P, _P],
    "shira_xgb_prune": [_P, _P, ctypes.c_double, ctypes.c_double, _I32, _P, _P, _P],
    "shira_sym_predict_f32": [_P, _I32, _P, _P, _P, _P, _P, _P, _I64, _I32, ctypes.c_double, _I32, _I32, _P],
    "shira_sym_leaf_f32": [_P, _I32, _P, _P, _P, _P, _P, _I64, _I32, _I32, _P],
    "shira_bin_raw_u8_f32": [_P, _I64, _I32, _P, _P, _I32, _I32, _P],
    "shira_bin_raw_u8_f64": [_P, _I64, _I32, _P, _P, _I32, _I32, _P],
    "shira_bin_lut_build": [_P, _P, _I32, _P],
    "shira_bin_lut_u8_f32": [_P, _I64, _I32, _P, _P, _I32, _P, _I32, _P],
    "shira_sym_predict_lut_f32": [_P, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _I32, _P, _I64, _I32,
                                  ctypes.c_double, _I32, _I32, _P],
    "shira_forest_predict_u8": [_P, _P, _I64, _I32, _P, _I32, _P],
    "shira_forest_leaf_u8": [_P, _P, _I64, _I32, _P],
    "shira_forest_predict_delta": [_P, _P, _P, _I64, _I32, _P, _I32, _P, _P, _P],
    "shira_sym_predict_u8": [_P, _I32, _P, _P, _P, _P, _P, _P, _I64, _I32, ctypes.c_double, _I32, _I32, _P],
    "shira_forest_leaf_raw_f32": [_P, _P, _P, _I32, _P, _I64, _I32, _P],
    "shira_forest_leaf_raw_f64": [_P, _P, _P, _I32, _P, _I64, _I32, _P],
    "shira_forest_predict_rawx_f32": [_P, _P, _P, _I32, _P, _P, _I64, _I32, _P, _I32, _I32, _P],
    "shira_forest_predict_rawx_f64": [_P, _P, _P, _I32, _P, _P, _I64, _I32, _P, _I32, _I32, _P],
    "shira_forest_leaf_rawx_f32": [_P, _P, _P, _I32, _P, _P, _I64, _I32, _I32, _P],
    "shira_forest_leaf_rawx_f64": [_P, _P, _P, _I32, _P, _P, _I64, _I32, _I32, _P],
    "shira_out_build_forest": [_I32, _P, _I32, _P, _P, _P, _P, _P, _P, _P, _P, _P],
    "shira_forest_prepare": [_P, _P, _P, _I32, _P, _I32, _I32, _P, _I32, _P],
    "shira_forest_prepared_free": [_P],
    "shira_prepared_predict_f32": [_P, _P, _I64, _I32, _P],
    "shira_prepared_predict_f64": [_P, _P, _I64, _I32, _P],
    "shira_prepared_leaf_f32": [_P, _P, _I64, _I32, _P],
    "shira_prepared_leaf_f64": [_P, _P, _I64, _I32, _P],
    "shira_out_from_lgb": [_I32] + [_P] * 15 + [_I32, _I32, _I32, _I64] + [_P] * 13,
    "shira_forest_prepare_cat": [_P, _P, _P, _I32, _I32, _I32, _P, _P, _P, _I32, _P],
    "shira_lgb_linear_predict": [_P, _I64, _I32, _P, _I32, _P, _P, _P, _P, _P, _P, _I32, _I32, _P],
    "shira_out_from_sym": [_I32, _P, _P, _P, _P, _P, _I32, _P, _I32] + [_P] * 8,
}


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        assert LIB_PATH.exists(), f"build first: {SHIRA / 'build.sh'}"
        lib = ctypes.CDLL(str(LIB_PATH))
        for name, args in _SIGS.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray | None):
    return None if a is None else a.ctypes.data


def _check(rc: int, what: str) -> None:
    assert rc == 0, f"{what} failed rc={rc}"


@dataclass
class RawTree:
    feat: np.ndarray
    thr: np.ndarray
    dleft: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    cover: np.ndarray | None = None
    gain: np.ndarray | None = None
    leaf_id: np.ndarray | None = None


@dataclass
class Forest:
    nf: int
    op_le: int
    pred_mode: int
    shap_mode: int
    base: np.ndarray
    tree_off: np.ndarray
    nodes: np.ndarray
    value: np.ndarray
    thr_off: np.ndarray
    thr: np.ndarray
    cover: np.ndarray | None = None
    gain: np.ndarray | None = None
    leaf_id: np.ndarray | None = None
    group: np.ndarray | None = None
    n_groups: int = 1
    base_weight: np.ndarray | None = None
    in_f32: bool = False
    sym: dict | None = None
    _keep: list = field(default_factory=list)
    _lut_cells: object = None
    _handles: dict = field(default_factory=dict)
    lgb_snap: bool = False
    lgb_linear: dict | None = None
    lgb_zero_nan: np.ndarray | None = None
    lgb_cat: dict | None = None
    no_stats: str | None = None

    def __del__(self) -> None:
        self._prep_drop()

    def _prep_drop(self) -> None:
        hs = getattr(self, "_handles", None)
        if not hs or _LIB is None:
            return
        for h, _ in hs.values():
            _LIB.shira_forest_prepared_free(h)
        hs.clear()

    def _handle(self, f32: bool, path: int) -> ctypes.c_void_p:
        key = (f32, path, self.pred_mode)
        refs = (self.tree_off, self.nodes, self.value, self.thr_off, self.thr, self.base, self.leaf_id, self.group)
        e = self._handles.get(key)
        if e is not None and all(a is b for a, b in zip(e[1], refs)):
            return e[0]
        if e is not None:
            _lib().shira_forest_prepared_free(e[0])
            del self._handles[key]
        h = ctypes.c_void_p()
        st = self.struct()
        if self.lgb_cat is not None:
            _check(_lib().shira_forest_prepare_cat(ctypes.byref(st), _ptr(self.thr_off), _ptr(self.thr), self.op_le,
                                                   int(f32), self.pred_mode, _ptr(self.base), _ptr(self.lgb_cat["feat"]),
                                                   _ptr(self.lgb_cat["bits"]), int(self.lgb_cat["bits"].shape[1]),
                                                   ctypes.byref(h)),
                   "shira_forest_prepare_cat")
            self._handles[key] = (h, refs)
            return h
        lut = self.lut() if f32 else None
        _check(_lib().shira_forest_prepare(ctypes.byref(st), _ptr(self.thr_off), _ptr(self.thr), self.op_le, _ptr(lut),
                                           int(f32), self.pred_mode, _ptr(self.base), path, ctypes.byref(h)),
               "shira_forest_prepare")
        self._handles[key] = (h, refs)
        return h

    def lut(self) -> np.ndarray | None:
        if self._lut_cells is None:
            ok = self.nf <= 64 and int(np.diff(self.thr_off).max(initial=0)) <= 254
            arr = np.zeros(self.nf * 65536, np.uint32) if ok else None
            if ok and _lib().shira_bin_lut_build(_ptr(self.thr_off), _ptr(self.thr), self.nf, _ptr(arr)) == 0:
                self._lut_cells = arr
            else:
                self._lut_cells = False
        return self._lut_cells if self._lut_cells is not False else None

    @property
    def n_trees(self) -> int:
        return int(self.tree_off.size - 1)

    def struct(self) -> _CForest:
        arrs = [self.tree_off, self.nodes, self.value, self.cover, self.gain, self.leaf_id, self.group]
        for a in arrs:
            assert a is None or a.flags.c_contiguous
        return _CForest(self.n_trees, self.n_groups, self.nf, 0, *[_ptr(a) for a in arrs])

    def _rows(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if self.lgb_snap:
            x = np.asarray(x, dtype=np.float64)
            return np.ascontiguousarray(np.where(np.abs(x) <= LGB_ZERO, 0.0, x))
        if self.lgb_zero_nan is not None:
            x = np.array(x, dtype=np.float64)
            c = self.lgb_zero_nan
            x[:, c] = np.where(np.abs(x[:, c]) <= LGB_ZERO, np.nan, x[:, c])
            return np.ascontiguousarray(x)
        if self.in_f32 or x.dtype == np.float32:
            return np.ascontiguousarray(x, dtype=np.float32)
        return np.ascontiguousarray(x, dtype=np.float64)

    def codes(self, x: np.ndarray, threads: int = 2) -> np.ndarray:
        x = self._rows(x)
        assert x.ndim == 2 and x.shape[1] == self.nf and 1 <= threads <= MAX_THREADS
        out = np.empty(x.shape, dtype=np.uint16)
        fn = _lib().shira_bin_raw_f32 if x.dtype == np.float32 else _lib().shira_bin_raw_f64
        if x.dtype != np.float32:
            x = np.ascontiguousarray(x, dtype=np.float64)
        _check(fn(_ptr(x), x.shape[0], self.nf, _ptr(self.thr_off), _ptr(self.thr), self.op_le, threads,
                  _ptr(out)), "shira_bin_raw")
        return out

    def predict(self, x: np.ndarray, threads: int = 2) -> np.ndarray:
        x = self._rows(x)
        assert x.ndim == 2 and x.shape[1] == self.nf
        if self.lgb_linear is not None:
            return self._predict_linear(x, threads)
        if self.sym is None:
            return self._predict_prepared(x, threads, -1)
        if int(np.diff(self.thr_off).max(initial=0)) <= 254:
            return self.predict_codes(self.codes8(x, threads), threads)
        return self.predict_float(x, threads)

    def _rawx_lut(self, x: np.ndarray) -> np.ndarray | None:
        return self.lut() if x.dtype == np.float32 else None

    def predict_float(self, x: np.ndarray, threads: int = 2, path: int = -1) -> np.ndarray:
        x = self._rows(x)
        assert x.ndim == 2 and x.shape[1] == self.nf and path in (-1, 0, 1, 2, 3)
        if self.lgb_linear is not None:
            return self._predict_linear(x, threads)
        f32 = x.dtype == np.float32
        if self.sym is not None and f32 and self.n_groups == 1:
            cut8, lt = self._sym_cut8(), self.lut()
            if cut8 is not None and lt is not None:
                return self._predict_sym_lut(x, cut8, lt, threads)
            return self._predict_sym(x, threads)
        return self._predict_prepared(x, threads, path)

    def _predict_linear(self, x: np.ndarray, threads: int) -> np.ndarray:
        lin = self.lgb_linear
        leaf = self.leaf(x, threads)
        out = np.empty(x.shape[0])
        _check(_lib().shira_lgb_linear_predict(_ptr(leaf), x.shape[0], leaf.shape[1], _ptr(x), self.nf,
                                               _ptr(lin["leaf_off"]), _ptr(lin["const"]), _ptr(lin["value"]),
                                               _ptr(lin["foff"]), _ptr(lin["feat"]), _ptr(lin["coef"]),
                                               int(lin["fused"]), threads, _ptr(out)), "shira_lgb_linear_predict")
        return out

    def _predict_prepared(self, x: np.ndarray, threads: int, path: int) -> np.ndarray:
        f32 = x.dtype == np.float32
        h = self._handle(f32, path)
        out = np.empty((x.shape[0], self.n_groups))
        fn = _LIB.shira_prepared_predict_f32 if f32 else _LIB.shira_prepared_predict_f64
        _check(fn(h, x.ctypes.data, x.shape[0], threads, out.ctypes.data), "shira_prepared_predict")
        return out[:, 0] if self.n_groups == 1 else out

    def _sym_tables(self, interactions: bool, threads: int):
        key = "itab" if interactions else "tab"
        if key in self.sym:
            return self.sym[key]
        sy, w = self.sym, self.nf + 1
        tabs = []
        for t in range(self.n_trees):
            d = int(sy["depth"][t])
            sub = build_subforest(self, t)
            sub.cover = np.ascontiguousarray(self.cover[self.tree_off[t]:self.tree_off[t + 1]])
            sub.base = np.zeros(1)
            feats = sy["feat"][sy["soff"][t]:sy["soff"][t + 1]]
            borders = sy["border"][sy["soff"][t]:sy["soff"][t + 1]].astype(np.float64)
            cuts = [int(np.searchsorted(self.thr[self.thr_off[f]:self.thr_off[f + 1]], bd)) + 1
                    for f, bd in zip(feats, borders)]
            rows = np.ones((1 << d, self.nf), np.uint16)
            ok = np.ones(1 << d, bool)
            for leaf in range(1 << d):
                lo = np.ones(self.nf, np.int64)
                hi = np.full(self.nf, 65535, np.int64)
                for lv in range(d):
                    f = feats[lv]
                    if (leaf >> lv) & 1:
                        lo[f] = max(lo[f], cuts[lv] + 1)
                    else:
                        hi[f] = min(hi[f], cuts[lv])
                ok[leaf] = bool((lo <= hi).all())
                rows[leaf] = np.minimum(lo, 65535)
            shape = (1 << d, w, w) if interactions else (1 << d, w)
            out = np.zeros(shape)
            fn = _lib().shira_forest_shap_interactions if interactions else _lib().shira_forest_shap
            st = sub.struct()
            _check(fn(ctypes.byref(st), _ptr(rows), rows.shape[0], 2, _ptr(sub.base), 1, _ptr(out)), "sym table")
            out[~ok] = 0.0
            tabs.append(out)
        self.sym[key] = tabs
        return tabs

    def _sym_shap(self, x: np.ndarray, threads: int, interactions: bool) -> np.ndarray:
        tabs = self._sym_tables(interactions, threads)
        lf = self.leaf(x, threads)
        w = self.nf + 1
        out = np.zeros((x.shape[0], w, w) if interactions else (x.shape[0], w))
        for t, tab in enumerate(tabs):
            out += tab[lf[:, t]]
        if interactions:
            out[:, w - 1, w - 1] += self.base[0]
        else:
            out[:, w - 1] += self.base[0]
        return out

    def _sym_cut8(self) -> np.ndarray | None:
        sy = self.sym
        if "cut8" not in sy:
            cuts = [int(np.searchsorted(self.thr[self.thr_off[f]:self.thr_off[f + 1]], float(bd))) + 1
                    for f, bd in zip(sy["feat"], sy["border"])]
            ok = max(cuts, default=0) <= 254 and int(sy["depth"].max(initial=0)) <= 8
            sy["cut8"] = np.ascontiguousarray(cuts, dtype=np.uint8) if ok else None
        return sy["cut8"]

    def _predict_sym_lut(self, x: np.ndarray, cut8: np.ndarray, lt: np.ndarray, threads: int) -> np.ndarray:
        sy = self.sym
        out = np.empty(x.shape[0])
        _check(_lib().shira_sym_predict_lut_f32(_ptr(sy["depth"]), sy["depth"].size, _ptr(sy["soff"]),
                                                _ptr(sy["feat"]), _ptr(cut8), _ptr(sy["loff"]), _ptr(sy["leaf"]),
                                                _ptr(lt), _ptr(self.thr_off), _ptr(self.thr), self.op_le, _ptr(x),
                                                x.shape[0], self.nf, float(self.base[0]), int(self.pred_mode == 3),
                                                threads, _ptr(out)), "shira_sym_predict_lut_f32")
        return out

    def _predict_sym_u8(self, c: np.ndarray, threads: int) -> np.ndarray:
        sy = self.sym
        assert self._sym_cut8() is not None
        out = np.empty(c.shape[0])
        _check(_lib().shira_sym_predict_u8(_ptr(sy["depth"]), sy["depth"].size, _ptr(sy["soff"]), _ptr(sy["feat"]),
                                           _ptr(sy["cut8"]), _ptr(sy["loff"]), _ptr(sy["leaf"]), _ptr(c), c.shape[0],
                                           self.nf, float(self.base[0]), int(self.pred_mode == 3), threads, _ptr(out)),
               "shira_sym_predict_u8")
        return out

    def _predict_sym(self, x: np.ndarray, threads: int) -> np.ndarray:
        sy = self.sym
        out = np.empty(x.shape[0])
        _check(_lib().shira_sym_predict_f32(_ptr(sy["depth"]), sy["depth"].size, _ptr(sy["soff"]), _ptr(sy["feat"]),
                                            _ptr(sy["border"]), _ptr(sy["loff"]), _ptr(sy["leaf"]), _ptr(x),
                                            x.shape[0], self.nf, float(self.base[0]), int(self.pred_mode == 3),
                                            threads, _ptr(out)), "shira_sym_predict_f32")
        return out

    def codes8(self, x: np.ndarray, threads: int = 2) -> np.ndarray:
        x = self._rows(x)
        out = np.empty(x.shape, dtype=np.uint8)
        lt = self.lut() if x.dtype == np.float32 else None
        if lt is not None:
            _check(_lib().shira_bin_lut_u8_f32(_ptr(x), x.shape[0], self.nf, _ptr(self.thr_off), _ptr(self.thr),
                                               self.op_le, _ptr(lt), threads, _ptr(out)), "shira_bin_lut_u8_f32")
            return out
        fn = _lib().shira_bin_raw_u8_f32 if x.dtype == np.float32 else _lib().shira_bin_raw_u8_f64
        _check(fn(_ptr(x), x.shape[0], self.nf, _ptr(self.thr_off), _ptr(self.thr), self.op_le, threads, _ptr(out)),
               "shira_bin_raw_u8")
        return out

    def predict_delta(self, new_codes: np.ndarray, old_codes: np.ndarray | None, leaves: np.ndarray,
                      threads: int = 2) -> tuple[np.ndarray, int]:
        nc = np.ascontiguousarray(new_codes, dtype=np.uint8)
        oc = None if old_codes is None else np.ascontiguousarray(old_codes, dtype=np.uint8)
        assert leaves.dtype == np.int32 and leaves.shape == (nc.shape[0], self.n_trees) and leaves.flags.c_contiguous
        assert self.pred_mode in (0, 1)
        out = np.empty((nc.shape[0], self.n_groups))
        walked = ctypes.c_int64(0)
        st = self.struct()
        _check(_lib().shira_forest_predict_delta(ctypes.byref(st), _ptr(nc), _ptr(oc), nc.shape[0], self.pred_mode,
                                                 _ptr(self.base), threads, _ptr(leaves), _ptr(out), ctypes.byref(walked)),
               "shira_forest_predict_delta")
        return (out[:, 0] if self.n_groups == 1 else out), int(walked.value)

    def predict_codes(self, codes: np.ndarray, threads: int = 2) -> np.ndarray:
        if self.lgb_linear is not None or self.lgb_cat is not None:
            raise NotImplementedError("linear / native categorical LightGBM forests need the raw rows: use predict()")
        c = np.ascontiguousarray(codes)
        u8 = c.dtype == np.uint8
        if u8 and self.sym is not None and self.n_groups == 1 and self._sym_cut8() is not None:
            return self._predict_sym_u8(c, threads)
        if not u8:
            c = np.ascontiguousarray(c, dtype=np.uint16)
        out = np.empty((c.shape[0], self.n_groups))
        st = self.struct()
        fn = _lib().shira_forest_predict_u8 if u8 else _lib().shira_forest_predict
        _check(fn(ctypes.byref(st), _ptr(c), c.shape[0], self.pred_mode, _ptr(self.base), threads, _ptr(out)),
               "shira_forest_predict_codes")
        return out[:, 0] if self.n_groups == 1 else out

    def leaf(self, x: np.ndarray, threads: int = 2, path: int = -1) -> np.ndarray:
        x = self._rows(x)
        assert path in (-1, 0, 1, 2, 3)
        out = np.empty((x.shape[0], self.n_trees), dtype=np.int32)
        if self.sym is not None and x.dtype == np.float32:
            sy = self.sym
            _check(_lib().shira_sym_leaf_f32(_ptr(sy["depth"]), sy["depth"].size, _ptr(sy["soff"]), _ptr(sy["feat"]),
                                             _ptr(sy["border"]), _ptr(sy["loff"]), _ptr(x), x.shape[0], self.nf,
                                             threads, _ptr(out)), "shira_sym_leaf_f32")
            return out
        f32 = x.dtype == np.float32
        h = self._handle(f32, path)
        fn = _LIB.shira_prepared_leaf_f32 if f32 else _LIB.shira_prepared_leaf_f64
        _check(fn(h, x.ctypes.data, x.shape[0], threads, out.ctypes.data), "shira_prepared_leaf")
        return out

    def importance(self, kind: str) -> np.ndarray:
        if self.no_stats:
            raise NotImplementedError(self.no_stats)
        out = np.zeros(self.nf)
        st = self.struct()
        _check(_lib().shira_forest_importance(ctypes.byref(st), IMPORTANCE[kind], _ptr(out)),
               "shira_forest_importance")
        return out

    def shap(self, x: np.ndarray, threads: int = 2, interactions: bool = False, mode: int | None = None) -> np.ndarray:
        if self.no_stats:
            raise NotImplementedError(self.no_stats)
        mode = (3 if self.shap_mode == 0 else 2) if mode is None else mode
        if self.sym is not None and mode == 2 and self.n_groups == 1 and self.pred_mode in (1, 3):
            return self._sym_shap(x, threads, interactions)
        c = self.codes(x, threads)
        w = self.nf + 1
        shape = (c.shape[0], self.n_groups, w, w) if interactions else (c.shape[0], self.n_groups, w)
        out = np.empty(shape)
        fn = _lib().shira_forest_shap_interactions if interactions else _lib().shira_forest_shap
        src = self
        if self.pred_mode == 2:
            per = np.bincount(self.group if self.group is not None else np.zeros(self.n_trees, np.int64),
                              minlength=self.n_groups)
            tree_of = np.repeat(np.arange(self.n_trees), np.diff(self.tree_off))
            g = self.group[tree_of] if self.group is not None else np.zeros(tree_of.size, np.int64)
            src = Forest(**{**self.__dict__, "value": np.ascontiguousarray(self.value / per[g]), "_keep": [],
                              "_handles": {}})
        st = src.struct()
        _check(fn(ctypes.byref(st), _ptr(c), c.shape[0], mode, _ptr(self.base), threads, _ptr(out)),
               "shira_forest_shap")
        return out[:, 0] if self.n_groups == 1 else out

    def node_sums(self, x: np.ndarray, w: np.ndarray | None = None, y: np.ndarray | None = None):
        if self.no_stats:
            raise NotImplementedError(self.no_stats)
        c = self.codes(x)
        nn = int(self.tree_off[-1])
        ow, owy, owy2 = np.zeros(nn), np.zeros(nn), np.zeros(nn)
        wv = None if w is None else np.ascontiguousarray(w, dtype=np.float64)
        yv = None if y is None else np.ascontiguousarray(y, dtype=np.float64)
        st = self.struct()
        _check(_lib().shira_forest_node_sums(ctypes.byref(st), _ptr(c), c.shape[0], _ptr(wv), _ptr(yv),
                                             _ptr(ow), _ptr(owy), _ptr(owy2)), "shira_forest_node_sums")
        return ow, owy, owy2


def build(trees: list[RawTree], nf: int, op_le: int, pred_mode: int, shap_mode: int, base=0.0,
          group: np.ndarray | None = None, n_groups: int = 1) -> Forest:
    assert trees and nf >= 1 and op_le in (0, 1)
    feat = np.ascontiguousarray(np.concatenate([t.feat for t in trees]), dtype=np.int32)
    thr = np.ascontiguousarray(np.concatenate([np.asarray(t.thr, dtype=np.float64) for t in trees]))
    dl, lf, rg = (np.ascontiguousarray(np.concatenate([getattr(t, n) for t in trees]), dtype=np.int32)
                  for n in ("dleft", "left", "right"))
    sizes = np.array([t.feat.size for t in trees], dtype=np.int32)
    thr_off, tab = np.empty(nf + 1, dtype=np.int32), np.empty(max(feat.size, 1))
    n_thr, nodes = ctypes.c_int32(0), np.empty((feat.size, 5), dtype=np.int32)
    _check(_lib().shira_out_build_forest(len(trees), _ptr(sizes), nf, _ptr(feat), _ptr(thr), _ptr(dl), _ptr(lf),
                                         _ptr(rg), _ptr(thr_off), _ptr(tab), ctypes.byref(n_thr), _ptr(nodes)),
           "shira_out_build_forest")

    def cat(name, dt):
        parts = [getattr(t, name) for t in trees]
        return None if any(p is None for p in parts) else np.ascontiguousarray(np.concatenate(parts), dtype=dt)

    return Forest(nf=nf, op_le=op_le, pred_mode=pred_mode, shap_mode=shap_mode,
                  base=np.ascontiguousarray(np.broadcast_to(np.asarray(base, np.float64), (n_groups,))),
                  tree_off=np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32),
                  nodes=nodes, value=cat("value", np.float64),
                  thr_off=thr_off, thr=np.ascontiguousarray(tab[:n_thr.value]),
                  cover=cat("cover", np.float64), gain=cat("gain", np.float64),
                  leaf_id=cat("leaf_id", np.int32), group=group, n_groups=n_groups)


def set_stats(fr: Forest, cover=None, gain=None) -> Forest:
    if cover is not None:
        fr.cover = np.ascontiguousarray(cover, dtype=np.float64)
    if gain is not None:
        fr.gain = np.ascontiguousarray(gain, dtype=np.float64)
    return fr


def _arr(a, dt=np.int32):
    return np.asarray(a, dtype=dt)


def from_shira_xgb(booster, ip: np.ndarray, cuts: np.ndarray, nf: int) -> Forest:
    trees = []
    for tree in booster.trees:
        k = len(tree)
        feat = _arr([-1 if nd.leaf is not None else nd.feature for nd in tree])
        thr = np.array([0.0 if nd.leaf is not None else float(cuts[ip[nd.feature] + nd.cut]) for nd in tree])
        trees.append(RawTree(feat, thr, _arr([nd.dir for nd in tree]), _arr([max(nd.left, 0) for nd in tree]),
                             _arr([max(nd.right, 0) for nd in tree]),
                             np.array([float(nd.leaf) if nd.leaf is not None else 0.0 for nd in tree]),
                             leaf_id=np.arange(k, dtype=np.int32)))
    fr = build(trees, nf, 0, 0, 0, base=float(np.float32(booster.recipe.base_score)))
    fr.in_f32 = True
    return fr


def xgb_replay(fr: Forest, x_train: np.ndarray, y: np.ndarray, base: float, objective: str, lam: float,
               mcw: float = 1.0) -> Forest:
    c = fr.codes(x_train)
    yy = np.ascontiguousarray(y, dtype=np.float32)
    nn = int(fr.tree_off[-1])
    cover, gain, weight = np.zeros(nn), np.zeros(nn), np.zeros(nn)
    st = fr.struct()
    obj = {"reg:pseudohubererror": 0, "binary:logistic": 1}[objective]
    _check(_lib().shira_xgb_replay_stats(ctypes.byref(st), _ptr(c), c.shape[0], _ptr(yy), float(np.float32(base)),
                                         obj, float(lam), float(mcw), _ptr(cover), _ptr(gain), _ptr(weight)),
           "shira_xgb_replay_stats")
    leaf = fr.nodes[:, 0] < 0
    weight[leaf] = fr.value[leaf]
    fr.base_weight = weight
    return set_stats(fr, cover, gain)


def prune_xgb(fr: Forest, eta: float, gamma: float, max_depth: int = 0) -> tuple[Forest, int]:
    assert fr.base_weight is not None and fr.gain is not None
    nodes, value = np.empty_like(fr.nodes), np.empty_like(fr.value)
    npr = ctypes.c_int32(0)
    st = fr.struct()
    _check(_lib().shira_xgb_prune(ctypes.byref(st), _ptr(np.ascontiguousarray(fr.base_weight)), float(eta),
                                  float(gamma), int(max_depth), _ptr(nodes), _ptr(value), ctypes.byref(npr)),
           "shira_xgb_prune")
    return Forest(**{**fr.__dict__, "nodes": nodes, "value": value, "_keep": [], "_handles": {}}), int(npr.value)


def xgboost_json(bst) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        bst.save_model(str(path))
        return json.loads(path.read_text())


def from_xgboost(bst, nf: int, base_margin: float) -> Forest:
    js = xgboost_json(bst)
    model = js["learner"]["gradient_booster"]["model"]
    trees, bw = [], []
    for t in model["trees"]:
        bw.append(np.asarray(t["base_weights"], dtype=np.float32).astype(np.float64))
        lc, rc = _arr(t["left_children"]), _arr(t["right_children"])
        leaf = lc < 0
        feat = np.where(leaf, -1, _arr(t["split_indices"]))
        cond = np.asarray(t["split_conditions"], dtype=np.float32).astype(np.float64)
        trees.append(RawTree(feat, np.where(leaf, 0.0, cond), _arr(t["default_left"]), np.maximum(lc, 0),
                             np.maximum(rc, 0), np.where(leaf, cond, 0.0),
                             np.asarray(t["sum_hessian"], dtype=np.float32).astype(np.float64),
                             np.asarray(t["loss_changes"], dtype=np.float32).astype(np.float64),
                             np.arange(lc.size, dtype=np.int32)))
    fr = build(trees, nf, 0, 0, 0, base=float(np.float32(base_margin)))
    fr.in_f32 = True
    fr.base_weight = np.ascontiguousarray(np.concatenate(bw))
    return fr


def lgb_text_gain(g: float) -> float:
    return float(np.float32(float("%g" % float(np.float32(g)))))


def _lgb_tree(nodes, leaf, gain, nleaves, upper) -> RawTree:
    ni = nleaves - 1
    k = ni + nleaves
    feat = np.full(k, -1, np.int32)
    thr, lft, rgt = np.zeros(k), np.zeros(k, np.int32), np.zeros(k, np.int32)
    val, gn = np.zeros(k), np.zeros(k)
    for v in range(ni):
        f, b, a, c = (int(z) for z in nodes[v])
        feat[v], thr[v] = f, float(upper[f][b])
        lft[v] = a if a >= 0 else ni + ~a
        rgt[v] = c if c >= 0 else ni + ~c
        gn[v] = lgb_text_gain(gain[v])
    val[ni:] = leaf[:nleaves]
    lid = np.concatenate([np.zeros(ni, np.int32), np.arange(nleaves, dtype=np.int32)])
    return RawTree(feat, thr, np.zeros(k, np.int32), lft, rgt, val, gain=gn, leaf_id=lid)


def from_shira_lgb(res: dict, prep: dict, k: int) -> Forest:
    upper = [m["upper"] for m in prep["mappers"][k]]
    trees = [_lgb_tree(res["nodes"][k, r], res["leaf"][k, r], res["gain"][k, r], int(res["nleaves"][k, r]), upper)
             for r in range(res["nleaves"].shape[1])]
    return build(trees, len(upper), 1, 1, 1)


LGB_KEYS = ("split_feature", "split_gain", "threshold", "decision_type", "left_child", "right_child", "leaf_value",
            "leaf_count", "internal_count")
LGB_ZERO = float(np.float32(1e-35))


def lgb_text_arrays(text: str) -> dict:
    out = {k: [] for k in LGB_KEYS}
    nl, cat_b, cat_w, toff, woff = [], [], [], [0], [0]
    lin = {"is": [], "const": [], "nfeat": [], "feat": [], "coef": []}
    blk: dict | None = None
    for line in text.splitlines():
        if line.startswith("Tree=") or line.startswith("end of trees"):
            if blk is not None:
                n = int(blk["num_leaves"])
                nl.append(n)
                for k in LGB_KEYS:
                    out[k] += blk.get(k, "").split()
                cb, cw = blk.get("cat_boundaries", "").split(), blk.get("cat_threshold", "").split()
                cat_b += cb
                cat_w += cw
                toff.append(toff[-1] + len(cb))
                woff.append(woff[-1] + len(cw))
                linear = blk.get("is_linear", "0") == "1"
                lin["is"].append(linear)
                nf_leaf = [int(v) for v in blk.get("num_features", "").split()] if linear else [0] * n
                lin["const"] += blk.get("leaf_const", "").split() if linear else blk.get("leaf_value", "").split()
                lin["nfeat"] += nf_leaf
                lin["feat"] += blk.get("leaf_features", "").split() if linear else []
                lin["coef"] += blk.get("leaf_coeff", "").split() if linear else []
            blk = {}
            if line.startswith("end of trees"):
                break
        elif blk is not None and "=" in line:
            k, v = line.split("=", 1)
            blk[k] = v
    arr = {"nleaves": np.asarray(nl, np.int32)}
    for k, dt in (("split_feature", np.int32), ("decision_type", np.int32), ("left_child", np.int32),
                  ("right_child", np.int32)):
        arr[k] = np.ascontiguousarray(np.asarray(out[k], dtype=np.int64).astype(dt))
    for k in ("threshold", "leaf_value", "leaf_count", "internal_count"):
        arr[k] = np.ascontiguousarray(np.asarray([float(v) for v in out[k]], dtype=np.float64))
    arr["split_gain"] = np.asarray([float(v) for v in out["split_gain"]], dtype=np.float64).astype(np.float32).astype(np.float64)
    arr["cat_toff"] = np.asarray(toff, np.int32)
    arr["cat_bounds"] = np.ascontiguousarray(np.asarray(cat_b, np.int64).astype(np.int32))
    arr["cat_woff"] = np.asarray(woff, np.int32)
    arr["cat_words"] = np.ascontiguousarray(np.asarray(cat_w, np.int64).astype(np.uint32))
    arr["linear"] = any(lin["is"])
    if arr["linear"]:
        nfl = np.asarray(lin["nfeat"], np.int64)
        arr["lin_const"] = np.ascontiguousarray([float(v) for v in lin["const"]], dtype=np.float64)
        arr["lin_foff"] = np.concatenate([[0], np.cumsum(nfl)]).astype(np.int32)
        arr["lin_feat"] = np.ascontiguousarray(np.asarray(lin["feat"], np.int64).astype(np.int32))
        arr["lin_coef"] = np.ascontiguousarray([float(v) for v in lin["coef"]], dtype=np.float64)
        assert arr["lin_feat"].size == arr["lin_coef"].size == int(nfl.sum())
    return arr


def _lgb_zero_nan(a: dict, nf: int) -> np.ndarray | None:
    if a["linear"]:
        return None
    sf, dt = a["split_feature"], a["decision_type"]
    zn = np.zeros(nf, np.uint8)
    for f in np.unique(sf).tolist():
        d = dt[sf == f]
        zn[f] = 1 if (((d >> 2) & 3) == 1).all() and not (d & 1).any() else 0
    return zn if zn.any() else None


def from_lightgbm(bst, nf: int) -> Forest:
    a = lgb_text_arrays(bst.model_to_string())
    zn = _lgb_zero_nan(a, nf)
    native = int(bool((a["decision_type"] & 1).any()))
    fr = _lgb_forest(a, nf, zn, native)
    if fr is not None and native and not _qs_fits(fr):
        fr = None
    if fr is None:
        fr = _lgb_forest(a, nf, zn, 0)
    return fr


QS_MAX_CW, QS_MAX_WORDS = 256, 1 << 26


def _cat_w(a: dict) -> int:
    cb = np.asarray(a["cat_bounds"], np.int64)
    return int(max(1, np.diff(cb).max())) if cb.size > 1 else 1


def _qs_fits(fr) -> bool:
    sizes = np.diff(np.asarray(fr.tree_off, np.int64))
    w0 = max(1, (int((sizes.max() + 1) // 2) + 31) // 32)
    if w0 > QS_MAX_CW:
        return False
    w = 1 << (w0 - 1).bit_length() if w0 <= 16 else (w0 + 15) // 16 * 16
    cw = max(16, w)
    tp = (sizes.size * w + cw - 1) // cw * cw
    nthr = np.diff(np.asarray(fr.thr_off, np.int64))
    cmax = np.asarray(fr.lgb_cat["feat"], np.int64)
    codes = np.where(cmax > 0, cmax + 1, np.where(nthr > 0, nthr + 2, 0))
    return int(codes.sum()) * tp <= QS_MAX_WORDS


def _lgb_forest(a: dict, nf: int, zn, native: int):
    nl = a["nleaves"]
    ins = (nl.size, _ptr(nl), _ptr(a["split_feature"]), _ptr(a["threshold"]), _ptr(a["decision_type"]),
           _ptr(a["left_child"]), _ptr(a["right_child"]), _ptr(a["leaf_value"]), _ptr(a["leaf_count"]),
           _ptr(a["internal_count"]), _ptr(a["split_gain"]), _ptr(a["cat_toff"]), _ptr(a["cat_bounds"]),
           _ptr(a["cat_woff"]), _ptr(a["cat_words"]), _ptr(zn), native, _cat_w(a), nf)
    k, tsz = ctypes.c_int64(0), np.empty(nl.size, np.int32)
    rc = _lib().shira_out_from_lgb(*ins, 0, ctypes.byref(k), _ptr(tsz), *([None] * 11))
    if rc == 2:
        return None
    _check(rc, "shira_out_from_lgb sizes")
    n = int(k.value)
    o = {nm: np.empty(n, np.int32) for nm in ("feat", "dleft", "left", "right", "leaf_id")}
    o.update({nm: np.empty(n) for nm in ("thr", "value", "cover", "gain")})
    cbits, cfeat = np.zeros((n, _cat_w(a)), np.uint32), np.zeros(nf, np.int32)
    _check(_lib().shira_out_from_lgb(*ins, n, ctypes.byref(k), _ptr(tsz), _ptr(o["feat"]), _ptr(o["thr"]),
                                     _ptr(o["dleft"]), _ptr(o["left"]), _ptr(o["right"]), _ptr(o["value"]),
                                     _ptr(o["cover"]), _ptr(o["gain"]), _ptr(o["leaf_id"]), _ptr(cbits), _ptr(cfeat)),
           "shira_out_from_lgb")
    fr = build(_split_trees(o, tsz.tolist(), ("cover", "gain")), nf, 1, 1, 1)
    if native:
        fr.lgb_cat = {"feat": cfeat, "bits": np.ascontiguousarray(cbits)}
    fr.lgb_snap = bool(a["linear"])
    fr.lgb_zero_nan = None if zn is None else np.flatnonzero(zn)
    cat_cols = np.flatnonzero(fr.lgb_cat["feat"]) if fr.lgb_cat is not None else np.zeros(0, np.int64)
    num = np.ones(fr.thr.size, bool)
    for f in cat_cols.tolist():
        num[fr.thr_off[f]:fr.thr_off[f + 1]] = False
    assert not ((fr.thr >= -LGB_ZERO) & (fr.thr < LGB_ZERO) & num).any()
    if n != int((2 * nl - 1).sum()) or a["linear"] or native:
        fr.no_stats = "LightGBM model with Zero / categorical splits or linear trees: prediction and leaf numbers only"
    if a["linear"]:
        fr.lgb_linear = {"leaf_off": np.concatenate([[0], np.cumsum(nl)]).astype(np.int32), "const": a["lin_const"],
                         "value": a["leaf_value"], "foff": a["lin_foff"], "feat": a["lin_feat"], "coef": a["lin_coef"],
                         "fused": LGB_LINEAR_FUSED}
    return fr


def _split_trees(o: dict, sizes: list, stats: tuple) -> list:
    trees, at = [], 0
    for k in sizes:
        sl = slice(at, at + k)
        trees.append(RawTree(o["feat"][sl], o["thr"][sl], o["dleft"][sl], o["left"][sl], o["right"][sl],
                             o["value"][sl], o["cover"][sl] if "cover" in stats else None,
                             o["gain"][sl] if "gain" in stats else None, o["leaf_id"][sl]))
        at += k
    return trees


def fill_counts(fr: Forest, x_train: np.ndarray) -> Forest:
    w, _, _ = fr.node_sums(x_train)
    return set_stats(fr, cover=w)


def _sk_tree(feature, threshold, left, right, value, cover=None, impurity=None) -> RawTree:
    feat = np.where(np.asarray(left) < 0, -1, np.asarray(feature)).astype(np.int32)
    return RawTree(feat, np.where(feat < 0, 0.0, threshold), np.zeros(feat.size, np.int32),
                   np.maximum(left, 0).astype(np.int32), np.maximum(right, 0).astype(np.int32),
                   np.asarray(value, np.float64).reshape(feat.size, -1)[:, 0], cover, impurity,
                   np.arange(feat.size, dtype=np.int32))


def from_shira_sk(arrs_list: list, nf: int) -> Forest:
    trees = [_sk_tree(a[0], a[1], a[2], a[3], a[4]) for a in arrs_list]
    fr = build(trees, nf, 1, 2 if len(trees) > 1 else 1, 1)
    fr.in_f32 = True
    return fr


def from_shira_skfit(model, nf: int) -> Forest:
    trees = []
    for t in model.trees_:
        tr = _sk_tree(t["feature"], t["threshold"], t["children_left"], t["children_right"],
                      np.asarray(t["value"]).reshape(len(t["feature"]), -1),
                      np.asarray(t["weighted_n_node_samples"], np.float64), np.asarray(t["impurity"], np.float64))
        tr.dleft = np.asarray(t["missing_go_to_left"], np.int32) * (tr.feat >= 0)
        trees.append(tr)
    fr = build(trees, nf, 1, 2 if len(trees) > 1 else 1, 1)
    fr.in_f32 = True
    return fr


def fill_sk_stats(fr: Forest, x_train: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Forest:
    nn = int(fr.tree_off[-1])
    cover, imp = np.zeros(nn), np.zeros(nn)
    for t in range(fr.n_trees):
        a, b = int(fr.tree_off[t]), int(fr.tree_off[t + 1])
        sub = build_subforest(fr, t)
        w, wy, wy2 = sub.node_sums(x_train, weights[t], y)
        cover[a:b] = w
        with np.errstate(invalid="ignore", divide="ignore"):
            imp[a:b] = wy2 / w - (wy / w) ** 2
    return set_stats(fr, cover, imp)


def build_subforest(fr: Forest, t: int) -> Forest:
    a, b = int(fr.tree_off[t]), int(fr.tree_off[t + 1])
    return Forest(nf=fr.nf, op_le=fr.op_le, pred_mode=1, shap_mode=fr.shap_mode, base=fr.base,
                  tree_off=np.array([0, b - a], np.int32), nodes=np.ascontiguousarray(fr.nodes[a:b]),
                  value=np.ascontiguousarray(fr.value[a:b]), thr_off=fr.thr_off, thr=fr.thr)


def from_sklearn(est, nf: int) -> Forest:
    ests = getattr(est, "estimators_", [est])
    trees = []
    for e in ests:
        t = e.tree_
        tr = _sk_tree(t.feature, t.threshold, t.children_left, t.children_right, t.value[:, 0, :],
                      t.weighted_n_node_samples.astype(np.float64), t.impurity.astype(np.float64))
        tr.dleft = np.asarray(t.missing_go_to_left, np.int32) * (tr.feat >= 0)
        trees.append(tr)
    fr = build(trees, nf, 1, 2 if len(trees) > 1 else 1, 1)
    fr.in_f32 = True
    return fr


def _hgb_tree(feat, thr, left, right, value, count=None) -> RawTree:
    feat = np.asarray(feat, np.int32)
    k = feat.size
    return RawTree(feat, np.where(feat < 0, 0.0, thr), np.zeros(k, np.int32), np.maximum(left, 0).astype(np.int32),
                   np.maximum(right, 0).astype(np.int32), np.asarray(value, np.float64), count, None,
                   np.arange(k, dtype=np.int32))


def _preorder(tr: RawTree) -> RawTree:
    order, stack = [], [0]
    for _ in range(2 * tr.feat.size + 2):
        if not stack:
            break
        v = stack.pop()
        order.append(v)
        if tr.feat[v] >= 0:
            stack += [int(tr.right[v]), int(tr.left[v])]
    assert not stack and len(order) == tr.feat.size
    o = np.asarray(order)
    new = np.empty_like(o)
    new[o] = np.arange(o.size)
    pick = {k: (None if getattr(tr, k) is None else np.asarray(getattr(tr, k))[o])
            for k in ("feat", "thr", "dleft", "value", "cover", "gain")}
    leaf = pick["feat"] < 0
    return RawTree(pick["feat"], pick["thr"], pick["dleft"], np.where(leaf, 0, new[tr.left[o]]).astype(np.int32),
                   np.where(leaf, 0, new[tr.right[o]]).astype(np.int32), pick["value"], pick["cover"], pick["gain"],
                   np.arange(o.size, dtype=np.int32))


def from_shira_hgb(res: dict, prep: dict, k: int) -> Forest:
    thr_k = prep["thresholds"][k]
    assert res["nnodes"].ndim == 2, "one tree per iteration only (regression / binary)"
    trees = []
    for r in range(int(np.asarray(res["n_iter"])[k])):
        nn = int(res["nnodes"][k, r])
        nd = res["nodes"][k, r, :nn]
        assert not (nd[:, 4] & 2).any(), "categorical splits not supported"
        thr = np.array([(float(thr_k[f][b]) if b < len(thr_k[f]) else np.inf) if f >= 0 else 0.0
                        for f, b in nd[:, :2]])
        tr = _hgb_tree(nd[:, 0], thr, nd[:, 2], nd[:, 3], res["value"][k, r, :nn])
        tr.dleft = (nd[:, 4] & 1).astype(np.int32) * (nd[:, 0] >= 0)
        trees.append(_preorder(tr))
    return build(trees, len(thr_k), 1, 1, 1, base=float(np.asarray(prep["base"][k]).ravel()[0]))


def from_hgb(est, nf: int) -> Forest:
    trees = []
    for preds in est._predictors:
        nd = preds[0].nodes
        feat = np.where(nd["is_leaf"].astype(bool), -1, nd["feature_idx"]).astype(np.int32)
        tr = _hgb_tree(feat, nd["num_threshold"], nd["left"], nd["right"], nd["value"],
                       nd["count"].astype(np.float64))
        tr.dleft = nd["missing_go_to_left"].astype(np.int32) * (feat >= 0)
        trees.append(tr)
    return build(trees, nf, 1, 1, 1, base=float(np.asarray(est._baseline_prediction).ravel()[0]))


def sym_trees(splits: list, leaves: list, weights: list | None, root_first: bool, nan_left: frozenset,
              nf: int) -> list:
    depth = np.array([len(s) for s in splits], np.int32)
    sf = np.ascontiguousarray([f for s in splits for f, _ in s], dtype=np.int32)
    sb = np.ascontiguousarray([b for s in splits for _, b in s], dtype=np.float32)
    lv = np.ascontiguousarray(np.concatenate([np.asarray(v, np.float64) for v in leaves]))
    w = None if weights is None else np.ascontiguousarray(np.concatenate([np.asarray(v, np.float64) for v in weights]))
    nanl = np.array([1 if f in nan_left else 0 for f in range(nf)], np.uint8)
    sizes = ((2 << depth.astype(np.int64)) - 1).tolist()
    k = int(sum(sizes))
    o = {n: np.empty(k, np.int32) for n in ("feat", "dleft", "left", "right", "leaf_id")}
    o.update({n: np.empty(k) for n in ("thr", "value", "cover")})
    _check(_lib().shira_out_from_sym(depth.size, _ptr(depth), _ptr(sf if sf.size else np.zeros(1, np.int32)),
                                     _ptr(sb if sb.size else np.zeros(1, np.float32)), _ptr(lv), _ptr(w),
                                     int(root_first), _ptr(nanl), nf, _ptr(o["feat"]), _ptr(o["thr"]), _ptr(o["dleft"]),
                                     _ptr(o["left"]), _ptr(o["right"]), _ptr(o["value"]), _ptr(o["cover"]),
                                     _ptr(o["leaf_id"])), "shira_out_from_sym")
    return _split_trees(o, sizes, ("cover",) if weights is not None else ())


def from_shira_cat(fm, nf: int, root_first: bool = False, nan_mode: str = "Min") -> Forest:
    assert nan_mode in ("Min", "Max", "Forbidden")
    nan_left = frozenset() if nan_mode == "Max" else frozenset(range(nf))
    trees = sym_trees(fm.splits, fm.leaves, None, root_first, nan_left, nf)
    fr = _cat_forest(trees, nf, fm.splits, fm.leaves, 0.0)
    if nan_mode == "Max":
        fr.sym = None
    return fr


def _cat_forest(trees: list, nf: int, splits: list, leaves: list, bias: float) -> Forest:
    fr = build(trees, nf, 1, 3, 1, base=bias)
    fr.in_f32 = True
    depth = np.array([len(sp) for sp in splits], np.int32)
    fr.sym = {"depth": depth, "soff": np.concatenate([[0], np.cumsum(depth)]).astype(np.int32),
              "feat": np.array([f for sp in splits for f, _ in sp], np.int32),
              "border": np.array([b for sp in splits for _, b in sp], np.float32),
              "loff": np.concatenate([[0], np.cumsum([len(lv) for lv in leaves])]).astype(np.int32),
              "leaf": np.ascontiguousarray(np.concatenate([np.asarray(lv, np.float64) for lv in leaves]))}
    assert all(len(lv) == 1 << d for lv, d in zip(leaves, depth))
    return fr


def fill_cat_weights(fr: Forest, x_train: np.ndarray) -> Forest:
    return fill_counts(fr, x_train)


def from_catboost(model, nf: int, root_first: bool = False) -> Forest:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        model.save_model(str(path), format="json")
        js = json.loads(path.read_text())
    ff = js.get("features_info", {}).get("float_features", [])
    nan_right = {int(f["feature_index"]) for f in ff if f.get("nan_value_treatment") == "AsTrue"}
    nan_left = frozenset(range(nf)) - nan_right
    all_splits, all_leaves, all_w = [], [], []
    for t in js["oblivious_trees"]:
        if any("float_feature_index" not in s for s in t["splits"]):
            raise NotImplementedError("CatBoost one-hot / CTR splits (categorical features) are not supported")
        all_splits.append([(int(s["float_feature_index"]), float(np.float32(s["border"]))) for s in t["splits"]])
        all_w.append(np.asarray(t.get("leaf_weights", np.ones(len(t["leaf_values"]))), np.float64))
        all_leaves.append(np.asarray(t["leaf_values"], np.float64))
    trees = sym_trees(all_splits, all_leaves, all_w, root_first, nan_left, nf)
    sb = js.get("scale_and_bias", [1.0, [0.0]])
    assert float(sb[0]) == 1.0
    fr = _cat_forest(trees, nf, all_splits, all_leaves, float(np.asarray(sb[1]).ravel()[0]))
    if any(f in nan_right for sp in all_splits for f, _ in sp):
        fr.sym = None
    return fr


def raw_trees(fr: Forest) -> list[RawTree]:
    out = []
    for t in range(fr.n_trees):
        a, b = int(fr.tree_off[t]), int(fr.tree_off[t + 1])
        nd = fr.nodes[a:b]
        feat = nd[:, 0].copy()
        thr = np.zeros(b - a)
        inner = feat >= 0
        thr[inner] = fr.thr[fr.thr_off[feat[inner]] + nd[inner, 1] - 1]
        pick = lambda arr: None if arr is None else arr[a:b].copy()
        out.append(RawTree(feat, thr, nd[:, 2].copy(), nd[:, 3].copy(), nd[:, 4].copy(), fr.value[a:b].copy(),
                           pick(fr.cover), pick(fr.gain), pick(fr.leaf_id)))
    return out


def stack(forests: list[Forest]) -> Forest:
    f0 = forests[0]
    assert all(f.nf == f0.nf and f.op_le == f0.op_le and f.pred_mode == f0.pred_mode and f.n_groups == 1
               for f in forests), "same family, features and modes"
    trees, group = [], []
    for k, f in enumerate(forests):
        rt = raw_trees(f)
        trees += rt
        group += [k] * len(rt)
    fr = build(trees, f0.nf, f0.op_le, f0.pred_mode, f0.shap_mode, base=np.array([f.base[0] for f in forests]),
               group=np.asarray(group, np.int32), n_groups=len(forests))
    fr.in_f32 = f0.in_f32
    return fr
