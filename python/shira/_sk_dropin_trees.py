from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
import sklearn.tree._classes as _tc
from sklearn.tree._tree import Tree

LOG = logging.getLogger("shira.sklearn")
RAND_R_MAX = 0x7FFFFFFF
CRIT = {"MSE": 0, "Gini": 1, "Entropy": 2, "FriedmanMSE": 3, "MAE": 4, "Poisson": 5}
CLF = ("Gini", "Entropy")
_REAL = {"splitters": dict(_tc.DENSE_SPLITTERS), "depth": _tc.DepthFirstTreeBuilder,
         "best": _tc.BestFirstTreeBuilder}
_NODE_DTYPE = Tree(1, np.array([1], dtype=np.intp), 1).__getstate__()["nodes"].dtype
_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"depth": 0, "shira": 0, "stock": 0, "reason": None}
_GB = threading.local()
_XGEN = [0]
_PRE = threading.local()


_FP: dict[str, Any] = {"depth": 0, "threads": 1, "items": {}, "built": 0, "trees": 0}


class _Pre:

    def __init__(self, X: np.ndarray) -> None:
        import ctypes

        assert isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float32
        self.x, self.xc = X, np.ascontiguousarray(X)
        self.handle, self.users, self.orphan, self.failed = ctypes.c_void_p(), 0, False, False
        self.lock = threading.Lock()


def _pre_free(entries: list) -> None:
    for e in entries:
        assert e.users == 0
        if e.handle.value:
            _lib_pre().shira_skx_pre_free(e.handle)
            e.handle.value = None


def _lib_pre() -> Any:
    from shira_cpp import sk_exact_cpp as sx

    return sx._lib_x()


class forest_session:

    def __init__(self, threads: int) -> None:
        assert 1 <= int(threads) <= 64
        self.threads = int(threads)

    def __enter__(self) -> None:
        with _LOCK:
            if _FP["depth"] == 0:
                _FP.update(built=0, trees=0)
            _FP["depth"] += 1
            _FP["threads"] = self.threads

    def __exit__(self, *exc: Any) -> None:
        dead: list = []
        with _LOCK:
            _FP["depth"] -= 1
            assert _FP["depth"] >= 0
            if _FP["depth"] == 0:
                for e in _FP["items"].values():
                    e.orphan = True
                dead = [e for e in _FP["items"].values() if e.users == 0]
                _FP["items"] = {}
        _pre_free(dead)


def _pre_acquire(X: np.ndarray) -> Any:
    import ctypes

    assert isinstance(X, np.ndarray) and X.ndim == 2
    dead: list = []
    with _LOCK:
        if _FP["depth"] == 0 or X.shape[0] >= 2 ** 31 or X.shape[0] < 1 or X.shape[1] < 1:
            return None
        items = _FP["items"]
        e = items.get(id(X))
        if e is None:
            if len(items) >= _FP["depth"]:
                dead = [v for v in items.values() if v.users == 0]
                for v in dead:
                    del items[id(v.x)]
            e = items[id(X)] = _Pre(X)
        assert e.x is X
        e.users += 1
        threads = _FP["threads"]
    _pre_free(dead)
    with e.lock:
        if not e.handle.value and not e.failed:
            rc = _lib_pre().shira_skx_pre_new(ctypes.byref(e.handle), e.xc.ctypes.data, e.xc.shape[0], e.xc.shape[1],
                                              threads)
            e.failed = rc != 0
            with _LOCK:
                _FP["built"] += 0 if e.failed else 1
    if e.failed:
        _pre_release(e, counted=False)
        return None
    return e


def _pre_release(e: Any, counted: bool = True) -> None:
    with _LOCK:
        assert e.users >= 1
        e.users -= 1
        _FP["trees"] += 1 if counted else 0
        dead = [e] if e.orphan and e.users == 0 else []
    _pre_free(dead)


def presort_stats() -> tuple[int, int]:
    with _LOCK:
        return int(_FP["built"]), int(_FP["trees"])


class presort_threads:

    def __init__(self, n: int) -> None:
        assert 1 <= int(n) <= 64
        self.n = int(n)

    def __enter__(self) -> None:
        self.prev = getattr(_PRE, "n", 1)
        _PRE.n = self.n

    def __exit__(self, *exc: Any) -> None:
        _PRE.n = self.prev


class _Splitter:

    def __init__(self, kind: str, args: tuple) -> None:
        assert kind in ("best", "random") and len(args) == 6
        self.kind, self.args = kind, args
        self.criterion, self.max_features = args[0], int(args[1])
        self.random_state, self.monotonic_cst = args[4], args[5]

    def real(self) -> Any:
        return _REAL["splitters"][self.kind](*self.args)


def _splitter_factory(kind: str) -> Any:
    def make(*args: Any) -> _Splitter:
        return _Splitter(kind, args)
    return make


def _note(shira: bool, reason: str | None = None) -> None:
    with _LOCK:
        _STATE["shira" if shira else "stock"] += 1
        if reason and _STATE["reason"] is None:
            _STATE["reason"] = reason


def _fill(tree: Any, t: dict) -> None:
    n = int(t["feature"].size)
    assert n >= 1 and t["value"].shape[0] == n
    nodes = np.zeros(n, dtype=_NODE_DTYPE)
    for src, dst in (("children_left", "left_child"), ("children_right", "right_child"),
                     ("feature", "feature"), ("threshold", "threshold"), ("impurity", "impurity"),
                     ("n_node_samples", "n_node_samples"),
                     ("weighted_n_node_samples", "weighted_n_node_samples"),
                     ("missing_go_to_left", "missing_go_to_left")):
        nodes[dst] = t[src]
    tree.__setstate__({"max_depth": int(t["depth"].max()), "node_count": n, "nodes": nodes,
                       "values": np.ascontiguousarray(t["value"], dtype=np.float64)})
    assert tree.node_count == n


def _ctx_lib() -> Any:
    import ctypes

    from shira_cpp import sk_exact_cpp as sx

    lib = sx._lib()
    if not hasattr(lib, "_skx_ctx_ready"):
        p = ctypes.c_void_p
        lib.shira_skx_ctx_build.restype = ctypes.c_int
        lib.shira_skx_ctx_build.argtypes = [ctypes.POINTER(p), ctypes.c_int64, p, ctypes.c_int64, ctypes.c_int64, p,
                                            p, ctypes.c_uint32, p, p, ctypes.c_int64] + [p] * 12
        lib.shira_skx_ctx_free.restype, lib.shira_skx_ctx_free.argtypes = None, [p]
        lib.shira_skx_ctx_threads.restype = ctypes.c_int
        lib.shira_skx_ctx_threads.argtypes = [ctypes.POINTER(p), ctypes.c_int32]
        lib._skx_ctx_ready = True
    return lib


MODE_ENV = "SHIRA_SK_MODE"


def tie_path_mode() -> int:
    import os

    m = os.environ.get(MODE_ENV, getattr(_MODE, "default", "ties"))
    if m not in ("exact", "ties"):
        raise ValueError(f"{MODE_ENV} must be 'exact' or 'ties', got {m!r}")
    return 4 if m == "ties" else 0


_MODE = threading.local()
EXACT_BY_DEFAULT = frozenset({"DecisionTreeRegressor"})


class default_mode:

    def __init__(self, estimator: str) -> None:
        assert isinstance(estimator, str) and estimator
        self.mode = "exact" if estimator in EXACT_BY_DEFAULT else "ties"

    def __enter__(self) -> None:
        self.prev = getattr(_MODE, "default", "ties")
        _MODE.default = self.mode

    def __exit__(self, *exc: Any) -> None:
        _MODE.default = self.prev


class gb_session:

    def __enter__(self) -> None:
        import ctypes

        _GB.ctx, _GB.x, _GB.xc, _GB.xgen = ctypes.c_void_p(), None, None, -1
        rc = _ctx_lib().shira_skx_ctx_threads(ctypes.byref(_GB.ctx), getattr(_PRE, "n", 1))
        if rc != 0:
            raise RuntimeError(f"shira_skx_ctx_threads returned {rc}")
        _GB.on = True

    def __exit__(self, *exc: Any) -> None:
        if getattr(_GB, "ctx", None) is not None and _GB.ctx.value:
            _ctx_lib().shira_skx_ctx_free(_GB.ctx)
        _GB.on, _GB.ctx, _GB.x, _GB.xc = False, None, None, None


def _gb_build(X: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int, crit: int, a: tuple, best_first: bool,
              path_mode: int = 0) -> dict:
    import ctypes

    if _GB.x is not X:
        with _LOCK:
            _XGEN[0] += 1
            _GB.xgen = _XGEN[0]
        _GB.x, _GB.xc = X, np.ascontiguousarray(X, dtype=np.float32)
    x = _GB.xc
    n, nf = x.shape
    mss, msl, mwl, md = int(a[0]), int(a[1]), float(a[2]), int(a[3])
    mln, mid = (int(a[4]), float(a[5])) if best_first else (-1, float(a[4]))
    m = int(np.count_nonzero(w))
    cap = max(2 * m - 1, 1)
    cap = min(cap, 2 * mln - 1) if mln > 0 else (min(cap, 2 ** (md + 1) - 1) if md < 40 else cap)
    yy = np.ascontiguousarray(y[:, 0], dtype=np.float64)
    ww = np.ascontiguousarray(w, dtype=np.float64)
    rec_i = np.array([crit, 0, md, mss, msl, int(a[-1]), mln, int(path_mode)], dtype=np.int64)
    rec_d = np.array([mid, 0.0, mwl], dtype=np.float64)
    i64 = [np.empty(cap, np.int64) for _ in range(5)]
    f64 = [np.empty(cap) for _ in range(4)]
    mgl, cnt, path = np.empty(cap, np.uint8), np.zeros(1, np.int64), np.zeros(1, np.uint8)
    q = [z.ctypes.data for z in (i64[0], f64[0], i64[1], i64[2], f64[3], i64[3], f64[1], f64[2], mgl, i64[4], cnt, path)]
    rc = _ctx_lib().shira_skx_ctx_build(ctypes.byref(_GB.ctx), _GB.xgen, x.ctypes.data, n, nf, yy.ctypes.data,
                                        ww.ctypes.data, seed, rec_i.ctypes.data, rec_d.ctypes.data, cap, *q)
    if rc != 0:
        raise RuntimeError(f"shira_skx_ctx_build returned {rc}")
    c = int(cnt[0])
    return {"feature": i64[0][:c], "threshold": f64[0][:c], "children_left": i64[1][:c], "children_right": i64[2][:c],
            "value": f64[3][:c].reshape(c, 1, 1), "n_node_samples": i64[3][:c], "weighted_n_node_samples": f64[1][:c],
            "impurity": f64[2][:c], "missing_go_to_left": mgl[:c], "depth": i64[4][:c], "path": int(path[0])}


class _Builder:

    best_first = False

    def __init__(self, splitter: Any, *args: Any) -> None:
        self.splitter, self.args = splitter, args

    def _stock(self, tree: Any, X: Any, y: Any, sw: Any, mask: Any, reason: str) -> None:
        sp = self.splitter.real() if isinstance(self.splitter, _Splitter) else self.splitter
        _REAL["best" if self.best_first else "depth"](sp, *self.args).build(tree, X, y, sw, mask)
        _note(False, reason)

    def _why_not(self, X: Any, y: Any) -> str | None:
        if not isinstance(self.splitter, _Splitter):
            return "a Splitter object was passed"
        if type(self.splitter.criterion).__name__ not in CRIT:
            return f"criterion {type(self.splitter.criterion).__name__}"
        ok = isinstance(X, np.ndarray) and X.ndim == 2 and X.dtype == np.float32
        return None if ok and isinstance(y, np.ndarray) and y.ndim == 2 else "input layout"

    def build(self, tree: Any, X: Any, y: Any, sample_weight: Any = None,
              missing_values_in_feature_mask: Any = None) -> None:
        why = self._why_not(X, y)
        if why:
            return self._stock(tree, X, y, sample_weight, missing_values_in_feature_mask, why)
        from shira_cpp import sk_exact_cpp as sx

        sp = self.splitter
        a = self.args
        mss, msl, mwl, md = int(a[0]), int(a[1]), float(a[2]), int(a[3])
        mln, mid = (int(a[4]), float(a[5])) if self.best_first else (-1, float(a[4]))
        cname = type(sp.criterion).__name__
        ncls = np.asarray(tree.n_classes, dtype=np.int64) if cname in CLF else None
        w = np.ones(X.shape[0]) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        miss = None if missing_values_in_feature_mask is None else np.asarray(
            missing_values_in_feature_mask, dtype=np.uint8)
        state = sp.random_state.get_state()
        seed = int(sp.random_state.randint(0, RAND_R_MAX))
        gb = (getattr(_GB, "on", False) and cname in ("MSE", "FriedmanMSE") and sp.kind == "best" and y.shape[1] == 1
              and miss is None and sp.monotonic_cst is None)
        pm = tie_path_mode()
        try:
            if gb:
                _fill(tree, _gb_build(X, y, w, seed, CRIT[cname], a + (sp.max_features,), self.best_first, pm))
            else:
                pre = _pre_acquire(X) if sp.kind == "best" else None
                try:
                    out = sx.build_ex(X if pre is None else pre.xc, y, ncls, w[None, :], [seed], CRIT[cname],
                                      0 if sp.kind == "best" else 1,
                                      md, mss, msl, sp.max_features, mln, [mwl], mid, 0.0, miss,
                                      sp.monotonic_cst, threads=getattr(_PRE, "n", 1),
                                      path_mode=pm, pre=None if pre is None else pre.handle)
                finally:
                    if pre is not None:
                        _pre_release(pre)
                _fill(tree, out[0])
        except (RuntimeError, AssertionError, ValueError) as exc:
            sp.random_state.set_state(state)
            return self._stock(tree, X, y, sample_weight, missing_values_in_feature_mask, f"engine: {exc}")
        _note(True)
        return None


class _BestBuilder(_Builder):
    best_first = True


class routed:

    def __enter__(self) -> dict:
        with _LOCK:
            if _STATE["depth"] == 0:
                _STATE.update(shira=0, stock=0, reason=None)
                _tc.DENSE_SPLITTERS = {k: _splitter_factory(k) for k in _REAL["splitters"]}
                _tc.DepthFirstTreeBuilder, _tc.BestFirstTreeBuilder = _Builder, _BestBuilder
            _STATE["depth"] += 1
        return _STATE

    def __exit__(self, *exc: Any) -> None:
        with _LOCK:
            _STATE["depth"] -= 1
            assert _STATE["depth"] >= 0
            if _STATE["depth"] == 0:
                _tc.DENSE_SPLITTERS = dict(_REAL["splitters"])
                _tc.DepthFirstTreeBuilder, _tc.BestFirstTreeBuilder = _REAL["depth"], _REAL["best"]


def counts() -> tuple[int, int, str | None]:
    with _LOCK:
        return int(_STATE["shira"]), int(_STATE["stock"]), _STATE["reason"]
