from __future__ import annotations

import ctypes
import os
import threading
from collections import OrderedDict

import numpy as np

from ._lib import check, lib, ptr

GRID = 4096
REF_ROWS = 512
MAX_STREAMS = 4
CAP_BYTES = 256 << 20
MAX_CANDIDATES = 8
PROBE_MIN, PROBE_MAX = 64, 4096
MAX_PROBES = 8
MAX_P = 126
MIN_ROWS = int(os.environ.get("SHIRA_REUSE_MIN", "16384"))
STATS = {"hit": 0, "new": 0, "fresh_blocks": 0, "cached_blocks": 0, "skipped": 0, "probe": 0}
_LOCK = threading.Lock()
_STREAMS: OrderedDict[int, "_Stream"] = OrderedDict()
_PROBES: list[tuple[int, int, np.ndarray]] = []


def enabled(n: int, p: int) -> bool:
    return os.environ.get("SHIRA_REUSE", "1") != "0" and n >= MIN_ROWS and n > p and p <= MAX_P


def payload_len(kind: str, p: int) -> int:
    q = p + 2
    return 2 * q * q if kind == "acc" else q * q


class _Stream:

    def __init__(self, h: np.ndarray, p: int, ref: np.ndarray, ref_y: float) -> None:
        self.h, self.n, self.p = h, int(h.size), p
        self.ref, self.ref_y = np.ascontiguousarray(ref, dtype=np.float64), float(ref_y)
        self.blocks: dict[str, dict[int, tuple[np.ndarray, int]]] = {}
        self.bytes = h.nbytes

    def extend(self, tail: np.ndarray) -> None:
        need = self.n + tail.size
        if need > self.h.size:
            grown = np.empty(max(need, 2 * self.h.size), dtype=np.uint64)
            grown[:self.n] = self.h[:self.n]
            self.bytes += grown.nbytes - self.h.nbytes
            self.h = grown
        self.h[self.n:need] = tail
        self.n = need

    def match(self, h: np.ndarray, cand: list[int]) -> int | None:
        for a in cand:
            m = min(h.size, self.n - a)
            if np.array_equal(self.h[a:a + m], h[:m]):
                if m < h.size:
                    self.extend(h[m:])
                return a
        return None


def _hash(X: np.ndarray, y: np.ndarray, threads: int) -> np.ndarray:
    n, p = X.shape
    out = np.empty(n, dtype=np.uint64)
    check(lib().shira_row_hash(ptr(X), ptr(y), n, p, int(threads), out.ctypes.data), "shira_row_hash")
    return out


def _evict() -> None:
    while len(_STREAMS) > MAX_STREAMS or (len(_STREAMS) > 1 and sum(s.bytes for s in _STREAMS.values()) > CAP_BYTES):
        _STREAMS.popitem(last=False)


def _probe_hit(X: np.ndarray, y: np.ndarray) -> bool:
    p = X.shape[1]
    with _LOCK:
        probes = [(st, s) for q, st, s in _PROBES if q == p]
    if not probes:
        return False
    head = _hash(X[:max(st for st, _ in probes)], y[:max(st for st, _ in probes)], 1)
    return any(bool(np.isin(head[:st], s).any()) for st, s in probes)


def _record_probe(X: np.ndarray, y: np.ndarray) -> None:
    n, p = X.shape
    st = int(min(PROBE_MAX, max(PROBE_MIN, int(np.sqrt(n)))))
    out = np.empty((n + st - 1) // st, dtype=np.uint64)
    check(lib().shira_row_hash_strided(ptr(X), ptr(y), n, p, st, out.ctypes.data), "shira_row_hash_strided")
    with _LOCK:
        _PROBES.append((p, st, out))
        del _PROBES[:-MAX_PROBES]
    STATS["probe"] += 1


def _find(X: np.ndarray, y: np.ndarray, threads: int) -> tuple[_Stream | None, int, np.ndarray | None]:
    h0 = np.uint64(_hash(X[:1], y[:1], 1)[0])
    p = X.shape[1]
    with _LOCK:
        cands = [(k, s, np.flatnonzero(s.h[:s.n] == h0)[:MAX_CANDIDATES].tolist())
                 for k, s in reversed(list(_STREAMS.items())) if s.p == p]
    cands = [c for c in cands if c[2]]
    if not cands:
        return None, 0, None
    h = _hash(X, y, threads)
    with _LOCK:
        for k, s, cand in cands:
            a = s.match(h, cand)
            if a is not None:
                if k in _STREAMS:
                    _STREAMS.move_to_end(k)
                STATS["hit"] += 1
                return s, a, h
    return None, 0, h


def _compute(kind: str, X: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray, ref: np.ndarray, ref_y: float,
             threads: int, hashes: np.ndarray | None) -> np.ndarray:
    p = X.shape[1]
    fresh = np.empty((lo.size, payload_len(kind, p)))
    fn = lib().shira_acc_segments if kind == "acc" else lib().shira_qr_segments
    hp = hashes.ctypes.data if hashes is not None else None
    check(fn(ptr(X), ptr(y), p, lo.ctypes.data, hi.ctypes.data, int(lo.size), ptr(ref), ref_y, int(threads), ptr(fresh),
             hp), f"reuse {kind} segments")
    return fresh


def _window(kind: str, X: np.ndarray, y: np.ndarray, threads: int) -> tuple[list[int], _Stream, list[np.ndarray]] | None:
    n, p = X.shape
    s, a, h = _find(X, y, threads)
    if s is None and h is None and not _probe_hit(X, y):
        _record_probe(X, y)
        return None
    if s is None:
        k = min(REF_ROWS, n)
        ref, ref_y = X[:k].mean(axis=0), float(y[:k].mean())
    else:
        ref, ref_y = s.ref, s.ref_y
    bs = np.arange(a // GRID, (a + n - 1) // GRID + 1, dtype=np.int64)
    lo, hi = np.maximum(a, bs * GRID) - a, np.minimum(a + n, (bs + 1) * GRID) - a
    full = ((hi - lo) == GRID).tolist()
    store = s.blocks.get(kind, {}) if s is not None else {}
    got = [store.get(b) if f else None for b, f in zip(bs.tolist(), full)]
    todo = np.array([i for i, g in enumerate(got) if g is None], dtype=np.int64)
    hashes = np.empty(n, dtype=np.uint64) if h is None else None
    fresh = _compute(kind, X, y, lo[todo], hi[todo], ref, ref_y, threads, hashes) if todo.size else np.empty((0, 1))
    base, step = fresh.ctypes.data, fresh.shape[1] * 8
    with _LOCK:
        if s is None:
            s = _Stream(h if h is not None else hashes, p, ref, ref_y)
            _STREAMS[id(s)] = s
            STATS["new"] += 1
        dst = s.blocks.setdefault(kind, {})
        for j, i in enumerate(todo.tolist()):
            got[i] = (fresh[j], base + j * step)
            if full[i] and s.bytes + step <= CAP_BYTES:
                dst[int(bs[i])] = got[i]
                s.bytes += step
        _evict()
    STATS["fresh_blocks"] += int(todo.size)
    STATS["cached_blocks"] += len(got) - int(todo.size)
    assert len(got) == bs.size and all(g is not None for g in got)
    return [g[1] for g in got], s, [g[0] for g in got]


def ridge_totals(X: np.ndarray, y: np.ndarray, threads: int) -> tuple | None:
    from . import linear as sl

    n, p = X.shape
    if not enabled(n, p):
        STATS["skipped"] += 1
        return None
    assert X.flags.c_contiguous and X.dtype == np.float64 and y.flags.c_contiguous and y.shape == (n,)
    win = _window("acc", X, y, threads)
    if win is None:
        return None
    addrs, s, keep = win
    out = np.empty(lib().shira_totals_len(p))
    arr = (ctypes.c_void_p * len(addrs))(*addrs)
    check(lib().shira_acc_merge(arr, len(addrs), p, ptr(out)), "shira_acc_merge")
    assert len(keep) == len(addrs)
    return sl.unpack(out, p), s.ref, s.ref_y


def linreg(X: np.ndarray, y: np.ndarray, threads: int) -> dict | None:
    n, p = X.shape
    if not enabled(n, p):
        STATS["skipped"] += 1
        return None
    assert X.flags.c_contiguous and X.dtype == np.float64 and y.flags.c_contiguous and y.shape == (n,)
    win = _window("qr", X, y, threads)
    if win is None:
        return None
    addrs, s, keep = win
    coef, icpt, sing, rank = np.empty(p), np.empty(1), np.empty(p), np.zeros(1, dtype=np.int32)
    colss = np.empty(p)
    arr = (ctypes.c_void_p * len(addrs))(*addrs)
    check(lib().shira_linreg_from_parts(arr, len(addrs), p, n, ptr(s.ref), s.ref_y, ptr(coef), ptr(icpt), ptr(sing),
                                        ptr(rank), ptr(colss)), "shira_linreg_from_parts")
    assert len(keep) == len(addrs)
    return {"coef": coef, "intercept": float(icpt[0]), "singular": sing, "rank": int(rank[0]), "colss": colss}


def clear() -> None:
    with _LOCK:
        _STREAMS.clear()
        _PROBES.clear()
