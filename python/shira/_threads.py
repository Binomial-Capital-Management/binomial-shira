from __future__ import annotations

import os
import platform
import sys

def affinity_cores() -> int:
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    assert n >= 1
    return n


MACHINE: str = f"{sys.platform}_{platform.machine().lower()}_{affinity_cores()}"
ENGINES = ("catboost", "lightgbm", "hgb", "xgboost")
TABLE: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {
    "darwin_arm64_28": {
        "catboost": ((30_000, 8), (60_000, 8)),
        "lightgbm": ((60_000, 8), (100_000, 8), (1_000_000, 8), (2_000_000, 8)),
        "hgb": ((60_000, 8), (100_000, 8), (300_000, 8), (1_000_000, 4), (2_000_000, 4)),
        "xgboost": ((60_000, 8), (100_000, 8), (1_000_000, 12), (2_000_000, 12)),
    },
    "linux_x86_64_8": {
        "hgb": ((100_000, 8), (1_000_000, 6)),
    },
}
USE_TABLE: bool = os.environ.get("SHIRA_THREAD_TABLE", "1") != "0"


def _env_cap() -> int | None:
    for key in ("SHIRA_THREADS", "OMP_NUM_THREADS"):
        v = os.environ.get(key, "").strip()
        if v.isdigit() and int(v) > 0:
            return int(v)
    return None


_CPU: list[int] = []


def _cpu_count() -> int:
    if not _CPU:
        from joblib import cpu_count

        _CPU.append(max(1, int(cpu_count())))
    assert _CPU[0] >= 1
    return _CPU[0]


def env_budget() -> int:
    from joblib import parallel

    backend, n_jobs = parallel.get_active_backend()
    if n_jobs is not None:
        return 1
    cap = _env_cap()
    if int(getattr(backend, "nesting_level", 0) or 0) >= 1:
        return max(1, min(64, cap)) if cap is not None else 1
    cores = _cpu_count()
    if cap is not None:
        cores = min(cores, cap)
    assert cores >= 1
    return max(1, min(64, cores))


def xgb_default_nthread() -> int:
    procs = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    omp = os.environ.get("OMP_NUM_THREADS", "").split(",")[0].strip()
    n = min(procs, int(omp)) if omp.isdigit() and int(omp) > 0 else procs
    assert n >= 1
    return n


def shira_cap() -> int | None:
    raw = os.environ.get("SHIRA_THREADS", "")
    try:
        n = int(raw) if raw.strip() else 0
    except ValueError:
        return None
    return n if n >= 1 else None


def table_cap(engine: str, rows: int, machine: str | None = None) -> int | None:
    assert engine in ENGINES and rows >= 0
    bands = TABLE.get(MACHINE if machine is None else machine, {}).get(engine, ())
    for max_rows, threads in bands:
        if rows <= max_rows:
            assert threads >= 1
            return threads
    return None


_PCORES: list[int | None] = []


def performance_cores() -> int | None:
    if not _PCORES:
        val = None
        if sys.platform == "darwin":
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c"))

            def get(name: str) -> int:
                v, size = ctypes.c_int(0), ctypes.c_size_t(ctypes.sizeof(ctypes.c_int))
                rc = libc.sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(size), None, ctypes.c_size_t(0))
                return int(v.value) if rc == 0 else 0

            perf, eff = get("hw.perflevel0.physicalcpu"), get("hw.perflevel1.physicalcpu")
            val = perf if perf > 0 and eff > 0 else None
        _PCORES.append(val)
    return _PCORES[0]


def default_threads(engine: str, rows: int, stock_default: int) -> int:
    assert engine in ENGINES and stock_default >= 1
    n = stock_default
    if USE_TABLE:
        n = min(n, affinity_cores())
    cap = shira_cap()
    if cap is not None:
        n = min(n, cap)
    tc = table_cap(engine, int(rows)) if USE_TABLE else None
    if tc is not None:
        n = min(n, tc)
    pc = performance_cores() if USE_TABLE else None
    if pc is not None:
        n = min(n, pc)
    assert 1 <= n <= stock_default
    return n


X86_SMALL_L2_BYTES = 2 * 1024 * 1024
_L2: list[int] = []


def l2_per_core_bytes() -> int:
    if not _L2:
        val = 0
        base = "/sys/devices/system/cpu/cpu0/cache"
        for k in range(8):
            try:
                with open(f"{base}/index{k}/level") as fh:
                    if fh.read().strip() != "2":
                        continue
                with open(f"{base}/index{k}/size") as fh:
                    raw = fh.read().strip().upper()
            except OSError:
                continue
            mult = 1024 * 1024 if raw.endswith("M") else (1024 if raw.endswith("K") else 1)
            digits = raw.rstrip("KM")
            val = int(digits) * mult if digits.isdigit() else 0
            break
        _L2.append(val)
    return _L2[0]


def x86_small_l2() -> bool:
    if platform.machine().lower() not in ("x86_64", "amd64", "x86", "i686"):
        return False
    l2 = l2_per_core_bytes()
    assert l2 >= 0
    return l2 == 0 or l2 <= X86_SMALL_L2_BYTES


def explicit_threads(requested: int) -> int:
    assert requested >= 1
    pc = performance_cores() if USE_TABLE else None
    n = min(requested, pc) if pc is not None else requested
    assert 1 <= n <= requested
    return n
