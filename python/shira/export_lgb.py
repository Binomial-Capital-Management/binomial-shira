from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._lgb_fit import FoldModel, Resolved

K_EPSILON = 1e-15
MISSING_SHIFT = 2


def _g17(v: float) -> str:
    return f"{float(v):.17g}"


def avoid_inf(v: float) -> float:
    if math.isnan(v):
        return 0.0
    return min(max(v, -1e300), 1e300)


def _gain_text(g: float) -> str:
    return "%g" % float(np.float32(g))


def _stats(m: FoldModel, t: int, nl: int) -> tuple:
    ni = nl - 1
    if m.stats is None:
        z = lambda k: " ".join(["0"] * k)
        return z(nl), z(nl), z(ni), z(ni), z(ni), None
    st = m.stats[t]
    L = (st.size + 2) // 5
    assert 5 * L - 2 == st.size and nl <= L
    g = lambda a: " ".join("%g" % float(v) for v in a)
    n = lambda a: " ".join(str(int(v)) for v in a)
    hi = lambda a: " ".join(_g17(v) for v in a)
    return (hi(st[:nl]), n(st[L:L + nl]), g(st[2 * L:2 * L + ni]), g(st[3 * L - 1:3 * L - 1 + ni]),
            n(st[4 * L - 2:4 * L - 2 + ni]), float(st[5 * L - 3]))


def tree_block(m: FoldModel, t: int) -> str:
    nl = int(m.nleaves[t])
    ni = nl - 1
    assert nl >= 1 and ni <= m.nodes.shape[1]
    nodes = m.nodes[t, :ni]
    feat = nodes[:, 0].astype(int).tolist()
    thr = [_g17(float(m.upper[f][b])) for f, b in zip(feat, nodes[:, 1].astype(int).tolist())]
    dec = [(int(m.missing_type[f]) << MISSING_SHIFT) | (2 if int(d) else 0)
           for f, d in zip(feat, m.default_left[t, :ni].tolist())]
    bounds, words = [0], []
    for i, f in enumerate(feat):
        if m.bin2cat is None or m.bin2cat[f] is None:
            continue
        bits = m.cat[t, i]
        cats = [int(m.bin2cat[f][b]) for b in range(len(m.bin2cat[f])) if (int(bits[b >> 5]) >> (b & 31)) & 1]
        cats = [c for c in cats if c >= 0]
        w = [0] * ((max(cats) // 32 + 1) if cats else 1)
        for c in cats:
            w[c >> 5] |= 1 << (c & 31)
        thr[i] = str(len(bounds) - 1)
        dec[i] = 1 | (int(m.missing_type[f]) << MISSING_SHIFT)
        words += w
        bounds.append(len(words))
    biased = t < m.num_class and abs(float(m.init[t % m.num_class])) > K_EPSILON and nl > 1
    lw, lc, iv, iw, ic, sh = _stats(m, t, nl)
    if (m.dart or getattr(m, "sched", False)) and sh is not None:
        shrink = "%g" % sh
    else:
        shrink = "1" if (biased or m.average) else "%g" % m.learning_rate
    lines = [f"Tree={t}", f"num_leaves={nl}", f"num_cat={len(bounds) - 1}",
             "split_feature=" + " ".join(str(f) for f in feat),
             "split_gain=" + " ".join(_gain_text(g) for g in m.gain[t, :ni].tolist()),
             "threshold=" + " ".join(thr),
             "decision_type=" + " ".join(str(d) for d in dec),
             "left_child=" + " ".join(str(int(c)) for c in nodes[:, 2].tolist()),
             "right_child=" + " ".join(str(int(c)) for c in nodes[:, 3].tolist()),
             "leaf_value=" + " ".join(_g17(v) for v in m.leaf[t, :nl].tolist()),
             "leaf_weight=" + lw, "leaf_count=" + lc,
             "internal_value=" + iv, "internal_weight=" + iw, "internal_count=" + ic]
    if len(bounds) > 1:
        lines += ["cat_boundaries=" + " ".join(str(v) for v in bounds), "cat_threshold=" + " ".join(str(v) for v in words)]
    lin = getattr(m, "linear", None)
    if lin is None:
        lines += ["is_linear=0", "shrinkage=" + shrink]
    else:
        const, fptr, lfeat, coeff = lin
        grp = lambda vals, fmt: "".join("".join(fmt(v) + " " for v in vals[fptr[t, j]:fptr[t, j + 1]]) + " " for j in range(nl))
        lines += ["is_linear=1", "leaf_const=" + " ".join(_g17(v) for v in const[t, :nl].tolist()),
                  "num_features=" + " ".join(str(int(fptr[t, j + 1] - fptr[t, j])) for j in range(nl)),
                  "leaf_features=" + grp(lfeat, lambda v: str(int(v))), "leaf_coeff=" + grp(coeff, lambda v: _g17(float(v))),
                  "shrinkage=" + shrink]
    return "\n".join(lines) + "\n\n\n"


def _blocks(m: FoldModel, n_trees: int) -> tuple[str, list[int]]:
    plain = (m.bin2cat is None or all(b is None for b in m.bin2cat)) and getattr(m, "linear", None) is None
    stats = None if m.stats is None else np.asarray(m.stats)
    if not plain or (stats is not None and (stats.ndim != 2 or stats.dtype == object)):
        blocks = [tree_block(m, t) for t in range(n_trees)]
        return "".join(blocks), [len(b) for b in blocks]
    from shira_cpp import lgb_cpp
    nl = np.asarray(m.nleaves[:n_trees])
    t = np.arange(n_trees)
    biased = (t < m.num_class) & (np.abs(np.asarray(m.init, dtype=np.float64)[t % m.num_class]) > K_EPSILON) & (nl > 1)
    if m.dart and stats is not None:
        shrink = stats[:n_trees, -1]
    else:
        shrink = np.where(biased | bool(m.average), 1.0, float(m.learning_rate))
    text, sizes = lgb_cpp.tree_text(nl, m.nodes, m.default_left, m.leaf, m.gain, stats, shrink, m.upper, m.missing_type)
    assert len(sizes) == n_trees
    return text, sizes.tolist()


def _importances(m: FoldModel, n_trees: int, names: list[str]) -> str:
    nl = np.asarray(m.nleaves[:n_trees]).astype(np.int64)
    used = np.arange(m.nodes.shape[1])[None, :] < (nl - 1)[:, None]
    cnt = np.bincount(np.asarray(m.nodes[:n_trees, :, 0])[used].astype(np.int64), minlength=len(names))
    order = sorted((j for j in range(len(names)) if cnt[j] > 0), key=lambda j: -int(cnt[j]))
    assert int(cnt.sum()) == int(np.maximum(m.nleaves[:n_trees] - 1, 0).sum())
    return "".join(f"{names[j]}={int(cnt[j])}\n" for j in order)


def to_text(m: FoldModel, rs: Resolved, names: list[str], n_iter: int | None = None,
            pandas_categorical: list | None = None) -> str:
    from lightgbm.basic import _dump_pandas_categorical
    n_c = m.num_class
    rounds = m.n_trees // n_c
    keep = rounds if not n_iter or n_iter <= 0 else min(int(n_iter), rounds)
    n_trees = keep * n_c
    assert 1 <= n_trees <= m.n_trees and len(names) == len(m.upper)
    blocks, sizes = _blocks(m, n_trees)
    head = ["tree", "version=v4", f"num_class={n_c}", f"num_tree_per_iteration={n_c}", "label_index=0",
            f"max_feature_idx={len(names) - 1}", rs.objective_line]
    if m.average:
        head.append("average_output")
    head += ["feature_names=" + " ".join(names),
            "feature_infos=" + " ".join(m.feature_infos), "tree_sizes=" + " ".join(str(v) for v in sizes)]
    return ("\n".join(head) + "\n\n" + blocks + "end of trees\n\nfeature_importances:\n"
            + _importances(m, n_trees, names) + "\n" + rs.params_block + _dump_pandas_categorical(pandas_categorical))


def engine_predict(m: FoldModel, bins: np.ndarray, n_iter: int | None = None) -> np.ndarray:
    from shira_cpp import lgb_cpp
    n_c = m.num_class
    n_trees = m.n_trees if not n_iter else min(m.n_trees, int(n_iter) * n_c)
    acc = np.zeros((bins.shape[0], n_c), dtype=np.float64)
    res = {"nleaves": m.nleaves[None], "leaf": m.leaf[None], "nodes": m.nodes[None],
           "default_left": m.default_left[None], "cat": None}
    prep = {"mtype": m.info[None, :, 1], "dbin": m.info[None, :, 2], "nbin": m.info[None, :, 0], "iscat": None}
    for t in range(n_trees):
        acc[:, t % n_c] = acc[:, t % n_c] + lgb_cpp.tree_output(res, 0, t, bins, prep)
    assert acc.shape[0] == bins.shape[0]
    return acc[:, 0] if n_c == 1 else acc


def bin_rows(m: FoldModel, x: np.ndarray) -> np.ndarray:
    from shira_cpp import binning_cpp
    ptrs = np.concatenate([[0], np.cumsum([u.size for u in m.upper])]).astype(np.int32)
    xa = np.ascontiguousarray(x, dtype=np.float64)
    if int(np.asarray(m.info).reshape(-1, 4)[:, 0].max()) > 256:
        out = binning_cpp.lgb_bin2(xa, ptrs, np.concatenate(m.upper), m.info, code_bytes=2)[0]
    else:
        out = binning_cpp.lgb_bin(xa, ptrs, np.concatenate(m.upper), m.info)
    assert out.shape == x.shape
    return out


def to_forest(booster: Any, nf: int) -> Any:
    assert nf >= 1
    assert booster is not None
    from shira_cpp import outputs_cpp
    return outputs_cpp.from_lightgbm(booster, nf)
