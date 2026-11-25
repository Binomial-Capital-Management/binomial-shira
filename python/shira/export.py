from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._lib import check as _check
from ._lib import lib as _lib
from .model import Node, predict_margin

F32 = np.float32
NO_PARENT = 2147483647
VERSION = [3, 2, 0]
OBJECTIVE_JSON = {
    "reg:pseudohubererror": {"name": "reg:pseudohubererror", "pseudo_huber_param": {"huber_slope": "1"}},
    "binary:logistic": {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}},
}
MAX_TREES = 1_000_000
MAX_NODES = 1 << 20
CODE_THREADS = 6


class LazyTrees(Sequence):

    def __init__(self, nodes: np.ndarray, leaf: np.ndarray, count: np.ndarray) -> None:
        assert nodes.ndim == 3 and nodes.shape[2] == 5 and leaf.shape == nodes.shape[:2]
        assert count.shape == (nodes.shape[0],)
        self.nodes, self.leaf, self.count = nodes, leaf, count
        self.cover: np.ndarray | None = None
        self.gain: np.ndarray | None = None
        self.bw: np.ndarray | None = None
        self._built: dict[int, list[Node]] = {}

    def __len__(self) -> int:
        return int(self.count.size)

    def __getitem__(self, i: int | slice) -> list[Node] | LazyTrees:
        if isinstance(i, slice):
            out = LazyTrees(self.nodes[i], self.leaf[i], self.count[i])
            if self.cover is not None and self.gain is not None:
                out.cover, out.gain = self.cover[i], self.gain[i]
            if self.bw is not None:
                out.bw = self.bw[i]
            return out
        i = int(i)
        if not -len(self) <= i < len(self):
            raise IndexError(i)
        i %= len(self)
        if i not in self._built:
            tree = []
            for v in range(int(self.count[i])):
                f, cut, d, lft, rgt = (int(a) for a in self.nodes[i, v])
                tree.append(Node(leaf=float(self.leaf[i, v])) if f < 0 else
                            Node(feature=f, cut=cut, dir=d, left=lft, right=rgt))
            self._built[i] = tree
        return self._built[i]

    def internal(self) -> tuple[np.ndarray, np.ndarray]:
        used = np.arange(self.nodes.shape[1])[None, :] < self.count[:, None]
        f = self.nodes[:, :, 0]
        keep = used & (f >= 0)
        return f[keep], self.nodes[:, :, 1][keep]


@dataclass
class ShiraModel:
    objective: str
    base_score: float
    base_margin: float
    num_feature: int
    ip: np.ndarray
    cuts: np.ndarray
    trees: Sequence = field(default_factory=list)
    num_class: int = 1
    objective_json: dict | None = None

    def head(self, n_rounds: int) -> ShiraModel:
        k = n_rounds * self.num_class
        assert 0 <= k <= len(self.trees) and self.num_class >= 1
        return ShiraModel(self.objective, self.base_score, self.base_margin, self.num_feature,
                          self.ip, self.cuts, self.trees[:k], self.num_class, self.objective_json)

    def codes(self, x: np.ndarray, clamp: bool = False) -> np.ndarray:
        assert x.ndim == 2 and x.shape[1] == self.num_feature, f"expected {self.num_feature} features"
        assert self.ip.size == self.num_feature + 1
        xc = np.ascontiguousarray(x, dtype=np.float32)
        ip = np.ascontiguousarray(self.ip, dtype=np.int64)
        cv = np.ascontiguousarray(self.cuts, dtype=np.float32)
        out = np.empty(x.shape, dtype=np.uint16)
        _check(_lib().shira_xgb_model_codes(xc.ctypes.data, xc.shape[0], xc.shape[1], ip.ctypes.data, cv.ctypes.data,
                                            1 if clamp else 0, CODE_THREADS, out.ctypes.data), "shira_xgb_model_codes")
        return out

    def internal(self) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.trees, LazyTrees):
            return self.trees.internal()
        pairs = [(nd.feature, nd.cut) for t in self.trees for nd in t if nd.leaf is None]
        arr = np.array(pairs, dtype=np.int64).reshape(-1, 2)
        return arr[:, 0], arr[:, 1]

    def below_min_split(self, x: np.ndarray) -> bool:
        f, cut = self.internal()
        feats = np.unique(f[cut == 0]).tolist()
        assert x.ndim == 2 and x.shape[1] == self.num_feature
        return any(bool(np.any(x[:, f] < self.cuts[self.ip[f]])) for f in feats)

    def predict_margin(self, x: np.ndarray, n_trees: int | None = None, clamp: bool = False) -> np.ndarray:
        assert self.num_class == 1, "multiclass margins: use stock XGBoost on the exported model"
        assert clamp or not self.below_min_split(x), "values below a cut-0 split: use stock XGBoost"
        trees = self.trees if n_trees is None else self.trees[:n_trees]
        return predict_margin(trees, self.codes(x, clamp), float(self.base_margin))

    def predict(self, x: np.ndarray, output_margin: bool = False, n_trees: int | None = None,
                clamp: bool = False) -> np.ndarray:
        m = self.predict_margin(x, n_trees, clamp)
        return m if output_margin else transform(self.objective, m)


def _expf(z: np.ndarray) -> np.ndarray:
    if not sys.platform.startswith("linux"):
        return np.exp(z)
    import ctypes

    from ._lib import lib, ptr
    zc = np.ascontiguousarray(z, dtype=F32)
    out = np.empty_like(zc)
    fn = lib().shira_expf_many
    fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p], ctypes.c_int
    assert fn(ptr(zc), zc.size, ptr(out)) == 0
    return out


def transform(objective: str, margin: np.ndarray) -> np.ndarray:
    assert margin.dtype == F32 and can_transform(objective), objective
    if objective in SIGMOID_LINK:
        z = np.minimum(-margin, F32(88.7))
        return (F32(1) / (_expf(z) + F32(1))).astype(F32)
    return margin.copy()


IDENTITY_LINK = frozenset({"reg:pseudohubererror", "reg:squarederror", "reg:absoluteerror", "reg:quantileerror",
                           "binary:logitraw"})
SIGMOID_LINK = frozenset({"binary:logistic", "reg:logistic"})


def can_transform(objective: str) -> bool:
    return objective in SIGMOID_LINK or objective in IDENTITY_LINK


def _tree_json(tree: list[Node], sm: ShiraModel, tid: int) -> dict:
    n = len(tree)
    assert 0 < n <= MAX_NODES and tree[0] is not None
    parents = [NO_PARENT] * n
    cond, feat, dleft, lch, rch, bw = [], [], [], [], [], []
    for i, nd in enumerate(tree):
        if nd.leaf is not None:
            cond.append(float(F32(nd.leaf)))
            bw.append(float(F32(nd.leaf)))
            feat.append(0)
            dleft.append(0)
            lch.append(-1)
            rch.append(-1)
            continue
        cj = sm.cuts[sm.ip[nd.feature]:sm.ip[nd.feature + 1]]
        assert 0 <= nd.cut < cj.size, f"tree {tid} node {i}: cut {nd.cut} outside 0..{cj.size - 1}"
        cond.append(float(cj[nd.cut]))
        bw.append(0.0)
        feat.append(int(nd.feature))
        dleft.append(1 if nd.dir == 1 else 0)
        lch.append(int(nd.left))
        rch.append(int(nd.right))
        parents[nd.left], parents[nd.right] = i, i
    assert parents[0] == NO_PARENT and all(p != NO_PARENT for p in parents[1:]), f"tree {tid}: not one tree"
    zeros = [0.0] * n
    return {"base_weights": bw, "categories": [], "categories_nodes": [], "categories_segments": [],
            "categories_sizes": [], "default_left": dleft, "id": tid, "left_children": lch,
            "loss_changes": zeros, "parents": parents, "right_children": rch, "split_conditions": cond,
            "split_indices": feat, "split_type": [0] * n, "sum_hessian": zeros,
            "tree_param": {"num_deleted": "0", "num_feature": str(sm.num_feature), "num_nodes": str(n),
                           "size_leaf_vector": "1"}}


def _tree_json_arrays(nd: np.ndarray, lf: np.ndarray, sm: ShiraModel, tid: int, cover: np.ndarray | None = None,
                      gain: np.ndarray | None = None, bw: np.ndarray | None = None) -> dict:
    n = nd.shape[0]
    f, cut, d, lc, rc = (nd[:, j].astype(np.int64) for j in range(5))
    is_leaf = f < 0
    fi = np.where(is_leaf, 0, f)
    width = sm.ip[fi + 1] - sm.ip[fi]
    assert 0 < n <= MAX_NODES and bool(np.all(is_leaf | ((cut >= 0) & (cut < width)))), f"tree {tid}: cut range"
    cond = np.where(is_leaf, lf, sm.cuts[sm.ip[fi] + np.where(is_leaf, 0, cut)]).astype(np.float64)
    parents = np.full(n, NO_PARENT, dtype=np.int64)
    idx = np.flatnonzero(~is_leaf)
    parents[lc[idx]] = idx
    parents[rc[idx]] = idx
    assert parents[0] == NO_PARENT and bool(np.all(parents[1:] != NO_PARENT)), f"tree {tid}: not one tree"
    zeros = [0.0] * n
    bwv = np.where(is_leaf, lf, F32(0)) if bw is None else np.asarray(bw, F32)
    return {"base_weights": bwv.astype(np.float64).tolist(), "categories": [],
            "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
            "default_left": np.where(is_leaf, 0, d == 1).astype(np.int64).tolist(), "id": tid,
            "left_children": np.where(is_leaf, -1, lc).tolist(),
            "loss_changes": zeros if gain is None else np.where(is_leaf, 0.0, gain.astype(np.float64)).tolist(),
            "parents": parents.tolist(), "right_children": np.where(is_leaf, -1, rc).tolist(),
            "split_conditions": cond.tolist(), "split_indices": fi.tolist(), "split_type": [0] * n,
            "sum_hessian": zeros if cover is None else cover.astype(np.float64).tolist(),
            "tree_param": {"num_deleted": "0", "num_feature": str(sm.num_feature), "num_nodes": str(n),
                           "size_leaf_vector": "1"}}


def _trees_json(sm: ShiraModel, k: int) -> list[dict]:
    if isinstance(sm.trees, LazyTrees):
        lt = sm.trees
        stats = lt.cover is not None and lt.gain is not None
        return [_tree_json_arrays(lt.nodes[t, :int(lt.count[t])], lt.leaf[t, :int(lt.count[t])], sm, t,
                                  lt.cover[t, :int(lt.count[t])] if stats else None,
                                  lt.gain[t, :int(lt.count[t])] if stats else None,
                                  None if lt.bw is None else lt.bw[t, :int(lt.count[t])])
                for t in range(k)]
    return [_tree_json(sm.trees[t], sm, t) for t in range(k)]


def to_dict(sm: ShiraModel, n_trees: int | None = None) -> dict:
    obj = sm.objective_json or OBJECTIVE_JSON.get(sm.objective)
    assert obj is not None, f"export: no objective config for {sm.objective}"
    C = sm.num_class
    k = len(sm.trees) if n_trees is None else n_trees
    assert 0 <= k <= min(len(sm.trees), MAX_TREES) and k % C == 0
    trees = _trees_json(sm, k)
    gbm = {"model": {"cats": {"enc": [], "feature_segments": [], "sorted_idx": []},
                     "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": str(k)},
                     "iteration_indptr": list(range(0, k + 1, C)), "tree_info": [t % C for t in range(k)],
                     "trees": trees}, "name": "gbtree"}
    base = ",".join([repr(float(F32(sm.base_score)))] * C)
    lmp = {"base_score": f"[{base}]", "boost_from_average": "1", "num_class": str(C) if C > 1 else "0",
           "num_feature": str(sm.num_feature), "num_target": "1"}
    learner = {"attributes": {}, "feature_names": [], "feature_types": [], "gradient_booster": gbm,
               "learner_model_param": lmp, "objective": obj}
    return {"learner": learner, "version": VERSION}


def to_json(sm: ShiraModel, n_trees: int | None = None) -> bytes:
    return json.dumps(to_dict(sm, n_trees), separators=(",", ":")).encode()


_UBJ_DTYPE = {"d": ">f4", "l": ">i4", "L": ">i8", "U": "u1"}
_UBJ_MAX_ITEMS = 50_000_000


class _RawUBJ:
    __slots__ = ("data",)

    def __init__(self, data: bytes) -> None:
        self.data = data


class _Typed:
    __slots__ = ("marker", "data")

    def __init__(self, marker: str, values) -> None:
        self.marker = marker
        self.data = np.ascontiguousarray(values, dtype=_UBJ_DTYPE[marker])


def _ubj_len(n: int) -> bytes:
    return b"L" + int(n).to_bytes(8, "big", signed=True)


def _ubj_scalar(v) -> bytes:
    if isinstance(v, _Typed):
        return b"[$" + v.marker.encode() + b"#" + _ubj_len(v.data.size) + v.data.tobytes()
    if isinstance(v, str):
        b = v.encode()
        return b"S" + _ubj_len(len(b)) + b
    if isinstance(v, (bool, np.bool_)):
        return b"T" if v else b"F"
    if isinstance(v, (int, np.integer)):
        return b"L" + int(v).to_bytes(8, "big", signed=True)
    if isinstance(v, (float, np.floating)):
        return b"D" + np.array(float(v), dtype=">f8").tobytes()
    raise TypeError(f"UBJSON export: unsupported value {type(v).__name__}")


def _ubj(doc: dict) -> bytes:
    out, stack, closers, items = [b"{"], [iter(doc.items())], [b"}"], 0
    while stack:
        nxt = next(stack[-1], None)
        if nxt is None:
            stack.pop()
            out.append(closers.pop())
            continue
        items += 1
        assert items <= _UBJ_MAX_ITEMS, "UBJSON export: document too large"
        key, val = nxt
        if key is not None:
            kb = key.encode()
            out.append(_ubj_len(len(kb)) + kb)
        if isinstance(val, _RawUBJ):
            out.append(val.data)
        elif isinstance(val, dict):
            out.append(b"{")
            stack.append(iter(val.items()))
            closers.append(b"}")
        elif isinstance(val, list):
            out.append(b"[#" + _ubj_len(len(val)))
            stack.append(iter([(None, e) for e in val]))
            closers.append(b"")
        else:
            out.append(_ubj_scalar(val))
    return b"".join(out)


def _tree_ubj_arrays(nd: np.ndarray, lf: np.ndarray, sm: ShiraModel, tid: int, cover: np.ndarray | None = None,
                     gain: np.ndarray | None = None, bw: np.ndarray | None = None) -> dict:
    n = nd.shape[0]
    f, cut, d, lc, rc = (nd[:, j].astype(np.int64) for j in range(5))
    is_leaf = f < 0
    fi = np.where(is_leaf, 0, f)
    width = sm.ip[fi + 1] - sm.ip[fi]
    assert 0 < n <= MAX_NODES and bool(np.all(is_leaf | ((cut >= 0) & (cut < width)))), f"tree {tid}: cut range"
    cond = np.where(is_leaf, lf, sm.cuts[sm.ip[fi] + np.where(is_leaf, 0, cut)]).astype(F32)
    parents = np.full(n, NO_PARENT, dtype=np.int64)
    idx = np.flatnonzero(~is_leaf)
    parents[lc[idx]] = idx
    parents[rc[idx]] = idx
    assert parents[0] == NO_PARENT and bool(np.all(parents[1:] != NO_PARENT)), f"tree {tid}: not one tree"
    zf, e32, e64 = np.zeros(n, F32), np.zeros(0, np.int32), np.zeros(0, np.int64)
    bwv = np.where(is_leaf, lf, F32(0)) if bw is None else np.asarray(bw, F32)
    return {"base_weights": _Typed("d", bwv), "categories": _Typed("l", e32),
            "categories_nodes": _Typed("l", e32), "categories_segments": _Typed("L", e64),
            "categories_sizes": _Typed("L", e64), "default_left": _Typed("U", np.where(is_leaf, 0, d == 1)),
            "id": tid, "left_children": _Typed("l", np.where(is_leaf, -1, lc)),
            "loss_changes": _Typed("d", zf if gain is None else np.where(is_leaf, F32(0), gain).astype(F32)),
            "parents": _Typed("l", parents), "right_children": _Typed("l", np.where(is_leaf, -1, rc)),
            "split_conditions": _Typed("d", cond), "split_indices": _Typed("l", fi),
            "split_type": _Typed("U", np.zeros(n, np.uint8)),
            "sum_hessian": _Typed("d", zf if cover is None else np.asarray(cover, F32)),
            "tree_param": {"num_deleted": "0", "num_feature": str(sm.num_feature), "num_nodes": str(n),
                           "size_leaf_vector": "1"}}


def trees_ubj(sm: ShiraModel, k: int) -> bytes:
    lt = sm.trees
    assert isinstance(lt, LazyTrees) and 0 <= k <= len(lt)
    nodes = np.ascontiguousarray(lt.nodes[:k], dtype=np.int32)
    leaf = np.ascontiguousarray(lt.leaf[:k], dtype=np.float32)
    count = np.ascontiguousarray(lt.count[:k], dtype=np.int32)
    ip = np.ascontiguousarray(sm.ip, dtype=np.int64)
    cv = np.ascontiguousarray(sm.cuts, dtype=np.float32)
    maxn = int(nodes.shape[1]) if k else 1
    stats = lt.cover is not None and lt.gain is not None
    cover = np.ascontiguousarray(lt.cover[:k], dtype=np.float32) if stats else None
    gain = np.ascontiguousarray(lt.gain[:k], dtype=np.float32) if stats else None
    assert cover is None or (cover.shape == nodes.shape[:2] and gain is not None and gain.shape == cover.shape)
    bw = None if lt.bw is None else np.ascontiguousarray(lt.bw[:k], dtype=np.float32)
    assert bw is None or bw.shape == nodes.shape[:2]
    args = (nodes.ctypes.data, leaf.ctypes.data, count.ctypes.data, k, maxn, ip.ctypes.data, cv.ctypes.data,
            int(sm.num_feature), None if cover is None else cover.ctypes.data, None if gain is None else gain.ctypes.data,
            None if bw is None else bw.ctypes.data)
    need = int(_lib().shira_xgb_trees_ubj_stats(*args, None, 0))
    assert need > 0, f"shira_xgb_trees_ubj_stats failed rc={need}"
    buf = np.empty(need, dtype=np.uint8)
    got = int(_lib().shira_xgb_trees_ubj_stats(*args, buf.ctypes.data, need))
    assert got == need, (got, need)
    return buf.tobytes()


def round_margins(sm: ShiraModel, codes: np.ndarray, n_rounds: int) -> np.ndarray:
    lt = sm.trees
    assert isinstance(lt, LazyTrees) and sm.num_class == 1 and 0 <= n_rounds <= len(lt)
    c = np.ascontiguousarray(codes, dtype=np.uint16)
    n, nf = c.shape
    nodes = np.ascontiguousarray(lt.nodes[:n_rounds], dtype=np.int32)
    leaf = np.ascontiguousarray(lt.leaf[:n_rounds], dtype=np.float32)
    count = np.ascontiguousarray(lt.count[:n_rounds], dtype=np.int32)
    out = np.empty((n_rounds, n), dtype=np.float32)
    _check(_lib().shira_xgb_round_margins(c.ctypes.data, n, nf, n_rounds, int(nodes.shape[1]) if n_rounds else 1,
                                          nodes.ctypes.data, leaf.ctypes.data, count.ctypes.data,
                                          float(np.float32(sm.base_margin)), CODE_THREADS, out.ctypes.data),
           "shira_xgb_round_margins")
    return out


def to_ubj(sm: ShiraModel, n_trees: int | None = None) -> bytes | None:
    if not isinstance(sm.trees, LazyTrees):
        return None
    doc = to_dict(sm, 0)
    C, lt = sm.num_class, sm.trees
    k = len(lt) if n_trees is None else n_trees
    assert 0 <= k <= min(len(lt), MAX_TREES) and k % C == 0
    model = doc["learner"]["gradient_booster"]["model"]
    model["trees"] = _RawUBJ(trees_ubj(sm, k))
    model["gbtree_model_param"]["num_trees"] = str(k)
    model["iteration_indptr"] = list(range(0, k + 1, C))
    model["tree_info"] = [t % C for t in range(k)]
    model["cats"] = {"enc": [], "feature_segments": _Typed("l", np.zeros(0, np.int32)),
                     "sorted_idx": _Typed("l", np.zeros(0, np.int32))}
    return _ubj(doc)


def save_xgboost_json(sm: ShiraModel, path: str | Path) -> Path:
    p = Path(path)
    assert p.suffix == ".json", "XGBoost picks the format from the suffix; use .json"
    p.write_bytes(to_json(sm))
    assert p.stat().st_size > 0
    return p


def zero_tree_json(objective: str, base_score: float, num_feature: int = 1, num_class: int = 1,
                   objective_json: dict | None = None) -> bytes:
    assert num_class >= 1 and num_feature >= 1
    sm = ShiraModel(objective, base_score, 0.0, num_feature, np.arange(num_feature + 1, dtype=np.int32),
                    np.zeros(num_feature, dtype=F32), [[Node(leaf=0.0)] for _ in range(num_class)],
                    num_class, objective_json)
    return to_json(sm)


def xgb_base_margin(objective: str, base_score: float, objective_json: dict | None = None,
                    num_class: int = 1) -> np.float32:
    import xgboost as xgb

    raw = zero_tree_json(objective, base_score, 1, num_class, objective_json)
    bst = xgb.core.Booster(model_file=bytearray(raw))
    out = np.asarray(bst.predict(xgb.core.DMatrix(np.zeros((1, 1), dtype=F32)), output_margin=True)).reshape(-1)
    assert out.size == num_class and out.dtype == F32 and np.all(out == out[0])
    return out[0]


def _check_importable(learner: dict) -> None:
    gb = learner["gradient_booster"]
    assert gb["name"] == "gbtree", f"import: booster {gb['name']} (gbtree only)"
    lmp = learner["learner_model_param"]
    assert lmp.get("num_target", "1") == "1" and lmp.get("num_class", "0") == "0", "import: single target only"
    assert gb["model"]["gbtree_model_param"]["num_parallel_tree"] == "1", "import: num_parallel_tree 1 only"
    for t in gb["model"]["trees"]:
        assert not any(t["split_type"]) and not t["categories"], "import: numeric splits only"
        assert t["tree_param"].get("size_leaf_vector", "1") in ("0", "1"), "import: scalar leaves only"


def _import_cuts(trees: list[dict], nf: int) -> tuple[np.ndarray, np.ndarray]:
    per = [set() for _ in range(nf)]
    for t in trees:
        for i, lc in enumerate(t["left_children"]):
            if lc != -1:
                per[t["split_indices"][i]].add(float(F32(t["split_conditions"][i])))
    cols = [np.concatenate([[-np.inf], np.sort(np.array(list(s), dtype=np.float64))]).astype(F32) for s in per]
    ip = np.concatenate([[0], np.cumsum([c.size for c in cols])]).astype(np.int64)
    return ip, np.concatenate(cols).astype(F32)


def from_dict(doc: dict) -> ShiraModel:
    learner = doc["learner"]
    _check_importable(learner)
    nf = int(learner["learner_model_param"]["num_feature"])
    objective = learner["objective"]["name"]
    base_score = float(F32(learner["learner_model_param"]["base_score"].strip("[]")))
    trees_j = learner["gradient_booster"]["model"]["trees"]
    assert nf > 0 and len(trees_j) <= MAX_TREES
    ip, cuts = _import_cuts(trees_j, nf)
    trees = []
    for t in trees_j:
        tree = []
        for i, lc in enumerate(t["left_children"]):
            if lc == -1:
                tree.append(Node(leaf=float(F32(t["split_conditions"][i]))))
                continue
            f = int(t["split_indices"][i])
            cj = cuts[ip[f]:ip[f + 1]]
            cut = int(np.searchsorted(cj, F32(t["split_conditions"][i]), side="left"))
            assert cj[cut] == F32(t["split_conditions"][i])
            tree.append(Node(feature=f, cut=cut, dir=int(t["default_left"][i]), left=int(lc),
                             right=int(t["right_children"][i])))
        trees.append(tree)
    return ShiraModel(objective, base_score, float(_doc_base_margin(doc, nf)), nf, ip, cuts, trees)


def _doc_base_margin(doc: dict, nf: int) -> np.float32:
    import xgboost as xgb

    one = json.loads(zero_tree_json("reg:pseudohubererror", 0.0, nf))["learner"]["gradient_booster"]
    learner = dict(doc["learner"], gradient_booster=one, feature_names=[], feature_types=[])
    bst = xgb.core.Booster(model_file=bytearray(json.dumps({"learner": learner, "version": doc["version"]}).encode()))
    out = bst.predict(xgb.core.DMatrix(np.zeros((1, nf), dtype=F32)), output_margin=True)
    assert out.shape == (1,) and out.dtype == F32
    return out[0]


def load_xgboost_json(path_or_bytes: str | Path | bytes | bytearray) -> ShiraModel:
    raw = bytes(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else Path(path_or_bytes).read_bytes()
    assert raw[:1] == b"{", "import: JSON models only"
    return from_dict(json.loads(raw))
