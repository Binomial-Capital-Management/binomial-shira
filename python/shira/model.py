from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._lib import check, lib

SCALE = 22
F32 = np.float32
OBJECTIVES = {"reg:pseudohubererror": "pseudohuber", "binary:logistic": "logistic"}
XGB_KEYS = {"objective", "huber_slope", "max_depth", "eta", "lambda", "min_child_weight",
            "base_score", "max_bin", "tree_method", "nthread", "gamma"}


@dataclass(frozen=True)
class Recipe:
    growth: str
    max_depth: int
    max_leaves: int
    lam: float
    min_hess: float
    min_gain: float
    eta: float
    objective: str
    base_score: float


@dataclass(frozen=True)
class Rec:
    growth: str
    max_depth: int
    max_leaves: int
    lam: float
    min_hess: float
    min_gain: float
    eta: float
    leaf_eps: float = 0.0
    zero_below_mcw: bool = False
    midpoint: bool = False
    min_count: int = 0
    count_rule: str = "true"


def recipe_xgboost(params: dict) -> Recipe:
    bad = set(params) - XGB_KEYS
    assert not bad, f"unsupported XGBoost params in shira v0: {bad}"
    obj = params.get("objective")
    assert obj in OBJECTIVES and params.get("huber_slope", 1.0) == 1.0, f"objective {obj}"
    assert params.get("tree_method", "hist") == "hist" and "base_score" in params
    base = float(params["base_score"])
    if obj == "binary:logistic":
        assert 0.0 < base < 1.0
        base = float(-np.log(F32(1) / F32(base) - F32(1)))
    return Recipe("depthwise", int(params.get("max_depth", 6)), 0, float(params.get("lambda", 1.0)),
                  float(params.get("min_child_weight", 1.0)), 1e-6 + float(params.get("gamma", 0.0)),
                  float(params.get("eta", 0.3)), OBJECTIVES[obj], base)


class DMatrix:

    def __init__(self, x: np.ndarray, label: np.ndarray, max_bin: int = 256):
        import xgboost as xgb

        assert x.ndim == 2 and x.shape[0] == label.shape[0] and x.dtype == F32
        self.qdm = xgb.QuantileDMatrix(x, label=label, max_bin=max_bin)
        self.ip, self.cuts = self.qdm.get_quantile_cut()
        self.y = label.astype(F32)
        self.codes = np.zeros(x.shape, dtype=np.int16)
        for j in range(x.shape[1]):
            cj = self.cuts[self.ip[j]:self.ip[j + 1]]
            col = x[:, j]
            self.codes[:, j] = np.where(np.isnan(col), 0, np.searchsorted(cj, col, side="right"))
        self.n_bins = int(self.codes.max()) + 1
        assert 2 <= self.n_bins <= 257


@dataclass
class Node:
    leaf: float | None = None
    feature: int = -1
    cut: int = -1
    dir: int = 0
    left: int = -1
    right: int = -1
    depth: int = 0
    gain64: float = 0.0
    parent64: float = 0.0


def pack_trees(trees: list[list[Node]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = [len(t) for t in trees]
    assert all(s > 0 for s in sizes), "empty tree"
    off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    nodes = np.array([[(-1 if nd.leaf is not None else nd.feature), nd.cut, nd.dir, nd.left, nd.right]
                      for t in trees for nd in t], dtype=np.int32).reshape(-1, 5)
    leaf = np.array([nd.leaf if nd.leaf is not None else 0.0 for t in trees for nd in t], dtype=F32)
    return off, np.ascontiguousarray(nodes), leaf


def predict_margin(trees: list[list[Node]], codes: np.ndarray, base: float) -> np.ndarray:
    c = np.ascontiguousarray(codes.view(np.uint16) if codes.dtype == np.int16 else codes.astype(np.uint16))
    n, nf = c.shape
    out = np.empty(n, dtype=F32)
    if not trees:
        out[:] = F32(base)
        return out
    off, nodes, leaf = pack_trees(trees)
    check(lib().shira_predict_margin(c.ctypes.data, n, nf, len(trees), off.ctypes.data, nodes.ctypes.data, leaf.ctypes.data,
             F32(base), out.ctypes.data), "shira_predict_margin")
    return out


@dataclass
class Booster:
    recipe: Recipe
    trees: list[list[Node]] = field(default_factory=list)

    def predict_margin(self, codes: np.ndarray) -> np.ndarray:
        return predict_margin(self.trees, codes, self.recipe.base_score)
