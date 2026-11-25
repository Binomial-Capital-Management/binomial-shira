from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

F32_LOWEST = np.float32(np.finfo(np.float32).min)
F32_MAX = np.float32(np.finfo(np.float32).max)
MAX_TREES = 1_000_000
MAX_DEPTH = 16


def _float_features(borders: list, names: list[str] | None) -> list[dict]:
    assert names is None or len(names) == len(borders), "one name per feature"
    out = []
    for f, b in enumerate(borders):
        b32 = np.asarray(b, dtype=np.float32)
        assert b32.ndim == 1 and (b32.size < 2 or bool(np.all(b32[1:] > b32[:-1]))), f"feature {f}: borders not ascending"
        low = b32.size > 0 and b32[0] == F32_LOWEST
        high = b32.size > 0 and b32[-1] == F32_MAX
        out.append({"borders": [float(v) for v in b32], "feature_id": "" if names is None else str(names[f]),
                    "feature_index": f, "flat_feature_index": f, "has_nans": bool(low or high),
                    "nan_value_treatment": "AsFalse" if low else ("AsTrue" if high else "AsIs")})
    return out


def _split_json(borders: list, offsets: np.ndarray, feature: int, border: float) -> dict:
    b32 = np.asarray(borders[feature], dtype=np.float32)
    v = np.float32(border)
    i = int(np.searchsorted(b32, v))
    assert i < b32.size and b32[i] == v, f"split border {border!r} is not a border of feature {feature}"
    return {"border": float(v), "float_feature_index": int(feature), "split_index": int(offsets[feature] + i),
            "split_type": "FloatFeature"}


def to_doc(borders: list, splits: list, leaves: list, classes: list | None = None,
           names: list[str] | None = None, leaf_weights: list | None = None, loss: str | None = None) -> dict:
    assert len(splits) == len(leaves) and 1 <= len(splits) <= MAX_TREES, "one leaf array per tree"
    assert leaf_weights is None or len(leaf_weights) == len(leaves)
    offsets = np.concatenate([[0], np.cumsum([len(b) for b in borders])]).astype(np.int64)
    trees = []
    for t, (sp, lv) in enumerate(zip(splits, leaves)):
        assert len(sp) <= MAX_DEPTH and len(lv) == 1 << len(sp), f"tree {t}: {len(lv)} leaves for depth {len(sp)}"
        lw = [0] * len(lv) if leaf_weights is None else [float(v) for v in leaf_weights[t]]
        trees.append({"leaf_values": [float(v) for v in lv], "leaf_weights": lw,
                      "splits": [_split_json(borders, offsets, int(f), float(b)) for f, b in sp]})
    doc: dict[str, Any] = {"features_info": {"float_features": _float_features(borders, names)},
                           "oblivious_trees": trees, "scale_and_bias": [1, [0]]}
    if classes is not None:
        assert len(classes) == 2, "binary classifiers only"
        doc["model_info"] = {"class_params": {
            "class_label_type": "Integer", "class_names": [int(c) for c in classes],
            "class_to_label": [0, 1], "classes_count": 0}}
    if loss == "CrossEntropy":
        doc.setdefault("model_info", {})["params"] = {"loss_function": {"type": "CrossEntropy", "params": {}}}
    return doc


def load_into(model: Any, doc: dict, leaves: list) -> None:
    flat = np.concatenate([np.asarray(lv, dtype=np.float64) for lv in leaves])
    assert flat.ndim == 1 and flat.size == sum(len(t["leaf_values"]) for t in doc["oblivious_trees"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shira_model.json"
        path.write_text(json.dumps(doc))
        model._object._load_model(str(path), "json")
    model._set_trained_model_attributes()
    model.set_leaf_values(flat)
    got = np.asarray(model.get_leaf_values(), dtype=np.float64)
    assert np.array_equal(got, flat), "stock CatBoost did not keep the exact leaf values"


def set_leaf_weights(model: Any, flat: np.ndarray) -> None:
    want = np.ascontiguousarray(flat, dtype="<f8")
    have = np.ascontiguousarray(model._object._get_leaf_weights(), dtype="<f8")
    assert have.shape == want.shape, "leaf weight count"
    if np.array_equal(have, want):
        return
    blob, old = model._serialize_model(), have.tobytes()
    if blob.count(old) != 1:
        raise RuntimeError("SHIRA: the model's leaf weights were not found exactly once in its serialized bytes")
    at = blob.find(old)
    model._deserialize_model(blob[:at] + want.tobytes() + blob[at + len(old):])
    assert np.array_equal(np.asarray(model._object._get_leaf_weights(), dtype=np.float64), want)


def stock_model(fm: Any, classes: list | None = None, names: list[str] | None = None, cls: Any = None) -> Any:
    import catboost

    assert len(fm.splits) > 0, "the engine model has no trees (train with trees_out=True)"
    kind = cls if cls is not None else (catboost.CatBoostClassifier if classes is not None else catboost.CatBoost)
    model = kind()
    load_into(model, to_doc(fm.borders, fm.splits, fm.leaves, classes, names), fm.leaves)
    return model


def save_cbm(fm: Any, path: Any, classes: list | None = None, names: list[str] | None = None) -> None:
    model = stock_model(fm, classes, names)
    model.save_model(str(path), format="cbm")
    assert Path(path).stat().st_size > 0


def save_json(fm: Any, path: Any, classes: list | None = None, names: list[str] | None = None) -> None:
    doc = to_doc(fm.borders, fm.splits, fm.leaves, classes, names)
    Path(path).write_text(json.dumps(doc))
    assert Path(path).stat().st_size > 0


def from_stock(model: Any) -> tuple[list, list, list, list]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "m.json"
        model.save_model(str(path), format="json")
        js = json.loads(path.read_text())
    assert "oblivious_trees" in js and not js["features_info"].get("categorical_features"), "float features only"
    ff = js["features_info"]["float_features"]
    borders = [np.asarray(f.get("borders", []), dtype=np.float32) for f in ff]
    nan = [f.get("nan_value_treatment", "AsIs") for f in ff]
    splits = [[(int(s["float_feature_index"]), float(np.float32(s["border"]))) for s in t["splits"]]
              for t in js["oblivious_trees"]]
    flat = np.asarray(model.get_leaf_values(), dtype=np.float64)
    sizes = [1 << len(sp) for sp in splits]
    assert flat.size == sum(sizes), "single-output models only"
    cuts = np.concatenate([[0], np.cumsum(sizes)])
    leaves = [flat[cuts[i]:cuts[i + 1]].copy() for i in range(len(sizes))]
    return borders, nan, splits, leaves


def leaf_index(x: np.ndarray, nan: list, sp: list) -> np.ndarray:
    assert x.ndim == 2 and len(sp) <= MAX_DEPTH
    idx = np.zeros(x.shape[0], dtype=np.int64)
    for k, (f, b) in enumerate(sp):
        col = x[:, f]
        bit = col > np.float32(b)
        if nan[f] == "AsTrue":
            bit = bit | np.isnan(col)
        idx |= bit.astype(np.int64) << k
    return idx


def predict_raw(x: np.ndarray, nan: list, splits: list, leaves: list) -> np.ndarray:
    xf = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros(xf.shape[0], dtype=np.float64)
    assert len(splits) == len(leaves)
    for sp, lv in zip(splits, leaves):
        out += np.asarray(lv, dtype=np.float64)[leaf_index(xf, nan, sp)]
    return out


def leaf_weights(x: np.ndarray, w: np.ndarray | None, doc: dict) -> list:
    xf = np.ascontiguousarray(x, dtype=np.float32)
    nan = [f["nan_value_treatment"] for f in doc["features_info"]["float_features"]]
    out = []
    for t in doc["oblivious_trees"]:
        sp = [(s["float_feature_index"], s["border"]) for s in t["splits"]]
        idx = leaf_index(xf, nan, sp)
        out.append(np.bincount(idx, weights=w, minlength=1 << len(sp)).astype(np.float64))
    assert len(out) == len(doc["oblivious_trees"])
    return out
