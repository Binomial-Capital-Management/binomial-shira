from __future__ import annotations

import json
import logging
import os
import platform
import sys
import tempfile
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import catboost as _cb
import numpy as np
from catboost import core as _core

from . import _cat_first_read as _cfr
from . import _cat_fit as _fit
from . import export_cat

LOG = logging.getLogger("shira.catboost")
FIRST_READ_ERRORS = (AssertionError, ValueError, RuntimeError, AttributeError, TypeError, ArithmeticError)
SUPPORTED = _fit.SUPPORTED
why_not = _fit.why_not
last_backend: str | None = None
backend_history: list[str] = []

_STOCK: dict[str, Any] = {"CatBoost": _core.CatBoost, "CatBoostRegressor": _core.CatBoostRegressor,
                          "CatBoostClassifier": _core.CatBoostClassifier, "train": _core.train}
Pool = _core.Pool
FIT_PASSIVE = ("verbose", "logging_level", "plot", "plot_file", "verbose_eval", "metric_period", "silent",
               "log_cout", "log_cerr")
FIT_HANDLED = ("sample_weight", "use_best_model", "eval_set", "early_stopping_rounds")
MAX_EVAL_SETS = 1
X86_GUARD: bool = sys.platform == "linux" and platform.machine().lower() in ("x86_64", "amd64")
X86_ALLOWED = ("RMSE", "MAE", "Quantile")


def _degenerate_reason(data: dict | None) -> str | None:
    if data is None:
        return None
    y = np.asarray(data["y"])
    if y.size == 0:
        return "no rows (stock decides)"
    if float(np.ptp(y)) == 0.0:
        return "all train targets are equal (stock decides)"
    return None


def _x86_reason(eng: dict, data: dict) -> str | None:
    if not X86_GUARD:
        return None
    if os.environ.get("SHIRA_CAT_X86_GUARD", "") == "0":
        LOG.warning("shira.catboost: SHIRA_CAT_X86_GUARD=0: the x86 allow-list guard is OFF (a test setting)")
        return None
    loss = eng["loss_function"].partition(":")[0]
    if loss not in X86_ALLOWED:
        return f"x86 guard: loss {loss} is not on the Linux x86 allow-list"
    if data.get("w") is not None:
        return "x86 guard: sample weights are not on the Linux x86 allow-list"
    return None


def _note(backend: str) -> None:
    global last_backend
    assert backend in ("shira", "catboost")
    last_backend = backend
    backend_history.append(backend)


def _call_reason(kw: dict[str, Any]) -> str | None:
    for key, val in kw.items():
        if key in FIT_PASSIVE or key in FIT_HANDLED:
            continue
        if val is None or (key in ("plot", "save_snapshot") and val is False):
            continue
        return f"fit argument {key}"
    return None


def _eval_pool(eval_set: Any) -> tuple[Any, str | None]:
    if eval_set is None:
        return None, None
    sets = eval_set if isinstance(eval_set, list) else [eval_set]
    if len(sets) != MAX_EVAL_SETS:
        return None, f"{len(sets)} evaluation sets (SHIRA replays one)"
    one = sets[0]
    if isinstance(one, _core.Pool):
        return one, _fit._pool_reason(one)
    if not (isinstance(one, tuple) and len(one) == 2):
        return None, "an evaluation set that is not a Pool or an (X, y) pair"
    _, _, why = _fit._features(one[0])
    return (None, f"evaluation set: {why}") if why else (_core.Pool(one[0], one[1]), None)


def _plan(model: Any, X: Any, y: Any, kw: dict[str, Any]) -> tuple[SimpleNamespace | None, str | None]:
    why = _call_reason(kw)
    if why:
        return None, why
    params = dict(model._init_params)
    for key in ("use_best_model", "early_stopping_rounds"):
        if kw.get(key) is not None:
            params[key] = kw[key]
    n_rows = X.num_row() if isinstance(X, _core.Pool) else (len(X) if hasattr(X, "__len__") else None)
    if n_rows == 0:
        return None, "no rows (stock decides)"
    eng, ctl, why = _fit.resolve(params, model._shira_kind, n_rows, kw.get("eval_set") is not None)
    if why:
        return None, why
    data, why = _fit.extract(X, y, kw.get("sample_weight"), model._shira_kind, eng["loss_function"].partition(":")[0])
    why = why or _degenerate_reason(data)
    why = why or _x86_reason(eng, data)
    if why:
        return None, why
    ev, why = _eval_pool(kw.get("eval_set"))
    if why is None and ev is None and ctl["use_best_model"]:
        why = "use_best_model without an evaluation set (stock CatBoost raises)"
    if why:
        return None, why
    assert eng is not None and ctl is not None and data is not None
    return SimpleNamespace(eng=eng, ctl=ctl, data=data, eval_pool=ev), None


def _metric_history(model: Any, pool: Any, ctl: dict) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        res = _core.CatBoost.eval_metrics(model, pool, [ctl["metric"]], eval_period=1,
                                          thread_count=ctl["threads"], tmp_dir=tmp)
    assert len(res) == 1, f"one metric expected, got {sorted(res)}"
    vals = np.asarray(next(iter(res.values())), dtype=np.float64)
    assert vals.ndim == 1 and vals.size == model.tree_count_
    return vals


def _detector(vals: np.ndarray, od_wait: int | None) -> tuple[int, int]:
    best, run = 0, int(vals.size)
    for i in range(vals.size):
        if vals[i] < vals[best]:
            best = i
        if od_wait is not None and i - best >= od_wait:
            run = i + 1
            break
    assert 0 <= best < run <= vals.size
    return best, run


def _train_pool(data: dict) -> Any:
    return _core.Pool(data["x"], data["y"], weight=data["w"])


def _replay(model: Any, plan: SimpleNamespace) -> None:
    st, ctl = model._shira, plan.ctl
    if plan.eval_pool is None:
        return
    val = _metric_history(model, plan.eval_pool, ctl)
    best, run = _detector(val, ctl["od_wait"])
    use_best = True if ctl["use_best_model"] is None else bool(ctl["use_best_model"])
    keep = best + 1 if use_best else run
    st.learn = _metric_history(model, _train_pool(plan.data), ctl)[:run]
    st.validation, st.best = val[:run], best
    if keep < model.tree_count_:
        _core.CatBoost.shrink(model, keep)
        model._set_trained_model_attributes()
        _stamp(model, ctl)
    assert model.tree_count_ == keep


class _StopAfterFirstTree:

    def after_iteration(self, info: Any) -> bool:
        del info
        return False


class _Sink:

    def write(self, text: str) -> None:
        del text

    def flush(self) -> None:
        return None


_RECORD_KEYS = ("params", "output_options", "catboost_version_info")
_CALLBACK_ENTRY = '"callbacks":"PythonUserDefinedPerObject"'
_STOCK_BY_KIND = {"generic": _core.CatBoost, "regressor": _core.CatBoostRegressor,
                  "classifier": _core.CatBoostClassifier}


def _stock_param_record(model: Any, call: tuple) -> dict[str, str]:
    X, y, kw = call
    kind = model._shira_kind
    init = deepcopy(model._init_params)
    helper = _STOCK_BY_KIND[kind](init) if kind == "generic" else _STOCK_BY_KIND[kind](**init)
    fit_kw = {k: v for k, v in kw.items() if k not in ("log_cout", "log_cerr")}
    helper.fit(X, y, callbacks=[_StopAfterFirstTree()], log_cout=_Sink(), log_cerr=_Sink(), **fit_kw)
    meta = _core.CatBoost.get_metadata(helper)
    rec = {key: str(meta[key]) for key in _RECORD_KEYS}
    text = rec["params"]
    for entry in ("," + _CALLBACK_ENTRY, _CALLBACK_ENTRY + ","):
        text = text.replace(entry, "")
    assert _CALLBACK_ENTRY not in text, "parameter record: the callback entry was not removed"
    rec["params"] = text
    return rec


_OUTPUT_TEMPLATE_1_2_10 = ('{"name":"experiment","verbose":0,"test_error_log":"test_error.tsv","json_log":"catboost_training.json",'
                           '"result_model_file":"model","roc_file":"","eval_file_name":"","use_best_model":false,'
                           '"allow_writing_files":true,"prediction_type":["RawFormulaVal"],"fstr_internal_file":"",'
                           '"output_columns":["SampleId","RawFormulaVal","Label"],"snapshot_interval":600,"time_left_log":'
                           '"time_left.tsv","fstr_type":"FeatureImportance","profile_log":"catboost_profile.log","train_dir":'
                           '"catboost_info","learn_error_log":"learn_error.tsv","training_options_file":"","snapshot_file":'
                           '"experiment.cbsnapshot","save_snapshot":false,"model_format":["CatboostBinary"],'
                           '"final_feature_calcer_computation_mode":"Default","metric_period":1,"output_borders":"",'
                           '"final_ctr_computation_mode":"Default","best_model_min_trees":1,"fstr_regular_file":""}')
_VERSION_INFO: dict = {}


def _version_info() -> str | None:
    if "text" in _VERSION_INFO:
        return _VERSION_INFO["text"]
    from catboost import _catboost

    blob = open(_catboost.__file__, "rb").read()
    found, at = set(), blob.find(b"Git info:\n")
    for _ in range(64):
        if at < 0:
            break
        end = blob.find(b"\x00", at)
        text = blob[at:end if end >= 0 else len(blob)]
        if b"Other info:" in text:
            found.add(text.decode("utf-8", "replace"))
        at = blob.find(b"Git info:\n", at + 1)
    _VERSION_INFO["text"] = next(iter(found)) if len(found) == 1 else None
    if _VERSION_INFO["text"] is None:
        LOG.info("shira.catboost: the library's build text was found %d times (not once): the record uses stock's helper", len(found))
    return _VERSION_INFO["text"]


def _meta_info(pool: Any) -> Any:
    from catboost import utils

    lab = np.asarray(pool.get_label(), dtype=np.float64)
    assert lab.ndim == 1 and lab.size == pool.num_row() > 0
    return utils.DataMetaInfo(object_count=pool.num_row(), feature_count=pool.num_col(), max_cat_features_uniq_values_on_learn=0,
                              target_stats=utils.TargetStats(min_value=float(lab.min()), max_value=float(lab.max())), has_pairs=False)


def _nofit_param_record(model: Any, call: tuple) -> dict[str, str]:
    from catboost import utils

    X, y, kw = call
    assert _cb.__version__ == "1.2.10", "the record's template is CatBoost 1.2.10's"
    kind = model._shira_kind
    init = deepcopy(model._init_params)
    helper = _STOCK_BY_KIND[kind](init) if kind == "generic" else _STOCK_BY_KIND[kind](**init)
    fit_kw = {k: v for k, v in kw.items() if k not in ("log_cout", "log_cerr")}
    tp = helper._prepare_train_params(X=X, y=y, **fit_kw)
    p = tp["params"]
    ev = tp["eval_sets"][0] if tp["eval_sets"] else None
    opts = utils.compute_training_options(p, _meta_info(tp["train_pool"]), None if ev is None else _meta_info(ev))
    if "logging_level" not in p:
        opts["logging_level"] = "Silent" if p.get("verbose", 1) in (0, False) else "Verbose"
    if "eval_metric" not in p:
        opts["metrics"]["eval_metric"] = deepcopy(opts["metrics"]["objective_metric"])
    out = json.loads(_OUTPUT_TEMPLATE_1_2_10)
    out["verbose"] = int(p["verbose"]) if "verbose" in p else int(p.get("metric_period", 1))
    out["metric_period"] = int(p.get("metric_period", 1))
    out["use_best_model"] = bool(p.get("use_best_model", ev is not None))
    out["allow_writing_files"] = bool(p.get("allow_writing_files", True))
    out["train_dir"] = str(p.get("train_dir", "catboost_info"))
    return {"params": json.dumps(opts, separators=(",", ":"), ensure_ascii=False),
            "output_options": json.dumps(out, separators=(",", ":"), ensure_ascii=False),
            "catboost_version_info": _checked_version_info()}


def _checked_version_info() -> str:
    text = _version_info()
    assert text is not None and text.startswith("Git info:"), "the library's build text was not found once"
    return text


def _stamp(model: Any, ctl: dict) -> None:
    model._random_seed = ctl["seed"]
    model._learning_rate = float(np.float32(ctl["learning_rate"]))


def _install(model: Any, fm: Any, plan: SimpleNamespace) -> None:
    doc = export_cat.to_doc(fm.borders, fm.splits, fm.leaves, plan.data["classes"], plan.data["names"],
                            loss=plan.eng["loss_function"].partition(":")[0])
    export_cat.load_into(model, doc, fm.leaves)
    if plan.eng.get("boost_from_average"):
        model.set_scale_and_bias(1.0, _fit.cat_cpp.start_value(plan.data["y"], plan.data["w"],
                                                                plan.eng["loss_function"]))
    _stamp(model, plan.ctl)
    model._shira = SimpleNamespace(doc=doc, leaves=fm.leaves, data=plan.data, ctl=plan.ctl, eng=plan.eng,
                                   learn=None, validation=None, best=None, weights_done=False,
                                   call=getattr(plan, "call", None), record=None, busy=False)


def _drop_rows(model: Any, plan: SimpleNamespace) -> None:
    st = model._shira
    assert st is not None and st.data is plan.data
    model._shira_finish()
    if st.learn is None:
        st.learn = _metric_history(model, _train_pool(plan.data), plan.ctl)
    st.data, st.call = None, None
    assert st.weights_done and st.record is not None and st.learn is not None


def _fit_shira(model: Any, plan: SimpleNamespace) -> str | None:
    keep = dict(model._init_params)
    try:
        fm = _fit.engine(plan.data, plan.eng, plan.ctl["threads"])[0]
    except AssertionError as err:
        return f"engine refused the data: {err}"
    _install(model, fm, plan)
    _replay(model, plan)
    _drop_rows(model, plan)
    model._init_params = keep
    assert model.is_fitted() and model._init_params == keep
    return None


class _ShiraFit:

    _shira_kind = "generic"
    _shira: SimpleNamespace | None = None
    _shira_fr: dict | None = None

    def _first_read(self, X: Any, y: Any, kw: dict, refused: bool = False) -> bool:
        assert isinstance(kw, dict)
        self._shira_fr = None
        try:
            if _cfr.why_not(self, X, y, kw, refused) is not None:
                return False
            passive = {k: v for k, v in kw.items() if k in _cfr.FIT_PASSIVE}
            line = _cfr.fit(self, lambda m, xs, ys, **ex: super(_ShiraFit, m).fit(xs, ys, **passive, **ex), X, y, kw,
                            refused)
        except FIRST_READ_ERRORS as e:
            LOG.warning("shira.catboost: the first read did not apply (%s: %s); today's route", type(e).__name__, e)
            self._shira_fr = None
            return False
        assert isinstance(line, str)
        LOG.info("shira.catboost: %s", line)
        return True

    def fit(self, X: Any, y: Any = None, *args: Any, **kw: Any) -> Any:
        self._shira = None
        if not args and self._first_read(X, y, kw):
            _note("shira")
            return self
        plan, why = (None, "positional fit arguments after y") if args else _plan(self, X, y, kw)
        if why is None:
            plan.call = (X, y, dict(kw))
            why = _fit_shira(self, plan)
        if why is not None and not args:
            self._shira = None
            if self._first_read(X, y, kw, refused=True):
                _note("shira")
                return self
        if why is not None:
            self._shira = None
            LOG.info("stock CatBoost: %s", why)
            _note("catboost")
            return super().fit(X, y, *args, **kw)
        _note("shira")
        return self

    def _learn_history(self) -> np.ndarray:
        st = self._shira
        assert st is not None
        if st.learn is None:
            st.learn = _metric_history(self, _train_pool(st.data), st.ctl)
        return st.learn

    def get_evals_result(self) -> dict:
        st, fr = self._shira, self._shira_fr
        if st is None and fr is not None:
            return {"learn": {fr["metric"]: [float(v) for v in fr["learn"]]},
                    "validation": {fr["metric"]: [float(v) for v in fr["validation"]]}}
        if st is None:
            return super().get_evals_result()
        out = {"learn": {st.ctl["metric"]: [float(v) for v in self._learn_history()]}}
        if st.validation is not None:
            out["validation"] = {st.ctl["metric"]: [float(v) for v in st.validation]}
        return out

    def get_best_iteration(self) -> int | None:
        st, fr = self._shira, self._shira_fr
        if st is None and fr is not None:
            return int(fr["best"])
        return super().get_best_iteration() if st is None else st.best

    def get_best_score(self) -> dict:
        st, fr = self._shira, self._shira_fr
        if st is None and fr is not None:
            return {"learn": {fr["metric"]: float(np.min(fr["learn"]))},
                    "validation": {fr["metric"]: float(fr["validation"][fr["best"]])}}
        if st is None:
            return super().get_best_score()
        out = {"learn": {st.ctl["metric"]: float(np.min(self._learn_history()))}}
        if st.validation is not None:
            out["validation"] = {st.ctl["metric"]: float(st.validation[st.best])}
        return out

    def get_all_params(self) -> dict:
        self._shira_finish()
        return super().get_all_params()

    def get_metadata(self) -> Any:
        self._shira_finish()
        return super().get_metadata()

    def get_leaf_weights(self) -> Any:
        self._shira_finish()
        return super().get_leaf_weights()

    def save_model(self, *args: Any, **kw: Any) -> Any:
        self._shira_finish()
        return super().save_model(*args, **kw)

    def __getstate__(self) -> Any:
        self._shira_finish()
        return super().__getstate__()

    def get_feature_importance(self, *args: Any, **kw: Any) -> Any:
        self._shira_finish()
        return super().get_feature_importance(*args, **kw)

    def _shira_finish(self) -> None:
        st = self._shira
        if st is None or st.busy:
            return
        st.busy = True
        try:
            self._shira_leaf_weights()
            self._shira_param_record()
        finally:
            st.busy = False

    def _shira_param_record(self) -> None:
        st = self._shira
        assert st is not None and (st.record is not None or st.call is not None), "SHIRA model without its fit call"
        if st.record is None:
            st.record = (_nofit_param_record(self, st.call) if _cb.__version__ == "1.2.10" and _version_info() is not None
                         else _stock_param_record(self, st.call))
        meta = _core.CatBoost.get_metadata(self)
        for key, text in st.record.items():
            if meta.get(key) != text:
                meta[key] = text
        assert all(meta.get(k) == v for k, v in st.record.items())

    def _shira_leaf_weights(self) -> None:
        st = self._shira
        if st is None or st.weights_done:
            return
        keep, n_keep = dict(self._init_params), self.tree_count_
        scale, bias = self.get_scale_and_bias()
        lw = export_cat.leaf_weights(st.data["x"], st.data["w"], st.doc)
        for tree, w in zip(st.doc["oblivious_trees"], lw):
            tree["leaf_weights"] = [float(v) for v in w]
        export_cat.load_into(self, st.doc, st.leaves)
        export_cat.set_leaf_weights(self, np.concatenate([np.asarray(w, dtype=np.float64) for w in lw]))
        if n_keep < self.tree_count_:
            _core.CatBoost.shrink(self, n_keep)
            self._set_trained_model_attributes()
        self.set_scale_and_bias(scale, bias)
        _stamp(self, st.ctl)
        self._init_params, st.weights_done = keep, True
        assert self.tree_count_ == n_keep and self.get_scale_and_bias() == (scale, bias)


class CatBoost(_ShiraFit, _core.CatBoost):

    _shira_kind = "generic"


class CatBoostRegressor(_ShiraFit, _core.CatBoostRegressor):

    _shira_kind = "regressor"


class CatBoostClassifier(_ShiraFit, _core.CatBoostClassifier):

    _shira_kind = "classifier"


def train(pool: Any = None, params: Any = None, dtrain: Any = None, logging_level: Any = None,
          verbose: Any = None, iterations: Any = None, num_boost_round: Any = None, evals: Any = None,
          eval_set: Any = None, early_stopping_rounds: Any = None, **kw: Any) -> Any:
    if params is None:
        raise _cb.CatBoostError("params should be set.")
    if dtrain is not None and pool is not None:
        raise _cb.CatBoostError("Only one of the parameters pool and dtrain should be set.")
    if num_boost_round is not None and iterations is not None:
        raise _cb.CatBoostError("Only one of the parameters iterations and num_boost_round should be set.")
    if evals is not None and eval_set is not None:
        raise _cb.CatBoostError("Only one of the parameters evals, eval_set should be set.")
    pool = dtrain if pool is None else pool
    rounds = num_boost_round if iterations is None else iterations
    p = deepcopy(params)
    if rounds is not None:
        p["iterations"] = rounds
    if early_stopping_rounds is not None:
        p.pop("od_pval", None)
        p.update({"od_type": "Iter", "od_wait": early_stopping_rounds})
    model = CatBoost(p)
    model.fit(pool, eval_set=evals if eval_set is None else eval_set, logging_level=logging_level,
              verbose=verbose, early_stopping_rounds=early_stopping_rounds, **kw)
    return model


def fit_folds(params: dict, X: Any, y: Any, folds: list, sample_weight: Any = None,
              kind: str = "regressor") -> list:
    cls = {"regressor": CatBoostRegressor, "classifier": CatBoostClassifier, "generic": CatBoost}[kind]
    assert len(folds) >= 1, "at least one fold"
    eng, ctl, why = _fit.resolve(params, kind, min(b - a for a, b in folds))
    if why is None:
        why = _fit.random_rows_reason(_fit.normalize(params)[0], eng["bootstrap_type"], max(b - a for a, b in folds))
    data = None
    if why is None:
        data, why = _fit.extract(X, y, sample_weight, kind, eng["loss_function"].partition(":")[0])
        why = why or _degenerate_reason(data)
        why = why or _x86_reason(eng, data)
    fms = None
    if why is None:
        try:
            fms = _fit.engine(data, eng, ctl["threads"], folds)
        except AssertionError as err:
            why = f"engine refused the data: {err}"
    if why is not None:
        LOG.info("stock CatBoost: %s", why)
        _note("catboost")
        return [_stock_fold(_STOCK[cls.__name__], kind, params, X, y, sample_weight, a, b) for a, b in folds]
    models = []
    for (a, b), fm in zip(folds, fms):
        model = cls(dict(params)) if kind == "generic" else cls(**params)
        keep = dict(model._init_params)
        part = dict(data, x=data["x"][a:b], y=data["y"][a:b], w=None if data["w"] is None else data["w"][a:b])
        xs = X.iloc[a:b] if hasattr(X, "iloc") else X[a:b]
        call = (xs, y[a:b], {} if sample_weight is None else {"sample_weight": sample_weight[a:b]})
        _install(model, fm, SimpleNamespace(eng=eng, ctl=ctl, data=part, eval_pool=None, call=call))
        model._init_params = keep
        models.append(model)
    _note("shira")
    assert len(models) == len(folds)
    return models


def _stock_fold(cls: Any, kind: str, params: dict, X: Any, y: Any, w: Any, a: int, b: int) -> Any:
    model = cls(dict(params)) if kind == "generic" else cls(**params)
    xs = X.iloc[a:b] if hasattr(X, "iloc") else X[a:b]
    model.fit(xs, y[a:b], sample_weight=None if w is None else w[a:b])
    return model


def patch() -> None:
    _cb.CatBoost, _cb.CatBoostRegressor, _cb.CatBoostClassifier = CatBoost, CatBoostRegressor, CatBoostClassifier
    _cb.train = train
    assert _cb.CatBoostRegressor is CatBoostRegressor and _cb.train is train


def unpatch() -> None:
    for name, obj in _STOCK.items():
        setattr(_cb, name, obj)
    assert _cb.CatBoostRegressor is _STOCK["CatBoostRegressor"] and _cb.train is _STOCK["train"]


def __getattr__(name: str) -> Any:
    return getattr(_cb, name)
