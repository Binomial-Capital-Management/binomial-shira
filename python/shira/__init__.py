def patch(xgboost: bool = True, sklearn: bool = True) -> None:
    if xgboost:
        from . import xgboost as _sx

        _sx.patch()
    if sklearn:
        from . import sklearn as _sk

        _sk.patch()
    _patch_catboost()
    _patch_lightgbm()


def _patch_lightgbm() -> None:
    import importlib.util

    if importlib.util.find_spec("lightgbm") is not None:
        from . import lightgbm as _sl

        _sl.patch()


def _patch_catboost() -> None:
    import importlib.util

    if importlib.util.find_spec("catboost") is not None:
        from . import catboost as _sc

        _sc.patch()


def unpatch() -> None:
    import sys

    for name in ("xgboost", "sklearn", "catboost", "lightgbm"):
        mod = sys.modules.get(f"{__name__}.{name}")
        if mod is not None:
            mod.unpatch()
