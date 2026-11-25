from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shira_cpp import LOSSES, boost_batch, grow_batch, train, train_xgb
from shira_cpp import hgb_cpp as hgb
from shira_cpp import lgb_cpp as lgbm
from shira_cpp import sk_exact_cpp as sklearn

from .model import (
    Booster, DMatrix, Node, Rec, Recipe, predict_margin, recipe_xgboost)
