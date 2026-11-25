"""Learning to rank on SHIRA: XGBRanker and LGBMRanker with query groups."""

import numpy as np

import shira.lightgbm as lgb
import shira.xgboost as xgb

rng = np.random.default_rng(0)
queries, per_query = 2_000, 20
X = rng.normal(size=(queries * per_query, 16))
relevance = np.clip(np.round(X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.7, size=X.shape[0]) + 1.5), 0, 4).astype(int)
groups = [per_query] * queries
query_id = np.repeat(np.arange(queries), per_query)

ranker = xgb.XGBRanker(objective="rank:ndcg", n_estimators=100, max_depth=6)
ranker.fit(X, relevance, qid=query_id)
print(f"XGBRanker   ran on: {xgb.last_backend}   first scores: {np.round(ranker.predict(X[:3]), 3)}")

ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=100, verbose=-1)
ranker.fit(X, relevance, group=groups)
print(f"LGBMRanker  ran on: {lgb.last_backend}   first scores: {np.round(ranker.predict(X[:3]), 3)}")
