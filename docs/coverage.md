# What is covered

Measured on the installed package, with made-up data, by calling each library the way a user does.

**By default** SHIRA runs these calls itself:

| Library | Calls |
|---|---|
| XGBoost | default regression, pseudo-Huber, absolute error, quantile, Poisson, Tweedie, binary logistic, multiclass softprob, sample weights, missing values, early stopping, lossguide, subsample + colsample, monotone constraints, max_depth 12, max_bin 512, scale_pos_weight 3, num_parallel_tree 4, booster dart, tree_method approx, tree_method exact, ranking pairwise, ranking ndcg, categorical feature, custom objective, XGBRegressor.fit, XGBClassifier.fit, XGBClassifier multiclass, XGBRanker.fit, xgb.cv, QuantileDMatrix, pandas DataFrame |
| LightGBM | default regression, L1, Huber, quantile, Poisson, binary, multiclass, lambdarank, sample weights, missing values, early stopping, categorical feature, pre-built Dataset, bagging + feature_fraction, GOSS, DART, random forest mode, linear_tree, extra_trees, monotone constraints, max_depth 6, max_bin 511, LGBMRegressor.fit, LGBMClassifier.fit, LGBMRanker.fit, lgb.cv, pandas DataFrame |
| CatBoost | default regression (RMSE), MAE, Quantile, Poisson, binary (Logloss), MultiClass, sample weights, missing values, early stopping, depth 10, Ordered boosting, Bernoulli bootstrap, Bayesian bootstrap, MVS bootstrap, rsm 0.5, grow_policy Depthwise, grow_policy Lossguide, categorical feature (CTR), categorical one-hot, categorical + binary, categorical + MultiClass, categorical + weights + MultiClass, Pool input, pandas DataFrame |
| scikit-learn | `DecisionTreeRegressor`, `DecisionTreeClassifier`, `ExtraTreeRegressor`, `ExtraTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor`, `ExtraTreesClassifier`, `IsolationForest`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`, `LinearRegression`, `Ridge`, `LogisticRegression`, `Lasso`, `ElasticNet`, `BayesianRidge`, `ARDRegression`, `Lars`, `LassoLars`, `TruncatedSVD`, `FactorAnalysis`, `GaussianNB`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis`, `PCA` |

**With the bit-identical settings** these calls give exactly the stock library's results:

| Library | Calls |
|---|---|
| XGBoost | default regression, pseudo-Huber, absolute error, quantile, Poisson, Tweedie, binary logistic, sample weights, missing values, early stopping, lossguide, subsample + colsample, monotone constraints, max_depth 12, max_bin 512, scale_pos_weight 3, ranking pairwise, ranking ndcg, custom objective, XGBRegressor.fit, XGBClassifier.fit, XGBRanker.fit, xgb.cv, QuantileDMatrix, pandas DataFrame |
| LightGBM | default regression, L1, Huber, quantile, Poisson, binary, multiclass, lambdarank, sample weights, missing values, early stopping, categorical feature, pre-built Dataset, bagging + feature_fraction, GOSS, DART, max_depth 6, max_bin 511, LGBMRegressor.fit, LGBMClassifier.fit, LGBMRanker.fit, lgb.cv, pandas DataFrame |
| CatBoost | default regression (RMSE), MAE, Quantile, sample weights, missing values, early stopping, depth 10, Bernoulli bootstrap, Bayesian bootstrap, MVS bootstrap, Pool input, pandas DataFrame |
| scikit-learn | `DecisionTreeRegressor`, `DecisionTreeClassifier`, `ExtraTreeRegressor`, `ExtraTreeClassifier`, `RandomForestRegressor`, `RandomForestClassifier`, `ExtraTreesRegressor`, `ExtraTreesClassifier`, `IsolationForest`, `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier`, `GradientBoostingRegressor`, `GradientBoostingClassifier`, `LinearRegression`, `Ridge`, `LogisticRegression`, `Lasso`, `ElasticNet`, `BayesianRidge`, `ARDRegression`, `Lars`, `LassoLars`, `TruncatedSVD`, `FactorAnalysis`, `GaussianNB`, `LinearDiscriminantAnalysis`, `QuadraticDiscriminantAnalysis` |

In that mode any other call runs on the stock library, so the result is the stock library's either way.

One difference on Linux: with the bit-identical settings, CatBoost fits with sample weights run on the stock library.
