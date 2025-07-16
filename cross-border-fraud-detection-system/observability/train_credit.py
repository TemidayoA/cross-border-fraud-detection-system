from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    "model__C": [0.1, 0.5, 1.0, 3.0],
    "model__solver": ["liblinear", "lbfgs"],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=5,
    scoring="roc_auc",
    cv=3,
    verbose=1,
)
search.fit(X_train, y_train)

pipeline = search.best_estimator_
