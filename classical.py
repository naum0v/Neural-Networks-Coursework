"""Сравнение классических регрессоров на зафиксированных фолдах."""

import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from validation import load_folds

NUMERIC = ["model_year", "milage"]


def preprocess(
    features: pd.DataFrame, encoding: str, missing: str, scale: bool
) -> ColumnTransformer:
    """Настроить обработку признаков внутри обучающего фолда."""
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale:
        numeric_steps.append(StandardScaler())
    transformers = [("numeric", make_pipeline(*numeric_steps), NUMERIC)]
    if encoding != "numeric":
        categorical = features.columns.difference(NUMERIC).tolist()
        encoder = (
            OneHotEncoder(handle_unknown="ignore")
            if encoding == "onehot"
            else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        )
        imputer = (
            SimpleImputer(strategy="constant", fill_value="__missing__")
            if missing == "category"
            else SimpleImputer(strategy="most_frequent")
        )
        transformers.append(("categorical", make_pipeline(imputer, encoder), categorical))
    return ColumnTransformer(transformers)


def experiments(seed: int) -> list:
    """Задать модели и пары для сравнения одной настройки за раз."""
    return [
        ("linear_onehot", LinearRegression(), "onehot", "category", True),
        ("ridge_onehot", Ridge(alpha=10, solver="lsqr", tol=1e-6), "onehot", "category", True),
        ("ridge_mode", Ridge(alpha=10, solver="lsqr", tol=1e-6), "onehot", "mode", True),
        ("ridge_ordinal", Ridge(alpha=10, solver="lsqr", tol=1e-6), "ordinal", "category", True),
        ("lasso_onehot", Lasso(alpha=25, max_iter=4000), "onehot", "category", True),
        ("elasticnet_onehot", ElasticNet(alpha=1, l1_ratio=0.9, max_iter=4000),
         "onehot", "category", True),
        ("knn_raw", KNeighborsRegressor(n_neighbors=50, algorithm="kd_tree"),
         "numeric", "median", False),
        ("knn_scaled", KNeighborsRegressor(n_neighbors=50, algorithm="kd_tree"),
         "numeric", "median", True),
        ("tree_ordinal", DecisionTreeRegressor(max_depth=16, min_samples_leaf=15,
                                               random_state=seed), "ordinal", "category", False),
        ("forest_ordinal", RandomForestRegressor(n_estimators=80, max_depth=16,
                                                 min_samples_leaf=10, max_samples=0.65,
                                                 random_state=seed, n_jobs=2),
         "ordinal", "category", False),
    ]


def evaluate(train: pd.DataFrame, config: dict, output: Path, folds_file: Path) -> None:
    """Обучить модели на общих фолдах и сохранить ошибки и время."""
    key, target = config["id_column"], config["target"]
    settings = config["validation"]
    expected_folds = set(range(1, settings["folds"] + 1))
    rows = train.sort_values(key)
    folds = load_folds(rows, key, settings["folds"], folds_file)
    features, price = rows.drop(columns=[key, target]), rows[target].to_numpy()
    records = []
    for name, model, encoding, missing, scale in experiments(settings["seed"]):
        for fold in sorted(expected_folds):
            valid = folds == fold
            pipeline = make_pipeline(preprocess(features, encoding, missing, scale), model)
            started = perf_counter()
            with warnings.catch_warnings(record=True) as fit_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                pipeline.fit(features.loc[~valid], price[~valid])
            fit_seconds = perf_counter() - started
            started = perf_counter()
            errors = pipeline.predict(features.loc[valid]) - price[valid]
            predict_seconds = perf_counter() - started
            if not np.isfinite(errors).all():
                raise ValueError(f"Некорректные прогнозы модели {name} на фолде {fold}.")
            records.append({
                "model": name, "fold": fold, "rows": int(valid.sum()),
                "squared_error": float(errors @ errors),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                "convergence_warnings": sum(
                    issubclass(item.category, ConvergenceWarning) for item in fit_warnings
                ),
            })
        print(f"Завершено: {name}", flush=True)

    details = pd.DataFrame(records)
    totals = details.groupby("model").sum(numeric_only=True)
    summary = totals[["fit_seconds", "predict_seconds", "convergence_warnings"]].assign(
        oof_rmse=np.sqrt(totals["squared_error"] / totals["rows"])
    ).sort_values("oof_rmse")
    output.mkdir(parents=True, exist_ok=True)
    details.to_csv(output / "rmse_by_fold.csv", index=False)
    summary.to_csv(output / "classical.csv")
    print("CLASSICAL " + json.dumps(summary.reset_index().to_dict(orient="records")), flush=True)
