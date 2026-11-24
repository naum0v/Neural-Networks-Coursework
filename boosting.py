"""Сравнение трёх бустингов на фиксированных фолдах."""

import importlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

FAMILIES = ("catboost", "lightgbm", "xgboost")
PROFILES = {
    "base": (250, 6, 0.10),
    "slower": (420, 6, 0.06),
    "deeper": (340, 8, 0.075),
}


def feature_sets(base: pd.DataFrame, reference_year: int) -> dict[str, pd.DataFrame]:
    """Собрать исходный набор и добавить к нему независимые группы признаков."""
    age = (reference_year - base["model_year"]).clip(lower=0)
    usage = pd.DataFrame({
        "vehicle_age": age,
        "miles_per_year": base["milage"] / age.clip(lower=1),
    })
    text = base["engine"].astype("string")
    patterns = {
        "engine_hp": r"(?i)(\d+(?:\.\d+)?)\s*hp\b",
        "engine_liters": r"(?i)(\d+(?:\.\d+)?)\s*l\b",
        "engine_cylinders": r"(?i)(\d+)\s+cylinders?\b",
    }
    engine = pd.DataFrame({
        name: pd.to_numeric(
            text.str.extract(pattern, expand=False), errors="coerce"
        ).astype("float64")
        for name, pattern in patterns.items()
    })
    crosses = pd.DataFrame({
        f"{left}_by_{right}": (
            base[left].astype("string").fillna("__missing__") + " | "
            + base[right].astype("string").fillna("__missing__")
        )
        for left, right in (("brand", "fuel_type"), ("model", "transmission"))
    })
    return {
        "base": base,
        "usage": pd.concat([base, usage], axis=1),
        "engine": pd.concat([base, engine], axis=1),
        "crosses": pd.concat([base, crosses], axis=1),
        "all": pd.concat([base, usage, engine, crosses], axis=1),
    }


def prepare_fold(
    features: pd.DataFrame, valid: np.ndarray, family: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Согласовать категории по обучающей части фолда."""
    training, validation = features.loc[~valid].copy(), features.loc[valid].copy()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    for column in categorical:
        train_values = training[column].astype("string").fillna("__missing__")
        valid_values = validation[column].astype("string").fillna("__missing__")
        if family == "catboost":
            training[column] = train_values.astype(object)
            validation[column] = valid_values.astype(object)
        else:
            categories = pd.Index(train_values.unique()).astype(object)
            training[column] = pd.Categorical(train_values, categories=categories)
            validation[column] = pd.Categorical(valid_values, categories=categories)
    return training, validation, categorical


def make_model(family: str, profile: str, seed: int, threads: int) -> Any:
    """Создать модель с фиксированным числом итераций."""
    rounds, depth, rate = PROFILES[profile]
    library = importlib.import_module(family)
    if family == "catboost":
        return library.CatBoostRegressor(
            iterations=rounds, depth=depth, learning_rate=rate,
            loss_function="RMSE", l2_leaf_reg=5, random_seed=seed,
            thread_count=threads, verbose=False, allow_writing_files=False,
            max_ctr_complexity=1, one_hot_max_size=10,
        )
    if family == "lightgbm":
        return library.LGBMRegressor(
            n_estimators=rounds, max_depth=depth, num_leaves=2 ** (depth - 1) - 1,
            learning_rate=rate, min_child_samples=40, reg_lambda=5,
            random_state=seed, n_jobs=threads, verbosity=-1,
        )
    return library.XGBRegressor(
        n_estimators=rounds, max_depth=depth, learning_rate=rate,
        min_child_weight=15, reg_lambda=5, tree_method="hist",
        enable_categorical=True, max_cat_threshold=32,
        random_state=seed, n_jobs=threads, objective="reg:squarederror",
    )


def score(
    family: str, name: str, profile: str, features: pd.DataFrame,
    price: np.ndarray, folds: np.ndarray, seed: int, threads: int,
) -> list[dict]:
    """Измерить ошибку на всех проверочных фолдах."""
    records = []
    for fold in np.unique(folds):
        valid = folds == fold
        started = perf_counter()
        training, validation, categorical = prepare_fold(features, valid, family)
        model = make_model(family, profile, seed, threads)
        fit_options = {"cat_features": categorical} if family == "catboost" else {}
        model.fit(training, price[~valid], **fit_options)
        fit_seconds = perf_counter() - started
        errors = model.predict(validation) - price[valid]
        if not np.isfinite(errors).all():
            raise ValueError(f"Некорректный прогноз {family}, фолд {fold}.")
        records.append({
            "model": family, "features": name, "profile": profile,
            "fold": int(fold), "rows": int(valid.sum()),
            "squared_error": float(errors @ errors),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "fit_seconds": fit_seconds,
        })
    print(f"Завершено: {family} / {name} / {profile}", flush=True)
    return records


def evaluate(
    train: pd.DataFrame, config: dict, output: Path, folds_file: Path,
    estimator: str | None = None,
) -> None:
    """Сравнить признаки и проверить два профиля на лучшем наборе."""
    from validation import load_folds

    key, target = config["id_column"], config["target"]
    settings, boosting = config["validation"], config["boosting"]
    rows = train.sort_values(key)
    folds = load_folds(rows, key, settings["folds"], folds_file)
    sets = feature_sets(rows.drop(columns=[key, target]), boosting["reference_year"])
    price = rows[target].to_numpy()
    records = []
    for family in (estimator,) if estimator else FAMILIES:
        errors_by_set = {}
        for name, features in sets.items():
            scores = score(
                family, name, "base", features, price, folds,
                settings["seed"], boosting["threads"],
            )
            records.extend(scores)
            errors_by_set[name] = sum(row["squared_error"] for row in scores)
        best_set = min(errors_by_set, key=errors_by_set.__getitem__)
        for profile in ("slower", "deeper"):
            records.extend(score(
                family, best_set, profile, sets[best_set], price, folds,
                settings["seed"], boosting["threads"],
            ))

    details = pd.DataFrame(records)
    totals = details.groupby(["model", "features", "profile"], as_index=False).agg(
        squared_error=("squared_error", "sum"), rows=("rows", "sum"),
        fit_seconds=("fit_seconds", "sum"),
    )
    summary = totals.assign(
        oof_rmse=np.sqrt(totals["squared_error"] / totals["rows"]),
    ).drop(columns=["squared_error", "rows"]).sort_values("oof_rmse")
    output.mkdir(parents=True, exist_ok=True)
    details.to_csv(output / "rmse_by_fold.csv", index=False)
    summary.to_csv(output / "boosting.csv", index=False)
    print("BOOSTING " + json.dumps(summary.to_dict(orient="records")), flush=True)
