"""Проверка ансамблей на вложенных фолдах и анализ ошибок цены."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import boosting
import classical
import neural
from validation import load_folds

BASE = ["catboost", "forest", "mlp"]
VOTES = np.array([0.5, 0.25, 0.25])
METHODS = BASE + ["mean", "voting", "stack_linear", "stack_ridge"]


def predict_base(
    features: pd.DataFrame, extended: pd.DataFrame, price: np.ndarray,
    valid: np.ndarray, config: dict, mlp_seed: int,
) -> np.ndarray:
    """Обучить три модели на общей обучающей части и вернуть прогнозы в долларах."""
    seed = config["validation"]["seed"]
    train, test, categories = boosting.prepare_fold(extended, valid, "catboost")
    cat = boosting.make_model("catboost", "slower", seed, config["boosting"]["threads"])
    cat.fit(train, price[~valid], cat_features=categories)
    cat_prices = cat.predict(test)

    _, forest, encoding, missing, scale = next(
        entry for entry in classical.experiments(seed) if entry[0] == "forest_ordinal"
    )
    pipeline = make_pipeline(classical.preprocess(features, encoding, missing, scale), forest)
    pipeline.fit(features.loc[~valid], price[~valid])
    forest_prices = pipeline.predict(features.loc[valid])

    tensors = neural.prepare_fold(features, price, valid, torch.device("cpu"))
    _, _, mlp_prices = neural.train_fold(*tensors, neural.experiments()["huber"], mlp_seed)
    predictions = np.column_stack([cat_prices, forest_prices, mlp_prices])
    if not np.isfinite(predictions).all():
        raise ValueError("Базовые модели вернули некорректные цены.")
    return predictions


def fold_predictions(
    features: pd.DataFrame, extended: pd.DataFrame, price: np.ndarray,
    folds: np.ndarray, fold: int, config: dict,
) -> dict[str, np.ndarray]:
    """Обучить метамодели только по внутренним отложенным прогнозам."""
    valid = folds == fold
    inner_x, inner_ext, inner_y = features.loc[~valid], extended.loc[~valid], price[~valid]
    seed = config["validation"]["seed"] + fold
    strata = pd.qcut(inner_y, config["validation"]["price_bins"], labels=False, duplicates="drop")
    splits = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    inner = np.full((len(inner_y), len(BASE)), np.nan)
    for part, (_, holdout) in enumerate(splits.split(inner_x, strata), start=1):
        mask = np.zeros(len(inner_y), dtype=bool)
        mask[holdout] = True
        inner[mask] = predict_base(inner_x, inner_ext, inner_y, mask, config, seed + part)
    if not np.isfinite(inner).all():
        raise ValueError("Неполное покрытие внутренней выборки.")

    outer = predict_base(features, extended, price, valid, config, seed)
    result = dict(zip(BASE, outer.T, strict=True))
    result["mean"] = outer.mean(axis=1)
    result["voting"] = outer @ VOTES
    result["stack_linear"] = LinearRegression().fit(inner, inner_y).predict(outer)
    # Средний квадрат ошибки плюс 0.01 * сумма квадратов стандартизованных коэффициентов.
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=0.01 * len(inner_y)))
    result["stack_ridge"] = ridge.fit(inner, inner_y).predict(outer)
    return result


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = prediction - actual
    return {
        "rows": len(error), "rmse": np.sqrt(np.mean(error**2)),
        "mae": np.abs(error).mean(), "bias": error.mean(),
    }


def error_groups(rows: pd.DataFrame, price: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    """Показать направление ошибки и вклад каждой группы в общий квадрат ошибки."""
    groups = {
        "price": pd.cut(price, [-np.inf, 20000, 50000, 100000, 200000, np.inf]),
        "milage": pd.cut(rows["milage"], [-np.inf, 30000, 100000, 200000, np.inf]),
        "model_year": pd.cut(rows["model_year"], [-np.inf, 2005, 2015, 2020, np.inf]),
    }
    records = []
    squared = (prediction - price) ** 2
    for feature, bins in groups.items():
        for label, indexes in pd.Series(bins).groupby(bins, observed=True, dropna=False).indices.items():
            records.append({
                "feature": feature, "group": str(label), **metrics(price[indexes], prediction[indexes]),
                "squared_error_share": squared[indexes].sum() / squared.sum(),
            })
    return pd.DataFrame(records)


def compare_stages(scores: pd.DataFrame) -> pd.DataFrame:
    """Собрать общую таблицу по сохранённым прогонам и текущей проверке."""
    root = Path(__file__).resolve().parent / "results"
    tables = []
    for stage, filename in [
        ("validation", "baselines"), ("classical", "classical"),
        ("boosting", "boosting"), ("mlp", "mlp"),
    ]:
        data = pd.read_csv(root / stage / f"{filename}.csv")
        if stage == "boosting":
            data["model"] = data["model"] + "/" + data["features"] + "/" + data["profile"]
        tables.append(data[["model", "oof_rmse"]].assign(stage=stage))
    ensembles = scores.loc[~scores["model"].isin(BASE), ["model", "rmse"]]
    tables.append(ensembles.rename(columns={"rmse": "oof_rmse"}).assign(stage="ensemble"))
    return pd.concat(tables, ignore_index=True).sort_values("oof_rmse")


def evaluate(train: pd.DataFrame, config: dict, output: Path, folds_file: Path) -> None:
    key, target = config["id_column"], config["target"]
    rows = train.sort_values(key).reset_index(drop=True)
    folds = load_folds(rows, key, config["validation"]["folds"], folds_file)
    features, price = rows.drop(columns=[key, target]), rows[target].to_numpy()
    extended = boosting.feature_sets(features, config["boosting"]["reference_year"])["all"]
    oof = pd.DataFrame(np.nan, index=rows.index, columns=METHODS)
    details = []
    for fold in np.unique(folds):
        valid = folds == fold
        for name, values in fold_predictions(features, extended, price, folds, int(fold), config).items():
            oof.loc[valid, name] = values
            details.append({"fold": fold, "model": name, **metrics(price[valid], values)})
        print(f"Завершён внешний фолд {fold}", flush=True)
    if not np.isfinite(oof.to_numpy()).all():
        raise ValueError("Пропущены или некорректны внешние прогнозы.")

    scores = pd.DataFrame([
        {"model": name, **metrics(price, oof[name].to_numpy())} for name in METHODS
    ]).sort_values("rmse")
    winner = scores.iloc[0]["model"]
    slices = error_groups(rows, price, oof[winner].to_numpy())
    overall = compare_stages(scores)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "ensemble.csv", index=False)
    overall.to_csv(output / "all_models.csv", index=False)
    pd.DataFrame(details).to_csv(output / "rmse_by_fold.csv", index=False)
    slices.to_csv(output / "errors_by_group.csv", index=False)
    oof.assign(**{key: rows[key], target: price, "fold": folds}).to_csv(
        output / "oof.csv.gz", index=False
    )

    best = overall.drop_duplicates("stage").sort_values("oof_rmse", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot(best["oof_rmse"], best["stage"], "o")
    axes[0].set(title="Лучший результат каждого этапа", xlabel="RMSE, $")
    ranked = scores.sort_values("rmse", ascending=False)
    axes[1].plot(ranked["rmse"], ranked["model"], "o")
    axes[1].set(title="Модели на пяти внешних фолдах", xlabel="RMSE, $")
    for axis in axes:
        axis.grid(axis="x", alpha=0.3)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
    fig.savefig(output / "comparison.svg", metadata={"Date": None})
    plt.close(fig)

    by_price = slices.loc[slices["feature"] == "price"]
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.barh(by_price["group"], by_price["squared_error_share"] * 100)
    axis.set(xlabel="Доля суммарного квадрата ошибки, %", ylabel="Фактическая цена, $",
             title=f"Какие автомобили определяют RMSE модели {winner}")
    fig.savefig(output / "price-errors.svg", metadata={"Date": None})
    plt.close(fig)
    print(scores.to_string(index=False), flush=True)
