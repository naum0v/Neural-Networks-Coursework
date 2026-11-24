"""Стратифицированная проверка базовых прогнозов цены."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold


def evaluate(train: pd.DataFrame, config: dict, output: Path) -> None:
    """Получить по одному проверочному прогнозу каждого метода для каждого id."""
    key, target = config["id_column"], config["target"]
    settings = config["validation"]
    rows = train.sort_values(key).reset_index(drop=True)
    price = rows[target]
    strata = pd.qcut(price, settings["price_bins"], labels=False, duplicates="drop")
    if strata.isna().any() or strata.value_counts().min() < settings["folds"]:
        raise ValueError("Недостаточно наблюдений для стратификации по цене.")
    splitter = StratifiedKFold(settings["folds"], shuffle=True, random_state=settings["seed"])
    predictions = pd.DataFrame(index=rows.index, columns=["median", "brand_model", "linear"], dtype=float)
    folds = np.zeros(len(rows), dtype=int)
    features = ["model_year", "milage"]
    for fold, (fit, valid) in enumerate(splitter.split(rows, strata), start=1):
        training, validation = rows.iloc[fit], rows.iloc[valid]
        median = training[target].median()
        groups = training.groupby(["brand", "model"])[target].median()
        pairs = pd.MultiIndex.from_frame(validation[["brand", "model"]])
        predictions.loc[valid, "median"] = median
        predictions.loc[valid, "brand_model"] = groups.reindex(pairs).fillna(median).to_numpy()
        fill = training[features].median()
        model = LinearRegression().fit(training[features].fillna(fill), training[target])
        predictions.loc[valid, "linear"] = model.predict(validation[features].fillna(fill))
        folds[valid] = fold

    if (folds == 0).any() or not np.isfinite(predictions).all().all():
        raise ValueError("Проверочные прогнозы заполнены не полностью.")
    squared = predictions.sub(price, axis=0).pow(2)
    scores = squared.groupby(folds).mean().pow(0.5)
    scores.index.name = "fold"
    pooled = squared.mean().pow(0.5)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({key: rows[key], "fold": folds}).to_csv(output / "folds.csv", index=False)
    scores.to_csv(output / "rmse_by_fold.csv")
    pooled.rename("oof_rmse").to_csv(output / "baselines.csv", index_label="model")
    print("BASELINES " + json.dumps({"oof_rmse": pooled.to_dict(),
          "fold_rmse": scores.to_dict(), "sizes": pd.Series(folds).value_counts().sort_index().to_dict()}))
