"""Сравнение конфигураций MLP на фиксированных фолдах."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn

from validation import load_folds

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRICE_SCALE = 100_000.0


@dataclass(frozen=True)
class Settings:
    """Параметры архитектуры и обучения MLP."""

    widths: tuple[int, ...] = (128,)
    activation: str = "relu"
    batch_norm: bool = False
    dropout: float = 0.0
    optimizer: str = "adam"
    learning_rate: float = 0.003
    scheduler: bool = False
    batch_size: int = 2048
    epochs: int = 12
    loss: str = "mse"


def experiments() -> dict[str, Settings]:
    """Поменять по одному фактору относительно общей конфигурации."""
    base = Settings()
    changes = {
        "deep": {"widths": (128, 64, 32)},
        "wide": {"widths": (256,)},
        "gelu": {"activation": "gelu"},
        "batchnorm": {"batch_norm": True},
        "dropout_10": {"dropout": 0.10},
        "dropout_25": {"dropout": 0.25},
        "adamw": {"optimizer": "adamw"},
        "sgd": {"optimizer": "sgd"},
        "cosine": {"scheduler": True},
        "low_lr": {"learning_rate": 0.001},
        "small_batch": {"batch_size": 1024},
        "longer": {"epochs": 20},
        "huber": {"loss": "huber"},
    }
    return {"base": base, **{name: replace(base, **options) for name, options in changes.items()}}


class PriceMLP(nn.Module):
    """Соединить числовые признаки и эмбеддинги категорий."""

    def __init__(self, numeric: int, cardinalities: list[int], settings: Settings) -> None:
        super().__init__()
        dimensions = [min(16, max(2, int(size**0.5))) for size in cardinalities]
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, dimension, padding_idx=0)
            for size, dimension in zip(cardinalities, dimensions, strict=True)
        )
        layers: list[nn.Module] = []
        previous = numeric + sum(dimensions)
        for width in settings.widths:
            layers.append(nn.Linear(previous, width))
            if settings.batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU() if settings.activation == "relu" else nn.GELU())
            if settings.dropout:
                layers.append(nn.Dropout(settings.dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        vectors = [embed(categorical[:, index]) for index, embed in enumerate(self.embeddings)]
        return self.layers(torch.cat([numeric, *vectors], dim=1)).squeeze(1)


def prepare_fold(
    features: pd.DataFrame, price: np.ndarray, valid: np.ndarray, device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], list[int]]:
    """Стандартизовать числа и зафиксировать словари на обучающей части."""
    train, test = features.loc[~valid], features.loc[valid]
    numeric = train.select_dtypes(include="number").columns
    train_num, test_num = train[numeric], test[numeric]
    fill = train_num.median().fillna(0)
    train_num, test_num = train_num.fillna(fill), test_num.fillna(fill)
    center, spread = train_num.mean(), train_num.std(ddof=0).replace(0, 1)
    train_num, test_num = (train_num - center) / spread, (test_num - center) / spread
    categories = train.columns.difference(numeric)
    train_codes, test_codes, cardinalities = [], [], []
    for column in categories:
        train_values = train[column].astype("string").fillna("__missing__")
        test_values = test[column].astype("string").fillna("__missing__")
        vocabulary = pd.Index(train_values.unique())
        train_codes.append(vocabulary.get_indexer(train_values) + 1)
        test_codes.append(vocabulary.get_indexer(test_values) + 1)
        cardinalities.append(len(vocabulary) + 1)

    def tensors(
        numbers: pd.DataFrame, codes: list[np.ndarray], targets: np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.as_tensor(numbers.to_numpy(), dtype=torch.float32, device=device),
            torch.as_tensor(np.column_stack(codes), dtype=torch.long, device=device),
            torch.as_tensor(targets / PRICE_SCALE, dtype=torch.float32, device=device),
        )

    return tensors(train_num, train_codes, price[~valid]), tensors(
        test_num, test_codes, price[valid]
    ), cardinalities


def train_fold(
    training: tuple[torch.Tensor, ...], validation: tuple[torch.Tensor, ...],
    cardinalities: list[int], settings: Settings, seed: int,
) -> tuple[list[dict], float]:
    """Обучить заданное число эпох и сохранить проверочную кривую."""
    numeric, categorical, target = training
    device = target.device
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = PriceMLP(numeric.shape[1], cardinalities, settings).to(device)
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    options = {"momentum": 0.9} if settings.optimizer == "sgd" else {}
    optimizer = optimizers[settings.optimizer](model.parameters(), lr=settings.learning_rate, **options)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.epochs)
        if settings.scheduler else None
    )
    loss_fn = nn.MSELoss() if settings.loss == "mse" else nn.HuberLoss(delta=1.0)
    curves = []
    started = perf_counter()
    for epoch in range(1, settings.epochs + 1):
        model.train()
        order = torch.randperm(len(target), device=device, generator=generator)
        epoch_loss = torch.zeros((), device=device)
        for batch in order.split(settings.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(numeric[batch], categorical[batch]), target[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach() * len(batch)
        if scheduler is not None:
            scheduler.step()
        model.eval()
        with torch.inference_mode():
            valid_num, valid_cat, valid_target = validation
            residuals = model(valid_num, valid_cat).double() - valid_target.double()
            squared = float(residuals.square().sum().item() * PRICE_SCALE**2)
        rmse = np.sqrt(squared / len(valid_target))
        if not np.isfinite(rmse):
            raise ValueError(f"Некорректный прогноз на эпохе {epoch}.")
        curves.append({
            "epoch": epoch, "train_loss": float(epoch_loss.item() / len(target)),
            "rows": len(valid_target), "squared_error": squared, "rmse": rmse,
        })
    return curves, perf_counter() - started


def save_curve(curves: pd.DataFrame, winner: str, output: Path) -> None:
    """Показать динамику RMSE базовой и лучшей конфигураций."""
    figure, axis = plt.subplots(figsize=(8, 4.4), layout="constrained")
    for name in dict.fromkeys(("base", winner)):
        grouped = curves.loc[curves["model"] == name].groupby("epoch")
        rmse = (grouped["squared_error"].sum() / grouped["rows"].sum()).pow(0.5)
        axis.plot(rmse.index, rmse, marker="o", markersize=3, label=name)
    axis.set(xlabel="Эпоха", ylabel="RMSE на проверочных фолдах", title="Кривые обучения MLP")
    axis.grid(alpha=0.25)
    axis.legend()
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        figure.savefig(output / "learning-curves.svg", metadata={"Date": None})
    plt.close(figure)


def evaluate(train: pd.DataFrame, config: dict, output: Path, folds_file: Path) -> None:
    """Сравнить настройки по прогнозам после заранее заданного числа эпох."""
    key, target = config["id_column"], config["target"]
    rows = train.sort_values(key)
    folds = load_folds(rows, key, config["validation"]["folds"], folds_file)
    features, price = rows.drop(columns=[key, target]), rows[target].to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MLP_DEVICE {device} · torch {torch.__version__}", flush=True)
    settings_by_name = experiments()
    details, curves = [], []
    for fold in np.unique(folds):
        training, validation, cardinalities = prepare_fold(features, price, folds == fold, device)
        for name, settings in settings_by_name.items():
            history, seconds = train_fold(
                training, validation, cardinalities, settings, config["validation"]["seed"] + int(fold),
            )
            curves.extend({"model": name, "fold": int(fold), **entry} for entry in history)
            details.append({"model": name, "fold": int(fold), "seconds": seconds, **history[-1]})
        print(f"MLP_FOLD {fold} complete", flush=True)

    fold_scores, curve_scores = pd.DataFrame(details), pd.DataFrame(curves)
    totals = fold_scores.groupby("model").sum(numeric_only=True)
    summary = totals[["seconds"]].assign(
        oof_rmse=np.sqrt(totals["squared_error"] / totals["rows"]),
    ).sort_values("oof_rmse")
    output.mkdir(parents=True, exist_ok=True)
    fold_scores.to_csv(output / "rmse_by_fold.csv", index=False)
    curve_scores.to_csv(output / "curves.csv", index=False)
    summary.to_csv(output / "mlp.csv")
    settings_json = json.dumps({name: asdict(value) for name, value in settings_by_name.items()}, indent=2)
    (output / "settings.json").write_text(settings_json + "\n", encoding="utf-8")
    save_curve(curve_scores, str(summary.index[0]), output)
    print("MLP_RESULTS " + json.dumps(summary.reset_index().to_dict(orient="records")), flush=True)
