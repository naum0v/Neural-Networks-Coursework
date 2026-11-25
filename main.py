"""Проверка входных данных соревнования Kaggle."""

import argparse
import json
import tomllib
from pathlib import Path

import pandas as pd


def find_data_dir(competition: str) -> Path:
    """Найти каталог соревнования среди источников Kaggle."""
    matches = [
        path.parent
        for path in Path("/kaggle/input").rglob(f"{competition}/train.csv")
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Ожидался один источник {competition!r} в /kaggle/input, "
            f"найдено {len(matches)}. Укажите --data-dir явно."
        )
    return matches[0]


def validate_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    submission: pd.DataFrame,
    id_column: str,
    target: str,
) -> None:
    """Проверить схему файлов, идентификаторы и целевую переменную."""
    if target not in train or target in test:
        raise ValueError(f"Столбец {target!r} должен быть только в train среди train и test.")
    if set(train.columns) - {target} != set(test.columns):
        raise ValueError("Наборы признаков train.csv и test.csv различаются.")
    if list(submission.columns) != [id_column, target]:
        raise ValueError("Столбцы sample_submission.csv отличаются от ожидаемых.")

    for name, frame in (("train", train), ("test", test), ("sample_submission", submission)):
        if frame.empty or id_column not in frame:
            raise ValueError(f"{name}.csv пуст или не содержит {id_column!r}.")
        ids = frame[id_column]
        if ids.isna().any() or not ids.is_unique:
            raise ValueError(f"В {name}.csv есть пропуски или повторы идентификаторов.")

    if train[id_column].isin(test[id_column]).any():
        raise ValueError("Идентификаторы train.csv и test.csv пересекаются.")
    if not test[id_column].equals(submission[id_column]):
        raise ValueError("Идентификаторы и их порядок в test и sample_submission различаются.")

    prices = pd.to_numeric(train[target], errors="coerce")
    if prices.isna().any() or prices.abs().eq(float("inf")).any():
        raise ValueError(f"В {target!r} есть пропуски или некорректные числа.")


def main() -> None:
    """Прочитать три CSV и вывести сводку после проверки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--data-dir", type=Path, help="Каталог с CSV соревнования")
    parser.add_argument(
        "--stage", choices=["check", "baselines", "classical", "boosting", "mlp"], default="check",
    )
    parser.add_argument("--estimator", choices=["catboost", "lightgbm", "xgboost"])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds-file", type=Path, default=Path("results/validation/folds.csv"))
    args = parser.parse_args()
    with args.config.open("rb") as config_file:
        config = tomllib.load(config_file)

    data_dir = args.data_dir or find_data_dir(config["competition"])
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    submission = pd.read_csv(data_dir / "sample_submission.csv")
    validate_data(train, test, submission, config["id_column"], config["target"])
    if args.stage == "baselines":
        from baselines import evaluate

        evaluate(train, config, args.output_dir or Path("results/validation"))
        return
    if args.stage == "classical":
        from classical import evaluate

        evaluate(train, config, args.output_dir or Path("results/classical"), args.folds_file)
        return
    if args.stage == "boosting":
        from boosting import evaluate

        evaluate(
            train, config, args.output_dir or Path("results/boosting"),
            args.folds_file, args.estimator,
        )
        return

    if args.stage == "mlp":
        from neural import evaluate

        evaluate(train, config, args.output_dir or Path("results/mlp"), args.folds_file)
        return

    summary = {
        "competition": config["competition"],
        "data_dir": str(data_dir),
        "pandas_version": pd.__version__,
        "target": config["target"],
        "files": {
            "train.csv": {"rows": len(train), "columns": len(train.columns)},
            "test.csv": {"rows": len(test), "columns": len(test.columns)},
            "sample_submission.csv": {
                "rows": len(submission),
                "columns": len(submission.columns),
            },
        },
        "features": [column for column in test.columns if column != config["id_column"]],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
