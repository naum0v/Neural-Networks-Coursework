"""Чтение фиксированных фолдов по идентификаторам строк."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_folds(train: pd.DataFrame, key: str, count: int, source: Path) -> np.ndarray:
    """Вернуть номера фолдов в порядке строк train."""
    assignments = pd.read_csv(source).set_index(key)
    if (
        not assignments.index.is_unique
        or len(assignments) != len(train)
        or not train[key].isin(assignments.index).all()
        or set(assignments["fold"]) != set(range(1, count + 1))
    ):
        raise ValueError("Назначения фолдов не соответствуют обучающим данным.")
    return assignments.loc[train[key], "fold"].to_numpy()
