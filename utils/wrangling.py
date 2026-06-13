from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_table(table: pl.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        table.write_parquet(path)
    elif path.suffix == ".csv":
        table.write_csv(path)
    else:
        raise ValueError("Unsupported file format: use .parquet or .csv")


def load_table(path: str | Path) -> pl.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return pl.read_csv(path)
    raise ValueError("Unsupported file format: use .parquet or .csv")


def merge_tables(left: pl.DataFrame, right: pl.DataFrame, on: str | list[str]) -> pl.DataFrame:
    return left.join(right, on=on, how="left")


def to_pandas(table: pl.DataFrame) -> Any:
    return table.to_pandas()
