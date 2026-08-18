"""Data loading helpers shared by the desktop app and scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from tslm_multitask.data.masks import observed_mask_from_values


def find_header_row(df: pd.DataFrame, data_type: str) -> int:
    """Locate the first likely header row for QAR or baseline engine data."""
    for row_idx in range(min(20, len(df))):
        if data_type == "qar":
            value = str(df.iloc[row_idx, 1]).upper() if df.shape[1] > 1 else ""
            if "FRAME" in value or "计数" in value:
                return row_idx
        else:
            value = str(df.iloc[row_idx, 0]).upper() if df.shape[1] > 0 else ""
            if "FLIGHT" in value:
                return row_idx
    return 0


def read_table_with_smart_header(file_path: str | Path, data_type: str) -> pd.DataFrame:
    """Read a CSV/Excel file and promote the detected header row to columns."""
    path = Path(file_path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, header=None, low_memory=False)
    else:
        df = pd.read_excel(path, header=None)

    header_idx = find_header_row(df, data_type)
    df.columns = df.iloc[header_idx].astype(str).str.strip()
    return df.iloc[header_idx + 1 :].reset_index(drop=True)


def load_multivariate_values(
    file_path: str | Path,
    columns: Sequence[str],
    data_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load selected columns as float values plus an explicit observed mask.

    Missing or non-numeric cells are preserved as NaN in the returned values.
    Numeric zero remains a valid observed value.
    """
    df = read_table_with_smart_header(file_path, data_type)

    data_list: list[np.ndarray] = []
    observed_mask_list: list[np.ndarray] = []
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"数据文件中找不到特征列: '{col}'。请检查数据类型选择是否正确。")
        col_data = pd.to_numeric(df[col], errors="coerce").values.astype(np.float32)
        data_list.append(col_data)
        observed_mask_list.append(observed_mask_from_values(col_data))

    return np.array(data_list, dtype=np.float32), np.array(observed_mask_list, dtype=bool)
