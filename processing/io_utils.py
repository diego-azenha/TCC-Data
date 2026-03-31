"""Reusable I/O helpers for reading Economatica and Bloomberg raw data."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def read_economatica_wide(path: Path, value_name: str) -> pd.DataFrame:
    """Read an Economatica wide CSV and return a long DataFrame(date, ticker, <value_name>)."""
    df = pd.read_csv(path, low_memory=False)

    # Drop the constant 'Ativo' column (col 0)
    df = df.drop(columns=[df.columns[0]])

    # Rename date column
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])

    # Extract ticker codes from column headers (format: "Metric|...|TICKER")
    data_cols = df.columns[1:]
    ticker_map = {col: col.split("|")[-1].strip() for col in data_cols}
    df = df.rename(columns=ticker_map)

    # Handle duplicate column names — keep first occurrence
    df = df.loc[:, ~df.columns.duplicated()]

    # Replace '-' with NaN and convert to numeric
    data_cols_new = df.columns[1:]
    df[data_cols_new] = df[data_cols_new].replace("-", np.nan)
    for col in data_cols_new:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that are entirely NaN (no data for any ticker)
    df = df.dropna(subset=data_cols_new, how="all")

    # Wide → long
    df_long = df.melt(id_vars=["date"], var_name="ticker", value_name=value_name)
    df_long = df_long.dropna(subset=[value_name])
    df_long = df_long.sort_values(["date", "ticker"]).reset_index(drop=True)

    return df_long


def read_bloomberg_indices(path: Path) -> pd.DataFrame:
    """Read all Bloomberg sheets, return wide DataFrame(date, col1, col2, ...)."""
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in xl.sheet_names:
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        # Row 0 = Bloomberg tickers, Row 1 = "Dates"/"Last Price", data from row 2
        raw_names = [str(c).strip() for c in df.iloc[0, 1:].values if pd.notna(c)]

        # Deduplicate within sheet (e.g. BCOMINTR Index appears twice)
        seen: dict[str, int] = {}
        deduped: list[str] = []
        for name in raw_names:
            if name in seen:
                seen[name] += 1
                deduped.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                deduped.append(name)

        # Only keep columns with valid names (skip trailing NaN cols)
        n_data_cols = len(deduped)
        df = df.iloc[2:, : n_data_cols + 1].copy()
        df.columns = ["date"] + deduped
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df.set_index("date"))

    combined = pd.concat(frames, axis=1)

    # Drop duplicate columns that survived concat
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Drop renamed duplicates (e.g. "BCOMINTR Index_1" when "BCOMINTR Index" exists)
    base_names = set(combined.columns)
    drop_cols = [
        c for c in combined.columns
        if re.search(r'_\d+$', c) and re.sub(r'_\d+$', '', c) in base_names
    ]
    if drop_cols:
        combined = combined.drop(columns=drop_cols)

    combined = combined.reset_index()
    return combined
