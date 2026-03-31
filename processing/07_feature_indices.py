"""07 — Compute index log-returns → features/index_returns.parquet

Reads cleaned/market_indices.parquet (wide: date + 30 level columns),
computes log-returns for each index, suffixes column names with '_ret'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CLEANED, FEATURES


def main() -> None:
    print("[07] Computing index returns ...")

    indices = pd.read_parquet(CLEANED / "market_indices.parquet")
    indices = indices.sort_values("date").reset_index(drop=True)

    value_cols = [c for c in indices.columns if c != "date"]

    # Log-returns for each index
    ret_df = indices[["date"]].copy()
    for col in value_cols:
        ret_col = f"{col}_ret"
        ret_df[ret_col] = np.log(indices[col] / indices[col].shift(1))
        ret_df[ret_col] = ret_df[ret_col].replace([np.inf, -np.inf], np.nan)

    # Drop first row (all NaN returns)
    ret_df = ret_df.iloc[1:].reset_index(drop=True)

    out = FEATURES / "index_returns.parquet"
    ret_df.to_parquet(out, index=False)

    ret_cols = [c for c in ret_df.columns if c != "date"]
    print(f"      Saved: {out}")
    print(f"      Shape: {ret_df.shape} ({len(ret_cols)} index return series)")


if __name__ == "__main__":
    main()
