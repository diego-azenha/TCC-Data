"""03 — Clean Bloomberg indices → cleaned/market_indices.parquet

Reads the 5-sheet Excel file, consolidates into a single wide DataFrame,
handles duplicates (BCOMINTR), interpolates small gaps, and filters dates.
"""
from __future__ import annotations

import pandas as pd

from config import BLOOMBERG_PATH, CLEANED, MIN_DATE
from io_utils import read_bloomberg_indices


def main() -> None:
    print("[03] Cleaning Bloomberg indices ...")

    indices = read_bloomberg_indices(BLOOMBERG_PATH)
    indices["date"] = pd.to_datetime(indices["date"])

    # Filter to period >= MIN_DATE
    indices = indices[indices["date"] >= pd.Timestamp(MIN_DATE)]
    indices = indices.sort_values("date").reset_index(drop=True)

    # Interpolate small gaps (≤ 3 days) per column
    value_cols = [c for c in indices.columns if c != "date"]
    for col in value_cols:
        indices[col] = indices[col].interpolate(method="linear", limit=3)

    out = CLEANED / "market_indices.parquet"
    indices.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Shape: {indices.shape} ({len(value_cols)} indices)")
    print(f"      Dates: {indices['date'].min().date()} → {indices['date'].max().date()}")

    # Coverage report
    for col in value_cols:
        pct = 100 * indices[col].notna().mean()
        if pct < 100:
            print(f"      {col}: {pct:.1f}% non-null")


if __name__ == "__main__":
    main()
