"""06 — Forward-fill fundamentals onto daily calendar → features/fundamentals_ffill.parquet

For each fundamental metric:
1. Load the cleaned quarterly parquet
2. Merge onto the daily calendar from prices.parquet
3. Forward-fill per ticker (NO bfill — avoids look-ahead bias)
4. Combine all 5 metrics into a single wide long-format parquet
"""
from __future__ import annotations

import pandas as pd

from config import CLEANED, FEATURES, FUNDAMENTAL_FILES


def main() -> None:
    print("[06] Forward-filling fundamentals ...")

    # Build daily calendar from prices (all unique dates)
    prices = pd.read_parquet(CLEANED / "prices.parquet", columns=["date", "ticker"])
    calendar = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Start with the calendar as base
    result = calendar.copy()

    for name, (_, col_name) in FUNDAMENTAL_FILES.items():
        path = CLEANED / f"{name}.parquet"
        fund = pd.read_parquet(path)  # date, ticker, <col_name>

        # Merge quarterly observations onto the daily grid
        merged = result[["date", "ticker"]].merge(
            fund, on=["date", "ticker"], how="left"
        )

        # Forward-fill per ticker (limited to 400 days ~= 1.6 years to avoid staleness)
        merged[col_name] = merged.groupby("ticker")[col_name].ffill(limit=400)

        result[col_name] = merged[col_name].values
        pct_fill = 100 * result[col_name].notna().mean()
        print(f"      {col_name}: {pct_fill:.1f}% filled after ffill")

    # Keep only date, ticker, and the 5 fundamental columns
    fund_cols = [v[1] for v in FUNDAMENTAL_FILES.values()]
    result = result[["date", "ticker"] + fund_cols]
    result = result.sort_values(["date", "ticker"]).reset_index(drop=True)

    out = FEATURES / "fundamentals_ffill.parquet"
    result.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Shape: {result.shape}")


if __name__ == "__main__":
    main()
