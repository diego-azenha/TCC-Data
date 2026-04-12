"""02 — Clean fundamental CSVs → cleaned/{metric}.parquet

Reads each of the 5 quarterly Economatica CSVs, converts to long,
drops the daily-grid NaN rows (keeping only ~200 actual quarter-end dates),
and filters tickers to those present in prices.parquet.
"""
from __future__ import annotations

import pandas as pd

from config import CLEANED, FUNDAMENTAL_FILES, MIN_DATE, RAW_ECO, TRAIN_END
from io_utils import read_economatica_wide


def main() -> None:
    print("[02] Cleaning fundamentals ...")

    # Load universe of valid tickers from prices
    prices = pd.read_parquet(CLEANED / "prices.parquet", columns=["ticker"])
    valid_tickers = set(prices["ticker"].unique())
    print(f"      Universe: {len(valid_tickers)} tickers from prices.parquet")

    for name, (filename, col_name) in FUNDAMENTAL_FILES.items():
        path = RAW_ECO / filename
        df = read_economatica_wide(path, col_name)

        # Filter to valid tickers and period
        df = df[df["ticker"].isin(valid_tickers)]
        df = df[df["date"] >= pd.Timestamp(MIN_DATE)]

        # Winsorize at 1st/99th percentile (calculated on train set only)
        train_df = df[df["date"] <= pd.Timestamp(TRAIN_END)]
        p01 = train_df[col_name].quantile(0.01)
        p99 = train_df[col_name].quantile(0.99)
        df[col_name] = df[col_name].clip(lower=p01, upper=p99)
        print(f"      {col_name} winsorized to [{p01:.2f}, {p99:.2f}] (train-only bounds)")

        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        out = CLEANED / f"{name}.parquet"
        df.to_parquet(out, index=False)

        print(f"      {name}: {df.shape[0]} rows, "
              f"{df['ticker'].nunique()} tickers, "
              f"{df['date'].nunique()} dates → {out.name}")


if __name__ == "__main__":
    main()
