"""01 — Clean closing prices → cleaned/prices.parquet

Reads fechamento.csv (Economatica wide format), converts to long,
filters for valid prices and dates >= MIN_DATE.
This parquet defines the master universe: (date, ticker) pairs with valid close > 0.
"""
from __future__ import annotations

import pandas as pd

from config import CLEANED, MIN_DATE, RAW_ECO
from io_utils import read_economatica_wide


def main() -> None:
    print("[01] Cleaning prices ...")

    prices = read_economatica_wide(RAW_ECO / "diario" / "fechamento.csv", "close")

    # Remove invalid prices
    prices = prices[prices["close"] > 0]

    # Filter to period >= MIN_DATE
    prices = prices[prices["date"] >= pd.Timestamp(MIN_DATE)]

    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

    out = CLEANED / "prices.parquet"
    prices.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Shape: {prices.shape}")
    print(f"      Tickers: {prices['ticker'].nunique()}")
    print(f"      Dates: {prices['date'].min().date()} → {prices['date'].max().date()}")


if __name__ == "__main__":
    main()
