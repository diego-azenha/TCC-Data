"""01 — Clean closing prices → cleaned/prices.parquet

Reads fechamento.csv (Economatica wide format), converts to long,
filters for valid prices and dates >= MIN_DATE.
Deduplicates ON/PN pairs by selecting ticker with highest average volume.
This parquet defines the master universe: (date, ticker) pairs with valid close > 0.
"""
from __future__ import annotations

import re
from collections import defaultdict

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

    # Deduplicate ON/PN by selecting ticker with highest average volume
    print("      Deduplicating ON/PN pairs by average volume ...")
    volume = read_economatica_wide(RAW_ECO / "diario" / "volume.csv", "volume")
    volume = volume[volume["date"] >= pd.Timestamp(MIN_DATE)]
    avg_vol = volume.groupby("ticker")["volume"].mean()

    # Extract company base (alphabetic part of ticker code)
    ticker_list = prices["ticker"].unique()
    base_map = {t: re.match(r"^([A-Z]+)", t).group(1) if re.match(r"^([A-Z]+)", t) else t 
                for t in ticker_list}

    base_to_tickers = defaultdict(list)
    for t, b in base_map.items():
        base_to_tickers[b].append(t)

    keep_tickers = set()
    n_deduped = 0
    for base, tickers in base_to_tickers.items():
        if len(tickers) == 1:
            keep_tickers.add(tickers[0])
        else:
            vols = {t: avg_vol.get(t, 0.0) for t in tickers}
            best = max(vols, key=vols.get)
            keep_tickers.add(best)
            dropped = [t for t in tickers if t != best]
            n_deduped += len(dropped)
            print(f"        {base}: kept {best} (vol={vols[best]:.0f}), dropped {dropped}")

    prices = prices[prices["ticker"].isin(keep_tickers)]
    print(f"      Deduplication: {n_deduped} tickers removed, {len(keep_tickers)} kept")

    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

    out = CLEANED / "prices.parquet"
    prices.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Shape: {prices.shape}")
    print(f"      Tickers: {prices['ticker'].nunique()}")
    print(f"      Dates: {prices['date'].min().date()} → {prices['date'].max().date()}")


if __name__ == "__main__":
    main()
