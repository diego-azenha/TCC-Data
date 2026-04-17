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

from config import (
    CLEANED, MIN_DATE, RAW_ECO,
    VOL_THRESHOLD_K, VOL_MIN_FRAC_ABOVE,
)
from io_utils import read_economatica_wide


def main() -> None:
    print("[01] Cleaning prices ...")

    prices = read_economatica_wide(RAW_ECO / "diario" / "fechamento.csv", "close")

    # Remove invalid prices
    prices = prices[prices["close"] > 0]

    # Filter to period >= MIN_DATE
    prices = prices[prices["date"] >= pd.Timestamp(MIN_DATE)]

    # --- Liquidity filter -------------------------------------------------
    # Remove tickers where < 90% of trading days have volume >= R$5M/day
    print(f"      Applying liquidity filter: >= {VOL_MIN_FRAC_ABOVE*100:.0f}% of days with volume >= R${VOL_THRESHOLD_K/1000:.0f}M/day ...")
    volume_wide = pd.read_csv(RAW_ECO / "diario" / "volume.csv", low_memory=False)
    volume_wide = volume_wide.drop(columns=[volume_wide.columns[0]])
    volume_wide = volume_wide.rename(columns={volume_wide.columns[0]: "date"})
    volume_wide["date"] = pd.to_datetime(volume_wide["date"])
    vcols = volume_wide.columns[1:]
    volume_wide = volume_wide.rename(columns={c: c.split("|")[-1].strip() for c in vcols})
    volume_wide = volume_wide.loc[:, ~volume_wide.columns.duplicated()]
    volume_wide[volume_wide.columns[1:]] = volume_wide[volume_wide.columns[1:]].replace("-", float("nan"))
    for c in volume_wide.columns[1:]:
        volume_wide[c] = pd.to_numeric(volume_wide[c], errors="coerce")
    volume_wide = volume_wide.dropna(subset=volume_wide.columns[1:].tolist(), how="all")
    volume_wide = volume_wide.sort_values("date").reset_index(drop=True)

    liquid_tickers = set()
    for t in volume_wide.columns[1:]:
        s = volume_wide[t].replace(0, float("nan")).dropna()
        if len(s) == 0:
            continue
        # Check what fraction of days have volume >= threshold (no rolling window)
        frac_above = (s >= VOL_THRESHOLD_K).sum() / len(s)
        if frac_above >= VOL_MIN_FRAC_ABOVE:
            liquid_tickers.add(t)

    n_before = prices["ticker"].nunique()
    prices = prices[prices["ticker"].isin(liquid_tickers)]
    n_after = prices["ticker"].nunique()
    print(f"      Liquidity filter: {n_before} -> {n_after} tickers "
          f"({n_before - n_after} removed)")

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
