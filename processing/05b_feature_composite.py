"""05b — Build composite indicators: FCF Yield and FCF/Dívida

FCF Yield  = FCF (quarterly, ffilled to daily) / Market Cap (daily)
             → features/fcf_yield.parquet  (date, ticker, fcf_yield)

FCF/Dívida = FCF (quarterly) / Dívida Total Bruta (quarterly)
             → ratio computed at quarterly report dates, then ffilled to daily
             → features/fcf_divida_ffill.parquet  (date, ticker, fcf_divida)

Rules:
- Both FCF and Dívida inputs winsorized at p1/p99 before division.
- FCF Yield ratio also winsorized at p1/p99 after division.
- FCF/Dívida ratio also winsorized at p1/p99 after division.
- Dívida = 0 or abs(dívida) < 1.0 (R$1k threshold) → NaN (masked before division).
- Market Cap <= 0 → NaN (masked before division).
- Only forward-fill, never backfill (anti-lookahead).
- Universe restricted to tickers present in cleaned/prices.parquet.
"""
from __future__ import annotations

import pandas as pd

from config import CLEANED, FEATURES, FCF_PATH, DIVIDA_TOTAL_PATH, MKTCAP_PATH, MIN_DATE
from io_utils import read_economatica_wide


def _winsorize(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Winsorize a series at empirical quantiles, ignoring NaN."""
    q_lo = s.quantile(lo)
    q_hi = s.quantile(hi)
    return s.clip(lower=q_lo, upper=q_hi)


def main() -> None:
    print("[05b] Building composite indicators (FCF Yield, FCF/Dívida) ...")

    # --- Load universe from cleaned prices ---
    prices = pd.read_parquet(CLEANED / "prices.parquet")
    valid_tickers = set(prices["ticker"].unique())
    daily_calendar = (
        prices[["date", "ticker"]]
        .drop_duplicates()
        .sort_values(["ticker", "date"])
    )
    print(f"      Universe: {len(valid_tickers)} tickers")

    # --- Load raw inputs ---
    print("      Reading FCF ...")
    fcf = read_economatica_wide(FCF_PATH, "fcf")
    print("      Reading Dívida Total Bruta ...")
    divida = read_economatica_wide(DIVIDA_TOTAL_PATH, "divida_total")
    print("      Reading Valor de Mercado ...")
    mktcap = read_economatica_wide(MKTCAP_PATH, "mktcap")

    # --- Filter to valid tickers and MIN_DATE ---
    for df, name in [(fcf, "fcf"), (divida, "divida_total"), (mktcap, "mktcap")]:
        before = len(df)
        df = df[df["ticker"].isin(valid_tickers) & (df["date"] >= pd.Timestamp(MIN_DATE))]
        print(f"      {name}: {before} → {len(df)} rows after ticker/date filter")

    fcf = fcf[fcf["ticker"].isin(valid_tickers) & (fcf["date"] >= pd.Timestamp(MIN_DATE))].copy()
    divida = divida[divida["ticker"].isin(valid_tickers) & (divida["date"] >= pd.Timestamp(MIN_DATE))].copy()
    mktcap = mktcap[mktcap["ticker"].isin(valid_tickers) & (mktcap["date"] >= pd.Timestamp(MIN_DATE))].copy()

    # --- Winsorize raw inputs ---
    fcf["fcf"] = _winsorize(fcf["fcf"])
    divida["divida_total"] = _winsorize(divida["divida_total"])
    mktcap["mktcap"] = _winsorize(mktcap["mktcap"])

    # =========================================================
    # FCF / Dívida
    # =========================================================
    print("      Building FCF/Dívida ...")

    # Merge quarterly FCF and Dívida on (date, ticker)
    fd = pd.merge(fcf, divida, on=["date", "ticker"], how="inner")

    # Mask zero/near-zero denominators (< R$1k in thousands → abs < 1.0)
    fd["divida_total"] = fd["divida_total"].where(fd["divida_total"].abs() >= 1.0, other=pd.NA)

    fd["fcf_divida"] = fd["fcf"] / fd["divida_total"]
    fd = fd[["date", "ticker", "fcf_divida"]].dropna(subset=["fcf_divida"])

    # Winsorize ratio
    fd["fcf_divida"] = _winsorize(fd["fcf_divida"])

    # Forward-fill to daily calendar
    fd_daily = pd.merge(daily_calendar, fd, on=["date", "ticker"], how="left")
    fd_daily = fd_daily.sort_values(["ticker", "date"])
    fd_daily["fcf_divida"] = fd_daily.groupby("ticker")["fcf_divida"].ffill()

    out_fd = FEATURES / "fcf_divida_ffill.parquet"
    fd_daily[["date", "ticker", "fcf_divida"]].to_parquet(out_fd, index=False)
    non_null = fd_daily["fcf_divida"].notna().sum()
    total = len(fd_daily)
    print(f"      FCF/Dívida: {non_null}/{total} non-null rows ({100*non_null/total:.1f}%)")
    print(f"      Saved: {out_fd}")

    # =========================================================
    # FCF Yield = FCF (ffilled to daily) / Market Cap (daily)
    # =========================================================
    print("      Building FCF Yield ...")

    # Forward-fill quarterly FCF to daily calendar
    fcf_daily = pd.merge(daily_calendar, fcf[["date", "ticker", "fcf"]], on=["date", "ticker"], how="left")
    fcf_daily = fcf_daily.sort_values(["ticker", "date"])
    fcf_daily["fcf"] = fcf_daily.groupby("ticker")["fcf"].ffill()

    # Merge with daily market cap
    fy = pd.merge(fcf_daily, mktcap[["date", "ticker", "mktcap"]], on=["date", "ticker"], how="left")

    # Mask non-positive market cap before division
    fy["mktcap"] = fy["mktcap"].where(fy["mktcap"] > 0, other=pd.NA)

    fy["fcf_yield"] = fy["fcf"] / fy["mktcap"]
    fy = fy[["date", "ticker", "fcf_yield"]]

    # Winsorize ratio
    fy["fcf_yield"] = _winsorize(fy["fcf_yield"])

    out_fy = FEATURES / "fcf_yield.parquet"
    fy.to_parquet(out_fy, index=False)
    non_null = fy["fcf_yield"].notna().sum()
    total = len(fy)
    print(f"      FCF Yield: {non_null}/{total} non-null rows ({100*non_null/total:.1f}%)")
    print(f"      Saved: {out_fy}")


if __name__ == "__main__":
    main()
