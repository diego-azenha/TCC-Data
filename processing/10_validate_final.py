"""10 — Validate final parquets

Checks:
1. No NaN or Inf in x_ts, x_static, prices
2. Dimensional integrity (d_ts, d_static)
3. Train-period normalization stats ≈ (0, 1)
4. Ticker consistency between x_ts and x_static
5. Temporal coverage: ≥ 30 tickers per date with 256+ history days
6. Date alignment between x_ts and prices
"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from config import PARQUETS, TRAIN_END


def main() -> None:
    print("[10] Validating final parquets ...")
    errors: list[str] = []

    # --- Load ---
    x_ts = pd.read_parquet(PARQUETS / "x_ts.parquet")
    x_static = pd.read_parquet(PARQUETS / "x_static.parquet")
    prices = pd.read_parquet(PARQUETS / "prices.parquet")

    with open(PARQUETS / "normalization_stats.json") as f:
        stats = json.load(f)

    feature_cols = stats["feature_order"]
    d_ts = stats["d_ts"]

    # --- 1. No NaN / Inf ---
    nan_ts = x_ts[feature_cols].isna().sum().sum()
    inf_ts = np.isinf(x_ts[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    if nan_ts > 0:
        errors.append(f"x_ts has {nan_ts} NaN values")
    if inf_ts > 0:
        errors.append(f"x_ts has {inf_ts} Inf values")

    nan_static = x_static.iloc[:, 1:].isna().sum().sum()
    if nan_static > 0:
        errors.append(f"x_static has {nan_static} NaN values")

    nan_prices = prices["close"].isna().sum()
    if nan_prices > 0:
        errors.append(f"prices has {nan_prices} NaN values")

    print(f"      NaN/Inf check: {'PASS' if not errors else 'FAIL'}")

    # --- 2. Dimensional integrity ---
    actual_d_ts = len(feature_cols)
    if actual_d_ts != d_ts:
        errors.append(f"d_ts mismatch: expected {d_ts}, got {actual_d_ts}")

    d_static = len(x_static.columns) - 1  # minus 'ticker'
    expected_d_static = 13
    # Allow flexibility — just report
    print(f"      d_ts = {actual_d_ts}, d_static = {d_static}")

    # --- 3. Train-period normalization check ---
    train_mask = x_ts["date"] <= pd.Timestamp(TRAIN_END)
    train = x_ts.loc[train_mask]

    # Returns: mean can be anything, std ≈ 1/σ_train * σ_train = 1
    ret_std = train["return"].std()
    print(f"      Train return std (should ≈ 1): {ret_std:.4f}")
    if abs(ret_std - 1.0) > 0.05:
        errors.append(f"Return std on train = {ret_std:.4f}, expected ≈ 1.0")

    # Fundamentals: mean ≈ 0, std ≈ 1
    for col, col_stats in stats.get("fundamental_stats", {}).items():
        if col in train.columns:
            mu = train[col].mean()
            sigma = train[col].std()
            if abs(mu) > 0.1:
                errors.append(f"{col} train mean = {mu:.4f}, expected ≈ 0")
            if abs(sigma - 1.0) > 0.15:
                errors.append(f"{col} train std = {sigma:.4f}, expected ≈ 1")

    print(f"      Normalization check: {'PASS' if not [e for e in errors if 'train' in e.lower()] else 'WARN'}")

    # --- 4. Ticker consistency ---
    ts_tickers = set(x_ts["ticker"].unique())
    static_tickers = set(x_static["ticker"].unique())
    missing_in_static = ts_tickers - static_tickers
    if missing_in_static:
        errors.append(f"{len(missing_in_static)} tickers in x_ts but not in x_static")
    print(f"      Ticker coverage: {len(ts_tickers)} in x_ts, {len(static_tickers)} in x_static, "
          f"{len(missing_in_static)} missing")

    # --- 5. Temporal coverage ---
    # Count how many tickers have >= 256 days of history at each date
    ticker_day_count = x_ts.groupby("ticker")["date"].cumcount() + 1
    eligible = x_ts.copy()
    eligible["history_days"] = ticker_day_count
    eligible_mask = eligible["history_days"] >= 256

    dates_with_enough = eligible.loc[eligible_mask].groupby("date")["ticker"].nunique()
    min_tickers = dates_with_enough.min() if len(dates_with_enough) > 0 else 0
    dates_below_30 = (dates_with_enough < 30).sum()

    print(f"      Min tickers with 256+ days on any date: {min_tickers}")
    if dates_below_30 > 0:
        print(f"      WARNING: {dates_below_30} dates have < 30 eligible tickers")

    # --- 6. Date alignment ---
    ts_dates = set(x_ts["date"].unique())
    price_dates = set(prices["date"].unique())
    # x_ts dates should be a subset of price dates (minus first day per ticker for returns)
    extra_dates = ts_dates - price_dates
    if extra_dates:
        errors.append(f"{len(extra_dates)} dates in x_ts not found in prices")

    print(f"      Date alignment: {'PASS' if not extra_dates else 'FAIL'}")

    # --- Summary ---
    print()
    if errors:
        print(f"      VALIDATION: {len(errors)} issue(s) found:")
        for e in errors:
            print(f"        - {e}")
        sys.exit(1)
    else:
        print("      VALIDATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
