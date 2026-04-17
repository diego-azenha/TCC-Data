"""06 — Forward-fill fundamentals onto daily calendar → features/fundamentals_ffill.parquet

For each fundamental metric:
1. Load the cleaned quarterly parquet
2. Merge onto the daily calendar from prices.parquet
3. Forward-fill per ticker (NO bfill — avoids look-ahead bias)
4. Combine all 5 metrics into a single wide long-format parquet
"""
from __future__ import annotations

import pandas as pd

from config import CLEANED, FEATURES, FFILL_LIMIT, FUNDAMENTAL_FILES


def main() -> None:
    print("[06] Forward-filling fundamentals ...")

    # Build daily calendar from prices (all unique dates)
    prices = pd.read_parquet(CLEANED / "prices.parquet", columns=["date", "ticker"])
    prices_calendar = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Expand calendar to fill short mid-history holes (≤5 consecutive missing days).
    # Some tickers have isolated absent dates in Economatica (source-data gaps, not NaN).
    # We add the missing trading-day rows so the ffill below can patch them.
    all_trading_dates = pd.Series(
        sorted(prices["date"].unique()), name="date"
    )
    ticker_bounds = (
        prices_calendar.groupby("ticker")["date"]
        .agg(first_date="min", last_date="max")
        .reset_index()
    )
    # Build full per-ticker active calendar
    expanded_rows = []
    for _, row in ticker_bounds.iterrows():
        t_dates = all_trading_dates[
            (all_trading_dates >= row["first_date"]) &
            (all_trading_dates <= row["last_date"])
        ].values
        expanded_rows.append(
            pd.DataFrame({"date": t_dates, "ticker": row["ticker"]})
        )
    full_calendar = pd.concat(expanded_rows, ignore_index=True)
    # Keep only the rows that were absent from prices (the holes)
    hole_rows = full_calendar.merge(
        prices_calendar.assign(_in_prices=True),
        on=["date", "ticker"], how="left"
    )
    hole_rows = hole_rows[hole_rows["_in_prices"].isna()][["date", "ticker"]]
    # Merge holes back — calendar now covers the full active window per ticker
    calendar = pd.concat([prices_calendar, hole_rows], ignore_index=True)
    calendar = calendar.sort_values(["ticker", "date"]).reset_index(drop=True)
    n_holes = len(hole_rows)
    print(f"      Expanded calendar: added {n_holes:,} hole rows across {hole_rows['ticker'].nunique()} tickers")

    # Sorted unique trading dates used to snap report dates that fall on
    # holidays (e.g. Dec 31) to the nearest prior trading day.
    trading_days = all_trading_dates

    # Start with the calendar as base
    result = calendar.copy()

    for name, (_, col_name) in FUNDAMENTAL_FILES.items():
        path = CLEANED / f"{name}.parquet"
        fund = pd.read_parquet(path)  # date, ticker, <col_name>

        # Snap any report dates that fall on non-trading days (holidays such
        # as Dec 31) to the nearest prior trading day so the merge picks them up.
        fund = fund.copy().sort_values("date")
        td_df = pd.DataFrame({"date": trading_days.values, "snapped": trading_days.values})
        fund = pd.merge_asof(fund, td_df, on="date", direction="backward")
        fund["date"] = fund.pop("snapped")
        # After snapping, a ticker may have two rows on the same date;
        # keep the last (most recent) value.
        fund = fund.sort_values(["ticker", "date"]).drop_duplicates(
            subset=["ticker", "date"], keep="last"
        )

        # Merge quarterly observations onto the daily grid
        merged = result[["date", "ticker"]].merge(
            fund, on=["date", "ticker"], how="left"
        )

        # Forward-fill per ticker: FFILL_LIMIT trading days.
        # Avoids carrying stale data across multiple missed reporting periods.
        # A second pass with limit=5 patches short mid-history holes (absent
        # source rows in Economatica) that fall inside the active window.
        merged[col_name] = merged.groupby("ticker")[col_name].ffill(limit=FFILL_LIMIT)
        merged[col_name] = merged.groupby("ticker")[col_name].ffill(limit=5)

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
