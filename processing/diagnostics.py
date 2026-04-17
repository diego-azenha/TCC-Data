"""
diagnostics.py -- Focused data quality checks for the TCC pipeline.

Run from the project root or from processing/:
    python processing/diagnostics.py

Analyses:
  1. Number of available stocks over time
  2. Return distribution at 12 dates (4 rows x 3 cols)
  3. Missing-data rate per series vs full calendar (post-pipeline ffill)
  4. Weekend / holiday removal check (raw CSV + cleaned parquets)
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    yf = None

warnings.filterwarnings("ignore")

# -- Path setup ---------------------------------------------------------------
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from config import (
    CLEANED, FEATURES, PARQUETS,
    TRAIN_END, VAL_END, MIN_DATE,
    FUNDAMENTAL_FILES,
    VOL_THRESHOLD_K, VOL_MIN_FRAC_ABOVE,
)

TRAIN_END_TS = pd.Timestamp(TRAIN_END)
VAL_END_TS   = pd.Timestamp(VAL_END)
MIN_DATE_TS  = pd.Timestamp(MIN_DATE)

SEP = "=" * 70


# -- 1. Stocks available over time + BOVESPA --------------------------------
def check_stocks_over_time(prices: pd.DataFrame) -> None:
    print(SEP)
    print("1. STOCKS AVAILABLE OVER TIME + BOVESPA")
    print(SEP)

    monthly = (
        prices.groupby(prices["date"].dt.to_period("M"))["ticker"]
        .nunique()
        .rename("n_tickers")
    )
    monthly.index = monthly.index.to_timestamp()

    print(f"{'Date':<12}  {'Tickers':>8}")
    print("-" * 22)
    for dt, n in monthly.items():
        if dt.month == 1:
            print(f"{str(dt.date()):<12}  {n:>8,}")

    print(f"\nOverall:  min={monthly.min():,}  max={monthly.max():,}  "
          f"median={monthly.median():.0f}")

    # Fetch BOVESPA data
    bovespa = None
    if yf is not None:
        try:
            print("  Fetching BOVESPA data from yfinance ...")
            import sys
            from io import StringIO
            # Suppress yfinance download output
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                bovespa = yf.download("^BVSP", start="2005-01-01", end="2026-04-02")
            finally:
                sys.stdout = old_stdout
            
            # Handle different column naming conventions
            if "Adj Close" in bovespa.columns:
                close_col = "Adj Close"
            elif "Close" in bovespa.columns:
                close_col = "Close"
            else:
                # Fallback: use the first numeric column
                close_col = [c for c in bovespa.columns if bovespa[c].dtype in ['float64', 'float32']][0]
            
            bovespa["returns"] = bovespa[close_col].pct_change()
            bovespa["cum_returns"] = (1 + bovespa["returns"]).cumprod()
            print(f"  BOVESPA: {len(bovespa)} trading days fetched (using '{close_col}' column)")
        except Exception as e:
            print(f"  [WARN] Could not fetch BOVESPA: {e}")
            bovespa = None

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # -- Subplot 1: Stock count --
    ax = axes[0]
    ax.plot(monthly.index, monthly.values, linewidth=1.2, color="steelblue", label="# stocks")
    ax.axvline(TRAIN_END_TS, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="train end (2019-12-31)")
    ax.axvline(VAL_END_TS,   color="red",    linestyle="--", linewidth=1.5, alpha=0.8, label="val end (2022-12-31)")
    ax.fill_between(monthly.index, monthly.min(), monthly.max(),
                     where=(monthly.index <= TRAIN_END_TS),
                     alpha=0.1, color="orange", label="train")
    ax.fill_between(monthly.index, monthly.min(), monthly.max(),
                     where=((monthly.index > TRAIN_END_TS) & (monthly.index <= VAL_END_TS)),
                     alpha=0.1, color="red", label="val")
    ax.fill_between(monthly.index, monthly.min(), monthly.max(),
                     where=(monthly.index > VAL_END_TS),
                     alpha=0.1, color="green", label="test")
    ax.set_title("1a -- Stocks available per month", fontsize=11, fontweight="bold")
    ax.set_ylabel("# tickers")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    # -- Subplot 2: BOVESPA cumulative returns --
    ax = axes[1]
    if bovespa is not None:
        ax.plot(bovespa.index, bovespa["cum_returns"], linewidth=1.2, color="darkgreen", label="BOVESPA cumulative return")
        ax.axvline(TRAIN_END_TS, color="orange", linestyle="--", linewidth=1.5, alpha=0.8, label="train end")
        ax.axvline(VAL_END_TS,   color="red",    linestyle="--", linewidth=1.5, alpha=0.8, label="val end")
        ax.fill_between(bovespa.index, bovespa["cum_returns"].min(), bovespa["cum_returns"].max(),
                         where=(bovespa.index <= TRAIN_END_TS),
                         alpha=0.1, color="orange")
        ax.fill_between(bovespa.index, bovespa["cum_returns"].min(), bovespa["cum_returns"].max(),
                         where=((bovespa.index > TRAIN_END_TS) & (bovespa.index <= VAL_END_TS)),
                         alpha=0.1, color="red")
        ax.fill_between(bovespa.index, bovespa["cum_returns"].min(), bovespa["cum_returns"].max(),
                         where=(bovespa.index > VAL_END_TS),
                         alpha=0.1, color="green")
        ax.set_title("1b -- BOVESPA index cumulative returns with train/val/test split", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative return")
        ax.set_xlabel("Date")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "BOVESPA data unavailable\n(yfinance not installed or connection error)",
                ha="center", va="center", fontsize=12, transform=ax.transAxes)
        ax.set_title("1b -- BOVESPA index (unavailable)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(_here.parent / "diagnostics_1_stocks_over_time.png", dpi=110)
    plt.close()
    print("  -> Chart saved: diagnostics_1_stocks_over_time.png")


# -- 2. Return distribution at 12 representative dates (4 rows x 3 cols) -----
def check_return_distributions(returns: pd.DataFrame) -> None:
    print()
    print(SEP)
    print("2. RETURN DISTRIBUTION -- 12 CROSS-SECTIONS (4 rows x 3 cols)")
    print(SEP)

    all_dates = sorted(returns["date"].unique())
    n = len(all_dates)
    # 12 evenly-spaced indices across the full timeline
    idx_picks = [int(n * k / 11) for k in range(12)]
    idx_picks[-1] = min(idx_picks[-1], n - 1)
    sample_dates = [pd.Timestamp(all_dates[i]) for i in idx_picks]

    fig, axes = plt.subplots(4, 3, figsize=(15, 14))
    axes = axes.flatten()

    for ax, dt in zip(axes, sample_dates):
        subset = returns[returns["date"] == dt]["return"].dropna()
        period = ("train" if dt <= TRAIN_END_TS
                  else "val" if dt <= VAL_END_TS else "test")

        if len(subset) < 5:
            ax.set_title(f"{dt.date()} ({period}) -- not enough data", fontsize=8)
            continue

        mu, sigma = subset.mean(), subset.std()
        clip = 5 * sigma
        disp = subset.clip(-clip, clip)

        ax.hist(disp, bins=40, density=True, color="steelblue",
                alpha=0.75, edgecolor="white", label="returns")
        xx = np.linspace(disp.min(), disp.max(), 300)
        ax.plot(xx, stats.norm.pdf(xx, mu, sigma), "r-", linewidth=1.5, label="Normal")

        skew = float(stats.skew(subset))
        kurt = float(stats.kurtosis(subset))
        ax.set_title(f"{dt.date()}  [{period}]  n={len(subset)}", fontsize=8)
        ax.text(0.97, 0.97,
                f"mu={mu:.4f}\nsigma={sigma:.4f}\nskew={skew:.2f}\nkurt={kurt:.2f}",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax.legend(fontsize=6)

        print(f"  {dt.date()} ({period}): n={len(subset):4d}  "
              f"mu={mu:.5f}  sigma={sigma:.5f}  "
              f"skew={skew:.2f}  kurt={kurt:.2f}")

    fig.suptitle("2 -- Cross-sectional return distribution at 12 dates", fontsize=12)
    plt.tight_layout()
    plt.savefig(_here.parent / "diagnostics_2_return_distributions.png", dpi=110)
    plt.close()
    print("  -> Chart saved: diagnostics_2_return_distributions.png")


# -- 3. Missing data rate per series vs full calendar ------------------------
def check_missing_rates() -> None:
    print()
    print(SEP)
    print("3. MISSING DATA RATE PER SERIES")
    print("   Denominator: full (date x ticker) calendar from prices.parquet.")
    print("   Quarterly: pipeline ffill limit = 65 trading days (~1 quarter, fundamentals_ffill.parquet).")
    print(SEP)

    FLAG_THRESHOLD = 10.0

    calendar = pd.read_parquet(CLEANED / "prices.parquet", columns=["date", "ticker"])
    cal_size  = len(calendar)

    rows = []

    # Quarterly -- use fundamentals_ffill.parquet (already on full calendar, limit=65)
    fund_ffill = pd.read_parquet(FEATURES / "fundamentals_ffill.parquet")
    for name in ["roa", "roe", "margem_bruta", "divida_bruta_ativo", "divida_liq_pl"]:
        col = FUNDAMENTAL_FILES[name][1]
        if col not in fund_ffill.columns:
            continue
        n_nan = fund_ffill[col].isna().sum()
        rows.append({
            "series"     : col,
            "type"       : "quarterly (65d ffill)",
            "cal_size"   : cal_size,
            "n_missing"  : int(n_nan),
            "pct_missing": round(100 * n_nan / cal_size, 2),
        })

    # Daily -- left-join cleaned parquet onto calendar
    for name in ["pvpa", "preco_lucro", "volume"]:
        col  = FUNDAMENTAL_FILES[name][1]
        path = CLEANED / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["date", "ticker", col])
        merged = calendar.merge(df, on=["date", "ticker"], how="left")
        n_nan  = merged[col].isna().sum()
        rows.append({
            "series"     : col,
            "type"       : "daily (no ffill)",
            "cal_size"   : cal_size,
            "n_missing"  : int(n_nan),
            "pct_missing": round(100 * n_nan / cal_size, 2),
        })

    # Return feature only (fcf_divida, fcf_yield, ev_ebitda dropped)
    for label, fname, col in [
        ("return", "returns.parquet", "return"),
    ]:
        path = FEATURES / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["date", "ticker", col])
        merged = calendar.merge(df, on=["date", "ticker"], how="left")
        n_nan  = merged[col].isna().sum()
        rows.append({
            "series"     : label,
            "type"       : "composite/feature",
            "cal_size"   : cal_size,
            "n_missing"  : int(n_nan),
            "pct_missing": round(100 * n_nan / cal_size, 2),
        })

    print(f"\n{'Series':<25}  {'Type':<25}  {'Calendar':>10}  {'Missing':>9}  {'%Missing':>9}  {'Flag':>7}")
    print("-" * 92)
    flagged = []
    for r in rows:
        flag = "[>10%]" if r["pct_missing"] > FLAG_THRESHOLD else ""
        if flag:
            flagged.append(r["series"])
        print(f"{r['series']:<25}  {r['type']:<25}  {r['cal_size']:>10,}  "
              f"{r['n_missing']:>9,}  {r['pct_missing']:>8.2f}%  {flag:>7}")

    print()
    if flagged:
        print(f"[FLAG] Series with >10% missing after ffill: {', '.join(flagged)}")
    else:
        print("[OK] No series exceeds 10% missing data.")


# -- 3b. Missing data rate -- in-window only ---------------------------------
def _in_window_stats(
    series_df: pd.DataFrame,
    value_col: str,
) -> tuple[int, int]:
    """
    For each ticker in series_df, find the first and last non-NaN date for
    value_col. Count NaN only within that per-ticker window using only dates
    that exist in series_df (avoids inflating counts with dates when the stock
    was not in the universe). Return (n_nan_in_window, window_size).
    """
    data_rows = series_df.dropna(subset=[value_col])
    if len(data_rows) == 0:
        return 0, 0

    bounds = (
        data_rows.groupby("ticker")["date"]
        .agg(first_date="min", last_date="max")
        .reset_index()
    )

    # Use only dates that actually exist in series_df for each ticker
    # (avoids counting dates when the stock wasn't in the universe at all)
    active = (
        series_df[["date", "ticker"]]
        .merge(bounds, on="ticker", how="inner")
        .query("date >= first_date and date <= last_date")
        [["date", "ticker"]]
        .reset_index(drop=True)
    )
    if len(active) == 0:
        return 0, 0

    merged = active.merge(
        series_df[["date", "ticker", value_col]],
        on=["date", "ticker"], how="left"
    )
    return int(merged[value_col].isna().sum()), len(active)


def check_missing_rates_in_window() -> None:
    print()
    print(SEP)
    print("3b. MISSING DATA RATE -- IN-WINDOW ONLY")
    print("    Denominator: trading days between first and last non-NaN observation")
    print("    for each (ticker, series) pair. Pre-listing and post-delisting NaNs excluded.")
    print(SEP)

    FLAG_THRESHOLD = 10.0

    rows = []

    # Quarterly
    fund_ffill = pd.read_parquet(FEATURES / "fundamentals_ffill.parquet")
    for name in ["roa", "roe", "margem_bruta", "divida_bruta_ativo", "divida_liq_pl"]:
        col = FUNDAMENTAL_FILES[name][1]
        if col not in fund_ffill.columns:
            continue
        n_nan, win_size = _in_window_stats(
            fund_ffill[["date", "ticker", col]], col
        )
        pct = round(100 * n_nan / win_size, 2) if win_size else 0.0
        rows.append(dict(series=col, type="quarterly (65d ffill)",
                         win_size=win_size, n_missing=n_nan, pct=pct))

    # Daily
    for name in ["pvpa", "preco_lucro", "volume"]:
        col  = FUNDAMENTAL_FILES[name][1]
        path = CLEANED / f"{name}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["date", "ticker", col])
        n_nan, win_size = _in_window_stats(df, col)
        pct = round(100 * n_nan / win_size, 2) if win_size else 0.0
        rows.append(dict(series=col, type="daily (no ffill)",
                         win_size=win_size, n_missing=n_nan, pct=pct))

    # Return only (fcf_divida, fcf_yield, ev_ebitda dropped)
    for label, fname, col in [
        ("return", "returns.parquet", "return"),
    ]:
        path = FEATURES / fname
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["date", "ticker", col])
        n_nan, win_size = _in_window_stats(df, col)
        pct = round(100 * n_nan / win_size, 2) if win_size else 0.0
        rows.append(dict(series=label, type="composite/feature",
                         win_size=win_size, n_missing=n_nan, pct=pct))

    print(f"{'Series':<25}  {'Type':<25}  {'Window':>10}  {'Missing':>9}  {'%Missing':>9}  {'Flag':>7}")
    print("-" * 94)
    flagged = []
    for r in rows:
        flag = "[>10%]" if r["pct"] > FLAG_THRESHOLD else ""
        if flag:
            flagged.append(r["series"])
        print(f"{r['series']:<25}  {r['type']:<25}  {r['win_size']:>10,}  "
              f"{r['n_missing']:>9,}  {r['pct']:>8.2f}%  {flag:>7}")

    print()
    if flagged:
        print(f"[FLAG] Series with >10% missing in active window: {', '.join(flagged)}")
    else:
        print("[OK] No series exceeds 10% missing data in active window.")


# -- 4. Liquidity filter validation -------------------------------------------
def check_liquidity_filter(prices: pd.DataFrame) -> None:
    print()
    print(SEP)
    print("4. LIQUIDITY FILTER VALIDATION")
    print(f"   Criterion: >= {VOL_MIN_FRAC_ABOVE*100:.0f}% of trading days with volume >= R${VOL_THRESHOLD_K/1000:.0f}M/day")
    print(SEP)

    raw_path = _here.parent / "raw" / "economatica" / "diario" / "volume.csv"
    if not raw_path.exists():
        print("   [WARN] Volume data not found. Skipping check.")
        return

    # Load and parse volume data
    volume_wide = pd.read_csv(raw_path, low_memory=False)
    volume_wide = volume_wide.drop(columns=[volume_wide.columns[0]])
    volume_wide = volume_wide.rename(columns={volume_wide.columns[0]: "date"})
    volume_wide["date"] = pd.to_datetime(volume_wide["date"])
    vcols = volume_wide.columns[1:]
    volume_wide = volume_wide.rename(columns={c: c.split("|")[-1].strip() for c in vcols})
    volume_wide = volume_wide.loc[:, ~volume_wide.columns.duplicated()]
    volume_wide[volume_wide.columns[1:]] = volume_wide[volume_wide.columns[1:]].replace("-", float("nan"))
    for c in volume_wide.columns[1:]:
        volume_wide[c] = pd.to_numeric(volume_wide[c], errors="coerce")
    volume_wide = volume_wide.sort_values("date").reset_index(drop=True)

    # Get tickers in cleaned prices
    cleaned_tickers = set(prices["ticker"].unique())

    # Check each ticker
    pass_count = 0
    fail_count = 0
    failures = []

    for t in volume_wide.columns[1:]:
        if t not in cleaned_tickers:
            continue
        s = volume_wide[t].replace(0, float("nan")).dropna()
        if len(s) == 0:
            fail_count += 1
            failures.append((t, 0, 0.0))
            continue
        frac_above = (s >= VOL_THRESHOLD_K).sum() / len(s)
        if frac_above >= VOL_MIN_FRAC_ABOVE:
            pass_count += 1
        else:
            fail_count += 1
            failures.append((t, len(s), frac_above))

    print(f"   Tickers in cleaned universe: {len(cleaned_tickers):,}")
    print(f"   Pass filter: {pass_count:,}")
    print(f"   Fail filter: {fail_count:,}")
    if fail_count > 0:
        print(f"\n   [FLAG] Tickers that DON'T meet criterion:")
        for t, n_days, frac in sorted(failures, key=lambda x: x[2])[:10]:
            print(f"     {t:<8} -> {frac*100:>6.1f}% ({n_days} trading days)")
        if fail_count > 10:
            print(f"     ... and {fail_count - 10} more")
    else:
        print(f"\n   [OK] All cleaned tickers pass the liquidity filter.")


# -- 5. Weekend / holiday removal check --------------------------------------
def check_no_weekends_holidays() -> None:
    print()
    print(SEP)
    print("4. WEEKEND / HOLIDAY REMOVAL CHECK")
    print(SEP)

    # ---- A. Inspect raw fechamento.csv before any processing ----------------
    raw_path = _here.parent / "raw" / "economatica" / "diario" / "fechamento.csv"
    print("A) RAW fechamento.csv -- before any processing:")
    if raw_path.exists():
        raw = pd.read_csv(raw_path, low_memory=False)
        date_col  = raw.columns[1]  # col 0 = Ativo label, col 1 = dates
        raw["_date"] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.dropna(subset=["_date"])

        data_cols = [c for c in raw.columns
                     if c not in [raw.columns[0], date_col, "_date"]]
        tmp = raw[data_cols].replace("-", float("nan"))
        for c in data_cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

        raw["all_nan"]  = tmp.isna().all(axis=1).values
        raw["weekday"]  = raw["_date"].dt.weekday
        raw["day_name"] = raw["_date"].dt.day_name()

        total    = len(raw)
        n_allnan = int(raw["all_nan"].sum())
        n_data   = total - n_allnan

        print(f"   Total rows in raw CSV        : {total:,}")
        print(f"   Rows where ALL tickers = NaN : {n_allnan:,}  (weekends + holidays)")
        print(f"   Rows with at least one price : {n_data:,}")
        print()
        print("   Day-of-week breakdown of ALL-NaN rows:")
        breakdown = raw[raw["all_nan"]]["day_name"].value_counts()
        for day in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]:
            cnt = int(breakdown.get(day, 0))
            if cnt == 0:
                continue
            marker = " <-- weekend" if day in ("Saturday","Sunday") else " <-- holiday"
            print(f"     {day:<12}: {cnt:>5,}{marker}")

        print()
        # Show one sample week to confirm 5-day pattern
        first_monday = raw[~raw["all_nan"]]["_date"].min()
        while first_monday.weekday() != 0:
            first_monday += pd.Timedelta(days=1)
        week_end = first_monday + pd.Timedelta(days=6)
        sample = raw[(raw["_date"] >= first_monday) & (raw["_date"] <= week_end)]
        print(f"   Sample week ({first_monday.date()} to {week_end.date()}):")
        for _, row in sample.iterrows():
            status = "ALL-NaN (no trade)" if row["all_nan"] else "has price data"
            print(f"     {str(row['_date'].date())} {row['day_name']:<12} -> {status}")

        print()
        print("   Removal mechanism (io_utils.read_economatica_wide):")
        print("     df.dropna(subset=data_cols, how='all')")
        print("     Drops every row where ALL tickers are NaN,")
        print("     covering all weekends and Brazilian holidays.")
    else:
        print(f"   [WARN] File not found: {raw_path}")

    # ---- B. Confirm no weekends in any cleaned parquet ----------------------
    print()
    print("B) Weekend check on cleaned / features parquets:")
    parquets_to_check = {
        "prices"             : CLEANED  / "prices.parquet",
        "market_indices"     : CLEANED  / "market_indices.parquet",
        "fundamentals_ffill" : FEATURES / "fundamentals_ffill.parquet",
        "returns"            : FEATURES / "returns.parquet",
        "index_returns"      : FEATURES / "index_returns.parquet",
        "x_ts"               : PARQUETS / "x_ts.parquet",
    }
    any_problem = False
    for name, path in parquets_to_check.items():
        if not path.exists():
            print(f"   {name:<24} -- file not found, skipped.")
            continue
        df = pd.read_parquet(path, columns=["date"])
        unique_dates = df["date"].unique()
        weekends = [d for d in unique_dates if pd.Timestamp(d).weekday() >= 5]
        if weekends:
            print(f"   [WARN]  {name:<24}: {len(weekends)} weekend dates!")
            for d in sorted(weekends)[:5]:
                print(f"           {pd.Timestamp(d).date()} ({pd.Timestamp(d).day_name()})")
            any_problem = True
        else:
            print(f"   [OK]  {name:<24}: no weekends ({len(unique_dates):,} unique dates).")

    if not any_problem:
        print("\n   All cleaned parquets: no weekend dates found.")


# -- Main --------------------------------------------------------------------
def main() -> None:
    print()
    print(SEP)
    print("TCC DATA DIAGNOSTICS")
    print(SEP)

    print("Loading prices and returns ...")
    prices  = pd.read_parquet(CLEANED  / "prices.parquet")
    returns = pd.read_parquet(FEATURES / "returns.parquet")
    print(f"  prices : {prices.shape}  |  {prices['ticker'].nunique()} tickers")
    print(f"  returns: {returns.shape}")

    check_stocks_over_time(prices)
    check_return_distributions(returns)
    check_liquidity_filter(prices)
    check_missing_rates()
    check_missing_rates_in_window()
    check_no_weekends_holidays()

    print()
    print(SEP)
    print("Done.")
    print(SEP)


if __name__ == "__main__":
    main()
