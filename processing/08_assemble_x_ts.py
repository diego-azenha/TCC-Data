"""08 — Assemble x_ts.parquet: merge, normalize, save

1. Merge returns + fundamentals + index_returns
2. Compute train-period normalization stats
3. Normalize: returns by σ only, fundamentals by z-score, indices by z-score
4. Fill remaining NaN with 0.0
5. Save x_ts.parquet + normalization_stats.json + prices.parquet (copy to parquets/)
"""
from __future__ import annotations

import json

import pandas as pd

from config import CLEANED, FEATURES, FUNDAMENTAL_FILES, PARQUETS, TRAIN_END


def main() -> None:
    print("[08] Assembling x_ts ...")

    # --- Load feature layers ---
    returns = pd.read_parquet(FEATURES / "returns.parquet")
    fundamentals = pd.read_parquet(FEATURES / "fundamentals_ffill.parquet")
    index_returns = pd.read_parquet(FEATURES / "index_returns.parquet")

    # --- Merge: returns define the universe ---
    x_ts = returns.copy()  # date, ticker, return

    # Merge fundamentals (left join on date+ticker)
    fund_cols = [v[1] for v in FUNDAMENTAL_FILES.values()]
    x_ts = x_ts.merge(fundamentals, on=["date", "ticker"], how="left")

    # Merge index returns (broadcast: same for all tickers on a given date)
    x_ts = x_ts.merge(index_returns, on="date", how="left")

    x_ts = x_ts.sort_values(["date", "ticker"]).reset_index(drop=True)

    # --- Identify column groups ---
    idx_ret_cols = [c for c in index_returns.columns if c != "date"]
    feature_cols = ["return"] + fund_cols + idx_ret_cols

    print(f"      Features: {len(feature_cols)} ({1} return + {len(fund_cols)} fund + {len(idx_ret_cols)} indices)")

    # --- Compute train-period stats ---
    train_mask = x_ts["date"] <= pd.Timestamp(TRAIN_END)
    train = x_ts.loc[train_mask]

    stats: dict = {
        "train_period": {"start": str(train["date"].min().date()), "end": TRAIN_END},
        "feature_order": feature_cols,
    }

    # Returns: normalize by σ only (no mean subtraction)
    ret_std = float(train["return"].std())
    stats["returns_std"] = ret_std
    x_ts["return"] = x_ts["return"] / ret_std
    print(f"      Return σ_train = {ret_std:.6f}")

    # Fundamentals: z-score with global train stats
    stats["fundamental_stats"] = {}
    for col in fund_cols:
        mu = float(train[col].mean())
        sigma = float(train[col].std())
        stats["fundamental_stats"][col] = {"mean": mu, "std": sigma}
        if sigma > 0:
            x_ts[col] = (x_ts[col] - mu) / sigma
        else:
            x_ts[col] = 0.0
        print(f"      {col}: μ={mu:.4f}, σ={sigma:.4f}")

    # Index returns: z-score per series with train stats
    stats["index_stats"] = {}
    for col in idx_ret_cols:
        mu = float(train[col].mean())
        sigma = float(train[col].std())
        stats["index_stats"][col] = {"mean": mu, "std": sigma}
        if sigma > 0:
            x_ts[col] = (x_ts[col] - mu) / sigma
        else:
            x_ts[col] = 0.0

    # --- Fill remaining NaN with 0.0 ---
    nan_before = x_ts[feature_cols].isna().sum().sum()
    x_ts[feature_cols] = x_ts[feature_cols].fillna(0.0)
    print(f"      NaN filled: {nan_before} cells → 0.0")

    # --- Add metadata ---
    d_ts = len(feature_cols)
    stats["d_ts"] = d_ts
    stats["n_tickers_total"] = int(x_ts["ticker"].nunique())
    stats["n_tickers_train"] = int(train["ticker"].nunique())

    # --- Save x_ts ---
    out_ts = PARQUETS / "x_ts.parquet"
    x_ts.to_parquet(out_ts, index=False)
    print(f"      Saved: {out_ts}")
    print(f"      Shape: {x_ts.shape}, d_ts={d_ts}")
    print(f"      Dates: {x_ts['date'].min().date()} → {x_ts['date'].max().date()}")

    # --- Save normalization stats ---
    out_stats = PARQUETS / "normalization_stats.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"      Saved: {out_stats}")

    # --- Copy prices to parquets/ for the dataset loader ---
    prices = pd.read_parquet(CLEANED / "prices.parquet")
    out_prices = PARQUETS / "prices.parquet"
    prices.to_parquet(out_prices, index=False)
    print(f"      Copied prices → {out_prices}")


if __name__ == "__main__":
    main()
