"""08 — Assemble x_ts.parquet: merge, normalize, save

1. Merge returns + fundamentals + index_returns
2. Compute train-period normalization stats
3. Normalize: returns by σ only, fundamentals by z-score, indices by z-score
4. Create missingness masks ({col}_obs) BEFORE filling NaN
5. Apply PCA to z-scored indices (reduce 29 → 10 components)
6. Fill remaining NaN with 0.0
7. Save x_ts.parquet + normalization_stats.json + prices.parquet (copy to parquets/)
"""
from __future__ import annotations

import json

import pandas as pd

from config import CLEANED, FEATURES, FUNDAMENTAL_FILES, PARQUETS, TRAIN_END

COMPOSITE_COLS = ["fcf_divida", "fcf_yield"]
# MSCI indices: keep only 4 most relevant for Brazil
MSCI_TO_KEEP = {"MXEF Index_ret", "MXUS Index_ret", "MXEU Index_ret", "MXLA Index_ret"}
MSCI_TO_DROP = {"MXCN Index_ret", "MXJP Index_ret", "MXGB Index_ret", "MXCA Index_ret", "MXPCJ Index_ret"}


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

    # Merge composite indicators
    fcf_divida = pd.read_parquet(FEATURES / "fcf_divida_ffill.parquet")
    fcf_yield = pd.read_parquet(FEATURES / "fcf_yield.parquet")
    x_ts = x_ts.merge(fcf_divida, on=["date", "ticker"], how="left")
    x_ts = x_ts.merge(fcf_yield, on=["date", "ticker"], how="left")

    x_ts = x_ts.sort_values(["date", "ticker"]).reset_index(drop=True)

    # --- Identify column groups ---
    all_idx_cols = [c for c in index_returns.columns if c != "date"]
    # Filter MSCI: drop irrelevant ones, keep only 4
    idx_ret_cols = [c for c in all_idx_cols if c not in MSCI_TO_DROP]
    feature_cols = ["return"] + fund_cols + COMPOSITE_COLS + idx_ret_cols

    print(f"      Features: {len(feature_cols)} ({1} return + {len(fund_cols)} fund + {len(COMPOSITE_COLS)} composite + {len(idx_ret_cols)} indices)")
    print(f"      MSCI kept: {sorted([c for c in idx_ret_cols if 'MSCI' in str(index_returns.columns)])}")
    print(f"      MSCI dropped: {sorted(MSCI_TO_DROP)}")

    # --- Compute train-period stats ---
    train_mask = x_ts["date"] <= pd.Timestamp(TRAIN_END)
    train = x_ts.loc[train_mask]

    stats: dict = {
        "train_period": {"start": str(train["date"].min().date()), "end": TRAIN_END},
        "feature_order": None,  # Will be set after PCA
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

    # Composite indicators: z-score with global train stats
    stats["composite_stats"] = {}
    for col in COMPOSITE_COLS:
        mu = float(train[col].mean())
        sigma = float(train[col].std())
        stats["composite_stats"][col] = {"mean": mu, "std": sigma}
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

    # --- Create missingness masks BEFORE filling NaN ---
    mask_cols = {}
    for col in fund_cols + COMPOSITE_COLS + idx_ret_cols:
        mask_col = f"{col}_obs"
        mask_cols[mask_col] = x_ts[col].notna().astype(int)
        x_ts[mask_col] = mask_cols[mask_col]
        pct = 100 * mask_cols[mask_col].mean()
        print(f"      {mask_col}: {pct:.1f}% obs")

    # --- Fill remaining NaN with 0.0 ---
    nan_before = x_ts[feature_cols + list(mask_cols.keys())].isna().sum().sum()
    x_ts[feature_cols + list(mask_cols.keys())] = x_ts[feature_cols + list(mask_cols.keys())].fillna(0.0)
    print(f"      NaN filled: {nan_before} cells → 0.0")

    # --- PCA NOT applied: keeping {len(idx_ret_cols)} indices as-is ---
    print(f"      Keeping {len(idx_ret_cols)} indices without PCA (no reduction)")
    stats["indices_stats"] = {
        "n_indices": len(idx_ret_cols),
        "indices_kept": idx_ret_cols,
        "msci_dropped": sorted(MSCI_TO_DROP),
    }

    # --- Build final feature order (no PCA columns) ---
    feature_cols = ["return"] + fund_cols + COMPOSITE_COLS + idx_ret_cols
    final_mask_cols = list(mask_cols.keys())
    
    stats["feature_order"] = feature_cols
    stats["mask_order"] = final_mask_cols

    # --- Add metadata ---
    d_ts = len(feature_cols)
    d_masks = len(final_mask_cols)
    stats["d_ts"] = d_ts
    stats["d_masks"] = d_masks
    stats["d_total"] = d_ts + d_masks
    stats["n_tickers_total"] = int(x_ts["ticker"].nunique())
    stats["n_tickers_train"] = int(train["ticker"].nunique())

    # --- Save x_ts ---
    out_ts = PARQUETS / "x_ts.parquet"
    x_ts.to_parquet(out_ts, index=False)
    print(f"      Saved: {out_ts}")
    print(f"      Shape: {x_ts.shape}, d_ts={d_ts}, d_masks={d_masks}, d_total={d_ts + d_masks}")
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
