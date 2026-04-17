"""09 — Assemble x_static.parquet: sector one-hot encoding

Reads cleaned/sectors.parquet, one-hot encodes subsetor (mid-level),
ensures all tickers in x_ts are covered (missing → "Outros").
"""
from __future__ import annotations

import pandas as pd

from config import CLEANED, PARQUETS


def main() -> None:
    print("[09] Assembling x_static ...")

    sectors = pd.read_parquet(CLEANED / "sectors.parquet")  # ticker, setor_economico, subsetor, sector_id, subsetor_id

    # Get all tickers present in x_ts
    x_ts = pd.read_parquet(PARQUETS / "x_ts.parquet", columns=["ticker"])
    ts_tickers = set(x_ts["ticker"].unique())

    # Find tickers in x_ts that are missing from sectors
    sector_tickers = set(sectors["ticker"].unique())
    missing = ts_tickers - sector_tickers
    if missing:
        print(f"      {len(missing)} tickers missing from sectors → mapped to 'Outros'")
        outros_subsetor_id = (
            sectors.loc[sectors["subsetor"] == "Outros", "subsetor_id"].iloc[0]
            if (sectors["subsetor"] == "Outros").any() else -1
        )
        missing_df = pd.DataFrame({
            "ticker": list(missing),
            "setor_economico": "Outros",
            "subsetor": "Outros",
            "sector_id": sectors.loc[sectors["setor_economico"] == "Outros", "sector_id"].iloc[0]
            if (sectors["setor_economico"] == "Outros").any() else -1,
            "subsetor_id": outros_subsetor_id,
        })
        sectors = pd.concat([sectors, missing_df], ignore_index=True)

    # Filter to only tickers that actually appear in x_ts
    sectors = sectors[sectors["ticker"].isin(ts_tickers)].copy()

    # One-hot encode subsetor (mid-level classification)
    onehot = pd.get_dummies(sectors["subsetor"], dtype=float)

    # Categories come from the data — sorted alphabetically for determinism
    x_static = pd.concat([sectors[["ticker"]], onehot], axis=1)
    x_static = x_static.drop_duplicates(subset="ticker", keep="first")
    x_static = x_static.sort_values("ticker").reset_index(drop=True)

    d_static = len(onehot.columns)

    out = PARQUETS / "x_static.parquet"
    x_static.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Tickers: {len(x_static)}")
    print(f"      d_static: {d_static}")
    print(f"      Subsetores: {sorted(onehot.columns.tolist())}")

    # Coverage report
    outros_count = int((x_static.get("Outros", pd.Series([0])) == 1).sum())
    coverage = 100 * (1 - outros_count / len(x_static))
    print(f"      Coverage (non-Outros): {coverage:.1f}%")


if __name__ == "__main__":
    main()
