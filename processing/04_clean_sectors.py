"""04 — Clean sector classification → cleaned/sectors.parquet

Reads setor_ibovespa.xlsx, extracts ticker, setor_economico, and subsetor.
Maps "-" to "Outros" for both levels and assigns integer IDs.
"""
from __future__ import annotations

import pandas as pd

from config import CLEANED, SECTOR_PATH


def main() -> None:
    print("[04] Cleaning sectors ...")

    df = pd.read_excel(SECTOR_PATH, header=3)

    # Select relevant columns
    # Column names have newlines from the Excel export
    codigo_col = "Código"
    setor_col    = [c for c in df.columns if "Setor Econ" in c][0]
    subsetor_col = [c for c in df.columns if "Subsetor" in c][0]

    sectors = df[[codigo_col, setor_col, subsetor_col]].copy()
    sectors.columns = ["ticker", "setor_economico", "subsetor"]

    # Clean
    sectors["ticker"]         = sectors["ticker"].astype(str).str.strip()
    sectors["setor_economico"] = sectors["setor_economico"].astype(str).str.strip()
    sectors["subsetor"]        = sectors["subsetor"].astype(str).str.strip()

    # Map "-" to "Outros" for both levels
    sectors.loc[sectors["setor_economico"] == "-", "setor_economico"] = "Outros"
    sectors.loc[sectors["subsetor"] == "-", "subsetor"] = "Outros"

    # Drop duplicates (keep first occurrence of each ticker)
    sectors = sectors.drop_duplicates(subset="ticker", keep="first")

    # Assign integer IDs (deterministic alphabetical order)
    setor_cats = sorted(sectors["setor_economico"].unique())
    sectors["sector_id"] = sectors["setor_economico"].map(
        {cat: i for i, cat in enumerate(setor_cats)}
    )

    subsetor_cats = sorted(sectors["subsetor"].unique())
    sectors["subsetor_id"] = sectors["subsetor"].map(
        {cat: i for i, cat in enumerate(subsetor_cats)}
    )

    sectors = sectors.sort_values("ticker").reset_index(drop=True)

    out = CLEANED / "sectors.parquet"
    sectors.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Tickers: {len(sectors)}")
    print(f"      Subsetores: {len(subsetor_cats)}")
    for cat in subsetor_cats:
        n = (sectors["subsetor"] == cat).sum()
        print(f"        {cat}: {n}")


if __name__ == "__main__":
    main()
