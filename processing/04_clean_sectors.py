"""04 — Clean sector classification → cleaned/sectors.parquet

Reads setor_ibovespa.xlsx, extracts ticker and setor_economico,
maps "-" to "Outros", and assigns integer sector_id.
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
    setor_col = [c for c in df.columns if "Setor Econ" in c][0]

    sectors = df[[codigo_col, setor_col]].copy()
    sectors.columns = ["ticker", "setor_economico"]

    # Clean
    sectors["ticker"] = sectors["ticker"].astype(str).str.strip()
    sectors["setor_economico"] = sectors["setor_economico"].astype(str).str.strip()

    # Map "-" to "Outros"
    sectors.loc[sectors["setor_economico"] == "-", "setor_economico"] = "Outros"

    # Drop duplicates (keep first occurrence of each ticker)
    sectors = sectors.drop_duplicates(subset="ticker", keep="first")

    # Assign integer sector_id (deterministic alphabetical order)
    categories = sorted(sectors["setor_economico"].unique())
    cat_to_id = {cat: i for i, cat in enumerate(categories)}
    sectors["sector_id"] = sectors["setor_economico"].map(cat_to_id)

    sectors = sectors.sort_values("ticker").reset_index(drop=True)

    out = CLEANED / "sectors.parquet"
    sectors.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Tickers: {len(sectors)}")
    print(f"      Sectors: {len(categories)}")
    for cat in categories:
        n = (sectors["setor_economico"] == cat).sum()
        print(f"        {cat}: {n}")


if __name__ == "__main__":
    main()
