"""Export ROA after ffill as a wide CSV (dates x tickers) for visualization.

NaN = missing after 65-day ffill plus short-hole patching on the active window.
Saved to diagnostics_roa_wide.csv.
"""
import pandas as pd
from pathlib import Path

ROOT   = Path(__file__).parent
FEAT   = ROOT / "features"
OUT    = ROOT / "diagnostics_roa_wide.csv"

df = pd.read_parquet(FEAT / "fundamentals_ffill.parquet", columns=["date", "ticker", "roa"])

wide = df.pivot(index="date", columns="ticker", values="roa")
wide.index = pd.to_datetime(wide.index).strftime("%Y-%m-%d")
wide.sort_index(inplace=True)

wide.to_csv(OUT)

total = wide.size
missing = wide.isna().sum().sum()
print(f"Shape : {wide.shape}  ({wide.shape[0]} dates x {wide.shape[1]} tickers)")
print(f"Missing: {missing:,} / {total:,}  ({100*missing/total:.2f}%)")
print(f"Saved : {OUT}")
