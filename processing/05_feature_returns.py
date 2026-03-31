"""05 — Compute log-returns → features/returns.parquet

Reads cleaned/prices.parquet, computes log-returns per ticker,
replaces ±Inf with NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CLEANED, FEATURES


def main() -> None:
    print("[05] Computing returns ...")

    prices = pd.read_parquet(CLEANED / "prices.parquet")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Log-return: ln(P_t / P_{t-1}) per ticker
    prices["return"] = prices.groupby("ticker")["close"].transform(
        lambda s: np.log(s / s.shift(1))
    )

    # Replace ±Inf with NaN
    prices["return"] = prices["return"].replace([np.inf, -np.inf], np.nan)

    # Drop first row per ticker (NaN return) and any invalid returns
    returns = prices[["date", "ticker", "return"]].dropna(subset=["return"])
    returns = returns.sort_values(["date", "ticker"]).reset_index(drop=True)

    out = FEATURES / "returns.parquet"
    returns.to_parquet(out, index=False)

    print(f"      Saved: {out}")
    print(f"      Shape: {returns.shape}")
    print(f"      Return stats: mean={returns['return'].mean():.6f}, "
          f"std={returns['return'].std():.6f}")


if __name__ == "__main__":
    main()
