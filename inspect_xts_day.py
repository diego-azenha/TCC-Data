"""Print the tensors the neural network receives for a given trading day.

Reconstructs the day-level sample described in README section 7:
- S: lookback tensor [N_t, L, d_ts]
- S_static: static sector tensor [N_t, d_static]
- r: next-day normalized target return [N_t]
- mask: valid sample mask [N_t]

Examples:
    python inspect_xts_day.py 2018-12-28
    python inspect_xts_day.py 2018-12-28 --ticker PETR4
    python inspect_xts_day.py 2018-12-28 --max-tickers 5 --tail 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PARQUETS = ROOT / "parquets"
LOOKBACK = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the tensors the network receives for a given day."
    )
    parser.add_argument("date", help="Trading day in YYYY-MM-DD format")
    parser.add_argument(
        "--ticker",
        help="If provided, print the full tensors for one ticker only",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=10,
        help="How many tickers to print in summary mode (default: 10)",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=5,
        help="How many last timesteps of S to print in summary mode (default: 5)",
    )
    return parser.parse_args()


def _load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, list[str], list[str]]:
    x_ts = pd.read_parquet(PARQUETS / "x_ts.parquet")
    x_static = pd.read_parquet(PARQUETS / "x_static.parquet")
    prices = pd.read_parquet(PARQUETS / "prices.parquet")
    with open(PARQUETS / "normalization_stats.json", encoding="utf-8") as f:
        stats = json.load(f)

    feature_cols = stats["feature_order"]
    static_cols = [c for c in x_static.columns if c != "ticker"]
    return x_ts, x_static, prices, stats, feature_cols, static_cols


def _build_sample(
    date_t: pd.Timestamp,
    x_ts: pd.DataFrame,
    x_static: pd.DataFrame,
    prices: pd.DataFrame,
    stats: dict,
    feature_cols: list[str],
    static_cols: list[str],
) -> tuple[pd.Timestamp, list[dict]]:
    x_ts = x_ts.sort_values(["ticker", "date"]).reset_index(drop=True)
    dates_array = x_ts["date"].values
    feature_matrix = x_ts[feature_cols].to_numpy(dtype=np.float32)
    ticker_groups = x_ts.groupby("ticker").indices
    all_dates = np.sort(x_ts["date"].unique())

    t_idx = np.searchsorted(all_dates, np.datetime64(date_t))
    if t_idx >= len(all_dates) or pd.Timestamp(all_dates[t_idx]) != date_t:
        raise ValueError(f"Date {date_t.date()} not found in x_ts.parquet")
    if t_idx + 1 >= len(all_dates):
        raise ValueError(f"Date {date_t.date()} is the last available date and has no target")

    date_t1 = pd.Timestamp(all_dates[t_idx + 1])
    tickers_t = x_ts.loc[x_ts["date"] == date_t, "ticker"].unique().tolist()
    static_dict = x_static.set_index("ticker")[static_cols]
    price_lookup = prices.pivot(index="date", columns="ticker", values="close")

    samples: list[dict] = []
    for ticker in tickers_t:
        rows_idx = ticker_groups[ticker]
        ticker_dates = dates_array[rows_idx]
        pos = np.searchsorted(ticker_dates, np.datetime64(date_t))

        history_len = int(pos + 1)
        static_vec = static_dict.loc[ticker].to_numpy(dtype=np.float32)

        if history_len < LOOKBACK:
            window = np.zeros((LOOKBACK, len(feature_cols)), dtype=np.float32)
            target = 0.0
            is_valid = False
        else:
            window_rows = rows_idx[pos - LOOKBACK + 1 : pos + 1]
            window = feature_matrix[window_rows]
            p_t = price_lookup.at[date_t, ticker] if ticker in price_lookup.columns and date_t in price_lookup.index else np.nan
            p_t1 = price_lookup.at[date_t1, ticker] if ticker in price_lookup.columns and date_t1 in price_lookup.index else np.nan
            has_target = pd.notna(p_t) and pd.notna(p_t1) and p_t > 0
            target = float(np.log(p_t1 / p_t) / stats["returns_std"]) if has_target else 0.0
            is_valid = bool(has_target)

        samples.append(
            {
                "ticker": ticker,
                "history_len": history_len,
                "valid_mask": is_valid,
                "target": target,
                "S": window,
                "S_static": static_vec,
            }
        )

    return date_t1, samples


def _print_summary(
    date_t: pd.Timestamp,
    date_t1: pd.Timestamp,
    samples: list[dict],
    feature_cols: list[str],
    static_cols: list[str],
    max_tickers: int,
    tail: int,
) -> None:
    valid_count = sum(int(s["valid_mask"]) for s in samples)
    print(f"date_t     : {date_t.date()}")
    print(f"date_t+1   : {date_t1.date()}")
    print(f"N_t        : {len(samples)}")
    print(f"lookback L : {LOOKBACK}")
    print(f"d_ts       : {len(feature_cols)}")
    print(f"d_static   : {len(static_cols)}")
    print(f"valid_mask : {valid_count}/{len(samples)} valid")
    print()
    print("feature_order:")
    print(", ".join(feature_cols))
    print()
    print("summary (first tickers):")

    for sample in samples[:max_tickers]:
        tail_rows = pd.DataFrame(sample["S"][-tail:], columns=feature_cols)
        print("-" * 100)
        print(
            f"ticker={sample['ticker']}  valid={sample['valid_mask']}  "
            f"history_len={sample['history_len']}  target={sample['target']:.6f}"
        )
        print("S_static:")
        print(pd.Series(sample["S_static"], index=static_cols).to_string())
        print()
        print(f"last {tail} timesteps of S:")
        print(tail_rows.to_string(index=False))
        print()


def _print_single_ticker(
    ticker: str,
    date_t: pd.Timestamp,
    date_t1: pd.Timestamp,
    samples: list[dict],
    feature_cols: list[str],
    static_cols: list[str],
) -> None:
    sample = next((s for s in samples if s["ticker"] == ticker), None)
    if sample is None:
        available = ", ".join(s["ticker"] for s in samples[:20])
        raise ValueError(
            f"Ticker {ticker} not present on {date_t.date()}. First available tickers: {available}"
        )

    print(f"date_t     : {date_t.date()}")
    print(f"date_t+1   : {date_t1.date()}")
    print(f"ticker     : {sample['ticker']}")
    print(f"valid_mask : {sample['valid_mask']}")
    print(f"history_len: {sample['history_len']}")
    print(f"target     : {sample['target']:.6f}")
    print()
    print("S_static:")
    print(pd.Series(sample["S_static"], index=static_cols).to_string())
    print()
    print(f"S tensor shape: {sample['S'].shape}")
    print("S (all lookback rows):")
    print(pd.DataFrame(sample["S"], columns=feature_cols).to_string(index=False))


def main() -> None:
    args = _parse_args()
    date_t = pd.Timestamp(args.date)
    x_ts, x_static, prices, stats, feature_cols, static_cols = _load_artifacts()
    date_t1, samples = _build_sample(date_t, x_ts, x_static, prices, stats, feature_cols, static_cols)

    if args.ticker:
        _print_single_ticker(args.ticker, date_t, date_t1, samples, feature_cols, static_cols)
    else:
        _print_summary(date_t, date_t1, samples, feature_cols, static_cols, args.max_tickers, args.tail)


if __name__ == "__main__":
    main()