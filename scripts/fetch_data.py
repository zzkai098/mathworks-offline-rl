"""
Fetch daily prices (yfinance) + FRED macro factors and write CSVs under data/.

Usage:
    pip install -r scripts/requirements.txt
    export FRED_API_KEY=<your_key>
    python scripts/fetch_data.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred


TICKERS = [
    "AAPL", "MSFT", "NVDA",
    "JPM", "BAC", "V",
    "JNJ", "UNH",
    "XOM", "CVX",
    "PG", "KO",
    "CAT",
    "TLT", "GLD",
]

MACRO_SERIES = ["DGS10", "T10Y2Y", "VIXCLS", "DFF"]

START = "2010-01-01"
END = "2026-01-01"

SPLITS = {
    "train": ("2010-01-01", "2019-12-31"),
    "val":   ("2020-01-01", "2021-12-31"),
    "test":  ("2022-01-01", "2025-12-31"),
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def fetch_prices() -> pd.DataFrame:
    print(f"[prices] downloading {len(TICKERS)} tickers from yfinance...")
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = TICKERS

    prices = prices[TICKERS]
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.index.name = "Date"

    nan_counts = prices.isna().sum()
    print(f"[prices] shape={prices.shape}, date range {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"[prices] NaN per ticker:\n{nan_counts.to_string()}")

    big_gaps = nan_counts[nan_counts > 2]
    if not big_gaps.empty:
        print(f"[prices] WARNING tickers with >2 NaN days: {big_gaps.to_dict()}")

    prices = prices.ffill(limit=2)
    return prices


def fetch_macro(trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    fred = Fred(api_key=api_key)
    series = {}
    for sid in MACRO_SERIES:
        print(f"[macro] fetching {sid}...")
        s = fred.get_series(sid, observation_start=START, observation_end=END)
        s.index = pd.to_datetime(s.index)
        series[sid] = s

    macro = pd.DataFrame(series)
    macro = macro.reindex(trading_index).ffill()
    macro.index.name = "Date"

    print(f"[macro] shape={macro.shape}")
    print(f"[macro] NaN per series after ffill:\n{macro.isna().sum().to_string()}")
    print(f"[macro] stats:\n{macro.describe().round(3).to_string()}")
    return macro


def sanity_check(prices: pd.DataFrame, macro: pd.DataFrame) -> None:
    print("\n=== Sanity checks ===")
    daily_ret = np.log(prices / prices.shift(1)).dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    print(f"Annualized vol per asset:\n{ann_vol.round(3).to_string()}")

    train_start, train_end = SPLITS["train"]
    train_ret = daily_ret.loc[train_start:train_end]
    corr = train_ret.corr()
    off_diag = corr.where(~np.eye(len(corr), dtype=bool))
    mean_off = off_diag.stack().mean()
    print(f"Train off-diagonal mean correlation: {mean_off:.3f}")

    if "VIXCLS" in macro.columns:
        covid_window = macro.loc["2020-03-01":"2020-04-30", "VIXCLS"]
        print(f"VIX peak 2020-03..04: {covid_window.max():.2f} (expect ~80+)")


def write_splits(df: pd.DataFrame, prefix: str) -> None:
    full_path = DATA_DIR / f"{prefix}_full.csv"
    df.to_csv(full_path)
    print(f"[write] {full_path}  ({df.shape[0]} rows)")

    for name, (s, e) in SPLITS.items():
        chunk = df.loc[s:e]
        path = DATA_DIR / f"{prefix}_{name}.csv"
        chunk.to_csv(path)
        print(f"[write] {path}  ({chunk.shape[0]} rows)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    prices = fetch_prices()
    macro = fetch_macro(prices.index)

    write_splits(prices, "prices")
    write_splits(macro, "macro")

    combined = prices.join(macro, how="left")
    combined_path = DATA_DIR / "dataset_full.csv"
    combined.to_csv(combined_path)
    print(f"[write] {combined_path}  ({combined.shape[0]} rows, {combined.shape[1]} cols)")

    sanity_check(prices, macro)
    print("\nDone.")


if __name__ == "__main__":
    main()
