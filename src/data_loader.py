import os
import logging
import pandas as pd
import yfinance as yf
from src.config import (
    TESLA_TICKER, SUPPLIER_TICKERS, MARKET_INDICATORS,
    START_DATE, END_DATE, DATA_DIR
)

logger = logging.getLogger(__name__)


def _fetch_close(ticker: str, name: str) -> pd.Series:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE,
                         auto_adjust=True, progress=False)
        if df.empty:
            logger.warning(f"No data for {name} ({ticker})")
            return pd.Series(dtype=float, name=name)
        close = df["Close"].squeeze()
        close.name = name
        logger.info(f"  ✓ {name:25s} — {len(close):>4d} trading days")
        return close
    except Exception as e:
        logger.error(f"  ✗ Failed to fetch {name} ({ticker}): {e}")
        return pd.Series(dtype=float, name=name)


def load_raw_data(force_refresh: bool = False) -> pd.DataFrame:
    cache_path = os.path.join(DATA_DIR, "raw_prices.csv")
    os.makedirs(DATA_DIR, exist_ok=True)

    if not force_refresh and os.path.exists(cache_path):
        logger.info("Loading cached raw prices …")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    logger.info("Fetching data from Yahoo Finance …")
    series_list = []

    series_list.append(_fetch_close(TESLA_TICKER, "Tesla"))

    for name, ticker in SUPPLIER_TICKERS.items():
        series_list.append(_fetch_close(ticker, name))

    for name, ticker in MARKET_INDICATORS.items():
        series_list.append(_fetch_close(ticker, name))

    df = pd.concat(series_list, axis=1)
    df.index.name = "Date"
    df = df[df["Tesla"].notna()]
    df.to_csv(cache_path)
    logger.info(f"Saved raw prices → {cache_path}  shape={df.shape}")
    return df
