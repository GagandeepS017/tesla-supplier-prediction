import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.config import (
    SHORT_LAG, LONG_LAG,
    SUPPLIER_TICKERS, MARKET_INDICATORS
)

SUPPLIER_COLS = list(SUPPLIER_TICKERS.keys())
MARKET_COLS   = list(MARKET_INDICATORS.keys())
EXTERNAL_COLS = SUPPLIER_COLS + MARKET_COLS


def build_features(raw: pd.DataFrame) -> dict:
    df = raw.copy().dropna()

    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns,
    )

    returns    = df.pct_change().dropna()
    target_col = "Tesla"

    def _align(X: pd.DataFrame, y: pd.Series):
        idx = X.index.intersection(y.index)
        return X.loc[idx].copy(), y.loc[idx].copy()

    base_X      = scaled[EXTERNAL_COLS].copy()
    ext_returns = returns[EXTERNAL_COLS].add_suffix("_ret")
    base_X      = pd.concat([base_X, ext_returns], axis=1).dropna()
    base_y      = df[target_col].shift(-1).dropna()
    base_X, base_y = _align(base_X, base_y)

    l1_X         = scaled[[target_col]].shift(SHORT_LAG).dropna()
    l1_X.columns = [f"Tesla_lag{SHORT_LAG}"]
    l1_y         = df[target_col].shift(-1).dropna()
    l1_X, l1_y   = _align(l1_X, l1_y)

    tesla_lag20         = scaled[[target_col]].shift(LONG_LAG)
    tesla_lag20.columns = [f"Tesla_lag{LONG_LAG}"]
    ext_scaled          = scaled[EXTERNAL_COLS]
    l2_X                = pd.concat([tesla_lag20, ext_scaled, ext_returns], axis=1).dropna()
    l2_y                = df[target_col].shift(-1).dropna()
    l2_X, l2_y          = _align(l2_X, l2_y)

    tesla_lag1         = scaled[[target_col]].shift(SHORT_LAG)
    tesla_lag1.columns = [f"Tesla_lag{SHORT_LAG}"]
    l3_X               = pd.concat([tesla_lag1, ext_scaled, ext_returns], axis=1).dropna()
    l3_y               = df[target_col].shift(-1).dropna()
    l3_X, l3_y         = _align(l3_X, l3_y)

    return {
        "base":   {"X": base_X, "y": base_y},
        "level1": {"X": l1_X,   "y": l1_y},
        "level2": {"X": l2_X,   "y": l2_y},
        "level3": {"X": l3_X,   "y": l3_y},
    }


def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)
