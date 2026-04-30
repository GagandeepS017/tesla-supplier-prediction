#!/usr/bin/env python

import sys, os, logging
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader          import load_raw_data
from src.feature_engineering  import build_features
from src.models               import StackingEnsemble
from src.config               import LSTM_LOOKBACK

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading latest market data …")
    raw          = load_raw_data()
    feature_sets = build_features(raw)

    fset = feature_sets["level3"]
    X    = fset["X"].values
    y    = fset["y"].values

    logger.info("Training Level-3 model on full dataset …")
    model = StackingEnsemble(lookback=LSTM_LOOKBACK)
    model.fit(X, y)

    recent_window     = X[-LSTM_LOOKBACK - 1:]
    y_pred            = model.predict(recent_window)
    last_actual_price = float(raw["Tesla"].dropna().iloc[-1])
    next_day_pred     = float(y_pred[-1])

    print("\n" + "="*45)
    print("  Tesla Next-Day Price Prediction")
    print("="*45)
    print(f"  Last known close : ${last_actual_price:,.2f}")
    print(f"  Predicted close  : ${next_day_pred:,.2f}")
    direction = "▲ UP" if next_day_pred > last_actual_price else "▼ DOWN"
    print(f"  Direction signal : {direction}")
    print("="*45)
    print("\n  ⚠  Research prototype. Not financial advice.\n")


if __name__ == "__main__":
    main()
