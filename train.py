#!/usr/bin/env python

import argparse
import logging
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader          import load_raw_data
from src.feature_engineering  import build_features
from src.models               import StackingEnsemble
from src.evaluation           import evaluate_model
from src.visualisation        import (
    plot_correlation_heatmap,
    plot_actual_vs_predicted,
    plot_results_dashboard,
)
from src.config               import LSTM_LOOKBACK

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LEVEL_NAMES = {
    "base":   "Base",
    "level1": "Level1",
    "level2": "Level2",
    "level3": "Level3",
}


def run(levels_to_run: list, make_plots: bool = True):
    logger.info("── Step 1: Loading data ──")
    raw = load_raw_data()
    logger.info(f"  Shape: {raw.shape}  |  Columns: {list(raw.columns)}")

    logger.info("\n── Step 2: Building feature sets ──")
    feature_sets = build_features(raw)

    if make_plots:
        logger.info("\n── Step 3: Correlation heatmap ──")
        plot_correlation_heatmap(raw)

    all_results = {}
    for key in levels_to_run:
        name = LEVEL_NAMES[key]
        logger.info(f"\n── Training: {name} ──")

        fset = feature_sets[key]
        X    = fset["X"].values
        y    = fset["y"].values

        result = evaluate_model(
            model_cls  = StackingEnsemble,
            X          = X,
            y          = y,
            model_name = name,
            lookback   = LSTM_LOOKBACK,
        )
        all_results[name] = result

        if make_plots:
            plot_actual_vs_predicted(result, name)

    if make_plots and len(all_results) > 1:
        logger.info("\n── Step 5: Generating results dashboard ──")
        plot_results_dashboard(all_results)

    logger.info("\n" + "="*60)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info("="*60)
    header = f"  {'Model':<12} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'DirAcc':>8} {'N':>6}"
    logger.info(header)
    logger.info("  " + "-"*56)
    for name, res in all_results.items():
        m = res["oof_metrics"]
        logger.info(
            f"  {name:<12} "
            f"{m['RMSE']:>7.2f} "
            f"{m['MAE']:>7.2f} "
            f"{m['R2']:>7.4f} "
            f"{m['Dir_Acc']:>7.1f}% "
            f"{m['Samples']:>6}"
        )
    logger.info("="*60)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tesla Supplier Stock Prediction")
    parser.add_argument(
        "--level", type=int, choices=[0, 1, 2, 3],
        help="Run a single level (0=base, 1, 2, 3). Omit to run all."
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    key_map  = {0: "base", 1: "level1", 2: "level2", 3: "level3"}
    all_keys = ["base", "level1", "level2", "level3"]

    if args.level is not None:
        levels = [key_map[args.level]]
    else:
        levels = all_keys

    run(levels, make_plots=not args.no_plots)
