import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from src.config import PLOTS_DIR, RESULTS_DIR, SUPPLIER_TICKERS, MARKET_INDICATORS

os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = {
    "Base":   "#E74C3C",
    "Level1": "#3498DB",
    "Level2": "#F39C12",
    "Level3": "#2ECC71",
    "Actual": "#2C3E50",
}


def plot_correlation_heatmap(raw_df: pd.DataFrame, save: bool = True):
    from sklearn.preprocessing import MinMaxScaler
    cols   = ["Tesla"] + list(SUPPLIER_TICKERS.keys())
    subset = raw_df[cols].dropna()
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(subset),
        columns=subset.columns, index=subset.index,
    )
    corr = scaled.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.zeros_like(corr, dtype=bool)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=0.6, vmax=1.0, linewidths=0.5,
        ax=ax, mask=mask,
    )
    ax.set_title("Correlation Matrix: Tesla and Supplier Scaled Prices", fontsize=14, pad=12)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "fig7_correlation_heatmap.png"), dpi=150)
    return fig


def plot_actual_vs_predicted(results: dict, level_name: str, save: bool = True):
    y_true = np.array(results["y_true"])
    y_pred = np.array(results["y_pred"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_true, label="Actual Price",    color=PALETTE["Actual"],  lw=1.5)
    ax.plot(y_pred, label="Predicted Price", color=PALETTE.get(level_name, "#9B59B6"),
            lw=1.5, linestyle="--")
    m = results["oof_metrics"]
    ax.set_title(
        f"Stacking Model ({level_name}): Actual vs Predicted Tesla Stock Prices\n"
        f"RMSE={m['RMSE']:.2f}  R²={m['R2']:.4f}  Dir={m['Dir_Acc']:.1f}%",
        fontsize=11,
    )
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Stock Price (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        safe = level_name.lower().replace(" ", "_")
        fig.savefig(os.path.join(PLOTS_DIR, f"actual_vs_pred_{safe}.png"), dpi=150)
    return fig


def plot_results_dashboard(all_results: dict, save: bool = True):
    labels    = list(all_results.keys())
    rmse_vals = [all_results[l]["oof_metrics"]["RMSE"] for l in labels]
    r2_vals   = [all_results[l]["oof_metrics"]["R2"]   for l in labels]
    colors    = [PALETTE.get(l, "#95A5A6") for l in labels]

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1  = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(labels, rmse_vals, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("Model Performance (RMSE)", fontweight="bold")
    ax1.set_ylabel("RMSE")
    ax1.grid(axis="y", alpha=0.3)

    ax2   = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(labels, r2_vals, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars2, r2_vals):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax2.set_title("Model Performance (R²)", fontweight="bold")
    ax2.set_ylabel("R² Score")
    ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[0, 2])
    if "Level1" in all_results and "Level3" in all_results:
        n    = 100
        true = np.array(all_results["Level3"]["y_true"])[-n:]
        p1   = np.array(all_results["Level1"]["y_pred"])[-n:]
        p3   = np.array(all_results["Level3"]["y_pred"])[-n:]
        ax3.plot(true, label="Actual",         color=PALETTE["Actual"],  lw=1.8)
        ax3.plot(p1,   label="Tesla Only",     color=PALETTE["Level1"],  lw=1.2, linestyle="--")
        ax3.plot(p3,   label="Tesla+Suppliers",color=PALETTE["Level3"],  lw=1.2, linestyle="--")
    ax3.set_title("Predictions vs Actual (Last 100 Days)", fontweight="bold")
    ax3.set_ylabel("Price (USD)")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 0])
    for lname, lcolor in [("Level1", PALETTE["Level1"]), ("Level3", PALETTE["Level3"])]:
        if lname in all_results:
            errs = np.abs(
                np.array(all_results[lname]["y_true"]) -
                np.array(all_results[lname]["y_pred"])
            )
            ax4.hist(errs, bins=40, alpha=0.6, color=lcolor, label=lname, density=True)
    ax4.set_title("Error Distribution", fontweight="bold")
    ax4.set_xlabel("Absolute Error")
    ax4.set_ylabel("Density")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    ax5    = fig.add_subplot(gs[1, 1])
    window = 30
    for lname, lcolor in [("Level1", PALETTE["Level1"]), ("Level3", PALETTE["Level3"])]:
        if lname in all_results:
            errs    = np.abs(
                np.array(all_results[lname]["y_true"]) -
                np.array(all_results[lname]["y_pred"])
            )
            rolling = pd.Series(errs).rolling(window).mean()
            ax5.plot(rolling, color=lcolor, label=lname, lw=1.4)
    ax5.set_title(f"Rolling MAE ({window}-day window)", fontweight="bold")
    ax5.set_xlabel("Sample Index")
    ax5.set_ylabel("MAE")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)

    ax6          = fig.add_subplot(gs[1, 2])
    comparisons  = []
    improvements = []
    imp_colors   = []

    def _pct_improvement(old, new):
        return (old - new) / old * 100

    pairs = [
        ("Base vs L1", "Base",   "Level1"),
        ("L1 vs L2",   "Level1", "Level2"),
        ("L1 vs L3",   "Level1", "Level3"),
        ("L2 vs L3",   "Level2", "Level3"),
    ]
    for label, a, b in pairs:
        if a in all_results and b in all_results:
            imp = _pct_improvement(all_results[a]["oof_metrics"]["RMSE"],
                                   all_results[b]["oof_metrics"]["RMSE"])
            comparisons.append(label)
            improvements.append(imp)
            imp_colors.append("#2ECC71" if imp > 0 else "#E74C3C")

    bars6 = ax6.bar(comparisons, improvements, color=imp_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars6, improvements):
        y_off = 1 if val >= 0 else -5
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax6.axhline(0, color="grey", lw=0.8)
    ax6.set_title("RMSE Improvement (%)", fontweight="bold")
    ax6.set_ylabel("Improvement (%)")
    ax6.grid(axis="y", alpha=0.3)

    fig.suptitle("Tesla Stock Prediction: Supplier Contribution Analysis",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "fig8_results_dashboard.png"),
                    dpi=150, bbox_inches="tight")
    return fig


def plot_feature_importance(rf_model, feature_names: list, top_n: int = 15,
                            save: bool = True):
    importances = rf_model.feature_importances_
    idx  = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx],
            color="#3498DB", edgecolor="white")
    ax.set_title(f"Random Forest — Top {top_n} Feature Importances", fontsize=13)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(PLOTS_DIR, "feature_importance_rf.png"), dpi=150)
    return fig
