# Tesla Supplier Stock Prediction

> **Can supplier stock movements predict Tesla's price?**  
> A stacking ensemble study LSTM + Random Forest + XGBoost across 2018–2024.  
> Published at the Ohrid Conference · SRH University Berlin

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)](https://xgboost.readthedocs.io)

---

## The Question

Tesla sits at the intersection of semiconductors, batteries, raw materials, and consumer electronics. Its 9 key suppliers span 4 continents and multiple industries. If a supplier's stock moves, does Tesla's follow?

This project tests that hypothesis rigorously building a two-layer stacking ensemble and running four controlled experiments that isolate exactly how much predictive power suppliers add on top of Tesla's own price history.

---

## Model Architecture

```
                     Input Features
                          │
           ┌──────────────┴──────────────┐
           │                             │
    ┌──────▼──────────┐        ┌─────────▼───────┐
    │   LSTM Network  │        │  Random Forest  │
    │  10-step window │        │  200 estimators │
    └──────┬──────────┘        └─────────┬───────┘
           │                             │
           └──────────┬──────────────────┘
                      │  OOF predictions (no leakage)
               ┌──────▼──────┐
               │   XGBoost   │  ← meta-learner
               │  (blending) │
               └──────┬──────┘
                      │
               Next-day Tesla price
```

The LSTM captures sequential momentum in 10-step sliding windows. Random Forest handles non-linear feature interactions without sequence assumptions. XGBoost learns the optimal blend of their out-of-fold predictions trained separately to prevent leakage.

---

## Experimental Setup

Four model levels isolate each variable's contribution:

```
┌──────────┬─────────────────────────────────────────────┐
│  Base    │  Supplier prices + market indicators only   │
│  Level 1 │  Tesla 1-day lagged price only              │
│  Level 2 │  Tesla 20-day lag + suppliers + market      │
│  Level 3 │  Tesla 1-day lag  + suppliers + market  ✦  │
└──────────┴─────────────────────────────────────────────┘
                                               ✦ best model
```

**Data:** Yahoo Finance · 2018–2024 · 5-fold TimeSeriesCV · ~1,160 samples per level

---

## Results

| Model | RMSE | MAE | R² | Dir. Acc. |
|---|---|---|---|---|
| Base (suppliers only) | 86.31 | 70.75 | 0.205 | 32.0% |
| Level 1 (Tesla 1d lag) | 58.96 | 38.08 | 0.629 | 26.7% |
| Level 2 (Tesla 20d lag + suppliers) | 73.50 | 61.94 | 0.401 | 23.2% |
| **Level 3 (Tesla 1d lag + suppliers)** | **70.07** | **49.52** | **0.476** | **34.7%** |

```
RMSE (lower = better)          R² (higher = better)

Base   ████████████████  86    Base   ██  0.21
Level1 ████████████  59        Level1 ████████████  0.63
Level2 ██████████████  74      Level2 ████████  0.40
Level3 █████████████  70       Level3 █████████  0.48
```

**What the numbers say:**

- Tesla's own 1-day lag (Level 1) explains 63% of variance momentum is the dominant signal
- Suppliers add a meaningful but secondary layer: Level 3 beats Level 1 by **16% on directional accuracy** (34.7% vs 26.7%), which is the metric that actually matters for trading signals
- The 20-day lag (Level 2) destroys performance autocorrelation decays fast in volatile stocks
- Suppliers alone (Base) are worse than a naive mean predictor on most folds (R² < 0 per-fold)

---

## Suppliers Analysed

| Company | Ticker | Supply Role |
|---|---|---|
| CATL | 300750.SZ | Battery packs |
| BYD | 002594.SZ | Battery cells / EV components |
| TSMC | TSM | Autopilot & infotainment chips |
| Sony | SONY | Camera sensors (Autopilot) |
| Fuyao Glass | 3606.HK | Windshields & glass |
| Samsung Display | 005930.KS | Interior display panels |
| TE Connectivity | TEL | Connectors & wiring harnesses |
| BHP | BHP | Lithium & nickel (raw materials) |
| Texas Instruments | TXN | MCUs & power management ICs |

Market indicators: **VIX · QQQ · XLI · ICLN · ARKQ**

---

## Setup

```bash
git clone https://github.com/GagandeepS017/tesla-supplier-prediction.git
cd tesla-supplier-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

```bash
# Full pipeline all 4 model levels + plots
python train.py

# Single level
python train.py --level 3      # Level 3: best model
python train.py --level 1      # Level 1: Tesla-only baseline
python train.py --level 0      # Base: suppliers only

# Skip plots (faster)
python train.py --no-plots

# Next-day price prediction demo
python predict.py
```

First run downloads ~7 years of data from Yahoo Finance and caches it to `data/raw_prices.csv`. Subsequent runs load from cache instantly.

---

## Project Structure

```
tesla-supplier-prediction/
│
├── src/
│   ├── config.py               # hyperparameters & tickers
│   ├── data_loader.py          # Yahoo Finance download + cache
│   ├── feature_engineering.py  # feature sets for all 4 levels
│   ├── models.py               # LSTM + RF + XGBoost stacking
│   ├── evaluation.py           # metrics + time-series CV
│   └── visualisation.py        # figures & dashboard
│
├── train.py                    # end-to-end training pipeline
├── predict.py                  # next-day inference demo
└── requirements.txt
```

Generated at runtime: `data/` · `results/` · `plots/` · `models/`

---

## Key Findings

1. **Short-range autocorrelation dominates.** A single lagged Tesla price explains 63% of the variance in next-day price. The market already prices in most public supplier information.

2. **Suppliers improve directional accuracy, not magnitude accuracy.** Level 3 RMSE is slightly worse than Level 1, but directional accuracy jumps from 26.7% → 34.7%. For a trading signal, direction matters more than precise dollar value.

3. **Lag length is critical.** A 1-day lag captures momentum; a 20-day lag loses it entirely (Level 2 R² drops from 0.63 to 0.40 vs Level 1).

4. **Supplier signals alone have no standalone power.** The Base model's negative per-fold R² values confirm suppliers cannot predict Tesla in isolation — they are a modifier, not a driver.

---

## Future Directions

- Sentiment signals from Reddit, X, and financial news filings
- Granger causality tests to formalise the supplier → Tesla causal direction
- SHAP values for per-supplier attribution
- Reframe as binary classification (up/down) to directly optimise directional accuracy
- Extend to other EV manufacturers: Rivian, NIO, BYD

---

## Citation

```bibtex
@inproceedings{ananthan2024tesla,
  title       = {Leveraging Supplier Stock Dynamics to Predict Tesla's Market Performance},
  author      = {Ananthan, Srihari and Shivanna, Gagandeep and Dhanawade, Rohan Rajendra
                 and Salunke, Kaustubh and Schwarz, Klaus},
  booktitle   = {Ohrid Conference Proceedings},
  year        = {2024},
  institution = {SRH University Berlin}
}
```

