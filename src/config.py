START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"

TESLA_TICKER = "TSLA"

SUPPLIER_TICKERS = {
    "CATL":              "300750.SZ",
    "BYD":               "002594.SZ",
    "TSMC":              "TSM",
    "Sony":              "SONY",
    "Fuyao Glass":       "3606.HK",
    "Samsung Display":   "005930.KS",
    "TE Connectivity":   "TEL",
    "BHP":               "BHP",
    "Texas Instruments": "TXN",
}

MARKET_INDICATORS = {
    "VIX":  "^VIX",
    "QQQ":  "QQQ",
    "XLI":  "XLI",
    "ICLN": "ICLN",
    "ARKQ": "ARKQ",
}

SHORT_LAG = 1
LONG_LAG  = 20

LSTM_UNITS      = 64
LSTM_DROPOUT    = 0.2
LSTM_EPOCHS     = 50
LSTM_BATCH_SIZE = 32
LSTM_LOOKBACK   = 10

RF_N_ESTIMATORS = 200
RF_MAX_DEPTH    = 10
RF_RANDOM_STATE = 42

XGB_N_ESTIMATORS  = 100
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH     = 4
XGB_RANDOM_STATE  = 42

N_SPLITS    = 5

DATA_DIR    = "data"
RESULTS_DIR = "results"
PLOTS_DIR   = "plots"
MODELS_DIR  = "models"
