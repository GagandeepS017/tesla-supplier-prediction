import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.config import (
    LSTM_UNITS, LSTM_DROPOUT, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LOOKBACK,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH, XGB_RANDOM_STATE,
)
from src.feature_engineering import make_sequences

logger = logging.getLogger(__name__)


def build_lstm(n_features: int, lookback: int) -> Sequential:
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(lookback, n_features), return_sequences=False),
        Dropout(LSTM_DROPOUT),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


class StackingEnsemble:
    def __init__(self, lookback: int = LSTM_LOOKBACK):
        self.lookback   = lookback
        self.rf         = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,
        )
        self.xgb        = XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            learning_rate=XGB_LEARNING_RATE,
            max_depth=XGB_MAX_DEPTH,
            random_state=XGB_RANDOM_STATE,
            verbosity=0,
        )
        self.lstm_model  = None
        self._n_features = None

    def _prepare_sequences(self, X: np.ndarray):
        n      = len(X)
        X_seq  = np.array([X[i - self.lookback: i] for i in range(self.lookback, n)])
        X_flat = X[self.lookback:]
        return X_seq, X_flat

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._n_features = X.shape[1]
        X_seq, X_flat    = self._prepare_sequences(X)
        y_trimmed        = y[self.lookback:]

        logger.info("  Training LSTM …")
        self.lstm_model = build_lstm(self._n_features, self.lookback)
        es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        self.lstm_model.fit(
            X_seq, y_trimmed,
            epochs=LSTM_EPOCHS,
            batch_size=LSTM_BATCH_SIZE,
            validation_split=0.1,
            callbacks=[es],
            verbose=0,
        )

        logger.info("  Training Random Forest …")
        self.rf.fit(X_flat, y_trimmed)

        lstm_preds = self.lstm_model.predict(X_seq, verbose=0).flatten()
        rf_preds   = self.rf.predict(X_flat)
        meta_X     = np.column_stack([lstm_preds, rf_preds])

        logger.info("  Training XGBoost meta-learner …")
        self.xgb.fit(meta_X, y_trimmed)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_seq, X_flat = self._prepare_sequences(X)
        lstm_preds    = self.lstm_model.predict(X_seq, verbose=0).flatten()
        rf_preds      = self.rf.predict(X_flat)
        meta_X        = np.column_stack([lstm_preds, rf_preds])
        return self.xgb.predict(meta_X)

    def predict_components(self, X: np.ndarray):
        X_seq, X_flat = self._prepare_sequences(X)
        lstm_preds    = self.lstm_model.predict(X_seq, verbose=0).flatten()
        rf_preds      = self.rf.predict(X_flat)
        meta_X        = np.column_stack([lstm_preds, rf_preds])
        final         = self.xgb.predict(meta_X)
        return lstm_preds, rf_preds, final
