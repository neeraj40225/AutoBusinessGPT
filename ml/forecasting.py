"""Sales forecasting on the engineered monthly series.

Trains Linear Regression, Random Forest, and (if available) XGBoost on lagged
monthly features, selects the winner by held-out RMSE, and produces a recursive
multi-step forecast with a confidence band that widens with the horizon (errors
compound when predictions feed the next step's lags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from core.config import settings
from utils.helpers import safe_divide
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:  # pragma: no cover
    _HAS_XGB = False

_FEATURES = ["time_index", "month_num", "month_sin", "month_cos",
             "lag_1", "lag_2", "lag_3", "roll_mean_3"]


@dataclass
class ModelScore:
    name: str
    rmse: float
    mae: float
    r2: float
    mape: float

    def to_dict(self) -> dict[str, Any]:
        return {"Model": self.name, "RMSE": round(self.rmse, 1),
                "MAE": round(self.mae, 1), "R²": round(self.r2, 3),
                "MAPE %": round(self.mape, 1)}


@dataclass
class ForecastResult:
    best_name: str
    scores: list[ModelScore]
    history: pd.DataFrame            # month, sales
    forecast: pd.DataFrame          # month, forecast, lower, upper
    horizon: int

    @property
    def scores_frame(self) -> pd.DataFrame:
        return pd.DataFrame([s.to_dict() for s in self.scores])


def _candidates() -> dict[str, Any]:
    rs = settings.ml.random_state
    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=2,
            random_state=rs, n_jobs=-1),
    }
    if _HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=rs, n_jobs=-1, verbosity=0)
    return models


def train(monthly: pd.DataFrame, horizon: int | None = None) -> ForecastResult:
    """Train candidates on the monthly series and forecast forward."""
    horizon = horizon or settings.ml.forecast_horizon
    df = monthly.dropna(subset=["lag_3"]).reset_index(drop=True)
    if len(df) < 4:
        raise ValueError("Not enough history after building lags to train.")

    # chronological split — never shuffle a time series
    split = max(3, int(len(df) * (1 - settings.ml.test_size)))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    if test_df.empty:
        test_df = train_df.iloc[-2:]

    x_tr, y_tr = train_df[_FEATURES], train_df["sales"]
    x_te, y_te = test_df[_FEATURES], test_df["sales"]

    scores: list[ModelScore] = []
    fitted: dict[str, Any] = {}
    for name, model in _candidates().items():
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        mae = float(mean_absolute_error(y_te, pred))
        r2 = float(r2_score(y_te, pred)) if len(y_te) > 1 else 0.0
        mape = float(np.mean([abs(safe_divide(t - p, t)) * 100
                              for t, p in zip(y_te.to_numpy(), pred)]))
        scores.append(ModelScore(name, rmse, mae, r2, mape))
        fitted[name] = model

    scores.sort(key=lambda s: s.rmse)
    best_name = scores[0].name
    best = fitted[best_name]

    # refit on full series before projecting
    best.fit(df[_FEATURES], df["sales"])
    forecast = _recursive_forecast(best, df, horizon, scores[0].rmse)

    history = pd.DataFrame({"month": df["month"].astype(str), "sales": df["sales"].to_numpy()})
    logger.info("Forecast best=%s RMSE=%.0f", best_name, scores[0].rmse)
    return ForecastResult(best_name, scores, history, forecast, horizon)


def _recursive_forecast(model: Any, df: pd.DataFrame, horizon: int, rmse: float) -> pd.DataFrame:
    """Roll forward one month at a time, feeding predictions back as lags."""
    hist = df["sales"].tolist()
    last_period = pd.Period(df["month"].iloc[-1], freq="M")
    rows: list[dict[str, Any]] = []

    for step in range(1, horizon + 1):
        nxt = last_period + step
        feat = {
            "time_index": float(df["time_index"].iloc[-1]) + step,
            "month_num": nxt.month,
            "month_sin": np.sin(2 * np.pi * nxt.month / 12),
            "month_cos": np.cos(2 * np.pi * nxt.month / 12),
            "lag_1": hist[-1],
            "lag_2": hist[-2] if len(hist) >= 2 else hist[-1],
            "lag_3": hist[-3] if len(hist) >= 3 else hist[-1],
            "roll_mean_3": float(np.mean(hist[-3:])),
        }
        x = pd.DataFrame([{k: feat[k] for k in _FEATURES}])
        pred = max(float(model.predict(x)[0]), 0.0)
        band = 1.96 * rmse * np.sqrt(step)
        rows.append({"month": str(nxt), "forecast": pred,
                     "lower": max(pred - band, 0.0), "upper": pred + band})
        hist.append(pred)

    return pd.DataFrame(rows)


__all__ = ["train", "ForecastResult", "ModelScore"]
