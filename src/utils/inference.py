"""
Inference utilities for demand forecasting.
"""

from pathlib import Path
from typing import Tuple
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_JSON_PATH = MODELS_DIR / "demand_forecast_model.json"

_MODEL_CACHE = None
_SCALER_CACHE = None
_MONTHLY_SALES_CACHE = None


def _build_lag_features(series: pd.Series, n_lags: int = 12) -> pd.DataFrame:
    df = pd.DataFrame(series, columns=["sales_diff"])
    for i in range(1, n_lags + 1):
        df[f"month_{i}"] = df["sales_diff"].shift(i)
    return df.dropna().reset_index(drop=True)


def _rebuild_scaler(n_lags: int = 12) -> MinMaxScaler:
    monthly_sales = _load_monthly_sales()
    supervised_data = _build_lag_features(monthly_sales["sales_diff"], n_lags=n_lags)
    feature_columns = ["sales_diff"] + [f"month_{i}" for i in range(1, n_lags + 1)]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(supervised_data[feature_columns])
    return scaler


def _load_models():
    global _MODEL_CACHE, _SCALER_CACHE
    if _MODEL_CACHE is None or _SCALER_CACHE is None:
        if MODEL_JSON_PATH.exists():
            booster = xgb.Booster()
            booster.load_model(str(MODEL_JSON_PATH))
            _MODEL_CACHE = booster
        else:
            with open(MODELS_DIR / "demand_forecast_model.pkl", "rb") as model_file:
                _MODEL_CACHE = pickle.load(model_file)
        try:
            with open(MODELS_DIR / "demand_scaler.pkl", "rb") as scaler_file:
                _SCALER_CACHE = pickle.load(scaler_file)
        except Exception:
            _SCALER_CACHE = _rebuild_scaler()
    return _MODEL_CACHE, _SCALER_CACHE


def _load_monthly_sales() -> pd.DataFrame:
    global _MONTHLY_SALES_CACHE
    if _MONTHLY_SALES_CACHE is None:
        store_sales = pd.read_csv(DATA_DIR / "store_sale.csv")
        store_sales = store_sales.drop(["store", "item"], axis=1)
        store_sales["date"] = pd.to_datetime(store_sales["date"])
        store_sales["date"] = store_sales["date"].dt.to_period("M")
        monthly_sales = store_sales.groupby("date").sum().reset_index()
        monthly_sales["date"] = monthly_sales["date"].dt.to_timestamp()
        monthly_sales["sales_diff"] = monthly_sales["sales"].diff()
        _MONTHLY_SALES_CACHE = monthly_sales.dropna().reset_index(drop=True)
    return _MONTHLY_SALES_CACHE.copy()


def _build_scaled_input(input_features: np.ndarray) -> np.ndarray:
    _, scaler = _load_models()
    feature_names = ["sales_diff"] + [f"month_{i}" for i in range(1, 13)]
    full_input = np.concatenate([[0.0], input_features]).reshape(1, -1)
    full_input_df = pd.DataFrame(full_input, columns=feature_names)
    full_input_scaled = scaler.transform(full_input_df)
    return full_input_scaled[:, 1:]


def predict_next_month_demand(n_lags: int = 12) -> Tuple[float, float]:
    """
    Predict the next month's sales and first-difference in original scale.
    """
    model_tuned, scaler = _load_models()
    monthly_sales = _load_monthly_sales()
    supervised_data = _build_lag_features(monthly_sales["sales_diff"], n_lags=n_lags)

    latest_features = supervised_data.iloc[-1]
    input_features = latest_features[[f"month_{i}" for i in range(1, n_lags + 1)]].to_numpy(dtype=float)
    x_input = _build_scaled_input(input_features)

    scaled_pred = model_tuned.predict(xgb.DMatrix(x_input))
    pred_row = np.concatenate([scaled_pred.reshape(-1, 1), x_input], axis=1)
    pred_unscaled = scaler.inverse_transform(pred_row)
    predicted_sales_diff = float(pred_unscaled[0][0])

    last_actual_sales = float(monthly_sales["sales"].iloc[-1])
    predicted_sales = predicted_sales_diff + last_actual_sales
    return predicted_sales, predicted_sales_diff


def predict_n_months(n: int = 3) -> pd.DataFrame:
    """
    Roll demand forecasts forward for the next N months.
    """
    model_tuned, scaler = _load_models()
    temp_monthly = _load_monthly_sales()
    predictions = []

    for _ in range(n):
        temp_supervised = _build_lag_features(temp_monthly["sales_diff"])
        latest_features = temp_supervised.iloc[-1]
        input_features = latest_features[[f"month_{j}" for j in range(1, 13)]].to_numpy(dtype=float)
        x_input = _build_scaled_input(input_features)

        scaled_pred = model_tuned.predict(xgb.DMatrix(x_input))
        pred_row = np.concatenate([scaled_pred.reshape(-1, 1), x_input], axis=1)
        pred_unscaled = scaler.inverse_transform(pred_row)
        predicted_diff = float(pred_unscaled[0][0])

        last_sales = float(temp_monthly["sales"].iloc[-1])
        predicted_sales = predicted_diff + last_sales
        next_date = temp_monthly["date"].iloc[-1] + pd.DateOffset(months=1)

        predictions.append(
            {
                "month": next_date.strftime("%B %Y"),
                "predicted_sales": round(predicted_sales),
                "sales_diff": round(predicted_diff),
            }
        )

        new_row = pd.DataFrame(
            {
                "date": [next_date],
                "sales": [predicted_sales],
                "sales_diff": [predicted_diff],
            }
        )
        temp_monthly = pd.concat([temp_monthly, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


if __name__ == "__main__":
    predicted_sales, predicted_diff = predict_next_month_demand()
    monthly_sales = _load_monthly_sales()
    next_month = monthly_sales["date"].iloc[-1] + pd.DateOffset(months=1)

    print("=" * 45)
    print(f"  Demand Forecast - {next_month.strftime('%B %Y')}")
    print("=" * 45)
    print(f"  Predicted sales:      {predicted_sales:,.0f} units")
    print(f"  Month-over-month diff:{predicted_diff:,.0f} units")
    print(f"  Last actual sales:    {monthly_sales['sales'].iloc[-1]:,.0f} units")
    print("=" * 45)

    forecast_df = predict_n_months(n=3)
    print("\n  3-Month Rolling Forecast")
    print("=" * 45)
    print(forecast_df.to_string(index=False))
    print("=" * 45)
