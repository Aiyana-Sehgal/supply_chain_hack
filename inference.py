import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import datetime

# ── Load saved models ──────────────────────────────────────────────
with open('demand_forecast_model.pkl', 'rb') as f:
    model_tuned = pickle.load(f)

with open('demand_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ── Load your dataset (needed to build lag features) ───────────────
store_sales = pd.read_csv('store_sale.csv')
store_sales = store_sales.drop(['store', 'item'], axis=1)
store_sales['date'] = pd.to_datetime(store_sales['date'])
store_sales['date'] = store_sales['date'].dt.to_period('M')
monthly_sales = store_sales.groupby('date').sum().reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna().reset_index(drop=True)

# ── Build supervised lag features ─────────────────────────────────
def build_lag_features(series, n_lags=12):
    df = pd.DataFrame(series, columns=['sales_diff'])
    for i in range(1, n_lags + 1):
        df[f'month_{i}'] = df['sales_diff'].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df

supervised_data = build_lag_features(monthly_sales['sales_diff'])

# ── Core inference function ────────────────────────────────────────
def predict_next_month_demand(n_lags=12):
    """
    Uses the last 12 months of sales_diff to predict
    the next month's total sales.
    Returns predicted sales value in original scale.
    """

    # Take the last row of supervised data as input
    latest_features = supervised_data.iloc[-1]

    # Input is month_1 to month_12 (lag features only, no sales_diff)
    input_features = latest_features[[f'month_{i}' for i in range(1, n_lags + 1)]].values

    # We need a full row for scaler (sales_diff + 12 lags)
    # Pad with 0 for sales_diff position (index 0) since we're predicting it
    full_input = np.concatenate([[0], input_features]).reshape(1, -1)

    # Scale
    feature_names = ['sales_diff'] + [f'month_{i}' for i in range(1, 13)]
    full_input_df = pd.DataFrame(full_input, columns=feature_names)
    full_input_scaled = scaler.transform(full_input_df)

    # Extract only the feature columns (drop index 0 which is the target)
    X_input = full_input_scaled[:, 1:]

    # Predict scaled sales_diff
    dmatrix_input = xgb.DMatrix(X_input)
    scaled_pred = model_tuned.predict(dmatrix_input)

    # Inverse transform: reconstruct full row with prediction in index 0
    pred_row = np.concatenate([scaled_pred.reshape(-1, 1), X_input], axis=1)
    pred_unscaled = scaler.inverse_transform(pred_row)
    predicted_sales_diff = pred_unscaled[0][0]

    # Recover actual sales value from last known sales
    last_actual_sales = monthly_sales['sales'].iloc[-1]
    predicted_sales = predicted_sales_diff + last_actual_sales

    return predicted_sales, predicted_sales_diff


# ── Run inference ──────────────────────────────────────────────────
predicted_sales, predicted_diff = predict_next_month_demand()

last_date = monthly_sales['date'].iloc[-1]
next_month = last_date + pd.DateOffset(months=1)

print("=" * 45)
print(f"  Demand Forecast — {next_month.strftime('%B %Y')}")
print("=" * 45)
print(f"  Predicted sales:      {predicted_sales:,.0f} units")
print(f"  Month-over-month diff:{predicted_diff:,.0f} units")
print(f"  Last actual sales:    {monthly_sales['sales'].iloc[-1]:,.0f} units")
print("=" * 45)


# ── Batch inference: predict next N months ─────────────────────────
def predict_n_months(n=3):
    """
    Rolls forward predictions for the next N months.
    Each predicted value feeds into the next prediction as a lag.
    """
    temp_monthly = monthly_sales.copy()
    predictions = []

    for i in range(n):
        temp_supervised = build_lag_features(temp_monthly['sales_diff'])
        latest_features = temp_supervised.iloc[-1]
        input_features = latest_features[[f'month_{j}' for j in range(1, 13)]].values

        full_input = np.concatenate([[0], input_features]).reshape(1, -1)
        full_input_scaled = scaler.transform(full_input)
        X_input = full_input_scaled[:, 1:]

        dmatrix_input = xgb.DMatrix(X_input)
        scaled_pred = model_tuned.predict(dmatrix_input)

        pred_row = np.concatenate([scaled_pred.reshape(-1, 1), X_input], axis=1)
        pred_unscaled = scaler.inverse_transform(pred_row)
        predicted_diff = pred_unscaled[0][0]

        last_sales = temp_monthly['sales'].iloc[-1]
        predicted_sales = predicted_diff + last_sales

        next_date = temp_monthly['date'].iloc[-1] + pd.DateOffset(months=1)

        predictions.append({
            'month': next_date.strftime('%B %Y'),
            'predicted_sales': round(predicted_sales),
            'sales_diff': round(predicted_diff)
        })

        # Roll forward: append predicted row into temp dataframe
        new_row = pd.DataFrame({
            'date': [next_date],
            'sales': [predicted_sales],
            'sales_diff': [predicted_diff]
        })
        temp_monthly = pd.concat([temp_monthly, new_row], ignore_index=True)

    return pd.DataFrame(predictions)


# ── Run batch inference ────────────────────────────────────────────
forecast_df = predict_n_months(n=3)
print("\n  3-Month Rolling Forecast")
print("=" * 45)
print(forecast_df.to_string(index=False))
print("=" * 45)