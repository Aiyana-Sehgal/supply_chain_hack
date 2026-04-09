"""
Refresh locally compatible model artifacts for the supply chain backend.

This script:
1. Converts the legacy XGBoost demand model pickle into a stable JSON artifact.
2. Rebuilds and saves the demand scaler in the current Python environment.
3. Retrains and saves the Q-learning agent pickle in the current environment.
"""

from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "store_sale.csv"
sys.path.insert(0, str(PROJECT_ROOT))


def build_lag_features(series, n_lags=12):
    df = pd.DataFrame(series, columns=["sales_diff"])
    for i in range(1, n_lags + 1):
        df[f"month_{i}"] = df["sales_diff"].shift(i)
    return df.dropna().reset_index(drop=True)


def load_monthly_sales():
    sales = pd.read_csv(DATA_PATH)
    sales = sales.drop(["store", "item"], axis=1)
    sales["date"] = pd.to_datetime(sales["date"])
    sales["date"] = sales["date"].dt.to_period("M")
    monthly = sales.groupby("date").sum().reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()
    monthly["sales_diff"] = monthly["sales"].diff()
    return monthly.dropna().reset_index(drop=True)


def refresh_demand_artifacts():
    print("Refreshing demand model artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    legacy_model_path = PROJECT_ROOT / "demand_forecast_model.pkl"
    if not legacy_model_path.exists():
        legacy_model_path = MODELS_DIR / "demand_forecast_model.pkl"

    with open(legacy_model_path, "rb") as model_file:
        legacy_model = pickle.load(model_file)

    model_json_path = MODELS_DIR / "demand_forecast_model.json"
    legacy_model.save_model(str(model_json_path))
    shutil.copy2(model_json_path, PROJECT_ROOT / "demand_forecast_model.json")

    monthly_sales = load_monthly_sales()
    supervised = build_lag_features(monthly_sales["sales_diff"], n_lags=12)
    feature_columns = ["sales_diff"] + [f"month_{i}" for i in range(1, 13)]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(supervised[feature_columns])

    scaler_path = MODELS_DIR / "demand_scaler.pkl"
    with open(scaler_path, "wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    shutil.copy2(scaler_path, PROJECT_ROOT / "demand_scaler.pkl")

    print(f"Saved {model_json_path}")
    print(f"Saved {scaler_path}")


def generate_training_scenarios():
    scenarios = []
    demands = [350000, 450000, 472276, 550000, 700000, 800000]
    inventories = [500, 1000, 5000, 8000, 10000, 25000, 30000, 50000, 100000]
    risk_scores = [0.15, 0.25, 0.5, 0.7, 0.8, 0.85, 0.92, 0.95]
    disruptions = [0.1, 0.2, 0.4, 0.5, 0.7, 0.75, 0.88, 0.95]

    for demand in demands:
        for inventory in inventories:
            for risk in risk_scores:
                for disruption in disruptions:
                    days = inventory / max(demand / 30, 1)
                    scenarios.append(
                        {
                            "predicted_demand": demand,
                            "current_inventory": inventory,
                            "supplier_risk_score": risk,
                            "disruption_signal": disruption,
                            "days_to_stockout": min(days, 30),
                        }
                    )
    return scenarios


def refresh_rl_agent(epochs=1000, scenario_sample=300):
    print("Refreshing RL agent artifact...")
    from src.core.layer4 import SupplyChainAgent, simulate_outcome, calculate_reward

    agent = SupplyChainAgent()
    all_scenarios = generate_training_scenarios()

    for epoch in range(epochs):
        sample_indices = np.random.choice(
            len(all_scenarios),
            size=min(scenario_sample, len(all_scenarios)),
            replace=False,
        )
        for idx in sample_indices:
            state = all_scenarios[idx]
            action = agent.choose_action(state, training=True)
            outcome = simulate_outcome(state, action)
            reward = calculate_reward(action, outcome)

            next_state = state.copy()
            next_state["predicted_demand"] *= np.random.uniform(0.95, 1.05)
            next_state["current_inventory"] *= np.random.uniform(0.90, 0.95)
            next_state["days_to_stockout"] = max(0, next_state["days_to_stockout"] - 1)

            agent.learn(state, action, reward, next_state)

        if (epoch + 1) % 200 == 0:
            print(f"  epoch {epoch + 1}/{epochs} complete")

    agent_path = MODELS_DIR / "supply_chain_agent.pkl"
    agent.save(str(agent_path))
    shutil.copy2(agent_path, PROJECT_ROOT / "supply_chain_agent.pkl")
    print(f"Saved {agent_path}")


def main():
    refresh_demand_artifacts()
    refresh_rl_agent()
    print("Artifact refresh complete.")


if __name__ == "__main__":
    main()
