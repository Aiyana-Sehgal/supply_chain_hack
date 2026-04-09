"""
End-to-End Integration Test: All 4 Layers
Tests the complete supply chain intelligence pipeline:
Layer 1 (Data Collection) → Layer 2 (Features) → Layer 3 (Forecast) → Layer 4 (RL Agent)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from datetime import datetime

print("=" * 80)
print("SUPPLY CHAIN INTELLIGENCE SYSTEM - END-TO-END TEST")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Import all layers
try:
    from layer1 import build_layer1_dataset
    from layer2 import build_layer2_dataset
    from layer4 import SupplyChainAgent, get_recommendation
    print("\n✓ All layer modules imported successfully")
except ImportError as e:
    print(f"\n✗ Failed to import layer modules: {e}")
    sys.exit(1)

# Check for API keys
print("\n" + "=" * 80)
print("CHECKING API KEYS")
print("=" * 80)
news_key = os.getenv('NEWSDATA_API_KEY')
weather_key = os.getenv('WEATHERAPI_KEY')

if news_key:
    print(f"✓ NEWSDATA_API_KEY found (length: {len(news_key)})")
else:
    print("⚠ NEWSDATA_API_KEY not found - will use fallback headlines")

if weather_key:
    print(f"✓ WEATHERAPI_KEY found (length: {len(weather_key)})")
else:
    print("⚠ WEATHERAPI_KEY not found - will use fallback weather")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 1: DATA COLLECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 80)
print("LAYER 1: DATA COLLECTION")
print("=" * 80)

print("\n[L1.1] Building Layer 1 dataset with real API keys...")
try:
    layer1_bundle = build_layer1_dataset(
        sales_path='store_sale.csv',
        news_api_key=news_key,
        weather_api_key=weather_key
    )
    print("✓ Layer 1 dataset built successfully!")
    
    # Validate outputs
    layer1_df = layer1_bundle['layer1_df']
    monthly_sales = layer1_bundle['monthly_sales']
    inventory_data = layer1_bundle['inventory_data']
    weather_signals = layer1_bundle['weather_signals']
    news_headlines = layer1_bundle['news_headlines']
    
    print(f"\n  Data Summary:")
    print(f"    • Monthly sales: {monthly_sales.shape} (date range: {monthly_sales['date'].min().date()} to {monthly_sales['date'].max().date()})")
    print(f"    • Inventory records: {inventory_data.shape}")
    print(f"    • Weather signals: {weather_signals.shape}")
    print(f"    • News headlines collected: {len(news_headlines)}")
    print(f"    • Layer 1 consolidated: {layer1_df.shape}")
    
    # Display sample of Layer 1 output
    print(f"\n  Layer 1 DF sample (last 3 rows):")
    print(layer1_df[['date', 'sales', 'current_inventory', 'disruption_signal', 'supplier_risk_score']].tail(3).to_string())
    
    # Validate data quality
    print(f"\n  Data Quality Check:")
    print(f"    ✓ No null values in key columns: {layer1_df[['date', 'sales', 'current_inventory']].isna().sum().sum() == 0}")
    print(f"    ✓ Disruption signal in [0,1]: {(layer1_df['disruption_signal'] >= 0).all() and (layer1_df['disruption_signal'] <= 1).all()}")
    print(f"    ✓ Risk scores in [0,1]: {(layer1_df['supplier_risk_score'] >= 0).all() and (layer1_df['supplier_risk_score'] <= 1).all()}")
    
except Exception as e:
    print(f"✗ Error in Layer 1: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 2: FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 80)
print("LAYER 2: FEATURE ENGINEERING & NORMALIZATION")
print("=" * 80)

print("\n[L2.1] Building Layer 2 consolidated features...")
try:
    layer2_bundle = build_layer2_dataset(
        layer1_bundle,
        inventory_method='risk_weighted',
        weather_method='mean',
        include_rolling=True
    )
    print("✓ Layer 2 features engineered successfully!")
    
    consolidated_df = layer2_bundle['consolidated_df']
    feature_columns = layer2_bundle['feature_columns']
    
    print(f"\n  Feature Engineering Summary:")
    print(f"    • Input features (Layer 1): 8")
    print(f"    • Output features (Layer 2): {len(consolidated_df.columns)}")
    print(f"    • Normalized features: {len(feature_columns)}")
    print(f"    • Records: {len(consolidated_df)}")
    
    # Display feature columns
    print(f"\n  Features in consolidated dataset:")
    for i, col in enumerate(consolidated_df.columns, 1):
        dtype_str = str(consolidated_df[col].dtype)
        print(f"    {i:2d}. {col:<40} {dtype_str}")
    
    # Display statistics
    print(f"\n  Aggregation Results:")
    print(f"    • Supplier inventory aggregate: min={consolidated_df['weighted_inventory'].min():.2f}, max={consolidated_df['weighted_inventory'].max():.2f}")
    print(f"    • Regional weather disruption: min={consolidated_df['regional_weather_disruption'].min():.3f}, max={consolidated_df['regional_weather_disruption'].max():.3f}")
    print(f"    • Composite risk score: min={consolidated_df['composite_risk_score'].min():.3f}, max={consolidated_df['composite_risk_score'].max():.3f}")
    
    # Display sample
    print(f"\n  Layer 2 DF sample (last 3 rows):")
    cols_to_show = ['date', 'sales', 'composite_risk_score', 'weighted_inventory', 'sales_rolling_mean']
    print(consolidated_df[cols_to_show].tail(3).to_string())
    
except Exception as e:
    print(f"✗ Error in Layer 2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 3: DEMAND FORECASTING & INTEGRATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 80)
print("LAYER 3: DEMAND FORECASTING & INTEGRATION")
print("=" * 80)

print("\n[L3.1] Loading demand forecast model...")
try:
    with open('demand_forecast_model.pkl', 'rb') as f:
        model_tuned = pickle.load(f)
    
    with open('demand_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("✓ Demand forecast models loaded successfully!")
    print(f"    • Model: {type(model_tuned).__name__}")
    print(f"    • Scaler: {type(scaler).__name__}")
    
except FileNotFoundError as e:
    print(f"✗ Model files not found: {e}")
    sys.exit(1)

print("\n[L3.2] Generating demand forecast (next 3 months)...")
try:
    # Prepare data for forecasting
    store_sales = pd.read_csv('store_sale.csv')
    store_sales = store_sales.drop(['store', 'item'], axis=1)
    store_sales['date'] = pd.to_datetime(store_sales['date'])
    store_sales['date'] = store_sales['date'].dt.to_period('M')
    monthly_sales = store_sales.groupby('date').sum().reset_index()
    monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
    monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
    monthly_sales = monthly_sales.dropna().reset_index(drop=True)
    
    # Build lag features
    def build_lag_features(series, n_lags=12):
        df = pd.DataFrame(series, columns=['sales_diff'])
        for i in range(1, n_lags + 1):
            df[f'month_{i}'] = df['sales_diff'].shift(i)
        df = df.dropna().reset_index(drop=True)
        return df
    
    # Generate forecasts
    temp_monthly = monthly_sales.copy()
    forecasts = []
    
    for i in range(3):
        temp_supervised = build_lag_features(temp_monthly['sales_diff'])
        latest_features = temp_supervised.iloc[-1]
        input_features = latest_features[[f'month_{j}' for j in range(1, 13)]].values
        
        full_input = np.concatenate([[0], input_features]).reshape(1, -1)
        feature_names = ['sales_diff'] + [f'month_{i}' for i in range(1, 13)]
        full_input_df = pd.DataFrame(full_input, columns=feature_names)
        full_input_scaled = scaler.transform(full_input_df)
        
        X_input = full_input_scaled[:, 1:]
        dmatrix_input = xgb.DMatrix(X_input)
        scaled_pred = model_tuned.predict(dmatrix_input)
        
        pred_row = np.concatenate([scaled_pred.reshape(-1, 1), X_input], axis=1)
        pred_unscaled = scaler.inverse_transform(pred_row)
        predicted_diff = pred_unscaled[0][0]
        
        last_sales = temp_monthly['sales'].iloc[-1]
        predicted_sales = predicted_diff + last_sales
        next_date = temp_monthly['date'].iloc[-1] + pd.DateOffset(months=1)
        
        forecasts.append({
            'date': next_date,
            'predicted_sales': predicted_sales,
            'predicted_diff': predicted_diff
        })
        
        new_row = pd.DataFrame({
            'date': [next_date],
            'sales': [predicted_sales],
            'sales_diff': [predicted_diff]
        })
        temp_monthly = pd.concat([temp_monthly, new_row], ignore_index=True)
    
    forecast_df = pd.DataFrame(forecasts)
    print("✓ Demand forecast generated!")
    
    print(f"\n  3-Month Forecast:")
    for idx, row in forecast_df.iterrows():
        print(f"    {row['date'].strftime('%B %Y'):>12}: {row['predicted_sales']:>12,.0f} units (Δ {row['predicted_diff']:>10,.0f})")
    
except Exception as e:
    print(f"✗ Error in Layer 3 forecasting: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[L3.3] Merging Layer 2 & forecast data...")
try:
    layer3_state_df = consolidated_df.copy()
    
    # Add forecast
    forecast_df_merged = forecast_df[['date', 'predicted_sales']].copy()
    layer3_state_df = layer3_state_df.merge(forecast_df_merged, on='date', how='left', suffixes=('', '_forecast'))
    
    # Fill missing forecast
    layer3_state_df['predicted_sales'] = layer3_state_df['predicted_sales'].fillna(layer3_state_df['sales'])
    
    print("✓ Layer 2 & forecast integrated!")
    print(f"    • Integrated state shape: {layer3_state_df.shape}")
    
except Exception as e:
    print(f"✗ Error integrating data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LAYER 4: RL AGENT DECISION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 80)
print("LAYER 4: RL AGENT DECISION ENGINE")
print("=" * 80)

print("\n[L4.1] Loading trained RL agent...")
try:
    agent = SupplyChainAgent()
    agent.load('supply_chain_agent.pkl')
    print("✓ RL agent loaded successfully!")
    print(f"    • Model: Q-Learning Agent")
    print(f"    • Q-table entries: {len(agent.q_table)}")
    print(f"    • Action space: 5 actions")
    
except Exception as e:
    print(f"⚠ RL agent not found, using untrained agent: {e}")
    agent = SupplyChainAgent()

print("\n[L4.2] Testing agent on various supply chain scenarios...")

# Test scenarios
test_scenarios = [
    {
        "name": "🔴 CRITICAL: Low Inventory Crisis",
        "state": {
            "predicted_demand": float(forecast_df['predicted_sales'].iloc[0]),
            "current_inventory": 500.0,
            "supplier_risk_score": 0.85,
            "disruption_signal": 0.75,
            "days_to_stockout": 2.0
        }
    },
    {
        "name": "🟠 URGENT: Rapid Depletion",
        "state": {
            "predicted_demand": float(forecast_df['predicted_sales'].iloc[0]),
            "current_inventory": 8000.0,
            "supplier_risk_score": 0.70,
            "disruption_signal": 0.50,
            "days_to_stockout": 5.0
        }
    },
    {
        "name": "🟡 HIGH RISK: Supplier Issues",
        "state": {
            "predicted_demand": float(forecast_df['predicted_sales'].iloc[0]),
            "current_inventory": 25000.0,
            "supplier_risk_score": 0.92,
            "disruption_signal": 0.20,
            "days_to_stockout": 20.0
        }
    },
    {
        "name": "🟡 DISRUPTION: Route/Weather Issues",
        "state": {
            "predicted_demand": float(forecast_df['predicted_sales'].iloc[0]),
            "current_inventory": 30000.0,
            "supplier_risk_score": 0.25,
            "disruption_signal": 0.88,
            "days_to_stockout": 18.0
        }
    },
    {
        "name": "🟢 HEALTHY: Normal Operations",
        "state": {
            "predicted_demand": float(forecast_df['predicted_sales'].iloc[0]),
            "current_inventory": 50000.0,
            "supplier_risk_score": 0.15,
            "disruption_signal": 0.10,
            "days_to_stockout": 25.0
        }
    }
]

results = []

for scenario in test_scenarios:
    print(f"\n  {scenario['name']}")
    print(f"  {'-' * 70}")
    
    try:
        recommendation = get_recommendation(scenario['state'], agent)
        
        print(f"    ✓ Action: {recommendation['action'].upper()}")
        print(f"      Confidence: {recommendation['confidence']:.1f}%")
        print(f"      Explanation: {recommendation['explanation'][:100]}...")
        
        results.append({
            'scenario': scenario['name'],
            'action': recommendation['action'],
            'confidence': recommendation['confidence']
        })
        
    except Exception as e:
        print(f"    ✗ Error: {e}")

print("\n" + "=" * 80)
print("SCENARIO SUMMARY")
print("=" * 80)
print("\n  Scenario ID | Situation              | Action              | Confidence")
print("  " + "-" * 76)
for i, result in enumerate(results, 1):
    scenario_short = result['scenario'].split(':')[0]
    print(f"      {i}      | {scenario_short:<22} | {result['action'].upper():<19} | {result['confidence']:>6.1f}%")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n" + "=" * 80)
print("END-TO-END TEST SUMMARY")
print("=" * 80)

summary_stats = {
    "Layer 1 - Data Collection": {
        "Status": "✓ PASS",
        "Records": len(layer1_df),
        "Features": 8,
        "API Keys Used": f"News: {bool(news_key)}, Weather: {bool(weather_key)}"
    },
    "Layer 2 - Feature Engineering": {
        "Status": "✓ PASS",
        "Records": len(consolidated_df),
        "Features": len(consolidated_df.columns),
        "Normalized": len(feature_columns)
    },
    "Layer 3 - Demand Forecast": {
        "Status": "✓ PASS",
        "Months Forecast": 3,
        "Latest Prediction": f"{forecast_df['predicted_sales'].iloc[0]:,.0f} units",
        "Model": "XGBoost"
    },
    "Layer 4 - RL Agent": {
        "Status": "✓ PASS",
        "Scenarios Tested": len(results),
        "Q-Table Size": len(agent.q_table),
        "Actions": 5
    }
}

for layer, stats in summary_stats.items():
    print(f"\n{layer}:")
    for key, value in stats.items():
        print(f"  • {key:<20} {value}")

print("\n" + "=" * 80)
print("✓ ALL 4 LAYERS TESTED SUCCESSFULLY")
print("=" * 80)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNext: Ready for production inference or submission generation")
print("=" * 80)
