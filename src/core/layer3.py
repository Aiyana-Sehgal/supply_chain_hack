"""
Layer 3: Integration & Demand Forecasting Pipeline

Combines Layer 1 (Data Collection), Layer 2 (Feature Engineering), and demand forecasting 
to prepare rich state representations for the RL agent (Layer 4).

Pipeline Overview:
1. Layer 1: Collect historical data + disruption signals
2. Layer 2: Engineer features + normalize + aggregate  
3. Layer 3: Integrate with demand forecast (this module)
4. Layer 4: RL agent makes supply chain decisions based on Layer 3 state
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

# Import custom layers
sys.path.append(os.path.dirname(__file__))
from layer1 import build_layer1_dataset
from layer2 import build_layer2_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_JSON_PATH = MODELS_DIR / 'demand_forecast_model.json'
MODEL_PKL_PATH = MODELS_DIR / 'demand_forecast_model.pkl'
SCALER_PKL_PATH = MODELS_DIR / 'demand_scaler.pkl'


def build_lag_features(series, n_lags=12):
    """Build lag features for time series forecasting"""
    df = pd.DataFrame(series, columns=['sales_diff'])
    for i in range(1, n_lags + 1):
        df[f'month_{i}'] = df['sales_diff'].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df


def rebuild_demand_scaler(sales_path='data/store_sale.csv', n_lags=12):
    """Rebuild the MinMaxScaler from the training data when pickle compatibility fails."""
    store_sales = pd.read_csv(PROJECT_ROOT / sales_path)
    store_sales = store_sales.drop(['store', 'item'], axis=1)
    store_sales['date'] = pd.to_datetime(store_sales['date'])
    store_sales['date'] = store_sales['date'].dt.to_period('M')
    monthly_sales = store_sales.groupby('date').sum().reset_index()
    monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
    monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
    monthly_sales = monthly_sales.dropna().reset_index(drop=True)

    supervised = build_lag_features(monthly_sales['sales_diff'], n_lags=n_lags)
    feature_columns = ['sales_diff'] + [f'month_{i}' for i in range(1, n_lags + 1)]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(supervised[feature_columns])
    return scaler


def load_demand_models():
    """Load trained demand forecast model and scaler"""
    try:
        if MODEL_JSON_PATH.exists():
            model_tuned = xgb.Booster()
            model_tuned.load_model(str(MODEL_JSON_PATH))
        else:
            with open(MODEL_PKL_PATH, 'rb') as f:
                model_tuned = pickle.load(f)
        
        try:
            with open(SCALER_PKL_PATH, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as scaler_error:
            print(f"Warning: Could not load serialized scaler, rebuilding from sales history: {scaler_error}")
            scaler = rebuild_demand_scaler()
        
        print("Demand forecast model loaded successfully")
        return model_tuned, scaler
    
    except FileNotFoundError:
        print("Error: Model files not found. Please run Modeltrain.ipynb first.")
        raise
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def generate_demand_forecast(model_tuned, scaler, n_months=3):
    """Generate demand forecast for next N months"""
    try:
        # Load and prepare sales data
        store_sales = pd.read_csv(DATA_DIR / 'store_sale.csv')
        store_sales = store_sales.drop(['store', 'item'], axis=1)
        store_sales['date'] = pd.to_datetime(store_sales['date'])
        store_sales['date'] = store_sales['date'].dt.to_period('M')
        monthly_sales = store_sales.groupby('date').sum().reset_index()
        monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
        monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
        monthly_sales = monthly_sales.dropna().reset_index(drop=True)
        
        # Forecast N months
        temp_monthly = monthly_sales.copy()
        forecasts = []
        
        for i in range(n_months):
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
        return forecast_df, monthly_sales
    
    except Exception as e:
        print(f"Error generating forecast: {e}")
        raise


def merge_forecast_with_features(consolidated_df, forecast_df):
    """Merge Layer 2 features with demand forecast"""
    try:
        # Merge forecast with consolidated Layer 2 data
        layer3_state = consolidated_df.copy()
        
        # Add forecast (match by month)
        forecast_df_merged = forecast_df[['date', 'predicted_sales']].copy()
        layer3_state = layer3_state.merge(forecast_df_merged, on='date', how='left', suffixes=('', '_forecast'))
        
        # Fill missing forecast values with current sales estimate
        layer3_state['predicted_sales'] = layer3_state['predicted_sales'].fillna(layer3_state['sales'])
        
        return layer3_state
    
    except Exception as e:
        print(f"Error merging data: {e}")
        raise


def create_rl_state(layer3_state):
    """Create ready-to-use state representations for RL agent"""
    try:
        # Extract latest state for RL agent
        latest_state = layer3_state.iloc[-1]
        
        rl_state = {
            "date": pd.Timestamp(latest_state['date']),
            "predicted_demand": float(latest_state['predicted_sales']),
            "current_inventory": float(latest_state['current_inventory']),
            "supplier_risk_score": float(latest_state['supplier_risk_score']),
            "disruption_signal": float(latest_state['disruption_signal']),
            "days_to_stockout": float(latest_state['days_to_stockout']),
            "composite_risk_score": float(latest_state['composite_risk_score']),
            "weighted_inventory": float(latest_state['weighted_inventory']),
            "sales_volatility": float(latest_state.get('sales_rolling_std', 0.0))
        }
        
        return rl_state
    
    except Exception as e:
        print(f"Error creating RL states: {e}")
        raise


def build_layer3_dataset(sales_path='data/store_sale.csv', 
                         news_api_key=None, 
                         weather_api_key=None,
                         n_forecast_months=3,
                         verbose=True):
    """
    Master function to build complete Layer 3 dataset
    
    Parameters:
    -----------
    sales_path : str
        Path to sales CSV file
    news_api_key : str, optional
        NewsData.io API key
    weather_api_key : str, optional
        WeatherAPI key
    n_forecast_months : int
        Number of months to forecast
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict
        Layer 3 bundle with state, forecast, and metadata
    """
    
    if verbose:
        print("=" * 70)
        print("Layer 3: Integration & Demand Forecasting")
        print("=" * 70)
    
    # Step 1: Build Layer 1 Dataset
    if verbose:
        print("\n[Step 1] Building Layer 1 dataset...")
    
    try:
        layer1_bundle = build_layer1_dataset(
            sales_path=sales_path,
            news_api_key=news_api_key or os.getenv('NEWSDATA_API_KEY'),
            weather_api_key=weather_api_key or os.getenv('WEATHERAPI_KEY')
        )
        if verbose:
            print("Layer 1 complete")
            print(f"  - Monthly sales: {layer1_bundle['monthly_sales'].shape}")
            print(f"  - Inventory data: {layer1_bundle['inventory_data'].shape}")
            print(f"  - Weather signals: {layer1_bundle['weather_signals'].shape}")
            print(f"  - Layer 1 consolidated: {layer1_bundle['layer1_df'].shape}")
    except Exception as e:
        print(f"Error in Layer 1: {e}")
        raise
    
    # Step 2: Build Layer 2 Dataset
    if verbose:
        print("\n[Step 2] Building Layer 2 features...")
    
    try:
        layer2_bundle = build_layer2_dataset(
            layer1_bundle,
            inventory_method='risk_weighted',
            weather_method='mean',
            include_rolling=True
        )
        if verbose:
            print("Layer 2 complete")
            
            consolidated_df = layer2_bundle['consolidated_df']
            print(f"  - Consolidated features: {consolidated_df.shape}")
            print(f"  - Features: {len(layer2_bundle['feature_columns'])} normalized columns")
            
            # Display key features
            print(f"\n  Key metrics in consolidated dataset:")
            print(f"    - Sales range: {consolidated_df['sales'].min():.0f} - {consolidated_df['sales'].max():.0f}")
            print(f"    - Inventory range: {consolidated_df['current_inventory'].min():.2f} - {consolidated_df['current_inventory'].max():.2f}")
            print(f"    - Composite risk: {consolidated_df['composite_risk_score'].min():.3f} - {consolidated_df['composite_risk_score'].max():.3f}")
    except Exception as e:
        print(f"Error in Layer 2: {e}")
        raise
    
    # Step 3: Load Demand Forecast Model
    if verbose:
        print("\n[Step 3] Loading demand forecast model...")
    
    try:
        model_tuned, scaler = load_demand_models()
        if verbose:
            print(f"  - Model type: {type(model_tuned).__name__}")
            print(f"  - Scaler: {type(scaler).__name__}")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    
    # Step 4: Generate Demand Forecast
    if verbose:
        print(f"\n[Step 4] Generating demand forecast (next {n_forecast_months} months)...")
    
    try:
        forecast_df, monthly_sales = generate_demand_forecast(
            model_tuned, scaler, n_months=n_forecast_months
        )
        if verbose:
            print("Demand forecast generated")
            print(f"\n  Forecast:")
            print(forecast_df.to_string(index=False))
    except Exception as e:
        print(f"Error generating forecast: {e}")
        raise
    
    # Step 5: Merge Layer 2 Features with Forecast
    if verbose:
        print("\n[Step 5] Merging Layer 2 features with demand forecast...")
    
    try:
        consolidated_df = layer2_bundle['consolidated_df']
        layer3_state = merge_forecast_with_features(consolidated_df, forecast_df)
        if verbose:
            print("Layer 2 and forecast merged")
            print(f"  - Layer 3 state shape: {layer3_state.shape}")
            print(f"  - Columns: {len(layer3_state.columns)}")
    except Exception as e:
        print(f"Error merging data: {e}")
        raise
    
    # Step 6: Create RL State Representations
    if verbose:
        print("\n[Step 6] Creating RL agent state representations...")
    
    try:
        rl_state = create_rl_state(layer3_state)
        if verbose:
            print("RL state representations created")
            print(f"\n  Latest state (for RL agent):")
            for key, value in rl_state.items():
                if isinstance(value, float):
                    print(f"    {key:<30} {value:.4f}")
                else:
                    print(f"    {key:<30} {value}")
    except Exception as e:
        print(f"Error creating RL states: {e}")
        raise
    
    # Step 7: Create Output Bundle
    layer3_bundle = {
        'layer3_state': rl_state,
        'layer3_dataframe': layer3_state,
        'forecast_df': forecast_df,
        'layer2_bundle': layer2_bundle,
        'layer1_bundle': layer1_bundle,
        'model_info': {
            'model_type': type(model_tuned).__name__,
            'scaler_type': type(scaler).__name__,
            'forecast_months': n_forecast_months
        },
        'timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        print(f"\n[Step 7] Layer 3 Output Summary")
        print("=" * 70)
        
        print(f"\nPipeline Summary:")
        print(f"  Layer 1 (Data Col.): {layer1_bundle['layer1_df'].shape[0]} months × {layer1_bundle['layer1_df'].shape[1]} features")
        print(f"  Layer 2 (Features):  {consolidated_df.shape[0]} records × {consolidated_df.shape[1]} features")
        print(f"  Layer 3 (Forecast):  {layer3_state.shape[0]} records × {layer3_state.shape[1]} features")
        print(f"\nReady for Layer 4 (RL Agent):")
        print(f"  State keys: {list(rl_state.keys())}")
        print(f"  Date: {rl_state['date']}")
        print(f"  Predicted demand: {rl_state['predicted_demand']:.0f} units")
        print(f"  Risk score: {rl_state['composite_risk_score']:.3f}")
        print(f"  Days to stockout: {rl_state['days_to_stockout']:.1f}")
        
        print("\n" + "=" * 70)
        print("Layer 3 Complete! Ready to pass to Layer 4 (RL Decision Engine)")
        print("=" * 70)
    
    return layer3_bundle


def export_state_for_layer4(rl_state, filename='layer3_state.json'):
    """Export RL state for Layer 4 consumption"""
    import json
    
    try:
        state_export = {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in rl_state.items()}
        with open(filename, 'w') as f:
            json.dump(state_export, f, indent=2)
        print(f"\nState exported to {filename}")
        return True
    except Exception as e:
        print(f"Error exporting state: {e}")
        return False


# Convenience function for quick state generation
def get_current_state(sales_path='data/store_sale.csv', verbose=True):
    """Quick function to get current RL state without full bundle"""
    bundle = build_layer3_dataset(sales_path=sales_path, verbose=verbose)
    return bundle['layer3_state']


# Example usage
if __name__ == "__main__":
    try:
        # Build complete Layer 3 dataset
        layer3_bundle = build_layer3_dataset(
            sales_path='data/store_sale.csv',
            news_api_key=os.getenv('NEWSDATA_API_KEY'),
            weather_api_key=os.getenv('WEATHERAPI_KEY'),
            n_forecast_months=3
        )
        
        # Export state for Layer 4
        export_state_for_layer4(layer3_bundle['layer3_state'])
        
        print("\nLayer 3 pipeline completed successfully!")
        
    except Exception as e:
        print(f"Error in Layer 3 pipeline: {e}")
        import traceback
        traceback.print_exc()
