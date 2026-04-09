# Supply Chain Intelligence System - Complete Architecture

## Project Overview

This is a **4-layer supply chain intelligence and decision system** that combines:
1. **Raw data collection** from multiple sources (sales, inventory, weather, news)
2. **Feature engineering** to create consolidated ML-ready features
3. **Demand forecasting** using trained XGBoost models
4. **RL-based decision making** for supply chain optimization

---

## Layer Architecture

### Layer 1: Data Collection (`layer1.py`)
**Purpose**: Gather raw data from multiple sources and compute initial disruption signals

**Key Functions**:
- `load_historical_sales()` - Aggregate Kaggle store sales to monthly granularity
- `simulate_inventory_data()` - Create supplier-level inventory states
- `fetch_weatherapi_history()` - Pull historical weather from WeatherAPI.com
- `score_weather_observation()` - Compute weather disruption scores (0-1)
- `fetch_newsdata_io()` - Pull India-focused news from NewsData.io
- `score_news_disruption()` - Extract disruption signals from headlines
- `build_layer1_dataset()` - Master function that orchestrates all collection

**Inputs**:
- `store_sale.csv` - Historical daily/monthly sales
- WeatherAPI key (optional, falls back to simulation)
- NewsData.io key (optional, falls back to static headlines)

**Outputs**:
```python
{
    'monthly_sales': DataFrame(59, 3),       # date, sales, sales_diff
    'supervised_sales': DataFrame(47, 13),   # lag features (month_1 to month_12)
    'inventory_data': DataFrame(59, 4),      # per-supplier inventory
    'weather_signals': DataFrame(232, 4),    # per-city weather
    'news_headlines': list,                  # raw headlines
    'layer1_df': DataFrame(59, 8)           # consolidated output
}
```

**Layer 1 DF Columns**:
- `date` - Month timestamp
- `sales` - Total monthly sales
- `sales_diff` - Month-over-month change
- `current_inventory` - Avg supplier inventory
- `days_to_stockout` - Estimated depletion time
- `weather_disruption_score` - Regional weather risk (0-1)
- `disruption_signal` - News-based disruption (0-1)
- `supplier_risk_score` - Derived from inventory metrics

---

### Layer 2: Feature Engineering (`layer2.py`)
**Purpose**: Transform Layer 1 outputs into standardized, aggregated features for ML models

**Key Functions**:
- `normalize_features()` - Scale features to [0,1] or standard normal
- `handle_outliers()` - Clip extreme values using IQR or Z-score
- `aggregate_weather_by_window()` - Consolidate per-city weather into regional signal
- `aggregate_suppliers()` - Combine supplier inventory (mean/min/risk-weighted)
- `compute_rolling_statistics()` - Calculate 3-month trend metrics
- `create_composite_risk_signal()` - Weighted combination of all risks
- `build_layer2_dataset()` - Master orchestrator function

**Inputs**:
- Layer 1 bundle (output from Layer 1)

**Outputs**:
```python
{
    'consolidated_df': DataFrame(59, 23),    # all features merged
    'inventory_agg': DataFrame(59, 3),       # aggregated inventory
    'weather_agg': DataFrame(58, 2),         # aggregated weather
    'risk_df': DataFrame(59, 2),             # composite risk scores
    'scaler': MinMaxScaler,                  # fitted scaler object
    'feature_columns': list                  # normalized column names
}
```

**Layer 2 Consolidated DF Columns** (23 total):
- **Base metrics** (7):
  - date, sales, sales_diff, current_inventory, days_to_stockout, supplier_risk_score, disruption_signal
- **Aggregated metrics** (3):
  - weighted_inventory, weighted_days_to_stockout, regional_weather_disruption
- **Composite signal** (1):
  - composite_risk_score (weighted: supplier 30%, disruption 40%, weather 30%)
- **Rolling statistics** (12):
  - sales/inventory/risk: rolling_mean, rolling_std, rolling_min, rolling_max (each)

**Feature Normalization**:
All features normalized to [0, 1] using fitted MinMaxScaler for consistent scaling across dimensions.

---

### Layer 3: Integration & Forecasting (`layer3.ipynb`)
**Purpose**: Combine Layer 2 features with demand predictions to create RL-ready states

**Dependencies**:
- `layer1.py` - For data collection
- `layer2.py` - For feature engineering
- `inference.py` - For demand forecasting
- `demand_forecast_model.pkl` - Trained XGBoost model
- `demand_scaler.pkl` - Fitted feature scaler

**Pipeline Steps**:
1. Build Layer 1 dataset (raw data)
2. Build Layer 2 features (engineered)
3. Load trained demand forecast model
4. Generate 3-month demand forecast
5. Merge forecast with Layer 2 features
6. Create RL agent state representations

**Outputs**:
```python
{
    'layer3_state': {
        'date': timestamp,
        'predicted_demand': float (units),
        'current_inventory': float,
        'supplier_risk_score': float (0-1),
        'disruption_signal': float (0-1),
        'days_to_stockout': float,
        'composite_risk_score': float (0-1),
        'weighted_inventory': float,
        'sales_volatility': float
    }
}
```

---

### Layer 4: RL Decision Engine (`layer4.ipynb`)
**Purpose**: Use Layer 3 state to make optimal supply chain decisions via Q-learning

**Agent Class**: `SupplyChainAgent`
- **Algorithm**: Tabular Q-Learning
- **State space**: Discretized (demand bucket, inventory ratio, risk bucket, disruption bucket, days_to_stockout)
- **Action space** (5 actions):
  - `0: do_nothing` - Passive monitoring
  - `1: reorder_stock` - Place standard order
  - `2: switch_supplier` - Change to backup supplier
  - `3: reroute_shipment` - Use alternate logistics route
  - `4: emergency_restock` - Urgent expensive restock

**Reward Function**:
- **Positive rewards**:
  - +80 if stockout avoided
  - +40 if delivery delay avoided
  - Variable reward for cost savings
- **Negative rewards**:
  - -300 for stockout (critical)
  - -50 to -200 for unnecessary actions
  - Variable penalty for cost increases

**Safety Layer**:
Rule-based overrides that take precedence when state is critical:
- If inventory < 2% of demand OR days_to_stockout ≤ 3 → emergency_restock
- If supplier_risk_score > 0.8 → switch_supplier
- If disruption_signal > 0.8 → reroute_shipment

**Outputs**:
```python
{
    'action': str,              # Recommended action name
    'confidence': float (0-100),# Confidence score from Q-table
    'explanation': str,         # Human-readable decision reason
    'state': dict              # Input state that was evaluated
}
```

---

## Data Flow Diagram

```
store_sale.csv
│
├──────────────────────────────────────┐
│                                      │
▼                                      ▼
Modeltrain.ipynb              layer1.py (Layer 1)
│                                │
├─ Train XGBoost               ├─ load_historical_sales()
├─ Save model                  ├─ simulate_inventory_data()
│                              ├─ fetch_weatherapi_history()
▼                              ├─ fetch_newsdata_io()
demand_forecast_model.pkl      │
demand_scaler.pkl              ▼
│                              layer1_bundle
├─────────────────────────────────────┤
│                                      │
▼                                      ▼
inference.py (Layer 3)      layer2.py (Layer 2)
│                                │
├─ Load models               ├─ normalize_features()
├─ Predict demand            ├─ aggregate_weather()
│                            ├─ aggregate_suppliers()
▼                            ├─ compute_rolling_stats()
forecast_df                  │
│                            ▼
│                            layer2_bundle
└────────────────┬───────────────────┘
                 │
                 ▼
           layer3.ipynb (Layer 3)
                 │
                 ├─ Merge forecast + Layer 2
                 ├─ Create RL state
                 │
                 ▼
           layer3_state
                 │
                 ▼
           layer4.ipynb (Layer 4)
                 │
                 ├─ Load RL agent
                 ├─ Predict action
                 │
                 ▼
           Decision + Confidence + Explanation
```

---

## Feature Summary

### Layer 1 Raw Features (8)
- Sales metrics: `sales`, `sales_diff`
- Inventory: `current_inventory`, `days_to_stockout`
- Risk: `supplier_risk_score`, `disruption_signal`
- Weather: `weather_disruption_score`

### Layer 2 Engineered Features (23)
- All Layer 1 features (8)
- Aggregated metrics (3): weighted inventory, weather disruption
- Composite risk (1)
- Rolling statistics (12): 3-month trends

### Layer 3 RL State Features (9)
- Predicted demand
- Current inventory & days to stockout
- Risk scores (supplier, disruption, composite)
- Weighted inventory
- Sales volatility

---

## Configuration & API Keys

Create `.env` file in project root:
```
WEATHERAPI_KEY=your_weatherapi_key
NEWSDATA_API_KEY=your_newsdata_api_key
```

**Optional**: Layer 1 has fallback simulation if keys are not provided.

---

## Usage Example

### Run Full Pipeline (Layer 1 → Layer 4)

```python
# Layer 1: Collect data
from layer1 import build_layer1_dataset
layer1_bundle = build_layer1_dataset('store_sale.csv')

# Layer 2: Engineer features
from layer2 import build_layer2_dataset
layer2_bundle = build_layer2_dataset(layer1_bundle)

# Layer 3: Integrate with forecast
# (Open and run layer3.ipynb)

# Layer 4: Get decision
from layer4 import SupplyChainAgent, get_recommendation
agent = SupplyChainAgent()
agent.load('supply_chain_agent.pkl')
decision = get_recommendation(layer3_state, agent)
print(f"Action: {decision['action']}")
print(f"Confidence: {decision['confidence']:.2f}")
```

---

## File Inventory

| File | Type | Purpose |
|------|------|---------|
| `layer1.py` | Python | Data collection module |
| `layer2.py` | Python | Feature engineering module |
| `layer3.ipynb` | Notebook | Integration & forecasting |
| `layer4.ipynb` | Notebook | RL decision engine |
| `Modeltrain.ipynb` | Notebook | Demand model training |
| `inference.py` | Python | Demand prediction |
| `test_layer1.py` | Python | Layer 1 tests |
| `test_layer2.py` | Python | Layer 2 tests |
| `.env` | Config | API keys |

**Trained Models** (pkl files):
- `demand_forecast_model.pkl` - XGBoost demand predictor
- `demand_scaler.pkl` - MinMaxScaler for features
- `supply_chain_agent.pkl` - Trained Q-learning agent

**Data** (csv files):
- `store_sale.csv` - Historical sales
- `test.csv` - Test dataset
- `sample_submission.csv` - Submission format

---

## Next Steps

1. ✓ Layer 1 implemented and tested
2. ✓ Layer 2 implemented and tested
3. ✓ Layer 3 integration notebook created
4. ⏳ **TODO**: Run full end-to-end pipeline
5. ⏳ **TODO**: Generate final submissions
6. ⏳ **TODO**: Fine-tune RL agent weights
7. ⏳ **TODO**: Add real API keys for live weather/news

---

**Last Updated**: April 9, 2026
**Status**: Layers 1-3 Complete, Testing in Progress
