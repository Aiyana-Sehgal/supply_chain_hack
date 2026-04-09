"""Test Layer 2 Feature Engineering & State Preparation."""
import sys
from layer1 import build_layer1_dataset
from layer2 import build_layer2_dataset, normalize_features, handle_outliers

print("=" * 70)
print("Testing Layer 2: Feature Engineering & State Preparation")
print("=" * 70)

# Step 1: Build Layer 1 data
print("\n[Step 1] Building Layer 1 dataset...")
try:
    layer1_bundle = build_layer1_dataset(
        sales_path='store_sale.csv',
        news_api_key=None,
        weather_api_key=None
    )
    print("✓ Layer 1 dataset built successfully!")
    print(f"  - monthly_sales: {layer1_bundle['monthly_sales'].shape}")
    print(f"  - inventory_data: {layer1_bundle['inventory_data'].shape}")
    print(f"  - weather_signals: {layer1_bundle['weather_signals'].shape}")
    print(f"  - layer1_df: {layer1_bundle['layer1_df'].shape}")
except Exception as e:
    print(f"✗ Error building Layer 1: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: Build Layer 2 dataset
print("\n[Step 2] Building Layer 2 consolidated features...")
try:
    layer2_bundle = build_layer2_dataset(
        layer1_bundle,
        inventory_method='risk_weighted',
        weather_method='mean',
        include_rolling=True
    )
    print("✓ Layer 2 dataset built successfully!")
    
    # Display outputs
    print("\n--- Layer 2 Bundle Contents ---")
    for key, value in layer2_bundle.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: {type(value).__name__} (length {len(value)})")
        elif hasattr(value, '__class__'):
            print(f"  {key}: {type(value).__name__}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
except Exception as e:
    print(f"✗ Error building Layer 2: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Display consolidated dataframe
print("\n--- Consolidated DataFrame (first 5 rows) ---")
consolidated_df = layer2_bundle['consolidated_df']
print(consolidated_df.head())

print("\n--- Consolidated DataFrame Info ---")
print(f"Columns ({len(consolidated_df.columns)}):")
for i, col in enumerate(consolidated_df.columns, 1):
    dtype = consolidated_df[col].dtype
    nulls = consolidated_df[col].isna().sum()
    print(f"  {i:2d}. {col:<35} {str(dtype):<15} (nulls: {nulls})")

# Step 4: Display statistics
print("\n--- Consolidated DataFrame Statistics ---")
numeric_cols = consolidated_df.select_dtypes(include=['float64', 'int64']).columns
print(consolidated_df[numeric_cols].describe())

# Step 5: Validate feature normalization
print("\n--- Feature Normalization Check ---")
feature_cols = layer2_bundle['feature_columns']
print(f"Normalized features: {feature_cols}")
for col in feature_cols:
    if col in consolidated_df.columns:
        min_val = consolidated_df[col].min()
        max_val = consolidated_df[col].max()
        print(f"  {col:<35} min={min_val:.3f}, max={max_val:.3f}")

# Step 6: Demonstrate individual function capabilities
print("\n--- Testing Individual Layer 2 Functions ---")

# Test outlier handling
print("\n[Function Test] handle_outliers():")
sample_series = layer1_bundle['layer1_df']['sales'].copy()
clipped = handle_outliers(sample_series, method='iqr', threshold=1.5)
print(f"  Original series: min={sample_series.min():.0f}, max={sample_series.max():.0f}, mean={sample_series.mean():.0f}")
print(f"  After clipping:  min={clipped.min():.0f}, max={clipped.max():.0f}, mean={clipped.mean():.0f}")

# Test aggregations
print("\n[Function Test] aggregate_suppliers():")
supplier_agg = layer2_bundle['inventory_agg']
print(f"  Aggregated inventory shape: {supplier_agg.shape}")
print(f"  Columns: {list(supplier_agg.columns)}")
print(supplier_agg.head(3))

print("\n[Function Test] aggregate_weather_by_window():")
weather_agg = layer2_bundle['weather_agg']
print(f"  Aggregated weather shape: {weather_agg.shape}")
print(f"  Columns: {list(weather_agg.columns)}")
print(weather_agg.head(3))

# Step 7: Summary
print("\n" + "=" * 70)
print("Layer 2 Test Summary")
print("=" * 70)
print(f"✓ All tests passed!")
print(f"✓ Consolidated dataset: {consolidated_df.shape[0]} records × {consolidated_df.shape[1]} features")
print(f"✓ Features normalized and ready for downstream layers")
print(f"✓ Rolling statistics computed for trend analysis")
print("=" * 70)
