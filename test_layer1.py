"""Test Layer 1 data collection module."""
import os
import sys
from layer1 import build_layer1_dataset

print("=" * 60)
print("Testing Layer 1 Data Collection")
print("=" * 60)

# Test 1: Build Layer 1 dataset without API keys (using fallback data)
print("\n[Test 1] Building Layer 1 dataset with fallback (no API keys)...")
try:
    data_bundle = build_layer1_dataset(
        sales_path='store_sale.csv',
        news_api_key=None,  # Using fallback
        weather_api_key=None  # Using fallback
    )
    print("✓ Layer 1 dataset built successfully!")
    
    # Display summary
    print("\n--- Data Bundle Contents ---")
    for key, value in data_bundle.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")
    
    # Display Layer 1 DataFrame
    print("\n--- Layer 1 DataFrame (first 5 rows) ---")
    print(data_bundle['layer1_df'].head())
    
    print("\n--- Layer 1 DataFrame Info ---")
    print(f"Columns: {list(data_bundle['layer1_df'].columns)}")
    print(f"Shape: {data_bundle['layer1_df'].shape}")
    print(f"Data types:\n{data_bundle['layer1_df'].dtypes}")
    
    # Display some stats
    print("\n--- Layer 1 DataFrame Statistics ---")
    print(data_bundle['layer1_df'].describe())
    
    print("\n✓ All tests passed!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
