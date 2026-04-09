"""
Test the new project structure
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

print("Testing new project structure...")

try:
    # Test core imports
    from src.core import build_layer1_dataset, build_layer2_dataset, build_layer3_dataset
    print("Core imports: OK")
except ImportError as e:
    print(f"Core imports failed: {e}")

try:
    # Test XAI imports
    from src.xai import create_enhanced_xai
    print("XAI imports: OK")
except ImportError as e:
    print(f"XAI imports failed: {e}")

try:
    # Test scenario imports
    from src.scenarios import create_digital_twin
    print("Scenario imports: OK")
except ImportError as e:
    print(f"Scenario imports failed: {e}")

try:
    # Test utils imports
    from src.utils import quick_analysis
    print("Utils imports: OK")
except ImportError as e:
    print(f"Utils imports failed: {e}")

print("Structure test completed!")
