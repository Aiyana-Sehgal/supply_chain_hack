"""
Supply Chain Intelligence System

A comprehensive AI-powered supply chain intelligence system that predicts demand,
detects disruptions, scores risk, learns optimal decisions over time using
reinforcement learning, and lets users simulate what-if scenarios.

Features:
- Layer 1: Data Collection (NewsData.io, weather, inventory)
- Layer 2: Feature Engineering & Normalization
- Layer 3: Integration & XGBoost Demand Forecasting
- Layer 4: Q-Learning RL Decision Engine
- Layer 5: Digital Twin & Enhanced Explainable AI with Ollama Llama 3.1
"""

# Core layers
from .core import (
    build_layer1_dataset,
    build_layer2_dataset,
    build_layer3_dataset,
    SupplyChainAgent,
    get_recommendation
)

# Enhanced XAI
from .xai import (
    EnhancedXAI,
    create_enhanced_xai,
    OllamaClient,
    CacheManager
)

# Scenario Analysis
# from .scenarios import (
#     DigitalTwin,
#     create_digital_twin,
#     ScenarioSimulator,
#     ScenarioInput
# )  # Commented out to avoid circular import

# Utilities
# from .utils import (
#     SupplyChainPipeline,
#     quick_analysis,
#     predict_next_month_demand
# )  # Commented out to avoid circular import

__version__ = "1.0.0"
__author__ = "Supply Chain Intelligence Team"

__all__ = [
    # Core layers
    'build_layer1_dataset',
    'build_layer2_dataset',
    'build_layer3_dataset',
    'SupplyChainAgent',
    'get_recommendation',
    
    # Enhanced XAI
    'EnhancedXAI',
    'create_enhanced_xai',
    'OllamaClient',
    'CacheManager',
    
    # Scenario Analysis
    'DigitalTwin',
    'create_digital_twin',
    'ScenarioSimulator',
    'ScenarioInput',
    
    # Utilities
    'SupplyChainPipeline',
    'quick_analysis',
    'predict_next_month_demand'
]
