"""
Utility Package

Contains utility modules for the supply chain intelligence system including
pipeline orchestration and inference capabilities.
"""

from .inference import predict_next_month_demand, predict_n_months

__all__ = [
    'predict_next_month_demand',
    'predict_n_months',
    'SupplyChainPipeline',
    'quick_analysis',
    'quick_scenario'
]


def __getattr__(name):
    if name in {'SupplyChainPipeline', 'quick_analysis', 'quick_scenario'}:
        from .pipeline import SupplyChainPipeline, quick_analysis, quick_scenario
        exports = {
            'SupplyChainPipeline': SupplyChainPipeline,
            'quick_analysis': quick_analysis,
            'quick_scenario': quick_scenario,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
