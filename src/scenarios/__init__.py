"""
Scenario Analysis Package

Provides what-if scenario analysis and digital twin capabilities for the supply chain intelligence system.
"""

# from .scenario_simulator import ScenarioInput, SCENARIO_TEMPLATES, run_predefined_scenario  # Commented out to avoid circular import
# from .digital_twin import DigitalTwin, DigitalTwinConfig, create_digital_twin  # Commented out to avoid circular import

__all__ = [
    'ScenarioInput',
    'SCENARIO_TEMPLATES',
    'run_predefined_scenario',
    'DigitalTwin',
    'DigitalTwinConfig',
    'create_digital_twin'
]
