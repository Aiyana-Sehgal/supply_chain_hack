"""
Scenario Analysis Package

Provides what-if scenario analysis and digital twin capabilities for the supply chain intelligence system.
"""

from .scenario_simulator import (
    ScenarioInput,
    ScenarioSimulator,
    SCENARIO_TEMPLATES,
    run_predefined_scenario,
)


def create_digital_twin(*args, **kwargs):
    from .digital_twin import create_digital_twin as _create_digital_twin
    return _create_digital_twin(*args, **kwargs)


def DigitalTwin(*args, **kwargs):
    from .digital_twin import DigitalTwin as _DigitalTwin
    return _DigitalTwin(*args, **kwargs)


def DigitalTwinConfig(*args, **kwargs):
    from .digital_twin import DigitalTwinConfig as _DigitalTwinConfig
    return _DigitalTwinConfig(*args, **kwargs)

__all__ = [
    'ScenarioInput',
    'ScenarioSimulator',
    'SCENARIO_TEMPLATES',
    'run_predefined_scenario',
    'DigitalTwin',
    'DigitalTwinConfig',
    'create_digital_twin'
]
