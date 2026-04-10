import sys
import os
sys.path.insert(0, os.path.abspath('src'))
from api.endpoints.simulation import _simulate_scenario
from api.models import ScenarioRequest

cases = [
    ('baseline', dict(demand_spike=0.0, supplier_failure=False, transport_delay=False, weather_disruption=False, inventory_adjustment=0.0)),
    ('medium', dict(demand_spike=0.20, supplier_failure=False, transport_delay=False, weather_disruption=False, inventory_adjustment=0.0)),
    ('high', dict(demand_spike=0.50, supplier_failure=False, transport_delay=False, weather_disruption=False, inventory_adjustment=-0.5)),
]

for name, kwargs in cases:
    res = _simulate_scenario(ScenarioRequest(**kwargs))
    print(name, res['recommendations']['action'], res['risk_breakdown']['days_to_stockout'], res['risk_breakdown']['stockout_risk'])
