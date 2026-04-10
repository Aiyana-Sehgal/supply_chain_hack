from api.endpoints.simulation import _simulate_scenario
from api.models import ScenarioRequest

scenarios = [
    ScenarioRequest(demand_spike=0.0, supplier_failure=False, transport_delay=False, weather_disruption=False, inventory_adjustment=0.0),
    ScenarioRequest(demand_spike=0.20, supplier_failure=False, transport_delay=False, weather_disruption=False, inventory_adjustment=0.0),
    ScenarioRequest(demand_spike=0.50, supplier_failure=False, transport_delay=False, weather_disruption=False, inventory_adjustment=-0.5),
    ScenarioRequest(demand_spike=0.0, supplier_failure=True, transport_delay=False, weather_disruption=False, inventory_adjustment=0.0),
]

for s in scenarios:
    result = _simulate_scenario(s)
    print('SCENARIO', s)
    print('  ACTION', result['recommendations']['action'])
    print('  RATIONALE', result['recommendations']['rationale'])
    print('  DAYS', result['risk_breakdown']['days_to_stockout'])
    print('  STOCKOUT', result['risk_breakdown']['stockout_risk'])
    print('  DELAY', result['risk_breakdown']['delay_risk'])
    print('  COST', result['impact_analysis']['cost_impact'])
    print('---')
