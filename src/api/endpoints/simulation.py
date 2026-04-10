"""
Scenario Simulation API Endpoint

Provides Digital Twin scenario simulation endpoints.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import asyncio
import logging
import copy

from ..models import ScenarioRequest

router = APIRouter()
logger = logging.getLogger(__name__)

def _simulate_scenario(scenario: ScenarioRequest) -> dict:
    """Run scenario simulation mirroring ScenarioSimulator logic."""
    # Base state — inventory sized for ~16–17 days of cover at baseline demand
    # (monthly demand / 30 = daily rate). Previously 25k vs 500k/mo implied ~1.5 days,
    # which forced emergency_restock for every scenario.
    base_state = {
        'predicted_demand': 500000,
        'current_inventory': 270000,
        'supplier_risk_score': 0.35,
        'disruption_signal': 0.3,
        'days_to_stockout': 16.2,
        'composite_risk_score': 0.3
    }

    mod = copy.deepcopy(base_state)

    # Apply demand spike
    mod['predicted_demand'] *= (1 + scenario.demand_spike)

    # Apply inventory adjustment
    if scenario.inventory_adjustment != 0:
        mod['current_inventory'] *= (1 + scenario.inventory_adjustment)

    # Apply supplier failure
    if scenario.supplier_failure:
        mod['supplier_risk_score'] = max(0.9, mod['supplier_risk_score'])

    # Apply transport delay
    if scenario.transport_delay:
        mod['disruption_signal'] = max(0.8, mod['disruption_signal'])

    # Apply weather disruption
    if scenario.weather_disruption:
        mod['disruption_signal'] = max(0.7, mod['disruption_signal'])
        mod['current_inventory'] *= 0.9

    # Recalculate days to stockout
    demand_per_day = mod['predicted_demand'] / 30.0
    mod['days_to_stockout'] = mod['current_inventory'] / max(demand_per_day, 1)

    # Composite risk
    mod['composite_risk_score'] = (
        mod['supplier_risk_score'] * 0.3 +
        mod['disruption_signal'] * 0.4 +
        0.3 * 0.3
    )

    # Stockout probability
    dts = mod['days_to_stockout']
    stockout_base = 0.8 if dts <= 3 else (0.5 if dts <= 7 else (0.2 if dts <= 14 else 0.05))
    stockout_prob = min(0.95, stockout_base + mod['supplier_risk_score'] * 0.3 + mod['disruption_signal'] * 0.2)
    delay_prob = min(0.9, mod['supplier_risk_score'] * 0.7 + mod['disruption_signal'] * 0.6)

    # Cost impact
    demand_change_pct = (mod['predicted_demand'] - base_state['predicted_demand']) / base_state['predicted_demand']
    base_cost = base_state['predicted_demand'] * 0.5
    cost_impact = demand_change_pct * base_cost + (stockout_prob * 1000 + delay_prob * 500) * (mod['predicted_demand'] / 100000)

    # Recommended action (safety layer logic from Layer 4)
    inv_ratio = mod['current_inventory'] / max(mod['predicted_demand'], 1)
    if dts <= 3 or inv_ratio < 0.02:
        action = 'emergency_restock'
        rationale = 'Critical inventory shortage. Emergency restocking required immediately.'
    elif mod['supplier_risk_score'] > 0.8:
        action = 'switch_supplier'
        rationale = 'Severe supplier risk detected. Switch to backup supplier to maintain supply continuity.'
    elif mod['disruption_signal'] > 0.8:
        action = 'reroute_shipment'
        rationale = 'High disruption detected. Activate alternative logistics routes.'
    elif dts < 7 or mod['composite_risk_score'] > 0.5:
        action = 'reorder_stock'
        rationale = 'Elevated risk conditions. Proactive stock reorder recommended.'
    else:
        action = 'do_nothing'
        rationale = 'Current conditions are within safe operating parameters. Continue monitoring.'

    return {
        'impact_analysis': {
            'cost_impact': cost_impact,
            'stockout_probability': stockout_prob,
            'delay_probability': delay_prob,
            'demand_change': demand_change_pct
        },
        'recommendations': {
            'action': action,
            'rationale': rationale
        },
        'risk_breakdown': {
            'supplier_risk': mod['supplier_risk_score'],
            'disruption_risk': mod['disruption_signal'],
            'composite_risk': mod['composite_risk_score'],
            'days_to_stockout': mod['days_to_stockout']
        },
        'scenario_summary': {
            'baseline_demand': base_state['predicted_demand'],
            'scenario_demand': mod['predicted_demand'],
            'baseline_inventory': base_state['current_inventory'],
            'scenario_inventory': mod['current_inventory']
        }
    }


@router.post("/simulate")
async def run_simulation(scenario: ScenarioRequest):
    """
    Run Digital Twin scenario simulation.
    
    Args:
        scenario: Scenario parameters including demand_spike, supplier_failure, etc.
    
    Returns:
        Complete scenario impact analysis with AI recommendations.
    """
    try:
        logger.info(f"Simulation endpoint called: demand_spike={scenario.demand_spike}, "
                    f"supplier_failure={scenario.supplier_failure}, "
                    f"transport_delay={scenario.transport_delay}, "
                    f"weather_disruption={scenario.weather_disruption}")

        results = await asyncio.to_thread(_simulate_scenario, scenario)

        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'success',
            **results
        }

    except Exception as e:
        logger.error(f"Error in simulation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@router.get("/simulate/templates")
async def get_scenario_templates():
    """Get predefined scenario templates."""
    return {
        'timestamp': datetime.now().isoformat(),
        'status': 'success',
        'templates': [
            {'name': 'demand_surge', 'label': 'Demand Surge +30%', 'demand_spike': 0.30, 'supplier_failure': False, 'transport_delay': False, 'weather_disruption': False},
            {'name': 'supplier_crisis', 'label': 'Supplier Crisis', 'demand_spike': 0.0, 'supplier_failure': True, 'transport_delay': False, 'weather_disruption': False},
            {'name': 'transport_disruption', 'label': 'Transport Disruption', 'demand_spike': 0.0, 'supplier_failure': False, 'transport_delay': True, 'weather_disruption': False},
            {'name': 'weather_crisis', 'label': 'Weather Crisis', 'demand_spike': 0.0, 'supplier_failure': False, 'transport_delay': False, 'weather_disruption': True},
            {'name': 'perfect_storm', 'label': 'Perfect Storm', 'demand_spike': 0.25, 'supplier_failure': True, 'transport_delay': True, 'weather_disruption': False},
        ]
    }
