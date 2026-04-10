"""
Scenario Simulation API Endpoint

Causal pipeline (strict order):
  1. Demand adjustment
  2. Inventory pressure (daily burn, days of cover)
  3. Stockout probability (inverse to days of cover; demand-only path)
  4. Delay probability (supplier failure, transport, disruption signal; w1>w2>w3)
  5. Cost impact (demand increase + delay exposure)
  6. Policy decision (thresholds on P_stockout, P_delay)
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import asyncio
import logging
import math

from ..models import ScenarioRequest

router = APIRouter()
logger = logging.getLogger(__name__)

# --- Baseline twin (units/month demand, units inventory) ---
D_BASE = 500_000.0
I_BASE = 270_000.0
BASE_DISRUPTION_SIGNAL = 0.30  # D_s ∈ [0, 1], logistics/supplier stress (continuous)

# Delay: independent branch using supplier, transport, and weather disruption weights
_W1, _W2, _W3 = 0.55, 0.45, 0.40

# Cost: α·max(0, D_adj - D_base) + β·P_delay (INR-style scale)
_ALPHA_DEMAND = 0.45  # per unit-month of demand increase
_BETA_DELAY = 220_000.0


def _simulate_scenario(scenario: ScenarioRequest) -> dict:
    """
    Deterministic causal simulation focused only on the 4 quantitative core metrics.
    Action recommendations are deferred to the separate Decision Engine agent.
    """
    # ----- 1. Demand adjustment -----
    demand_change_frac = scenario.demand_spike
    d_adj = D_BASE * (1.0 + demand_change_frac)

    # ----- 2. Delay probability (mathematical independent union) -----
    p_delay = BASE_DISRUPTION_SIGNAL
    if scenario.supplier_failure:
        p_delay = p_delay + 0.60 - (p_delay * 0.60)
    if scenario.transport_delay:
        p_delay = p_delay + 0.40 - (p_delay * 0.40)
    if scenario.weather_disruption:
        p_delay = p_delay + 0.20 - (p_delay * 0.20)

    # ----- 3. Stockout probability (exponential decay) -----
    effective_supply_rate = 1.0
    if scenario.supplier_failure:
        effective_supply_rate *= 0.2
    if scenario.transport_delay:
        effective_supply_rate *= 0.6
    if scenario.weather_disruption:
        effective_supply_rate *= 0.85

    inventory = I_BASE * (1.0 + scenario.inventory_adjustment)
    
    # Calculate effective days of cover
    d_daily = d_adj / 30.0
    effective_buffer = inventory * effective_supply_rate
    days_cover = effective_buffer / max(d_daily, 1e-9)
    
    # Exponential probability based on coverage
    # Assuming half-life of 15 days cover provides 36% risk
    p_stockout = math.exp(-days_cover / 15.0)
    p_stockout = min(1.0, max(0.0, p_stockout))

    # ----- 4. Cost impact -----
    delta_d_abs = d_adj - D_BASE
    cost_demand_term = _ALPHA_DEMAND * max(0.0, delta_d_abs)
    
    # Exponential cost scaling vs fixed buckets
    cost_delay_term = 400_000.0 * p_delay
    cost_stockout_term = 600_000.0 * p_stockout
    
    # Baseline expected cost
    baseline_p_stockout = math.exp(- (I_BASE) / (D_BASE/30) / 15.0)
    baseline_cost = (400_000.0 * BASE_DISRUPTION_SIGNAL) + (600_000.0 * baseline_p_stockout)
    
    cost_impact = cost_demand_term + cost_delay_term + cost_stockout_term - baseline_cost

    pipeline = {
        "step_1_demand": {"d_adj": d_adj},
        "step_2_delay": {"p_delay": p_delay},
        "step_3_stockout": {"days_cover": days_cover, "p_stockout": p_stockout},
        "step_4_cost": {"cost_impact": cost_impact},
    }

    return {
        "impact_analysis": {
            "cost_impact": cost_impact,
            "stockout_probability": p_stockout,
            "delay_probability": p_delay,
            "demand_change": demand_change_frac,
        },
        "scenario_summary": {
            "baseline_demand": D_BASE,
            "scenario_demand": d_adj,
            "baseline_inventory": I_BASE,
            "scenario_inventory": inventory,
        },
        "pipeline": pipeline,
    }


@router.post("/simulate")
async def run_simulation(scenario: ScenarioRequest):
    """
    Run Digital Twin scenario simulation (causal pipeline).
    """
    try:
        logger.info(
            f"Simulation: demand_spike={scenario.demand_spike}, "
            f"supplier_failure={scenario.supplier_failure}, "
            f"transport_delay={scenario.transport_delay}, "
            f"weather_disruption={scenario.weather_disruption}"
        )

        results = await asyncio.to_thread(_simulate_scenario, scenario)

        return {"timestamp": datetime.now().isoformat(), "status": "success", **results}

    except Exception as e:
        logger.error(f"Error in simulation endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@router.get("/simulate/templates")
async def get_scenario_templates():
    """Get predefined scenario templates."""
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "success",
        "templates": [
            {
                "name": "demand_surge",
                "label": "Demand Surge +30%",
                "demand_spike": 0.30,
                "supplier_failure": False,
                "transport_delay": False,
                "weather_disruption": False,
            },
            {
                "name": "supplier_crisis",
                "label": "Supplier Crisis",
                "demand_spike": 0.0,
                "supplier_failure": True,
                "transport_delay": False,
                "weather_disruption": False,
            },
            {
                "name": "transport_disruption",
                "label": "Transport Disruption",
                "demand_spike": 0.0,
                "supplier_failure": False,
                "transport_delay": True,
                "weather_disruption": False,
            },
            {
                "name": "weather_crisis",
                "label": "Weather Crisis",
                "demand_spike": 0.0,
                "supplier_failure": False,
                "transport_delay": False,
                "weather_disruption": True,
            },
            {
                "name": "perfect_storm",
                "label": "Perfect Storm",
                "demand_spike": 0.25,
                "supplier_failure": True,
                "transport_delay": True,
                "weather_disruption": False,
            },
        ],
    }
