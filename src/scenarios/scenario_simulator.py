"""
Layer 5a: Scenario Simulator (Digital Twin)

Allows users to input what-if conditions and see projected impacts
by re-running existing prediction models with modified inputs.
"""

import copy
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pickle

# Import existing layers
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..core.layer1 import build_layer1_dataset
from ..core.layer2 import build_layer2_dataset
from ..core.layer4 import SupplyChainAgent, get_recommendation
from ..utils.inference import predict_next_month_demand, predict_n_months


@dataclass
class ScenarioInput:
    """What-if scenario input parameters"""
    demand_spike: float = 0.0          # +30% = 0.30, -20% = -0.20
    supplier_failure: bool = False     # Force supplier failure
    transport_delay: bool = False      # Force transport delay
    weather_disruption: bool = False   # Force weather disruption
    custom_disruption_signal: Optional[float] = None
    custom_supplier_risk: Optional[float] = None
    inventory_adjustment: float = 0.0  # +/- inventory percentage


class ScenarioSimulator:
    """
    Main scenario simulation engine that applies what-if conditions
    and re-runs the complete prediction pipeline.
    """
    
    def __init__(self):
        """Initialize with all required models and pipelines"""
        self.layer1_bundle = None
        self.layer2_bundle = None
        self.rl_agent = None
        self.base_state = None
        
        # Load models
        try:
            self.rl_agent = SupplyChainAgent()
            self.rl_agent.load('supply_chain_agent.pkl')
            print("Scenario Simulator: RL agent loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load RL agent: {e}")
            self.rl_agent = None
    
    def load_base_pipeline(self, data_path='store_sale.csv'):
        """Load the complete base pipeline (Layers 1-4)"""
        print("Loading base pipeline...")
        
        # Layer 1: Data Collection
        self.layer1_bundle = build_layer1_dataset(data_path)
        
        # Layer 2: Feature Engineering
        self.layer2_bundle = build_layer2_dataset(self.layer1_bundle)
        
        # Create base state for latest month
        self.base_state = self._create_latest_state()
        
        print("Base pipeline loaded successfully")
        return self.base_state
    
    def _create_latest_state(self) -> Dict[str, Any]:
        """Create state dict from latest Layer 2 data"""
        if self.layer2_bundle is None:
            raise ValueError("Layer 2 bundle not loaded")
        
        consolidated_df = self.layer2_bundle['consolidated_df']
        latest_row = consolidated_df.iloc[-1]
        
        # Get demand forecast
        try:
            predicted_sales, predicted_diff = predict_next_month_demand()
        except Exception as e:
            print(f"Warning: Could not get demand forecast: {e}")
            predicted_sales = latest_row.get('sales', 500000)
        
        # Create state dict matching RL agent expectations
        state = {
            'predicted_demand': float(predicted_sales),
            'current_inventory': float(latest_row.get('current_inventory', 50000)),
            'supplier_risk_score': float(latest_row.get('supplier_risk_score', 0.3)),
            'disruption_signal': float(latest_row.get('disruption_signal', 0.2)),
            'days_to_stockout': float(latest_row.get('days_to_stockout', 15.0)),
            'composite_risk_score': float(latest_row.get('composite_risk_score', 0.25)),
            'weighted_inventory': float(latest_row.get('weighted_inventory', 45000)),
            'sales_volatility': float(latest_row.get('sales_volatility', 0.1))
        }
        
        return state
    
    def apply_scenario_modifications(self, base_state: Dict[str, Any], 
                                   scenario: ScenarioInput) -> Dict[str, Any]:
        """Apply what-if conditions to base state"""
        modified_state = copy.deepcopy(base_state)
        
        # Apply demand spike
        if scenario.demand_spike != 0.0:
            modified_state['predicted_demand'] *= (1 + scenario.demand_spike)
            print(f"Applied demand spike: {scenario.demand_spike:+.1%}")
        
        # Apply inventory adjustment
        if scenario.inventory_adjustment != 0.0:
            modified_state['current_inventory'] *= (1 + scenario.inventory_adjustment)
            modified_state['weighted_inventory'] *= (1 + scenario.inventory_adjustment)
            
            # Recalculate days to stockout
            demand = modified_state['predicted_demand']
            inventory = modified_state['current_inventory']
            modified_state['days_to_stockout'] = inventory / max(demand / 30.0, 1.0)
            print(f"Applied inventory adjustment: {scenario.inventory_adjustment:+.1%}")
        
        # Apply custom disruption signal
        if scenario.custom_disruption_signal is not None:
            modified_state['disruption_signal'] = float(scenario.custom_disruption_signal)
            print(f"Applied custom disruption signal: {scenario.custom_disruption_signal:.2f}")
        
        # Apply custom supplier risk
        if scenario.custom_supplier_risk is not None:
            modified_state['supplier_risk_score'] = float(scenario.custom_supplier_risk)
            print(f"Applied custom supplier risk: {scenario.custom_supplier_risk:.2f}")
        
        # Apply binary conditions
        if scenario.supplier_failure:
            modified_state['supplier_risk_score'] = max(0.9, modified_state['supplier_risk_score'])
            print("Applied supplier failure condition")
        
        if scenario.transport_delay:
            modified_state['disruption_signal'] = max(0.8, modified_state['disruption_signal'])
            print("Applied transport delay condition")
        
        if scenario.weather_disruption:
            # Increase disruption signal and adjust inventory (weather affects supply)
            modified_state['disruption_signal'] = max(0.7, modified_state['disruption_signal'])
            modified_state['current_inventory'] *= 0.9  # Weather reduces effective inventory
            modified_state['weighted_inventory'] *= 0.9
            print("Applied weather disruption condition")
        
        # Recalculate composite risk score
        supplier_weight = 0.3
        disruption_weight = 0.4
        weather_weight = 0.3
        
        # Use weather disruption score if available, otherwise use disruption signal
        weather_score = modified_state.get('weather_disruption_score', modified_state['disruption_signal'])
        
        modified_state['composite_risk_score'] = (
            modified_state['supplier_risk_score'] * supplier_weight +
            modified_state['disruption_signal'] * disruption_weight +
            weather_score * weather_weight
        )
        
        return modified_state
    
    def calculate_risk_probabilities(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stockout and delay probabilities from state"""
        demand = state['predicted_demand']
        inventory = state['current_inventory']
        days_to_stockout = state['days_to_stockout']
        supplier_risk = state['supplier_risk_score']
        disruption_signal = state['disruption_signal']
        
        # Stockout probability based on inventory ratio and time
        inventory_ratio = inventory / max(demand, 1)
        
        if days_to_stockout <= 3:
            stockout_base = 0.8
        elif days_to_stockout <= 7:
            stockout_base = 0.5
        elif days_to_stockout <= 14:
            stockout_base = 0.2
        else:
            stockout_base = 0.05
        
        # Adjust for supplier risk and disruption
        stockout_probability = min(0.95, stockout_base + supplier_risk * 0.3 + disruption_signal * 0.2)
        
        # Delay probability based on supplier and disruption signals
        delay_probability = min(0.9, supplier_risk * 0.7 + disruption_signal * 0.6)
        
        return {
            'stockout_probability': stockout_probability,
            'delay_probability': delay_probability
        }
    
    def estimate_cost_impact(self, base_state: Dict[str, Any], 
                           modified_state: Dict[str, Any]) -> Dict[str, float]:
        """Estimate cost impact of scenario changes"""
        base_demand = base_state['predicted_demand']
        modified_demand = modified_state['predicted_demand']
        
        demand_change = modified_demand - base_demand
        demand_change_pct = demand_change / max(base_demand, 1)
        
        # Base cost estimate (simplified)
        base_cost = base_demand * 0.5  # Assume 50% of demand value as cost
        
        # Cost impact factors
        stockout_prob = self.calculate_risk_probabilities(modified_state)['stockout_probability']
        delay_prob = self.calculate_risk_probabilities(modified_state)['delay_probability']
        
        # Cost impact from demand change
        demand_cost_impact = demand_change_pct * base_cost
        
        # Cost impact from risks
        risk_cost_impact = (stockout_prob * 1000 + delay_prob * 500) * (modified_demand / 100000)
        
        total_cost_impact = demand_cost_impact + risk_cost_impact
        
        return {
            'demand_cost_impact': demand_cost_impact,
            'risk_cost_impact': risk_cost_impact,
            'total_cost_impact': total_cost_impact,
            'demand_change_pct': demand_change_pct
        }
    
    def run_scenario_simulation(self, scenario: ScenarioInput) -> Dict[str, Any]:
        """Execute complete scenario simulation"""
        if self.base_state is None:
            self.load_base_pipeline()
        
        print(f"\n{'='*60}")
        print("SCENARIO SIMULATION - DIGITAL TWIN")
        print(f"{'='*60}")
        
        # Apply scenario modifications
        modified_state = self.apply_scenario_modifications(self.base_state, scenario)
        
        # Get RL agent recommendation
        if self.rl_agent:
            recommendation = get_recommendation(modified_state, self.rl_agent)
        else:
            recommendation = {
                'action': 'do_nothing',
                'confidence': 0.0,
                'explanation': 'RL agent not available'
            }
        
        # Calculate probabilities and impacts
        risk_probabilities = self.calculate_risk_probabilities(modified_state)
        cost_impact = self.estimate_cost_impact(self.base_state, modified_state)
        
        # Generate scenario report
        results = {
            'scenario_summary': {
                'conditions_applied': {
                    'demand_spike': scenario.demand_spike,
                    'supplier_failure': scenario.supplier_failure,
                    'transport_delay': scenario.transport_delay,
                    'weather_disruption': scenario.weather_disruption,
                    'custom_disruption_signal': scenario.custom_disruption_signal,
                    'custom_supplier_risk': scenario.custom_supplier_risk,
                    'inventory_adjustment': scenario.inventory_adjustment
                },
                'baseline_vs_scenario': {
                    'baseline_demand': self.base_state['predicted_demand'],
                    'scenario_demand': modified_state['predicted_demand'],
                    'baseline_inventory': self.base_state['current_inventory'],
                    'scenario_inventory': modified_state['current_inventory']
                }
            },
            'impact_analysis': {
                'cost_impact': cost_impact['total_cost_impact'],
                'stockout_probability': risk_probabilities['stockout_probability'],
                'delay_probability': risk_probabilities['delay_probability'],
                'demand_change': cost_impact['demand_change_pct']
            },
            'recommendations': {
                'action': recommendation['action'],
                'confidence': recommendation['confidence'],
                'rationale': recommendation['explanation']
            },
            'risk_breakdown': {
                'supplier_risk': modified_state['supplier_risk_score'],
                'disruption_risk': modified_state['disruption_signal'],
                'inventory_risk': 1.0 - (modified_state['current_inventory'] / max(modified_state['predicted_demand'], 1)),
                'composite_risk': modified_state['composite_risk_score']
            },
            'state_details': modified_state
        }
        
        # Print summary
        self._print_scenario_summary(results)
        
        return results
    
    def _print_scenario_summary(self, results: Dict[str, Any]):
        """Print formatted scenario summary"""
        print(f"\nSCENARIO RESULTS:")
        print(f"{'-'*40}")
        
        # Conditions applied
        conditions = results['scenario_summary']['conditions_applied']
        active_conditions = [k for k, v in conditions.items() if v not in [0.0, False, None]]
        if active_conditions:
            print(f"Active Conditions: {', '.join(active_conditions)}")
        else:
            print("Active Conditions: None (baseline scenario)")
        
        # Impact analysis
        impact = results['impact_analysis']
        print(f"\nImpact Analysis:")
        print(f"  Demand Change: {impact['demand_change']:+.1%}")
        print(f"  Cost Impact: ${impact['cost_impact']:,.0f}")
        print(f"  Stockout Probability: {impact['stockout_probability']:.1%}")
        print(f"  Delay Probability: {impact['delay_probability']:.1%}")
        
        # Recommendation
        rec = results['recommendations']
        print(f"\nRecommendation:")
        print(f"  Action: {rec['action'].upper()}")
        print(f"  Confidence: {rec['confidence']:.1f}%")
        
        # Risk breakdown
        risk = results['risk_breakdown']
        print(f"\nRisk Breakdown:")
        print(f"  Supplier Risk: {risk['supplier_risk']:.2f}")
        print(f"  Disruption Risk: {risk['disruption_risk']:.2f}")
        print(f"  Inventory Risk: {risk['inventory_risk']:.2f}")
        print(f"  Composite Risk: {risk['composite_risk']:.2f}")
        
        print(f"\n{'='*60}")


# Predefined scenario templates
SCENARIO_TEMPLATES = {
    "demand_surge": ScenarioInput(demand_spike=0.30),
    "supplier_crisis": ScenarioInput(supplier_failure=True, custom_supplier_risk=0.9),
    "transport_disruption": ScenarioInput(transport_delay=True),
    "weather_crisis": ScenarioInput(weather_disruption=True),
    "inventory_shortage": ScenarioInput(inventory_adjustment=-0.5),
    "perfect_storm": ScenarioInput(
        demand_spike=0.25,
        supplier_failure=True,
        transport_delay=True,
        inventory_adjustment=-0.3
    )
}


def run_predefined_scenario(scenario_name: str) -> Dict[str, Any]:
    """Run a predefined scenario template"""
    if scenario_name not in SCENARIO_TEMPLATES:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    simulator = ScenarioSimulator()
    scenario = SCENARIO_TEMPLATES[scenario_name]
    
    print(f"Running predefined scenario: {scenario_name}")
    return simulator.run_scenario_simulation(scenario)


if __name__ == "__main__":
    # Example usage
    simulator = ScenarioSimulator()
    
    # Test a custom scenario
    custom_scenario = ScenarioInput(
        demand_spike=0.20,  # 20% demand increase
        supplier_failure=True,
        inventory_adjustment=-0.3  # 30% inventory reduction
    )
    
    results = simulator.run_scenario_simulation(custom_scenario)
    print("\nScenario simulation completed!")
