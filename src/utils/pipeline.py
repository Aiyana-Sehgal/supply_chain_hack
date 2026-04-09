"""
Unified Supply Chain Intelligence Pipeline

End-to-end orchestrator for Layers 1-5:
- Layer 1: Data Collection
- Layer 2: Feature Engineering  
- Layer 3: Integration & Forecasting
- Layer 4: RL Decision Engine
- Layer 5a: Scenario Simulator (Digital Twin)
- Layer 5b: Explainable AI
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import json

# Import all layers
from ..core import build_layer1_dataset, build_layer2_dataset, build_layer3_dataset, SupplyChainAgent, get_recommendation
# from ..scenarios import ScenarioSimulator, ScenarioInput  # Commented out to avoid circular import
from ..xai import EnhancedXAI, create_enhanced_xai


class SupplyChainPipeline:
    """
    Unified pipeline orchestrator for the complete supply chain intelligence system.
    """
    
    def __init__(self, sales_path='data/store_sale.csv', 
                 news_api_key=None, 
                 weather_api_key=None,
                 verbose=True):
        """
        Initialize the pipeline with configuration.
        
        Parameters:
        -----------
        sales_path : str
            Path to historical sales data
        news_api_key : str, optional
            NewsData.io API key
        weather_api_key : str, optional
            WeatherAPI key
        verbose : bool
            Enable verbose logging
        """
        self.sales_path = sales_path
        self.news_api_key = news_api_key or os.getenv('NEWSDATA_API_KEY')
        self.weather_api_key = weather_api_key or os.getenv('WEATHERAPI_KEY')
        self.verbose = verbose
        
        # Initialize components
        self.layer1_bundle = None
        self.layer2_bundle = None
        self.layer3_bundle = None
        self.rl_agent = None
        self.scenario_simulator = None
        self.enhanced_xai = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load trained models and initialize components"""
        try:
            # Load RL agent
            self.rl_agent = SupplyChainAgent()
            self.rl_agent.load('supply_chain_agent.pkl')
            if self.verbose:
                print("RL agent loaded successfully")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load RL agent: {e}")
            self.rl_agent = None
        
        # Initialize Layer 5 components
        self.scenario_simulator = ScenarioSimulator()
        self.enhanced_xai = create_enhanced_xai(enable_enhancement=True)
    
    def run_full_pipeline(self, n_forecast_months=3) -> Dict[str, Any]:
        """
        Run the complete pipeline from Layer 1 to Layer 4.
        
        Parameters:
        -----------
        n_forecast_months : int
            Number of months to forecast
        
        Returns:
        --------
        dict
            Complete pipeline results with current state and recommendation
        """
        if self.verbose:
            print("=" * 80)
            print("SUPPLY CHAIN INTELLIGENCE PIPELINE")
            print("=" * 80)
            print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Layer 1: Data Collection
            if self.verbose:
                print("\n[Layer 1] Data Collection...")
            
            self.layer1_bundle = build_layer1_dataset(
                sales_path=self.sales_path,
                news_api_key=self.news_api_key,
                weather_api_key=self.weather_api_key
            )
            
            # Layer 2: Feature Engineering
            if self.verbose:
                print("\n[Layer 2] Feature Engineering...")
            
            self.layer2_bundle = build_layer2_dataset(
                self.layer1_bundle,
                inventory_method='risk_weighted',
                weather_method='mean',
                include_rolling=True
            )
            
            # Layer 3: Integration & Forecasting
            if self.verbose:
                print("\n[Layer 3] Integration & Forecasting...")
            
            self.layer3_bundle = build_layer3_dataset(
                sales_path=self.sales_path,
                news_api_key=self.news_api_key,
                weather_api_key=self.weather_api_key,
                n_forecast_months=n_forecast_months,
                verbose=self.verbose
            )
            
            # Layer 4: RL Decision Engine
            if self.verbose:
                print("\n[Layer 4] RL Decision Engine...")
            
            current_state = self.layer3_bundle['layer3_state']
            
            if self.rl_agent:
                recommendation = get_recommendation(current_state, self.rl_agent)
            else:
                recommendation = {
                    'action': 'do_nothing',
                    'confidence': 0.0,
                    'explanation': 'RL agent not available'
                }
            
            # Generate XAI explanations
            if self.verbose:
                print("\n[Layer 5b] Explainable AI...")
            
            explanations = self.xai_explainer.generate_comprehensive_explanation(
                state=current_state,
                action=recommendation['action'],
                confidence=recommendation['confidence']
            )
            
            # Assemble results
            pipeline_results = {
                'timestamp': datetime.now().isoformat(),
                'current_state': current_state,
                'recommendation': recommendation,
                'explanations': explanations,
                'layer_summaries': {
                    'layer1': {
                        'shape': self.layer1_bundle['layer1_df'].shape,
                        'date_range': f"{self.layer1_bundle['layer1_df']['date'].min()} to {self.layer1_bundle['layer1_df']['date'].max()}"
                    },
                    'layer2': {
                        'shape': self.layer2_bundle['consolidated_df'].shape,
                        'features_count': len(self.layer2_bundle['feature_columns'])
                    },
                    'layer3': {
                        'shape': self.layer3_bundle['layer3_dataframe'].shape,
                        'forecast_months': n_forecast_months
                    }
                },
                'pipeline_status': 'success'
            }
            
            if self.verbose:
                print("\n" + "=" * 80)
                print("PIPELINE COMPLETED SUCCESSFULLY")
                print("=" * 80)
                self._print_pipeline_summary(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            error_msg = f"Pipeline execution failed: {e}"
            if self.verbose:
                print(f"\nERROR: {error_msg}")
                import traceback
                traceback.print_exc()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'pipeline_status': 'error',
                'error_message': error_msg
            }
    
    def run_scenario_analysis(self, scenario: ScenarioInput) -> Dict[str, Any]:
        """
        Run what-if scenario analysis using the Digital Twin.
        
        Parameters:
        -----------
        scenario : ScenarioInput
            What-if scenario parameters
        
        Returns:
        --------
        dict
            Scenario analysis results with explanations
        """
        if self.verbose:
            print("\n" + "=" * 80)
            print("SCENARIO ANALYSIS - DIGITAL TWIN")
            print("=" * 80)
        
        try:
            # Ensure base pipeline is loaded
            if self.layer3_bundle is None:
                self.run_full_pipeline()
            
            # Load base state in simulator
            self.scenario_simulator.load_base_pipeline(self.sales_path)
            
            # Run scenario simulation
            scenario_results = self.scenario_simulator.run_scenario_simulation(scenario)
            
            # Generate XAI explanations for scenario
            scenario_explanations = self.xai_explainer.explain_scenario_impact(scenario_results)
            
            # Generate recommendation explanation
            recommendation = scenario_results['recommendations']
            rec_explanation = self.xai_explainer.explain_recommendation(
                state=scenario_results['state_details'],
                action=recommendation['action'],
                confidence=recommendation['confidence']
            )
            
            # Assemble complete scenario results
            complete_results = {
                'scenario_input': {
                    'demand_spike': scenario.demand_spike,
                    'supplier_failure': scenario.supplier_failure,
                    'transport_delay': scenario.transport_delay,
                    'weather_disruption': scenario.weather_disruption,
                    'custom_disruption_signal': scenario.custom_disruption_signal,
                    'custom_supplier_risk': scenario.custom_supplier_risk,
                    'inventory_adjustment': scenario.inventory_adjustment
                },
                'scenario_results': scenario_results,
                'scenario_explanations': scenario_explanations,
                'recommendation_explanation': rec_explanation,
                'timestamp': datetime.now().isoformat(),
                'analysis_status': 'success'
            }
            
            if self.verbose:
                self._print_scenario_summary(complete_results)
            
            return complete_results
            
        except Exception as e:
            error_msg = f"Scenario analysis failed: {e}"
            if self.verbose:
                print(f"\nERROR: {error_msg}")
                import traceback
                traceback.print_exc()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'analysis_status': 'error',
                'error_message': error_msg
            }
    
    def get_current_risk_assessment(self) -> Dict[str, Any]:
        """Get current risk assessment without running full pipeline"""
        if self.layer3_bundle is None:
            self.run_full_pipeline()
        
        current_state = self.layer3_bundle['layer3_state']
        risk_explanations = self.xai_explainer.explain_risk_score(current_state)
        
        return {
            'current_state': current_state,
            'risk_explanations': risk_explanations,
            'risk_breakdown': {
                'supplier_risk': current_state['supplier_risk_score'],
                'disruption_risk': current_state['disruption_signal'],
                'inventory_risk': 1.0 - (current_state['current_inventory'] / max(current_state['predicted_demand'], 1)),
                'composite_risk': current_state['composite_risk_score']
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def export_results(self, results: Dict[str, Any], filename: str):
        """Export results to JSON file"""
        try:
            # Convert non-serializable objects
            serializable_results = self._make_serializable(results)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            if self.verbose:
                print(f"Results exported to {filename}")
            return True
        except Exception as e:
            if self.verbose:
                print(f"Error exporting results: {e}")
            return False
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _print_pipeline_summary(self, results: Dict[str, Any]):
        """Print formatted pipeline summary"""
        print(f"\nCURRENT STATE SUMMARY:")
        print(f"{'-'*50}")
        
        state = results['current_state']
        print(f"  Date: {state['date']}")
        print(f"  Predicted Demand: {state['predicted_demand']:,.0f} units")
        print(f"  Current Inventory: {state['current_inventory']:,.0f} units")
        print(f"  Days to Stockout: {state['days_to_stockout']:.1f} days")
        print(f"  Supplier Risk Score: {state['supplier_risk_score']:.3f}")
        print(f"  Disruption Signal: {state['disruption_signal']:.3f}")
        print(f"  Composite Risk: {state['composite_risk_score']:.3f}")
        
        print(f"\nRECOMMENDATION:")
        print(f"{'-'*50}")
        rec = results['recommendation']
        print(f"  Action: {rec['action'].upper()}")
        print(f"  Confidence: {rec['confidence']:.1f}%")
        print(f"  Explanation: {rec['explanation']}")
        
        print(f"\nRISK ASSESSMENT:")
        print(f"{'-'*50}")
        risk_exp = results['explanations']['risk_explanations']
        for risk_type, explanation in risk_exp.items():
            print(f"  {risk_type.replace('_', ' ').title()}: {explanation}")
    
    def _print_scenario_summary(self, results: Dict[str, Any]):
        """Print formatted scenario analysis summary"""
        print(f"\nSCENARIO ANALYSIS SUMMARY:")
        print(f"{'-'*60}")
        
        # Scenario conditions
        conditions = results['scenario_input']
        active_conditions = [k for k, v in conditions.items() if v not in [0.0, False, None]]
        if active_conditions:
            print(f"Active Conditions: {', '.join(active_conditions)}")
        else:
            print("Active Conditions: None (baseline scenario)")
        
        # Impact analysis
        impact = results['scenario_results']['impact_analysis']
        print(f"\nImpact Analysis:")
        print(f"  Demand Change: {impact['demand_change']:+.1%}")
        print(f"  Cost Impact: ${impact['cost_impact']:,.0f}")
        print(f"  Stockout Probability: {impact['stockout_probability']:.1%}")
        print(f"  Delay Probability: {impact['delay_probability']:.1%}")
        
        # Recommendation
        rec = results['scenario_results']['recommendations']
        print(f"\nScenario Recommendation:")
        print(f"  Action: {rec['action'].upper()}")
        print(f"  Confidence: {rec['confidence']:.1f}%")
        
        print(f"\n{'='*80}")


# Convenience functions for quick usage
def quick_analysis(sales_path='data/store_sale.csv', verbose=True):
    """Quick end-to-end analysis"""
    pipeline = SupplyChainPipeline(sales_path=sales_path, verbose=verbose)
    return pipeline.run_full_pipeline()


def quick_scenario(scenario_params: Dict[str, Any], sales_path='data/store_sale.csv', verbose=True):
    """Quick scenario analysis"""
    pipeline = SupplyChainPipeline(sales_path=sales_path, verbose=verbose)
    scenario = ScenarioInput(**scenario_params)
    return pipeline.run_scenario_analysis(scenario)


# Example usage
if __name__ == "__main__":
    # Example 1: Full pipeline analysis
    print("Running full pipeline analysis...")
    pipeline = SupplyChainPipeline(sales_path='data/store_sale.csv', verbose=True)
    results = pipeline.run_full_pipeline()
    
    # Example 2: Scenario analysis
    print("\nRunning scenario analysis...")
    scenario = ScenarioInput(
        demand_spike=0.25,  # 25% demand increase
        supplier_failure=True,
        inventory_adjustment=-0.3  # 30% inventory reduction
    )
    scenario_results = pipeline.run_scenario_analysis(scenario)
    
    # Export results
    pipeline.export_results(results, 'pipeline_results.json')
    pipeline.export_results(scenario_results, 'scenario_results.json')
    
    print("\nAnalysis completed!")
