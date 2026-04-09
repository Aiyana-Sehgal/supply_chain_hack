"""
Layer 5: Digital Twin Interface

Unified interface for Layer 5a (Scenario Simulator) and Layer 5b (Explainable AI)
This is the marketing-facing "Digital Twin" component that provides what-if analysis
and explainable AI capabilities.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime

from .scenario_simulator import ScenarioSimulator, ScenarioInput, SCENARIO_TEMPLATES
from ..xai import XAIExplainer
from ..utils.pipeline import SupplyChainPipeline
from ..core import get_recommendation


@dataclass
class DigitalTwinConfig:
    """Configuration for Digital Twin operations"""
    sales_path: str = 'store_sale.csv'
    news_api_key: Optional[str] = None
    weather_api_key: Optional[str] = None
    verbose: bool = True
    auto_load_pipeline: bool = True


class DigitalTwin:
    """
    Main Digital Twin interface that combines scenario simulation and explainable AI.
    
    This is the centerpiece feature that allows users to:
    1. Run what-if scenarios on their supply chain
    2. Get human-readable explanations for all recommendations
    3. Understand risk factors and decision rationale
    """
    
    def __init__(self, config: DigitalTwinConfig = None):
        """Initialize Digital Twin with configuration"""
        self.config = config or DigitalTwinConfig()
        
        # Initialize components
        self.pipeline = SupplyChainPipeline(
            sales_path=self.config.sales_path,
            news_api_key=self.config.news_api_key,
            weather_api_key=self.config.weather_api_key,
            verbose=self.config.verbose
        )
        
        self.scenario_simulator = ScenarioSimulator()
        self.xai_explainer = XAIExplainer()
        
        # Load base pipeline if configured
        if self.config.auto_load_pipeline:
            self._ensure_pipeline_loaded()
    
    def _ensure_pipeline_loaded(self):
        """Ensure base pipeline is loaded"""
        if self.pipeline.layer3_bundle is None:
            self.pipeline.run_full_pipeline()
    
    def analyze_current_state(self) -> Dict[str, Any]:
        """
        Analyze current supply chain state with full explanations.
        
        Returns:
        --------
        dict
            Current state analysis with risk explanations and recommendations
        """
        if self.config.verbose:
            print("=" * 80)
            print("DIGITAL TWIN - CURRENT STATE ANALYSIS")
            print("=" * 80)
        
        # Ensure pipeline is loaded
        self._ensure_pipeline_loaded()
        
        # Get current state and risk assessment
        risk_assessment = self.pipeline.get_current_risk_assessment()
        current_state = risk_assessment['current_state']
        
        # Get current recommendation
        if self.pipeline.rl_agent:
            recommendation = get_recommendation(current_state, self.pipeline.rl_agent)
        else:
            recommendation = {
                'action': 'do_nothing',
                'confidence': 0.0,
                'explanation': 'RL agent not available'
            }
        
        # Generate comprehensive explanations
        explanations = self.xai_explainer.generate_comprehensive_explanation(
            state=current_state,
            action=recommendation['action'],
            confidence=recommendation['confidence']
        )
        
        # Assemble analysis
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'current_state',
            'current_state': current_state,
            'risk_assessment': risk_assessment,
            'recommendation': recommendation,
            'explanations': explanations,
            'digital_twin_status': 'active'
        }
        
        if self.config.verbose:
            self._print_current_analysis(analysis)
        
        return analysis
    
    def run_what_if_scenario(self, scenario_name: str = None, 
                            custom_scenario: ScenarioInput = None) -> Dict[str, Any]:
        """
        Run what-if scenario analysis.
        
        Parameters:
        -----------
        scenario_name : str, optional
            Name of predefined scenario template
        custom_scenario : ScenarioInput, optional
            Custom scenario parameters
        
        Returns:
        --------
        dict
            Complete scenario analysis with explanations
        """
        if self.config.verbose:
            print("=" * 80)
            print("DIGITAL TWIN - WHAT-IF SCENARIO ANALYSIS")
            print("=" * 80)
        
        # Determine scenario to use
        if custom_scenario:
            scenario = custom_scenario
            scenario_display_name = "Custom Scenario"
        elif scenario_name and scenario_name in SCENARIO_TEMPLATES:
            scenario = SCENARIO_TEMPLATES[scenario_name]
            scenario_display_name = f"Predefined: {scenario_name}"
        else:
            raise ValueError("Must provide either scenario_name or custom_scenario")
        
        if self.config.verbose:
            print(f"Running scenario: {scenario_display_name}")
        
        # Run scenario analysis
        scenario_results = self.pipeline.run_scenario_analysis(scenario)
        
        # Add Digital Twin metadata
        scenario_results['digital_twin_metadata'] = {
            'scenario_name': scenario_display_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'digital_twin_version': '1.0'
        }
        
        if self.config.verbose:
            self._print_scenario_analysis(scenario_results)
        
        return scenario_results
    
    def compare_scenarios(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple scenarios side-by-side.
        
        Parameters:
        -----------
        scenarios : List[Dict[str, Any]]
            List of scenario definitions with names and parameters
        
        Returns:
        --------
        dict
            Comparative analysis of all scenarios
        """
        if self.config.verbose:
            print("=" * 80)
            print("DIGITAL TWIN - SCENARIO COMPARISON")
            print("=" * 80)
        
        comparison_results = []
        
        for scenario_def in scenarios:
            name = scenario_def.get('name', 'Unnamed Scenario')
            params = scenario_def.get('parameters', {})
            
            # Create scenario input
            scenario = ScenarioInput(**params)
            
            # Run analysis
            result = self.run_what_if_scenario(custom_scenario=scenario)
            result['scenario_comparison_name'] = name
            comparison_results.append(result)
        
        # Generate comparison summary
        comparison_summary = self._generate_comparison_summary(comparison_results)
        
        comparison_analysis = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'scenario_comparison',
            'scenarios': comparison_results,
            'comparison_summary': comparison_summary,
            'total_scenarios': len(scenarios)
        }
        
        if self.config.verbose:
            self._print_comparison_analysis(comparison_analysis)
        
        return comparison_analysis

    def _generate_comparison_summary(self, comparison_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a compact side-by-side scenario comparison summary."""
        summary = []
        for result in comparison_results:
            scenario_name = result.get('scenario_comparison_name', 'Unnamed Scenario')
            impact = result.get('scenario_results', {}).get('impact_analysis', {})
            recommendation = result.get('scenario_results', {}).get('recommendations', {})
            summary.append({
                'name': scenario_name,
                'cost_impact': impact.get('cost_impact', 0.0),
                'stockout_probability': impact.get('stockout_probability', 0.0),
                'delay_probability': impact.get('delay_probability', 0.0),
                'recommended_action': recommendation.get('action', 'do_nothing')
            })
        return summary
    
    def get_risk_alerts(self, alert_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Get current risk alerts based on thresholds.
        
        Parameters:
        -----------
        alert_thresholds : Dict[str, float], optional
            Custom alert thresholds for different risk factors
        
        Returns:
        --------
        dict
            Risk alerts and recommendations
        """
        if self.config.verbose:
            print("=" * 80)
            print("DIGITAL TWIN - RISK ALERT SYSTEM")
            print("=" * 80)
        
        # Default thresholds
        default_thresholds = {
            'supplier_risk': 0.7,
            'disruption_signal': 0.6,
            'inventory_ratio': 0.1,  # inventory/demand ratio
            'days_to_stockout': 7,
            'composite_risk': 0.6
        }
        
        thresholds = alert_thresholds or default_thresholds
        
        # Get current state
        self._ensure_pipeline_loaded()
        current_state = self.pipeline.layer3_bundle['layer3_state']
        
        # Check alerts
        alerts = []
        
        # Supplier risk alert
        if current_state['supplier_risk_score'] > thresholds['supplier_risk']:
            alerts.append({
                'type': 'supplier_risk',
                'severity': 'high' if current_state['supplier_risk_score'] > 0.8 else 'medium',
                'message': f"Supplier risk score ({current_state['supplier_risk_score']:.2f}) exceeds threshold ({thresholds['supplier_risk']:.1f})",
                'recommendation': 'Consider switching to backup supplier'
            })
        
        # Disruption signal alert
        if current_state['disruption_signal'] > thresholds['disruption_signal']:
            alerts.append({
                'type': 'disruption_signal',
                'severity': 'high' if current_state['disruption_signal'] > 0.8 else 'medium',
                'message': f"Disruption signal ({current_state['disruption_signal']:.2f}) exceeds threshold ({thresholds['disruption_signal']:.1f})",
                'recommendation': 'Activate alternative logistics routes'
            })
        
        # Inventory alert
        inventory_ratio = current_state['current_inventory'] / max(current_state['predicted_demand'], 1)
        if inventory_ratio < thresholds['inventory_ratio']:
            alerts.append({
                'type': 'inventory_shortage',
                'severity': 'critical' if inventory_ratio < 0.05 else 'high',
                'message': f"Inventory ratio ({inventory_ratio:.2%}) below threshold ({thresholds['inventory_ratio']:.1%})",
                'recommendation': 'Immediate restocking required'
            })
        
        # Days to stockout alert
        if current_state['days_to_stockout'] < thresholds['days_to_stockout']:
            alerts.append({
                'type': 'stockout_timeline',
                'severity': 'critical' if current_state['days_to_stockout'] < 3 else 'high',
                'message': f"Days to stockout ({current_state['days_to_stockout']:.1f}) below threshold ({thresholds['days_to_stockout']})",
                'recommendation': 'Emergency restocking recommended'
            })
        
        # Composite risk alert
        if current_state['composite_risk_score'] > thresholds['composite_risk']:
            alerts.append({
                'type': 'composite_risk',
                'severity': 'high' if current_state['composite_risk_score'] > 0.8 else 'medium',
                'message': f"Composite risk score ({current_state['composite_risk_score']:.2f}) exceeds threshold ({thresholds['composite_risk']:.1f})",
                'recommendation': 'Review overall supply chain strategy'
            })
        
        # Generate alert summary
        alert_summary = {
            'total_alerts': len(alerts),
            'critical_alerts': len([a for a in alerts if a['severity'] == 'critical']),
            'high_alerts': len([a for a in alerts if a['severity'] == 'high']),
            'medium_alerts': len([a for a in alerts if a['severity'] == 'medium']),
            'overall_status': 'critical' if alerts and any(a['severity'] == 'critical' for a in alerts) 
                           else 'high' if alerts and any(a['severity'] == 'high' for a in alerts)
                           else 'normal' if not alerts else 'medium'
        }
        
        risk_alerts = {
            'timestamp': datetime.now().isoformat(),
            'current_state': current_state,
            'alerts': alerts,
            'alert_summary': alert_summary,
            'thresholds_used': thresholds,
            'recommendations': [alert['recommendation'] for alert in alerts]
        }
        
        if self.config.verbose:
            self._print_risk_alerts(risk_alerts)
        
        return risk_alerts
    
    def generate_executive_summary(self) -> Dict[str, Any]:
        """
        Generate executive-friendly summary of current supply chain status.
        
        Returns:
        --------
        dict
            Executive summary with key metrics and recommendations
        """
        if self.config.verbose:
            print("=" * 80)
            print("DIGITAL TWIN - EXECUTIVE SUMMARY")
            print("=" * 80)
        
        # Get current analysis
        current_analysis = self.analyze_current_state()
        risk_alerts = self.get_risk_alerts()
        
        # Extract key metrics
        state = current_analysis['current_state']
        
        # Calculate key performance indicators
        kpis = {
            'service_level_risk': 'Low' if state['days_to_stockout'] > 21 else 
                                'Medium' if state['days_to_stockout'] > 7 else 'High',
            'supplier_stability': 'Stable' if state['supplier_risk_score'] < 0.4 else
                                'At Risk' if state['supplier_risk_score'] < 0.7 else 'Unstable',
            'disruption_exposure': 'Low' if state['disruption_signal'] < 0.3 else
                                  'Medium' if state['disruption_signal'] < 0.6 else 'High',
            'inventory_health': 'Optimal' if 0.3 <= state['current_inventory'] / max(state['predicted_demand'], 1) <= 0.6 else
                              'Low' if state['current_inventory'] / max(state['predicted_demand'], 1) < 0.3 else 'Excess',
            'overall_risk_level': risk_alerts['alert_summary']['overall_status'].title()
        }
        
        # Generate executive recommendations
        executive_recommendations = []
        
        if risk_alerts['alert_summary']['critical_alerts'] > 0:
            executive_recommendations.append("IMMEDIATE ACTION REQUIRED: Critical risks detected")
        
        if state['supplier_risk_score'] > 0.6:
            executive_recommendations.append("Diversify supplier base to reduce dependency risk")
        
        if state['days_to_stockout'] < 14:
            executive_recommendations.append("Increase safety stock to improve service levels")
        
        if state['disruption_signal'] > 0.5:
            executive_recommendations.append("Develop contingency plans for logistics disruptions")
        
        if not executive_recommendations:
            executive_recommendations.append("Current supply chain posture is optimal - maintain monitoring")
        
        executive_summary = {
            'timestamp': datetime.now().isoformat(),
            'report_period': f"Current as of {datetime.now().strftime('%Y-%m-%d')}",
            'key_metrics': {
                'predicted_demand': f"{state['predicted_demand']:,.0f} units",
                'current_inventory': f"{state['current_inventory']:,.0f} units",
                'days_of_supply': f"{state['days_to_stockout']:.1f} days",
                'supplier_risk': f"{state['supplier_risk_score']:.1%}",
                'disruption_risk': f"{state['disruption_signal']:.1%}"
            },
            'kpis': kpis,
            'risk_summary': risk_alerts['alert_summary'],
            'current_recommendation': current_analysis['recommendation']['action'],
            'executive_recommendations': executive_recommendations,
            'confidence_level': f"{current_analysis['recommendation']['confidence']:.0f}%"
        }
        
        if self.config.verbose:
            self._print_executive_summary(executive_summary)
        
        return executive_summary
    
    def export_analysis(self, analysis: Dict[str, Any], filename: str, format: str = 'json'):
        """Export analysis to file"""
        try:
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            if self.config.verbose:
                print(f"Analysis exported to {filename}")
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"Error exporting analysis: {e}")
            return False
    
    # Private printing methods
    def _print_current_analysis(self, analysis: Dict[str, Any]):
        """Print current state analysis"""
        print(f"\nCURRENT STATE ANALYSIS:")
        print(f"{'-'*50}")
        
        state = analysis['current_state']
        print(f"  Predicted Demand: {state['predicted_demand']:,.0f} units")
        print(f"  Inventory Coverage: {state['days_to_stockout']:.1f} days")
        print(f"  Supplier Risk: {state['supplier_risk_score']:.1%}")
        print(f"  Disruption Risk: {state['disruption_signal']:.1%}")
        
        rec = analysis['recommendation']
        print(f"\nRecommended Action: {rec['action'].upper()}")
        print(f"Confidence: {rec['confidence']:.1f}%")
        
        print(f"\n{'='*80}")
    
    def _print_scenario_analysis(self, analysis: Dict[str, Any]):
        """Print scenario analysis results"""
        print(f"\nSCENARIO ANALYSIS RESULTS:")
        print(f"{'-'*60}")
        
        results = analysis['scenario_results']
        impact = results['impact_analysis']
        
        print(f"Demand Change: {impact['demand_change']:+.1%}")
        print(f"Cost Impact: ${impact['cost_impact']:,.0f}")
        print(f"Stockout Risk: {impact['stockout_probability']:.1%}")
        print(f"Delay Risk: {impact['delay_probability']:.1%}")
        
        rec = results['recommendations']
        print(f"\nRecommended Action: {rec['action'].upper()}")
        print(f"Confidence: {rec['confidence']:.1f}%")
        
        print(f"\n{'='*80}")
    
    def _print_comparison_analysis(self, analysis: Dict[str, Any]):
        """Print scenario comparison results"""
        print(f"\nSCENARIO COMPARISON SUMMARY:")
        print(f"{'-'*60}")
        
        summary = analysis['comparison_summary']
        print(f"Total Scenarios: {analysis['total_scenarios']}")
        
        for scenario_summary in summary:
            print(f"\n{scenario_summary['name']}:")
            print(f"  Cost Impact: ${scenario_summary['cost_impact']:,.0f}")
            print(f"  Stockout Risk: {scenario_summary['stockout_probability']:.1%}")
            print(f"  Recommended Action: {scenario_summary['recommended_action']}")
        
        print(f"\n{'='*80}")
    
    def _print_risk_alerts(self, alerts: Dict[str, Any]):
        """Print risk alerts"""
        print(f"\nRISK ALERT SUMMARY:")
        print(f"{'-'*50}")
        
        summary = alerts['alert_summary']
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Total Alerts: {summary['total_alerts']}")
        
        if summary['critical_alerts'] > 0:
            print(f"Critical Alerts: {summary['critical_alerts']}")
        if summary['high_alerts'] > 0:
            print(f"High Priority: {summary['high_alerts']}")
        
        if alerts['alerts']:
            print(f"\nActive Alerts:")
            for alert in alerts['alerts']:
                print(f"  {alert['type'].replace('_', ' ').title()}: {alert['message']}")
        
        print(f"\n{'='*80}")
    
    def _print_executive_summary(self, summary: Dict[str, Any]):
        """Print executive summary"""
        print(f"\nEXECUTIVE SUMMARY:")
        print(f"{'-'*50}")
        
        print(f"Report Period: {summary['report_period']}")
        print(f"Overall Risk Level: {summary['kpis']['overall_risk_level']}")
        
        print(f"\nKey Metrics:")
        metrics = summary['key_metrics']
        for key, value in metrics.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nKPI Summary:")
        kpis = summary['kpis']
        for key, value in kpis.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"\nTop Recommendation: {summary['current_recommendation'].upper()}")
        print(f"Confidence Level: {summary['confidence_level']}")
        
        print(f"\nExecutive Recommendations:")
        for i, rec in enumerate(summary['executive_recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n{'='*80}")


# Factory function for easy instantiation
def create_digital_twin(sales_path='store_sale.csv', 
                        news_api_key=None, 
                        weather_api_key=None,
                        verbose=True) -> DigitalTwin:
    """Create Digital Twin instance with default configuration"""
    config = DigitalTwinConfig(
        sales_path=sales_path,
        news_api_key=news_api_key,
        weather_api_key=weather_api_key,
        verbose=verbose
    )
    return DigitalTwin(config)


# Example usage
if __name__ == "__main__":
    # Create Digital Twin
    dt = create_digital_twin(verbose=True)
    
    # Run current state analysis
    current = dt.analyze_current_state()
    
    # Run what-if scenario
    scenario = dt.run_what_if_scenario(scenario_name="demand_surge")
    
    # Get risk alerts
    alerts = dt.get_risk_alerts()
    
    # Generate executive summary
    executive = dt.generate_executive_summary()
    
    print("\nDigital Twin analysis completed!")
