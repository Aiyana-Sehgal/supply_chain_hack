"""
Recommendation Service

Business logic for recommendation engine with XAI explanations.
Integrates with existing Layers 4 (RL) and 5 (Enhanced XAI).
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from ..models import (
    RecommendationResponse, 
    RecommendationExplanation, 
    ValidationMetrics,
    ActionType,
    ScenarioRequest
)

from ..endpoints.simulation import _simulate_scenario

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service for recommendation engine and XAI explanations"""
    
    def __init__(self):
        """Initialize recommendation service"""
        self.cache_timeout = 300  # 5 minutes
        self.last_update = None
        self.cached_recommendation = None
        
        # Available actions
        self.actions = list(ActionType)
        
        # Action descriptions for explanations
        self.action_descriptions = {
            ActionType.DO_NOTHING: "Continue monitoring current situation without taking action",
            ActionType.REORDER_STOCK: "Place standard order to replenish inventory levels",
            ActionType.SWITCH_SUPPLIER: "Change to backup supplier to mitigate risk",
            ActionType.REROUTE_SHIPMENT: "Use alternate logistics route to avoid disruptions",
            ActionType.EMERGENCY_RESTOCK: "Initiate urgent expensive restocking procedure"
        }
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state from layer3"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            
            from src.core.layer3 import build_layer3_dataset
            
            layer3_bundle = build_layer3_dataset(verbose=False)
            state = layer3_bundle['layer3_state']
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            mock_state = {
                'predicted_demand': 500000,
                'current_inventory': 25000,
                'supplier_risk_score': 0.65,
                'disruption_signal': 0.45,
                'days_to_stockout': 5.0,
                'composite_risk_score': 0.55
            }
            return mock_state
    
    def _get_enhanced_explanation(self, base_explanation: str, state: Dict[str, Any], action: str) -> Optional[str]:
        """Get enhanced explanation from XAI system"""
        try:
            # Import enhanced XAI
            from src.xai import create_enhanced_xai
            
            xai = create_enhanced_xai(enable_enhancement=True)
            xai_state = self._sanitize_state_for_xai(state)
            
            # Generate enhanced explanation
            enhanced_exp = xai.explain_recommendation(xai_state, action, 85.0)
            
            return enhanced_exp.get('rationale', base_explanation)
            
        except Exception as e:
            logger.error(f"Error getting enhanced explanation: {e}")
            return base_explanation

    def _sanitize_state_for_xai(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Limit XAI inputs to the core numeric fields it actually uses."""
        return {
            'predicted_demand': float(state.get('predicted_demand', 0.0)),
            'current_inventory': float(state.get('current_inventory', 0.0)),
            'supplier_risk_score': float(state.get('supplier_risk_score', 0.0)),
            'disruption_signal': float(state.get('disruption_signal', 0.0)),
            'days_to_stockout': float(state.get('days_to_stockout', 0.0)),
            'composite_risk_score': float(state.get('composite_risk_score', 0.0)),
        }
    
    def _generate_explanation(self, state: Dict[str, Any], action: str, confidence: float) -> RecommendationExplanation:
        """Generate detailed explanation for recommendation"""
        # Base rationale
        base_rationale = self.action_descriptions.get(ActionType(action), "Take recommended action")
        
        # Risk factors
        risk_factors = []
        if state.get('supplier_risk_score', 0) > 0.6:
            risk_factors.append(f"High supplier risk ({state['supplier_risk_score']:.1%})")
        if state.get('disruption_signal', 0) > 0.5:
            risk_factors.append(f"Elevated disruption risk ({state['disruption_signal']:.1%})")
        if state.get('days_to_stockout', 0) < 7:
            risk_factors.append(f"Low inventory coverage ({state['days_to_stockout']:.1f} days)")
        
        # Confidence factors
        confidence_factors = []
        if confidence > 80:
            confidence_factors.append("Strong historical performance for this action")
        elif confidence > 60:
            confidence_factors.append("Moderate confidence based on current conditions")
        else:
            confidence_factors.append("Limited historical data for this scenario")
        
        # Get enhanced explanation
        enhanced_explanation = self._get_enhanced_explanation(base_rationale, state, action)
        
        return RecommendationExplanation(
            rationale=base_rationale,
            enhanced_explanation=enhanced_explanation,
            risk_factors=risk_factors,
            confidence_factors=confidence_factors
        )
    
    def _calculate_cost_impact(self, action: str, state: Dict[str, Any]) -> float:
        """Calculate estimated cost impact for recommendation"""
        # Mock cost calculation based on action type
        cost_multipliers = {
            ActionType.DO_NOTHING: 0.0,
            ActionType.REORDER_STOCK: 0.05,  # 5% increase
            ActionType.SWITCH_SUPPLIER: 0.08,  # 8% increase
            ActionType.REROUTE_SHIPMENT: 0.06,  # 6% increase
            ActionType.EMERGENCY_RESTOCK: 0.15   # 15% increase
        }
        
        base_cost = state.get('predicted_demand', 500000) * 0.1  # Assume 10% of demand as base cost
        multiplier = cost_multipliers.get(ActionType(action), 0.05)
        
        return base_cost * multiplier
    
    def _get_alternatives(self, current_action: str) -> List[ActionType]:
        """Get alternative actions that were considered"""
        return [action for action in self.actions if action != ActionType(current_action)]
    
    def _get_historical_performance(self, action: str) -> Dict[str, float]:
        """Get historical performance data for action"""
        # Mock historical performance data
        return {
            'success_rate': np.random.uniform(0.7, 0.95),
            'avg_cost_impact': np.random.uniform(-0.1, 0.15),
            'avg_implementation_time': np.random.uniform(1, 14)  # days
        }
    
    def get_current_recommendation(self) -> RecommendationResponse:
        """Get current recommendation with XAI explanation"""
        try:
            # Check cache
            if (self.cached_recommendation and 
                self.last_update and 
                (datetime.now() - self.last_update).seconds < self.cache_timeout):
                logger.info("Returning cached recommendation")
                return self.cached_recommendation
            
            logger.info("Generating new recommendation")
            
            # Get current state
            state = self._get_current_state()
            
            # Translate to simulation scenario
            scenario = ScenarioRequest(
                demand_spike=0.0,
                supplier_failure=state.get('supplier_risk_score', 0) > 0.8,
                transport_delay=state.get('disruption_signal', 0) > 0.7,
                weather_disruption=False,
                inventory_adjustment=0.0
            )
            
            # Run baseline simulation
            baseline_result = _simulate_scenario(scenario)
            base_p_stockout = baseline_result['impact_analysis']['stockout_probability']
            base_p_delay = baseline_result['impact_analysis']['delay_probability']
            base_cost = baseline_result['impact_analysis']['cost_impact']

            # Make deterministic decision layer
            if base_p_stockout > 0.75:
                action_str = "emergency_restock"
            elif base_p_delay > 0.6 and scenario.supplier_failure:
                action_str = "switch_supplier"
            elif base_p_delay > 0.6 and scenario.transport_delay:
                action_str = "reroute_shipment"
            elif base_p_stockout > 0.4:
                action_str = "reorder_stock"
            else:
                action_str = "do_nothing"
                
            confidence = 88.0

            try:
                action = ActionType(action_str)
            except ValueError:
                action = ActionType.DO_NOTHING

            # Generate explanation
            explanation = self._generate_explanation(state, action_str, confidence)

            # Validation Layer
            val_scenario = ScenarioRequest(**scenario.dict())
            if action_str == "emergency_restock":
                val_scenario.inventory_adjustment = 0.5
            elif action_str == "reorder_stock":
                val_scenario.inventory_adjustment = 0.2
            elif action_str == "switch_supplier":
                val_scenario.supplier_failure = False
            elif action_str == "reroute_shipment":
                val_scenario.transport_delay = False
                
            val_result = _simulate_scenario(val_scenario)
            val_p_stockout = val_result['impact_analysis']['stockout_probability']
            val_p_delay = val_result['impact_analysis']['delay_probability']
            val_cost = val_result['impact_analysis']['cost_impact']
            
            stockout_reduction = base_p_stockout - val_p_stockout
            delay_reduction = base_p_delay - val_p_delay
            
            action_cost = self._calculate_cost_impact(action_str, state)
            cost_impact = action_cost # the response uses this for what it costs
            cost_saved = base_cost - val_cost - action_cost

            val_status = "VALIDATED_OPTIMAL" if cost_saved >= 0 or max(stockout_reduction, delay_reduction) > 0.1 else "VALIDATED_SUBOPTIMAL"

            validation_layer = ValidationMetrics(
                simulated_stockout_reduction=stockout_reduction,
                simulated_delay_reduction=delay_reduction,
                cost_saved=cost_saved,
                validation_status=val_status
            )
            
            # Get implementation time
            implementation_times = {
                ActionType.DO_NOTHING: None,
                ActionType.REORDER_STOCK: "3-5 days",
                ActionType.SWITCH_SUPPLIER: "7-14 days",
                ActionType.REROUTE_SHIPMENT: "2-4 days",
                ActionType.EMERGENCY_RESTOCK: "1-2 days"
            }
            
            # Get alternatives and historical performance
            alternatives = self._get_alternatives(action_str)
            historical_performance = self._get_historical_performance(action_str)
            
            # Create response
            response = RecommendationResponse(
                timestamp=datetime.now(),
                action=action,
                confidence=confidence,
                explanation=explanation,
                cost_impact=cost_impact,
                implementation_time=implementation_times.get(action),
                alternatives=alternatives,
                historical_performance=historical_performance,
                validation_layer=validation_layer
            )
            
            # Cache the response
            self.cached_recommendation = response
            self.last_update = datetime.now()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            raise
    
    def get_recommendation_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical recommendation data"""
        try:
            history = []
            current_time = datetime.now()
            
            for day in range(days):
                date = current_time - timedelta(days=day)
                
                # Generate mock historical data
                action = np.random.choice(self.actions)
                confidence = np.random.uniform(60, 95)
                
                history.append({
                    'date': date.isoformat(),
                    'action': action.value,
                    'confidence': confidence,
                    'cost_impact': np.random.uniform(-10000, 50000)
                })
            
            return sorted(history, key=lambda x: x['date'])
            
        except Exception as e:
            logger.error(f"Error getting recommendation history: {e}")
            return []
    
    def clear_cache(self):
        """Clear the recommendation service cache"""
        self.cached_recommendation = None
        self.last_update = None
        logger.info("Recommendation service cache cleared")
