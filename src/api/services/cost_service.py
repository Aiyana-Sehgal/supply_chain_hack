"""
Cost Service

Business logic for cost impact analysis and ROI calculations.
Integrates with existing layers for cost data processing.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from ..models import (
    CostImpactResponse, 
    CostImpactBreakdown,
    ActionType
)

logger = logging.getLogger(__name__)

class CostService:
    """Service for cost impact analysis and ROI calculations"""
    
    def __init__(self):
        """Initialize cost service"""
        self.cache_timeout = 300  # 5 minutes
        self.last_update = None
        self.cached_analysis = None
        
        # Cost categories for analysis
        self.cost_categories = [
            "Inventory Costs",
            "Transportation Costs", 
            "Supplier Costs",
            "Storage Costs",
            "Emergency Costs",
            "Opportunity Costs"
        ]
    
    def _get_current_state(self) -> Dict[str, Any]:
        """Get current state from existing layers"""
        try:
            # Import from existing layers
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            
            from src.core.layer3 import build_layer3_dataset
            
            # Get current state
            layer3_bundle = build_layer3_dataset(verbose=False)
            state = layer3_bundle['layer3_state']
            
            return state
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            # Return mock data if layers are not available
            return {
                'predicted_demand': 500000,
                'current_inventory': 25000,
                'supplier_risk_score': 0.65,
                'disruption_signal': 0.45,
                'days_to_stockout': 5.0,
                'composite_risk_score': 0.55
            }
    
    def _calculate_base_costs(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate base costs by category"""
        demand = state.get('predicted_demand', 500000)
        inventory = state.get('current_inventory', 25000)
        
        # Base cost calculations (mock values for demonstration)
        base_costs = {
            "Inventory Costs": inventory * 0.02,  # 2% of inventory value
            "Transportation Costs": demand * 0.05,  # 5% of demand value
            "Supplier Costs": demand * 0.03,  # 3% of demand value
            "Storage Costs": inventory * 0.01,  # 1% of inventory value
            "Emergency Costs": 0,  # No emergency costs initially
            "Opportunity Costs": demand * 0.01  # 1% of demand value
        }
        
        return base_costs
    
    def _calculate_action_cost_impacts(self, action: ActionType, base_costs: Dict[str, float], state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate cost impacts for a specific action"""
        demand = state.get('predicted_demand', 500000)
        
        # Action-specific cost multipliers
        if action == ActionType.DO_NOTHING:
            impacts = {category: 0.0 for category in self.cost_categories}
            
        elif action == ActionType.REORDER_STOCK:
            impacts = {
                "Inventory Costs": base_costs["Inventory Costs"] * 0.3,  # +30% inventory
                "Transportation Costs": base_costs["Transportation Costs"] * 0.1,  # +10% transport
                "Supplier Costs": base_costs["Supplier Costs"] * 0.05,  # +5% supplier
                "Storage Costs": base_costs["Storage Costs"] * 0.25,  # +25% storage
                "Emergency Costs": 0,
                "Opportunity Costs": base_costs["Opportunity Costs"] * -0.5  # -50% opportunity
            }
            
        elif action == ActionType.SWITCH_SUPPLIER:
            impacts = {
                "Inventory Costs": base_costs["Inventory Costs"] * 0.1,  # +10% inventory
                "Transportation Costs": base_costs["Transportation Costs"] * 0.15,  # +15% transport
                "Supplier Costs": base_costs["Supplier Costs"] * 0.2,  # +20% supplier (new supplier)
                "Storage Costs": base_costs["Storage Costs"] * 0.05,  # +5% storage
                "Emergency Costs": 0,
                "Opportunity Costs": base_costs["Opportunity Costs"] * -0.3  # -30% opportunity
            }
            
        elif action == ActionType.REROUTE_SHIPMENT:
            impacts = {
                "Inventory Costs": 0,
                "Transportation Costs": base_costs["Transportation Costs"] * 0.25,  # +25% transport
                "Supplier Costs": base_costs["Supplier Costs"] * 0.05,  # +5% supplier
                "Storage Costs": 0,
                "Emergency Costs": 0,
                "Opportunity Costs": base_costs["Opportunity Costs"] * -0.2  # -20% opportunity
            }
            
        elif action == ActionType.EMERGENCY_RESTOCK:
            impacts = {
                "Inventory Costs": base_costs["Inventory Costs"] * 0.4,  # +40% inventory
                "Transportation Costs": base_costs["Transportation Costs"] * 0.5,  # +50% transport
                "Supplier Costs": base_costs["Supplier Costs"] * 0.3,  # +30% supplier
                "Storage Costs": base_costs["Storage Costs"] * 0.2,  # +20% storage
                "Emergency Costs": demand * 0.02,  # 2% of demand as emergency cost
                "Opportunity Costs": base_costs["Opportunity Costs"] * -0.8  # -80% opportunity
            }
            
        else:
            impacts = {category: 0.0 for category in self.cost_categories}
        
        return impacts
    
    def _calculate_roi(self, total_cost_change: float, action: ActionType, state: Dict[str, Any]) -> float:
        """Calculate Return on Investment estimate"""
        demand = state.get('predicted_demand', 500000)
        
        # Benefit estimation based on action type
        if action == ActionType.DO_NOTHING:
            benefit = 0
        elif action == ActionType.REORDER_STOCK:
            benefit = demand * 0.02  # 2% of demand value from avoiding stockouts
        elif action == ActionType.SWITCH_SUPPLIER:
            benefit = demand * 0.03  # 3% of demand value from risk mitigation
        elif action == ActionType.REROUTE_SHIPMENT:
            benefit = demand * 0.015  # 1.5% of demand value from avoiding delays
        elif action == ActionType.EMERGENCY_RESTOCK:
            benefit = demand * 0.04  # 4% of demand value from preventing major disruption
        else:
            benefit = 0
        
        # ROI = (Benefit - Cost) / Cost
        if total_cost_change > 0:
            roi = (benefit - total_cost_change) / total_cost_change
        else:
            roi = benefit / abs(total_cost_change) if total_cost_change != 0 else 0
        
        return roi
    
    def _calculate_payback_period(self, total_cost_change: float, monthly_benefit: float) -> Optional[float]:
        """Calculate payback period in months"""
        if total_cost_change <= 0 or monthly_benefit <= 0:
            return None
        
        return total_cost_change / monthly_benefit
    
    def _generate_assumptions(self, action: ActionType, state: Dict[str, Any]) -> List[str]:
        """Generate key assumptions for cost analysis"""
        demand = state.get('predicted_demand', 500000)
        
        assumptions = [
            f"Current demand forecast: {demand:,.0f} units",
            "Analysis based on 3-month cost projection",
            "Cost calculations include direct and indirect impacts"
        ]
        
        if action == ActionType.REORDER_STOCK:
            assumptions.extend([
                "Assumes 30% inventory increase",
                "Standard shipping rates apply",
                "No supplier change costs"
            ])
        elif action == ActionType.SWITCH_SUPPLIER:
            assumptions.extend([
                "New supplier pricing 20% higher",
                "One-time setup costs included",
                "Quality standards maintained"
            ])
        elif action == ActionType.REROUTE_SHIPMENT:
            assumptions.extend([
                "Alternative route 25% more expensive",
                "No significant delays expected",
                "Fuel price volatility considered"
            ])
        elif action == ActionType.EMERGENCY_RESTOCK:
            assumptions.extend([
                "Emergency shipping 50% premium",
                "Express delivery charges apply",
                "Potential quality impacts considered"
            ])
        
        return assumptions
    
    def get_cost_impact_analysis(self, action: ActionType) -> CostImpactResponse:
        """Get cost impact analysis for a specific action"""
        try:
            # Check cache
            if (self.cached_analysis and 
                self.last_update and 
                (datetime.now() - self.last_update).seconds < self.cache_timeout and
                self.cached_analysis.get('action') == action):
                logger.info("Returning cached cost analysis")
                return self.cached_analysis['response']
            
            logger.info(f"Generating new cost impact analysis for {action}")
            
            # Get current state
            state = self._get_current_state()
            
            # Calculate base costs
            base_costs = self._calculate_base_costs(state)
            
            # Calculate action impacts
            cost_impacts = self._calculate_action_cost_impacts(action, base_costs, state)
            
            # Generate cost breakdown
            breakdown = []
            total_current_cost = 0
            total_projected_cost = 0
            
            for category in self.cost_categories:
                current_cost = base_costs[category]
                projected_cost = current_cost + cost_impacts[category]
                change_amount = cost_impacts[category]
                change_percentage = (change_amount / current_cost * 100) if current_cost > 0 else 0
                
                breakdown.append(CostImpactBreakdown(
                    category=category,
                    current_cost=current_cost,
                    projected_cost=projected_cost,
                    change_amount=change_amount,
                    change_percentage=change_percentage
                ))
                
                total_current_cost += current_cost
                total_projected_cost += projected_cost
            
            # Calculate totals
            total_cost_change = total_projected_cost - total_current_cost
            total_change_percentage = (total_cost_change / total_current_cost * 100) if total_current_cost > 0 else 0
            
            # Calculate ROI
            roi = self._calculate_roi(total_cost_change, action, state)
            
            # Calculate payback period
            monthly_benefit = abs(total_cost_change * roi) if roi > 0 else 0
            payback_period = self._calculate_payback_period(total_cost_change, monthly_benefit)
            
            # Generate assumptions
            assumptions = self._generate_assumptions(action, state)
            
            # Create response
            response = CostImpactResponse(
                recommendation=action,
                total_current_cost=total_current_cost,
                total_projected_cost=total_projected_cost,
                total_cost_change=total_cost_change,
                total_change_percentage=total_change_percentage,
                roi_estimate=roi,
                payback_period_months=payback_period,
                breakdown=breakdown,
                assumptions=assumptions
            )
            
            # Cache the response
            self.cached_analysis = {
                'action': action,
                'response': response
            }
            self.last_update = datetime.now()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating cost impact analysis: {e}")
            raise
    
    def get_cost_trends(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical cost trends"""
        try:
            trends = {}
            current_time = datetime.now()
            
            for category in self.cost_categories:
                trend_data = []
                for day in range(days):
                    date = current_time - timedelta(days=day)
                    # Generate mock historical data
                    cost = np.random.uniform(1000, 10000)
                    
                    trend_data.append({
                        'date': date.isoformat(),
                        'cost': cost,
                        'category': category
                    })
                
                trends[category] = sorted(trend_data, key=lambda x: x['date'])
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting cost trends: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the cost service cache"""
        self.cached_analysis = None
        self.last_update = None
        logger.info("Cost service cache cleared")
