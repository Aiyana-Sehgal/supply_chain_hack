"""
Risk Service

Business logic for risk heatmap calculations and regional/supplier risk assessment.
Integrates with existing Layers 1-5 for risk data processing.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from ..models import RegionRisk, SupplierRisk, RiskLevel

logger = logging.getLogger(__name__)

class RiskService:
    """Service for risk assessment and heatmap data generation"""
    
    def __init__(self):
        """Initialize risk service"""
        self.cache_timeout = 300  # 5 minutes
        self.last_update = None
        self.cached_data = None
        
        # Mock data for regions and suppliers (in production, this would come from database)
        self.regions = [
            "North India", "South India", "East India", "West India", 
            "Central India", "Northeast India"
        ]
        
        self.suppliers = [
            {"id": "SUP001", "name": "Global Logistics Ltd", "region": "North India"},
            {"id": "SUP002", "name": "Regional Transport Co", "region": "South India"},
            {"id": "SUP003", "name": "Express Freight Services", "region": "East India"},
            {"id": "SUP004", "name": "National Supply Chain", "region": "West India"},
            {"id": "SUP005", "name": "Quick Delivery Systems", "region": "Central India"},
            {"id": "SUP006", "name": "Mountain Transport", "region": "Northeast India"},
        ]
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """Convert risk score to risk level category"""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_current_state_data(self) -> Dict[str, Any]:
        """Get current state data from existing layers"""
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
    
    def _generate_mock_risk_data(self) -> Dict[str, Any]:
        """Generate mock risk data for demonstration"""
        current_time = datetime.now()
        
        # Generate regional risk data
        regions_data = []
        for region in self.regions:
            # Base risk with some randomness
            base_risk = np.random.uniform(0.2, 0.8)
            disruption_signal = np.random.uniform(0.1, 0.9)
            
            regions_data.append(RegionRisk(
                region=region,
                risk_score=base_risk,
                risk_level=self._get_risk_level(base_risk),
                supplier_count=np.random.randint(1, 4),
                disruption_signal=disruption_signal,
                last_updated=current_time
            ))
        
        # Generate supplier risk data
        suppliers_data = []
        for supplier in self.suppliers:
            # Individual supplier risk
            supplier_risk = np.random.uniform(0.1, 0.9)
            inventory_ratio = np.random.uniform(0.1, 0.8)
            days_to_stockout = np.random.uniform(3, 30)
            
            suppliers_data.append(SupplierRisk(
                supplier_id=supplier["id"],
                supplier_name=supplier["name"],
                region=supplier["region"],
                risk_score=supplier_risk,
                risk_level=self._get_risk_level(supplier_risk),
                inventory_ratio=inventory_ratio,
                days_to_stockout=days_to_stockout,
                last_updated=current_time
            ))
        
        # Calculate overall risk level
        all_risk_scores = [r.risk_score for r in regions_data] + [s.risk_score for s in suppliers_data]
        avg_risk = np.mean(all_risk_scores)
        overall_risk_level = self._get_risk_level(avg_risk)
        
        return {
            'regions': regions_data,
            'suppliers': suppliers_data,
            'overall_risk_level': overall_risk_level,
            'total_suppliers': len(suppliers_data),
            'high_risk_regions': len([r for r in regions_data if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]),
            'high_risk_suppliers': len([s for s in suppliers_data if s.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]])
        }
    
    def get_risk_heatmap_data(self) -> Dict[str, Any]:
        """Get risk heatmap data for API response"""
        try:
            # Check cache
            if (self.cached_data and 
                self.last_update and 
                (datetime.now() - self.last_update).seconds < self.cache_timeout):
                logger.info("Returning cached risk heatmap data")
                return self.cached_data
            
            logger.info("Generating new risk heatmap data")
            
            # Get current state from layers
            state_data = self._get_current_state_data()
            
            # Generate risk data (using mock data for demonstration)
            risk_data = self._generate_mock_risk_data()
            
            # Enhance with real state data where possible
            if state_data:
                # Update supplier risk with real data
                for supplier in risk_data['suppliers']:
                    if supplier.region == "North India":  # Example mapping
                        supplier.risk_score = state_data.get('supplier_risk_score', supplier.risk_score)
                        supplier.risk_level = self._get_risk_level(supplier.risk_score)
                        supplier.disruption_signal = state_data.get('disruption_signal', 0.5)
            
            # Cache the data
            self.cached_data = risk_data
            self.last_update = datetime.now()
            
            return risk_data
            
        except Exception as e:
            logger.error(f"Error generating risk heatmap data: {e}")
            raise
    
    def get_regional_risk_trends(self, days: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """Get historical risk trends for regions"""
        try:
            trends = {}
            current_time = datetime.now()
            
            for region in self.regions:
                trend_data = []
                for day in range(days):
                    date = current_time - timedelta(days=day)
                    # Generate mock historical data
                    risk_score = np.random.uniform(0.2, 0.8)
                    
                    trend_data.append({
                        'date': date.isoformat(),
                        'risk_score': risk_score,
                        'risk_level': self._get_risk_level(risk_score).value
                    })
                
                trends[region] = sorted(trend_data, key=lambda x: x['date'])
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting regional risk trends: {e}")
            return {}
    
    def get_supplier_risk_history(self, supplier_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical risk data for a specific supplier"""
        try:
            history = []
            current_time = datetime.now()
            
            for day in range(days):
                date = current_time - timedelta(days=day)
                # Generate mock historical data
                risk_score = np.random.uniform(0.1, 0.9)
                
                history.append({
                    'date': date.isoformat(),
                    'risk_score': risk_score,
                    'risk_level': self._get_risk_level(risk_score).value,
                    'inventory_ratio': np.random.uniform(0.1, 0.8),
                    'days_to_stockout': np.random.uniform(3, 30)
                })
            
            return sorted(history, key=lambda x: x['date'])
            
        except Exception as e:
            logger.error(f"Error getting supplier risk history: {e}")
            return []
    
    def clear_cache(self):
        """Clear the risk service cache"""
        self.cached_data = None
        self.last_update = None
        logger.info("Risk service cache cleared")
