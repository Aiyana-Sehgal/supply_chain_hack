"""
Alert Service

Business logic for live risk alerts with NewsData.io integration.
Integrates with existing layers for real-time risk monitoring.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import uuid
import random

from ..models import RiskAlert, RiskAlertsResponse, AlertSeverity

logger = logging.getLogger(__name__)

class AlertService:
    """Service for live risk alerts and monitoring"""
    
    def __init__(self):
        """Initialize alert service"""
        self.cache_timeout = 300  # 5 minutes
        self.last_update = None
        self.cached_alerts = None
        
        # Alert thresholds
        self.risk_thresholds = {
            'supplier_risk': 0.7,
            'disruption_signal': 0.6,
            'inventory_ratio': 0.2,
            'days_to_stockout': 7
        }
        
        # Alert types
        self.alert_types = [
            "supplier_risk_high",
            "disruption_detected", 
            "inventory_low",
            "stockout_imminent",
            "cost_spike",
            "weather_disruption",
            "transport_delay"
        ]
        
        # Mock alert storage (in production, this would be a database)
        self.alert_store = []
    
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
    
    def _get_severity_from_risk_score(self, risk_score: float) -> AlertSeverity:
        """Convert risk score to alert severity"""
        if risk_score >= 0.9:
            return AlertSeverity.CRITICAL
        elif risk_score >= 0.7:
            return AlertSeverity.HIGH
        elif risk_score >= 0.4:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        return str(uuid.uuid4())
    
    def _check_supplier_risk_alerts(self, state: Dict[str, Any]) -> List[RiskAlert]:
        """Check for supplier risk alerts"""
        alerts = []
        supplier_risk = state.get('supplier_risk_score', 0)
        
        if supplier_risk >= self.risk_thresholds['supplier_risk']:
            severity = self._get_severity_from_risk_score(supplier_risk)
            
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                alert_type="supplier_risk_high",
                severity=severity,
                title="Supplier Risk Alert",
                message=f"Supplier risk score is {supplier_risk:.1%}, exceeding threshold of {self.risk_thresholds['supplier_risk']:.1%}",
                region="All Regions",
                supplier="Primary Supplier",
                risk_score=supplier_risk,
                threshold=self.risk_thresholds['supplier_risk'],
                created_at=datetime.now(),
                resolved=False
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_disruption_alerts(self, state: Dict[str, Any]) -> List[RiskAlert]:
        """Check for disruption signal alerts"""
        alerts = []
        disruption_signal = state.get('disruption_signal', 0)
        
        if disruption_signal >= self.risk_thresholds['disruption_signal']:
            severity = self._get_severity_from_risk_score(disruption_signal)
            
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                alert_type="disruption_detected",
                severity=severity,
                title="Disruption Signal Alert",
                message=f"Disruption signal is {disruption_signal:.1%}, indicating potential supply chain disruptions",
                region="Affected Regions",
                risk_score=disruption_signal,
                threshold=self.risk_thresholds['disruption_signal'],
                created_at=datetime.now(),
                resolved=False
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_inventory_alerts(self, state: Dict[str, Any]) -> List[RiskAlert]:
        """Check for inventory-related alerts"""
        alerts = []
        inventory = state.get('current_inventory', 0)
        demand = state.get('predicted_demand', 1)
        days_to_stockout = state.get('days_to_stockout', 0)
        
        # Low inventory alert
        inventory_ratio = inventory / demand if demand > 0 else 0
        if inventory_ratio <= self.risk_thresholds['inventory_ratio']:
            severity = AlertSeverity.CRITICAL if inventory_ratio <= 0.1 else AlertSeverity.HIGH
            
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                alert_type="inventory_low",
                severity=severity,
                title="Low Inventory Alert",
                message=f"Inventory ratio is {inventory_ratio:.1%}, below threshold of {self.risk_thresholds['inventory_ratio']:.1%}",
                risk_score=inventory_ratio,
                threshold=self.risk_thresholds['inventory_ratio'],
                created_at=datetime.now(),
                resolved=False
            )
            alerts.append(alert)
        
        # Stockout imminent alert
        if days_to_stockout <= self.risk_thresholds['days_to_stockout']:
            severity = AlertSeverity.CRITICAL if days_to_stockout <= 3 else AlertSeverity.HIGH
            stockout_threshold_score = 1.0 - (self.risk_thresholds['days_to_stockout'] / 30.0)
            
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                alert_type="stockout_imminent",
                severity=severity,
                title="Stockout Imminent Alert",
                message=f"Stockout expected in {days_to_stockout:.1f} days",
                risk_score=1.0 - (days_to_stockout / 30),  # Higher risk for fewer days
                threshold=stockout_threshold_score,
                created_at=datetime.now(),
                resolved=False
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_mock_news_alerts(self) -> List[RiskAlert]:
        """Generate mock news-based alerts (would integrate with NewsData.io in production)"""
        alerts = []
        
        # Mock news alerts
        mock_news_items = [
            {
                "type": "weather_disruption",
                "title": "Weather Disruption Alert",
                "message": "Heavy rainfall expected in North India affecting transport routes",
                "region": "North India",
                "severity": AlertSeverity.HIGH
            },
            {
                "type": "transport_delay",
                "title": "Transport Delay Alert", 
                "message": "Major highway closure reported in East India",
                "region": "East India",
                "severity": AlertSeverity.MEDIUM
            }
        ]
        
        for news_item in mock_news_items:
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                alert_type=news_item["type"],
                severity=news_item["severity"],
                title=news_item["title"],
                message=news_item["message"],
                region=news_item["region"],
                created_at=datetime.now(),
                resolved=False
            )
            alerts.append(alert)
        
        return alerts
    
    def _generate_mock_alerts(self) -> List[RiskAlert]:
        """Generate mock alerts for demonstration"""
        alerts = []
        current_time = datetime.now()
        
        # Generate some historical alerts
        for i in range(5):
            created_time = current_time - timedelta(hours=i*2)
            severity = random.choice(list(AlertSeverity))
            alert_type = np.random.choice(self.alert_types)
            
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                alert_type=alert_type,
                severity=severity,
                title=f"{alert_type.replace('_', ' ').title()} Alert",
                message=f"Mock alert message for {alert_type}",
                region=np.random.choice(["North India", "South India", "East India", "West India"]),
                supplier=np.random.choice(["SUP001", "SUP002", "SUP003"]) if np.random.random() > 0.5 else None,
                risk_score=np.random.uniform(0.3, 0.9),
                threshold=np.random.uniform(0.5, 0.8),
                created_at=created_time,
                resolved=np.random.random() > 0.7,  # 30% chance of being resolved
                resolved_at=created_time + timedelta(hours=np.random.randint(1, 12)) if np.random.random() > 0.7 else None
            )
            alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(self) -> RiskAlertsResponse:
        """Get all active risk alerts"""
        try:
            # Check cache
            if (self.cached_alerts and 
                self.last_update and 
                (datetime.now() - self.last_update).seconds < self.cache_timeout):
                logger.info("Returning cached alerts")
                return self.cached_alerts
            
            logger.info("Generating new risk alerts")
            
            # Get current state
            state = self._get_current_state()
            
            # Generate alerts from different sources
            all_alerts = []
            
            # System-generated alerts
            all_alerts.extend(self._check_supplier_risk_alerts(state))
            all_alerts.extend(self._check_disruption_alerts(state))
            all_alerts.extend(self._check_inventory_alerts(state))
            
            # News-based alerts
            all_alerts.extend(self._generate_mock_news_alerts())
            
            # Mock historical alerts
            all_alerts.extend(self._generate_mock_alerts())
            
            # Filter active alerts (not resolved)
            active_alerts = [alert for alert in all_alerts if not alert.resolved]
            
            # Sort by creation time (newest first)
            active_alerts.sort(key=lambda x: x.created_at, reverse=True)
            
            # Count alerts by severity
            critical_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL])
            high_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.HIGH])
            medium_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.MEDIUM])
            low_alerts = len([a for a in active_alerts if a.severity == AlertSeverity.LOW])
            
            # Create response
            response = RiskAlertsResponse(
                timestamp=datetime.now(),
                alerts=active_alerts,
                total_alerts=len(active_alerts),
                critical_alerts=critical_alerts,
                high_alerts=high_alerts,
                medium_alerts=medium_alerts,
                low_alerts=low_alerts,
                last_updated=datetime.now()
            )
            
            # Cache the response
            self.cached_alerts = response
            self.last_update = datetime.now()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating risk alerts: {e}")
            raise
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        try:
            # Find and resolve alert
            for alert in self.alert_store:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Alert {alert_id} resolved")
                    return True
            
            logger.warning(f"Alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return False
    
    def get_alert_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get alert history for the specified period"""
        try:
            history = []
            current_time = datetime.now()
            
            # Generate mock historical data
            for day in range(days):
                date = current_time - timedelta(days=day)
                daily_alerts = np.random.randint(0, 10)
                
                history.append({
                    'date': date.isoformat(),
                    'total_alerts': daily_alerts,
                    'critical_alerts': np.random.randint(0, daily_alerts),
                    'resolved_alerts': np.random.randint(0, daily_alerts)
                })
            
            return sorted(history, key=lambda x: x['date'])
            
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []
    
    def clear_cache(self):
        """Clear the alert service cache"""
        self.cached_alerts = None
        self.last_update = None
        logger.info("Alert service cache cleared")
