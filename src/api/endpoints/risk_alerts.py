"""
Risk Alerts API Endpoint

Provides live risk alerts with NewsData.io integration.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
import asyncio
import logging

from ..models import RiskAlertsResponse
from ..services.alert_service import AlertService

router = APIRouter()

# Initialize service
alert_service = AlertService()
logger = logging.getLogger(__name__)

@router.get("/risk-alerts", response_model=RiskAlertsResponse)
async def get_risk_alerts():
    """
    Get live risk alerts with NewsData.io integration
    
    Returns:
        RiskAlertsResponse: Complete risk alerts data including:
        - Active risk alerts with severity levels
        - Alert statistics and counts
        - Alert metadata and timestamps
    """
    try:
        logger.info("Risk alerts endpoint called")

        alerts_response = await asyncio.to_thread(alert_service.get_active_alerts)

        logger.info(
            f"Risk alerts generated: {alerts_response.total_alerts} total, {alerts_response.critical_alerts} critical"
        )
        return alerts_response

    except Exception as e:
        logger.error(f"Error in risk alerts endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating risk alerts: {str(e)}")

@router.get("/risk-alerts/history")
async def get_alert_history(days: int = Query(default=30, ge=1, le=365)):
    """
    Get historical alert data
    
    Args:
        days: Number of days of historical data to return (1-365)
    
    Returns:
        List of historical alert statistics
    """
    try:
        logger.info(f"Alert history endpoint called with days={days}")
        
        # Get alert history
        history = alert_service.get_alert_history(days)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "days": days,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error in alert history endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting alert history: {str(e)}")

@router.post("/risk-alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    Mark an alert as resolved
    
    Args:
        alert_id: Unique identifier of the alert to resolve
    
    Returns:
        Success message
    """
    try:
        logger.info(f"Resolve alert endpoint called for alert_id: {alert_id}")
        
        # Resolve alert
        success = alert_service.resolve_alert(alert_id)
        
        if success:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "message": f"Alert {alert_id} resolved successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail=f"Error resolving alert: {str(e)}")

@router.get("/risk-alerts/types")
async def get_alert_types():
    """
    Get list of available alert types
    
    Returns:
        List of alert types with descriptions
    """
    try:
        logger.info("Alert types endpoint called")
        
        alert_types = [
            {
                "type": "supplier_risk_high",
                "description": "Supplier risk score exceeds threshold",
                "severity": "High to Critical"
            },
            {
                "type": "disruption_detected",
                "description": "Disruption signal indicates potential supply chain issues",
                "severity": "Medium to High"
            },
            {
                "type": "inventory_low",
                "description": "Inventory levels below safe threshold",
                "severity": "High to Critical"
            },
            {
                "type": "stockout_imminent",
                "description": "Stockout expected within few days",
                "severity": "Critical"
            },
            {
                "type": "weather_disruption",
                "description": "Weather-related disruptions detected",
                "severity": "Medium to High"
            },
            {
                "type": "transport_delay",
                "description": "Transportation delays affecting supply chain",
                "severity": "Medium"
            }
        ]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "alert_types": alert_types
        }
        
    except Exception as e:
        logger.error(f"Error getting alert types: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting alert types: {str(e)}")

@router.post("/risk-alerts/cache/clear")
async def clear_alert_cache():
    """
    Clear the risk alerts cache
    
    Returns:
        Success message
    """
    try:
        logger.info("Clear alert cache endpoint called")
        
        # Clear cache
        alert_service.clear_cache()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Risk alerts cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing alert cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing alert cache: {str(e)}")
