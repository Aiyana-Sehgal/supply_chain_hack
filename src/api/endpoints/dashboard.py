"""
Dashboard API Endpoint

Comprehensive dashboard aggregation endpoint for all Layer 6 components.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging

from ..models import DashboardResponse
from ..services.risk_service import RiskService
from ..services.recommendation_service import RecommendationService
from ..services.cost_service import CostService
from ..services.alert_service import AlertService

router = APIRouter()

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize services
risk_service = RiskService()
recommendation_service = RecommendationService()
cost_service = CostService()
alert_service = AlertService()

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard():
    """
    Get comprehensive dashboard data aggregating all Layer 6 components
    
    Returns:
        DashboardResponse: Complete dashboard data including:
        - Risk heatmap visualization data
        - Current recommendations with XAI explanations
        - Cost impact analysis
        - Live risk alerts
        - System health status
    """
    try:
        logger.info("Dashboard endpoint called - aggregating all components")
        
        # Get data from all services
        risk_data = risk_service.get_risk_heatmap_data()
        recommendation = recommendation_service.get_current_recommendation()
        
        # Get cost impact for current recommendation
        cost_analysis = cost_service.get_cost_impact_analysis(recommendation.action)
        
        # Get active alerts
        alerts_response = alert_service.get_active_alerts()
        
        # System health check
        system_health = {
            "risk_service": "operational",
            "recommendation_service": "operational", 
            "cost_service": "operational",
            "alert_service": "operational",
            "cache_status": "active",
            "overall_status": "healthy"
        }
        
        # Create response models
        from ..models import RiskHeatmapResponse, RiskAlertsResponse
        
        risk_heatmap_response = RiskHeatmapResponse(
            timestamp=datetime.now(),
            status="success",
            regions=risk_data['regions'],
            suppliers=risk_data['suppliers'],
            overall_risk_level=risk_data['overall_risk_level'],
            total_suppliers=risk_data['total_suppliers'],
            high_risk_regions=risk_data['high_risk_regions'],
            high_risk_suppliers=risk_data['high_risk_suppliers']
        )
        
        # Create dashboard response
        dashboard_response = DashboardResponse(
            timestamp=datetime.now(),
            status="success",
            risk_heatmap=risk_heatmap_response,
            recommendations=recommendation,
            cost_impact=cost_analysis,
            risk_alerts=alerts_response,
            system_health=system_health,
            last_data_update=datetime.now()
        )
        
        logger.info(f"Dashboard data aggregated successfully: {len(risk_data['regions'])} regions, {len(risk_data['suppliers'])} suppliers, {alerts_response.total_alerts} alerts")
        return dashboard_response
        
    except Exception as e:
        logger.error(f"Error in dashboard endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating dashboard data: {str(e)}")

@router.get("/dashboard/summary")
async def get_dashboard_summary():
    """
    Get dashboard summary with key metrics
    
    Returns:
        Dictionary with key dashboard metrics
    """
    try:
        logger.info("Dashboard summary endpoint called")
        
        # Get summary data from services
        risk_data = risk_service.get_risk_heatmap_data()
        recommendation = recommendation_service.get_current_recommendation()
        alerts_response = alert_service.get_active_alerts()
        
        # Calculate summary metrics
        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "key_metrics": {
                "overall_risk_level": risk_data['overall_risk_level'].value,
                "total_suppliers": risk_data['total_suppliers'],
                "high_risk_suppliers": risk_data['high_risk_suppliers'],
                "current_recommendation": recommendation.action.value,
                "recommendation_confidence": recommendation.confidence,
                "total_cost_impact": recommendation.cost_impact,
                "active_alerts": alerts_response.total_alerts,
                "critical_alerts": alerts_response.critical_alerts
            },
            "health_status": {
                "risk_monitoring": "active",
                "recommendation_engine": "active",
                "cost_analysis": "active",
                "alert_system": "active"
            },
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info("Dashboard summary generated successfully")
        return summary
        
    except Exception as e:
        logger.error(f"Error in dashboard summary endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating dashboard summary: {str(e)}")

@router.post("/dashboard/refresh")
async def refresh_dashboard():
    """
    Refresh dashboard data and clear all caches
    
    Returns:
        Success message with updated timestamp
    """
    try:
        logger.info("Dashboard refresh endpoint called")
        
        # Clear all service caches
        risk_service.clear_cache()
        recommendation_service.clear_cache()
        cost_service.clear_cache()
        alert_service.clear_cache()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Dashboard data refreshed successfully",
            "caches_cleared": [
                "risk_service",
                "recommendation_service", 
                "cost_service",
                "alert_service"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error refreshing dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error refreshing dashboard: {str(e)}")
