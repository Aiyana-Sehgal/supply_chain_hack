"""
Cost Analysis API Endpoint

Provides cost impact analysis for recommended decisions.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from ..models import CostImpactResponse, ActionType
from ..services.cost_service import CostService

router = APIRouter()

# Initialize service
cost_service = CostService()
logger = logging.getLogger(__name__)

@router.get("/cost-impact", response_model=CostImpactResponse)
async def get_cost_impact(action: str = Query(..., description="Action to analyze")):
    """
    Get cost impact analysis for a specific action
    
    Args:
        action: Action to analyze (do_nothing, reorder_stock, switch_supplier, reroute_shipment, emergency_restock)
    
    Returns:
        CostImpactResponse: Complete cost impact analysis including:
        - Current vs projected costs
        - Cost breakdown by category
        - ROI estimate and payback period
        - Key assumptions
    """
    try:
        logger.info(f"Cost impact endpoint called for action: {action}")
        
        # Validate action
        try:
            action_enum = ActionType(action)
        except ValueError:
            available_actions = [a.value for a in ActionType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid action '{action}'. Available actions: {available_actions}"
            )
        
        # Get cost impact analysis
        analysis = cost_service.get_cost_impact_analysis(action_enum)
        
        logger.info(f"Cost impact analysis completed for {action}: {analysis.total_cost_change:.2f} total change")
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in cost impact endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating cost impact analysis: {str(e)}")

@router.get("/cost-impact/trends")
async def get_cost_trends(days: int = Query(default=30, ge=1, le=365)):
    """
    Get historical cost trends by category
    
    Args:
        days: Number of days of historical data to return (1-365)
    
    Returns:
        Dictionary with cost trend data by category
    """
    try:
        logger.info(f"Cost trends endpoint called with days={days}")
        
        # Get cost trends
        trends = cost_service.get_cost_trends(days)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "days": days,
            "trends": trends
        }
        
    except Exception as e:
        logger.error(f"Error in cost trends endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting cost trends: {str(e)}")

@router.get("/cost-impact/actions")
async def get_available_actions():
    """
    Get list of available actions for cost analysis
    
    Returns:
        List of available actions with descriptions
    """
    try:
        logger.info("Available actions endpoint called")
        
        actions = []
        for action in ActionType:
            actions.append({
                "action": action.value,
                "description": action.value.replace("_", " ").title()
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "actions": actions
        }
        
    except Exception as e:
        logger.error(f"Error in available actions endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting available actions: {str(e)}")

@router.post("/cost-impact/cache/clear")
async def clear_cost_cache():
    """
    Clear the cost analysis cache
    
    Returns:
        Success message
    """
    try:
        logger.info("Clear cost cache endpoint called")
        
        # Clear cache
        cost_service.clear_cache()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Cost analysis cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cost cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing cost cache: {str(e)}")
