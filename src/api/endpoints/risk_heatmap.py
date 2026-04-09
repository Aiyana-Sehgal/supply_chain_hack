"""
Risk Heatmap API Endpoint

Provides risk visualization data by region and supplier.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
from datetime import datetime
import logging

from ..models import RiskHeatmapResponse
from ..services.risk_service import RiskService

router = APIRouter()

# Initialize service
risk_service = RiskService()
logger = logging.getLogger(__name__)

@router.get("/risk-heatmap", response_model=RiskHeatmapResponse)
async def get_risk_heatmap():
    """
    Get risk heatmap data with regional and supplier risk scores
    
    Returns:
        RiskHeatmapResponse: Complete risk heatmap data including:
        - Regional risk data with color coding
        - Supplier-specific risk mapping
        - Overall risk level assessment
        - Risk statistics and counts
    """
    try:
        logger.info("Risk heatmap endpoint called")
        
        # Get risk heatmap data
        risk_data = risk_service.get_risk_heatmap_data()
        
        # Create response
        response = RiskHeatmapResponse(
            timestamp=datetime.now(),
            status="success",
            regions=risk_data['regions'],
            suppliers=risk_data['suppliers'],
            overall_risk_level=risk_data['overall_risk_level'],
            total_suppliers=risk_data['total_suppliers'],
            high_risk_regions=risk_data['high_risk_regions'],
            high_risk_suppliers=risk_data['high_risk_suppliers']
        )
        
        logger.info(f"Risk heatmap data generated: {len(response.regions)} regions, {len(response.suppliers)} suppliers")
        return response
        
    except Exception as e:
        logger.error(f"Error in risk heatmap endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating risk heatmap: {str(e)}")

@router.get("/risk-heatmap/trends")
async def get_risk_trends(days: int = Query(default=30, ge=1, le=365)):
    """
    Get historical risk trends for regions
    
    Args:
        days: Number of days of historical data to return (1-365)
    
    Returns:
        Dictionary with regional risk trend data
    """
    try:
        logger.info(f"Risk trends endpoint called with days={days}")
        
        # Get risk trends
        trends = risk_service.get_regional_risk_trends(days)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "days": days,
            "trends": trends
        }
        
    except Exception as e:
        logger.error(f"Error in risk trends endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting risk trends: {str(e)}")

@router.get("/risk-heatmap/supplier/{supplier_id}/history")
async def get_supplier_risk_history(
    supplier_id: str,
    days: int = Query(default=30, ge=1, le=365)
):
    """
    Get historical risk data for a specific supplier
    
    Args:
        supplier_id: Supplier identifier
        days: Number of days of historical data to return (1-365)
    
    Returns:
        List of historical risk data points
    """
    try:
        logger.info(f"Supplier risk history endpoint called for {supplier_id}")
        
        # Get supplier risk history
        history = risk_service.get_supplier_risk_history(supplier_id, days)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "supplier_id": supplier_id,
            "days": days,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error in supplier risk history endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting supplier risk history: {str(e)}")

@router.post("/risk-heatmap/cache/clear")
async def clear_risk_cache():
    """
    Clear the risk heatmap cache
    
    Returns:
        Success message
    """
    try:
        logger.info("Clear risk cache endpoint called")
        
        # Clear cache
        risk_service.clear_cache()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Risk heatmap cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing risk cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing risk cache: {str(e)}")
