"""
Recommendations API Endpoint

Provides RL agent recommendations with XAI explanations.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime
import logging

from ..models import RecommendationResponse
from ..services.recommendation_service import RecommendationService

router = APIRouter()

# Initialize service
recommendation_service = RecommendationService()
logger = logging.getLogger(__name__)

@router.get("/recommendations", response_model=RecommendationResponse)
async def get_current_recommendations():
    """
    Get current recommendation with XAI explanation
    
    Returns:
        RecommendationResponse: Complete recommendation data including:
        - RL agent's chosen action
        - Confidence level
        - Enhanced XAI explanations
        - Cost impact estimate
        - Historical performance data
    """
    try:
        logger.info("Recommendations endpoint called")
        
        # Get current recommendation
        recommendation = recommendation_service.get_current_recommendation()
        
        logger.info(f"Recommendation generated: {recommendation.action} with {recommendation.confidence:.1f}% confidence")
        return recommendation
        
    except Exception as e:
        logger.error(f"Error in recommendations endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@router.get("/recommendations/history")
async def get_recommendation_history(days: int = 30):
    """
    Get historical recommendation data
    
    Args:
        days: Number of days of historical data to return (1-365)
    
    Returns:
        List of historical recommendations
    """
    try:
        logger.info(f"Recommendation history endpoint called with days={days}")
        
        # Get recommendation history
        history = recommendation_service.get_recommendation_history(days)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "days": days,
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error in recommendation history endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendation history: {str(e)}")

@router.post("/recommendations/cache/clear")
async def clear_recommendation_cache():
    """
    Clear the recommendation cache
    
    Returns:
        Success message
    """
    try:
        logger.info("Clear recommendation cache endpoint called")
        
        # Clear cache
        recommendation_service.clear_cache()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "message": "Recommendation cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing recommendation cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing recommendation cache: {str(e)}")
