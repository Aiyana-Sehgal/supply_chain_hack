"""
Supply Chain Intelligence API - Main FastAPI Application

Layer 6 Backend API providing endpoints for supply chain intelligence system.
Access Swagger UI at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging
import sys
import os
from dotenv import load_dotenv

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Supply Chain Intelligence API",
    description="""
    Backend API for supply chain intelligence system providing:
    
    ## Core Components
    - **Risk Heatmap**: Regional and supplier risk visualization
    - **Recommendation Engine**: RL agent decisions with XAI explanations
    - **Cost Impact Analysis**: Cost change analysis per decision
    - **Live Risk Alerts**: Real-time alerts with NewsData.io integration
    
    ## Integration
    - Layers 1-5 integration for complete data flow
    - Enhanced XAI with Ollama Llama 3.1
    - Real-time data processing and caching
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
_cache = {
    'last_updated': None,
    'data': {}
}

@app.on_event("startup")
async def startup_event():
    """Initialize the API application"""
    logger.info("Starting Supply Chain Intelligence API...")
    logger.info(f"Swagger UI available at: http://localhost:8000/docs")
    logger.info(f"ReDoc available at: http://localhost:8000/redoc")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Supply Chain Intelligence API...")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Supply Chain Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "api": "operational",
            "cache": "operational",
            "layers": "connected"
        }
    }

# Import and include endpoint routers
from .endpoints.risk_heatmap import router as risk_heatmap_router
from .endpoints.recommendations import router as recommendations_router
from .endpoints.cost_analysis import router as cost_analysis_router
from .endpoints.risk_alerts import router as risk_alerts_router
from .endpoints.dashboard import router as dashboard_router

# Include routers
app.include_router(risk_heatmap_router, prefix="/api/v1", tags=["Risk Heatmap"])
app.include_router(recommendations_router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(cost_analysis_router, prefix="/api/v1", tags=["Cost Analysis"])
app.include_router(risk_alerts_router, prefix="/api/v1", tags=["Risk Alerts"])
app.include_router(dashboard_router, prefix="/api/v1", tags=["Dashboard"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
