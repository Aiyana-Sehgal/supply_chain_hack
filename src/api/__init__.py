"""
Supply Chain Intelligence API

Layer 6 Backend API providing endpoints for:
- Risk Heatmap visualization
- Recommendation Engine with XAI explanations
- Cost Impact Analysis
- Live Risk Alerts

Access Swagger UI at: http://localhost:8000/docs
"""

from .main import app

__all__ = ['app']
