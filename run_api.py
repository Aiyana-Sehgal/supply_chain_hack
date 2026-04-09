"""
Supply Chain Intelligence API Runner

Start the Layer 6 Backend API with FastAPI and Swagger UI.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main function to start the API server"""
    print("=" * 80)
    print("SUPPLY CHAIN INTELLIGENCE API - LAYER 6 BACKEND")
    print("=" * 80)
    print("Starting FastAPI server with Swagger UI...")
    print()
    print("API Documentation:")
    print("  Swagger UI: http://localhost:8000/docs")
    print("  ReDoc:      http://localhost:8000/redoc")
    print("  OpenAPI:    http://localhost:8000/openapi.json")
    print()
    print("Available Endpoints:")
    print("  GET /api/v1/dashboard              - Complete dashboard aggregation")
    print("  GET /api/v1/risk-heatmap          - Risk visualization data")
    print("  GET /api/v1/recommendations      - RL agent recommendations")
    print("  GET /api/v1/cost-impact          - Cost impact analysis")
    print("  GET /api/v1/risk-alerts          - Live risk alerts")
    print()
    print("Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop the server")
    print("=" * 80)
    
    try:
        # Import and run the FastAPI app
        from src.api.main import app
        import uvicorn
        
        # Run the server
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install API dependencies:")
        print("  pip install -r requirements-api.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
