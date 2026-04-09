# Supply Chain Intelligence API Documentation

Layer 6 Backend API providing endpoints for supply chain intelligence system with complete Swagger UI testing capabilities.

## Quick Start

### 1. Install API Dependencies
```bash
pip install -r requirements-api.txt
```

### 2. Start the API Server
```bash
python run_api.py
```

### 3. Access Swagger UI
Open your browser and navigate to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Dashboard Endpoints

#### GET /api/v1/dashboard
Complete dashboard aggregation of all Layer 6 components.

**Response**: Complete dashboard data including risk heatmap, recommendations, cost analysis, and alerts.

#### GET /api/v1/dashboard/summary
Key dashboard metrics for quick overview.

**Response**: Summary with key metrics and health status.

#### POST /api/v1/dashboard/refresh
Refresh all dashboard data and clear caches.

**Response**: Success message with cache clearance confirmation.

### Risk Heatmap Endpoints

#### GET /api/v1/risk-heatmap
Get risk visualization data by region and supplier.

**Response**: Regional and supplier risk scores with color coding.

#### GET /api/v1/risk-heatmap/trends
Get historical risk trends for regions.

**Query Parameters**:
- `days`: Number of days of historical data (1-365, default: 30)

#### GET /api/v1/risk-heatmap/supplier/{supplier_id}/history
Get historical risk data for specific supplier.

**Query Parameters**:
- `days`: Number of days of historical data (1-365, default: 30)

### Recommendation Endpoints

#### GET /api/v1/recommendations
Get current RL agent recommendation with XAI explanations.

**Response**: Current action, confidence, enhanced explanations, and cost impact.

#### GET /api/v1/recommendations/history
Get historical recommendation data.

**Query Parameters**:
- `days`: Number of days of historical data (1-365, default: 30)

### Cost Analysis Endpoints

#### GET /api/v1/cost-impact
Get cost impact analysis for specific action.

**Query Parameters**:
- `action`: Action to analyze (do_nothing, reorder_stock, switch_supplier, reroute_shipment, emergency_restock)

**Response**: Current vs projected costs, ROI, payback period, and assumptions.

#### GET /api/v1/cost-impact/trends
Get historical cost trends by category.

**Query Parameters**:
- `days`: Number of days of historical data (1-365, default: 30)

#### GET /api/v1/cost-impact/actions
Get list of available actions for cost analysis.

**Response**: Available actions with descriptions.

### Risk Alerts Endpoints

#### GET /api/v1/risk-alerts
Get live risk alerts with NewsData.io integration.

**Response**: Active alerts with severity levels and metadata.

#### GET /api/v1/risk-alerts/history
Get historical alert statistics.

**Query Parameters**:
- `days`: Number of days of historical data (1-365, default: 30)

#### POST /api/v1/risk-alerts/{alert_id}/resolve
Mark an alert as resolved.

**Path Parameters**:
- `alert_id`: Unique identifier of the alert to resolve

#### GET /api/v1/risk-alerts/types
Get list of available alert types.

**Response**: Alert types with descriptions and severity levels.

## Testing with Swagger UI

### Step 1: Access Swagger UI
Navigate to http://localhost:8000/docs in your browser.

### Step 2: Test Dashboard Endpoint
1. Find the `/api/v1/dashboard` endpoint
2. Click "Try it out"
3. Click "Execute"
4. Review the complete dashboard response

### Step 3: Test Individual Components
1. **Risk Heatmap**: Test `/api/v1/risk-heatmap`
2. **Recommendations**: Test `/api/v1/recommendations`
3. **Cost Analysis**: Test `/api/v1/cost-impact` with action parameter
4. **Risk Alerts**: Test `/api/v1/risk-alerts`

### Step 4: Test Cost Analysis
1. Navigate to `/api/v1/cost-impact`
2. Click "Try it out"
3. Enter action parameter (e.g., "reorder_stock")
4. Click "Execute"
5. Review cost impact analysis

### Step 5: Test Alert Management
1. Get alerts: `/api/v1/risk-alerts`
2. Note an alert_id from the response
3. Resolve alert: `/api/v1/risk-alerts/{alert_id}/resolve`

## Response Examples

### Dashboard Response
```json
{
  "timestamp": "2026-04-09T15:45:00",
  "status": "success",
  "risk_heatmap": {
    "regions": [...],
    "suppliers": [...],
    "overall_risk_level": "medium"
  },
  "recommendations": {
    "action": "reorder_stock",
    "confidence": 85.2,
    "explanation": {
      "rationale": "Low inventory levels indicate need for restocking",
      "enhanced_explanation": "AI-enhanced business context..."
    }
  },
  "cost_impact": {
    "recommendation": "reorder_stock",
    "total_cost_change": 25000.0,
    "roi_estimate": 0.15
  },
  "risk_alerts": {
    "alerts": [...],
    "total_alerts": 3,
    "critical_alerts": 1
  }
}
```

### Risk Heatmap Response
```json
{
  "timestamp": "2026-04-09T15:45:00",
  "status": "success",
  "regions": [
    {
      "region": "North India",
      "risk_score": 0.65,
      "risk_level": "medium",
      "supplier_count": 3
    }
  ],
  "suppliers": [
    {
      "supplier_id": "SUP001",
      "supplier_name": "Global Logistics Ltd",
      "region": "North India",
      "risk_score": 0.72,
      "risk_level": "high"
    }
  ]
}
```

## Error Handling

### Error Response Format
```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "timestamp": "2026-04-09T15:45:00",
  "details": {...}
}
```

### Common Error Codes
- **400**: Invalid parameters (e.g., invalid action)
- **404**: Resource not found (e.g., invalid alert_id)
- **500**: Internal server error

## Performance & Caching

### Cache Management
All endpoints use 5-minute caching for performance. Clear caches using:
- `POST /api/v1/dashboard/refresh` - Clear all caches
- Individual endpoint cache clear endpoints

### Response Times
- Dashboard aggregation: < 2 seconds
- Individual endpoints: < 500ms
- Cached responses: < 100ms

## Integration with Existing Layers

The API integrates with existing Layers 1-5:
- **Layer 1**: Data collection and disruption signals
- **Layer 2**: Feature engineering and normalization
- **Layer 3**: Demand forecasting with XGBoost
- **Layer 4**: RL agent decision engine
- **Layer 5**: Enhanced XAI with Ollama Llama 3.1

## Production Deployment

### Environment Variables
```bash
# Optional: NewsData.io API for enhanced alerts
NEWSDATA_API_KEY=your_api_key

# Optional: WeatherAPI for weather-based alerts
WEATHERAPI_KEY=your_api_key
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_api.py"]
```

### Health Check
```bash
curl http://localhost:8000/health
```

## Monitoring & Logging

### Log Levels
- INFO: Normal operation
- WARNING: Non-critical issues
- ERROR: Service errors

### Key Metrics
- Response times
- Cache hit rates
- Error rates
- Active alert counts

## Troubleshooting

### Common Issues
1. **Import Errors**: Install requirements with `pip install -r requirements-api.txt`
2. **Port Conflicts**: Change port in `run_api.py` or stop conflicting services
3. **CORS Issues**: Ensure proper CORS configuration for browser access

### Debug Mode
Run with debug logging:
```bash
python run_api.py --log-level debug
```

This API provides complete browser-based testing capabilities through Swagger UI, allowing you to test all Layer 6 components interactively without any frontend implementation.
