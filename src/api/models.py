"""
API Data Models

Pydantic models for request/response validation and Swagger documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Enums for better documentation
class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ActionType(str, Enum):
    DO_NOTHING = "do_nothing"
    REORDER_STOCK = "reorder_stock"
    SWITCH_SUPPLIER = "switch_supplier"
    REROUTE_SHIPMENT = "reroute_shipment"
    EMERGENCY_RESTOCK = "emergency_restock"

# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields"""
    timestamp: datetime = Field(..., description="Response timestamp")
    status: str = Field(default="success", description="Response status")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# Risk Heatmap Models
class RegionRisk(BaseModel):
    """Regional risk data"""
    region: str = Field(..., description="Region name")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk level category")
    supplier_count: int = Field(..., ge=0, description="Number of suppliers in region")
    disruption_signal: float = Field(..., ge=0, le=1, description="Disruption signal strength")
    last_updated: datetime = Field(..., description="Last update timestamp")

class SupplierRisk(BaseModel):
    """Supplier-specific risk data"""
    supplier_id: str = Field(..., description="Supplier identifier")
    supplier_name: str = Field(..., description="Supplier name")
    region: str = Field(..., description="Supplier region")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk level category")
    inventory_ratio: float = Field(..., ge=0, description="Inventory to demand ratio")
    days_to_stockout: float = Field(..., ge=0, description="Days until stockout")
    disruption_signal: float = Field(..., ge=0, le=1, description="Disruption signal strength")
    last_updated: datetime = Field(..., description="Last update timestamp")

class RiskHeatmapResponse(BaseResponse):
    """Risk heatmap response model"""
    regions: List[RegionRisk] = Field(..., description="Regional risk data")
    suppliers: List[SupplierRisk] = Field(..., description="Supplier risk data")
    overall_risk_level: RiskLevel = Field(..., description="Overall system risk level")
    total_suppliers: int = Field(..., ge=0, description="Total number of suppliers")
    high_risk_regions: int = Field(..., ge=0, description="Number of high-risk regions")
    high_risk_suppliers: int = Field(..., ge=0, description="Number of high-risk suppliers")

# Recommendation Models
class RecommendationExplanation(BaseModel):
    """Recommendation explanation model"""
    rationale: str = Field(..., description="Basic explanation")
    enhanced_explanation: Optional[str] = Field(None, description="AI-enhanced explanation")
    risk_factors: List[str] = Field(..., description="Key risk factors considered")
    confidence_factors: List[str] = Field(..., description="Factors affecting confidence")

class ValidationMetrics(BaseModel):
    """Validation layer feedback loop metrics comparing action vs baseline"""
    simulated_stockout_reduction: float = Field(..., description="Absolute drop in stockout probability")
    simulated_delay_reduction: float = Field(..., description="Absolute drop in delay probability")
    cost_saved: float = Field(..., description="Monetary cost offset by simulated action")
    validation_status: str = Field(..., description="Outcome of the simulation validation ('VALIDATED_OPTIMAL', 'VALIDATED_SUBOPTIMAL')")

class RecommendationResponse(BaseResponse):
    """Recommendation response model"""
    action: ActionType = Field(..., description="Recommended action")
    confidence: float = Field(..., ge=0, le=100, description="Confidence percentage")
    explanation: RecommendationExplanation = Field(..., description="Action explanation")
    cost_impact: float = Field(..., description="Estimated cost impact")
    implementation_time: Optional[str] = Field(None, description="Estimated implementation time")
    alternatives: List[ActionType] = Field(..., description="Alternative actions considered")
    historical_performance: Optional[Dict[str, float]] = Field(None, description="Historical action performance")
    validation_layer: ValidationMetrics = Field(..., description="Simulation-backed validation layer metrics")

# Cost Impact Models
class CostImpactBreakdown(BaseModel):
    """Cost impact breakdown by category"""
    category: str = Field(..., description="Cost category")
    current_cost: float = Field(..., ge=0, description="Current cost")
    projected_cost: float = Field(..., ge=0, description="Projected cost after action")
    change_amount: float = Field(..., description="Cost change amount")
    change_percentage: float = Field(..., description="Cost change percentage")

class CostImpactResponse(BaseResponse):
    """Cost impact analysis response model"""
    recommendation: ActionType = Field(..., description="Action being analyzed")
    total_current_cost: float = Field(..., ge=0, description="Total current cost")
    total_projected_cost: float = Field(..., ge=0, description="Total projected cost")
    total_cost_change: float = Field(..., description="Total cost change")
    total_change_percentage: float = Field(..., description="Total cost change percentage")
    roi_estimate: float = Field(..., description="Return on investment estimate")
    payback_period_months: Optional[float] = Field(None, description="Payback period in months")
    breakdown: List[CostImpactBreakdown] = Field(..., description="Cost breakdown by category")
    assumptions: List[str] = Field(..., description="Key assumptions in calculation")

# Risk Alert Models
class RiskAlert(BaseModel):
    """Individual risk alert model"""
    alert_id: str = Field(..., description="Unique alert identifier")
    alert_type: str = Field(..., description="Type of alert")
    severity: AlertSeverity = Field(..., description="Alert severity level")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    region: Optional[str] = Field(None, description="Affected region")
    supplier: Optional[str] = Field(None, description="Affected supplier")
    risk_score: Optional[float] = Field(None, ge=0, le=1, description="Current risk score")
    threshold: Optional[float] = Field(None, ge=0, le=1, description="Risk threshold that triggered alert")
    created_at: datetime = Field(..., description="Alert creation timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Alert resolution timestamp")
    resolved: bool = Field(default=False, description="Alert resolution status")

class RiskAlertsResponse(BaseResponse):
    """Risk alerts response model"""
    alerts: List[RiskAlert] = Field(..., description="List of active alerts")
    total_alerts: int = Field(..., ge=0, description="Total number of alerts")
    critical_alerts: int = Field(..., ge=0, description="Number of critical alerts")
    high_alerts: int = Field(..., ge=0, description="Number of high severity alerts")
    medium_alerts: int = Field(..., ge=0, description="Number of medium severity alerts")
    low_alerts: int = Field(..., ge=0, description="Number of low severity alerts")
    last_updated: datetime = Field(..., description="Last update timestamp")

# Dashboard Aggregation Model
class DashboardResponse(BaseResponse):
    """Complete dashboard response aggregating all components"""
    risk_heatmap: RiskHeatmapResponse = Field(..., description="Risk heatmap data")
    recommendations: RecommendationResponse = Field(..., description="Current recommendations")
    cost_impact: CostImpactResponse = Field(..., description="Cost impact analysis")
    risk_alerts: RiskAlertsResponse = Field(..., description="Risk alerts")
    system_health: Dict[str, str] = Field(..., description="System health status")
    last_data_update: datetime = Field(..., description="Last data update timestamp")

# Request Models for future extensibility
class ScenarioRequest(BaseModel):
    """Scenario analysis request model"""
    demand_spike: float = Field(default=0.0, ge=-1, le=1, description="Demand spike multiplier")
    supplier_failure: bool = Field(default=False, description="Force supplier failure")
    transport_delay: bool = Field(default=False, description="Force transport delay")
    weather_disruption: bool = Field(default=False, description="Force weather disruption")
    inventory_adjustment: float = Field(default=0.0, ge=-1, le=1, description="Inventory adjustment multiplier")

class TimeSeriesRequest(BaseModel):
    """Time series data request model"""
    start_date: datetime = Field(..., description="Start date for time series")
    end_date: datetime = Field(..., description="End date for time series")
    granularity: str = Field(default="daily", description="Data granularity (daily, weekly, monthly)")
    metrics: List[str] = Field(..., description="Metrics to include")
