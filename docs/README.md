# Supply Chain Intelligence System

An AI-powered supply chain intelligence system that predicts demand, detects disruptions, scores risk, learns optimal decisions over time using reinforcement learning, and lets users simulate what-if scenarios with enhanced explainable AI powered by Ollama Llama 3.1.

## Features

### Core Intelligence Layers
- **Layer 1**: Data Collection from multiple sources (historical sales, NewsData.io hyperlocal India intelligence, weather signals, inventory data)
- **Layer 2**: Advanced Feature Engineering with normalization, aggregation, and rolling statistics
- **Layer 3**: Integration & XGBoost Demand Forecasting with 3-month predictions
- **Layer 4**: Q-Learning Reinforcement Learning Agent for optimal decision-making

### Digital Twin & Enhanced XAI (Layer 5)
- **Scenario Simulator**: What-if analysis for demand spikes, supplier failures, transport delays, weather disruptions
- **Enhanced Explainable AI**: Rule-based explanations enhanced with Ollama Llama 3.1 for business context
- **Intelligent Caching**: Fast responses with fallback to rule-based system
- **Executive Summaries**: C-suite friendly reports and recommendations

## Quick Start

### Prerequisites
- Python 3.8+
- Ollama installed (optional but recommended for enhanced XAI)
- Required model files (see Installation)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd supply_chain_intelligence
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup Ollama (optional but recommended)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.1 model
ollama pull llama3.1:8b

# Start Ollama service
ollama serve
```

4. **Verify installation**
```bash
python tests/test_product.py
```

### Basic Usage

```python
from src import create_digital_twin, quick_analysis

# Quick end-to-end analysis
results = quick_analysis()
print(f"Recommendation: {results['recommendation']['action']}")

# Create Digital Twin for advanced analysis
dt = create_digital_twin()

# Analyze current state
current = dt.analyze_current_state()

# Run what-if scenario
scenario = dt.run_what_if_scenario(scenario_name="demand_surge")

# Get risk alerts
alerts = dt.get_risk_alerts()
```

## Architecture

### Data Flow
```
store_sale.csv (Historical Data)
    |
    v
Layer 1: Data Collection (NewsData.io + Weather + Inventory)
    |
    v
Layer 2: Feature Engineering (Normalization + Aggregation)
    |
    v
Layer 3: Integration + XGBoost Forecasting
    |
    v
Layer 4: Q-Learning RL Agent
    |
    v
Layer 5: Digital Twin + Enhanced XAI (Ollama Llama 3.1)
```

### Key Components

#### Enhanced Explainable AI
- **Hybrid Approach**: Rule-based logic + Llama 3.1 enhancement
- **Business Context**: Industry-specific terminology and executive-friendly language
- **Caching**: Intelligent caching for performance optimization
- **Fallback**: Graceful degradation to rule-based system

#### Scenario Simulator
- **Predefined Templates**: Common supply chain scenarios
- **Custom Scenarios**: Full parameter customization
- **Impact Analysis**: Cost, risk, and probability assessments
- **Digital Twin**: Real-time what-if capabilities

## Configuration

### Environment Variables
```bash
# NewsData.io API (hyperlocal India intelligence)
NEWSDATA_API_KEY=your_newsdata_api_key

# WeatherAPI (optional, falls back to simulation)
WEATHERAPI_KEY=your_weather_api_key

# Ollama configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_MODEL=llama3.1:8b
```

### Model Files
Required model files should be placed in the `models/` directory:
- `demand_forecast_model.pkl` - XGBoost demand forecasting model
- `demand_scaler.pkl` - Feature scaling parameters
- `supply_chain_agent.pkl` - Trained Q-learning agent

## Usage Examples

### Current State Analysis
```python
from src import create_digital_twin

dt = create_digital_twin()
analysis = dt.analyze_current_state()

print(f"Predicted Demand: {analysis['current_state']['predicted_demand']:,.0f}")
print(f"Risk Level: {analysis['risk_assessment']['risk_breakdown']['composite_risk']:.1%}")
print(f"Recommendation: {analysis['recommendation']['action']}")
```

### What-If Scenario Analysis
```python
from src.scenarios import ScenarioInput

# Create custom scenario
scenario = ScenarioInput(
    demand_spike=0.25,  # 25% demand increase
    supplier_failure=True,
    inventory_adjustment=-0.3  # 30% inventory reduction
)

# Run scenario analysis
results = dt.run_what_if_scenario(custom_scenario=scenario)

print(f"Cost Impact: ${results['scenario_results']['impact_analysis']['cost_impact']:,.0f}")
print(f"Stockout Risk: {results['scenario_results']['impact_analysis']['stockout_probability']:.1%}")
```

### Enhanced Explanations
```python
from src.xai import create_enhanced_xai

xai = create_enhanced_xai(enable_enhancement=True)

# Generate enhanced risk explanation
risk_exp = xai.explain_risk_score(state)
print(risk_exp['supplier_risk'])

# Generate enhanced recommendation explanation
rec_exp = xai.explain_recommendation(state, "switch_supplier", 78.5)
print(rec_exp['rationale'])
```

## API Reference

### Digital Twin Interface
```python
class DigitalTwin:
    def analyze_current_state() -> Dict[str, Any]
    def run_what_if_scenario(scenario_name: str, custom_scenario: ScenarioInput) -> Dict[str, Any]
    def compare_scenarios(scenarios: List[Dict]) -> Dict[str, Any]
    def get_risk_alerts(alert_thresholds: Dict[str, float]) -> Dict[str, Any]
    def generate_executive_summary() -> Dict[str, Any]
```

### Enhanced XAI
```python
class EnhancedXAI:
    def explain_risk_score(state: Dict[str, Any]) -> Dict[str, str]
    def explain_recommendation(state: Dict[str, Any], action: str, confidence: float) -> Dict[str, Any]
    def explain_scenario_impact(scenario_results: Dict[str, Any]) -> Dict[str, Any]
    def generate_comprehensive_explanation(state: Dict[str, Any], action: str, confidence: float) -> Dict[str, Any]
```

### Scenario Simulator
```python
class ScenarioSimulator:
    def run_scenario_simulation(scenario: ScenarioInput) -> Dict[str, Any]
    def apply_scenario_modifications(base_state: Dict[str, Any], scenario: ScenarioInput) -> Dict[str, Any]
    def calculate_risk_probabilities(state: Dict[str, Any]) -> Dict[str, float]
    def estimate_cost_impact(base_state: Dict[str, Any], modified_state: Dict[str, Any]) -> Dict[str, float]
```

## Performance & Reliability

### Caching System
- **Intelligent Caching**: Hash-based key generation for fast lookups
- **TTL Management**: Configurable cache expiration (default: 24 hours)
- **Size Limits**: LRU eviction with configurable limits
- **Fallback**: Graceful degradation when cache is unavailable

### Ollama Integration
- **Connection Management**: Auto-reconnect and health monitoring
- **Retry Logic**: Configurable retry attempts and delays
- **Performance**: Sub-5 second response times with cache
- **Fallback**: Rule-based system when Ollama unavailable

### Benchmarks
- **Quick Analysis**: < 30 seconds for end-to-end pipeline
- **XAI Enhancement**: < 5 seconds for enhanced explanations
- **Cache Hit Rate**: > 80% for repeated queries
- **Uptime**: 100% with fallback mechanisms

## Testing

### Run Comprehensive Tests
```bash
python tests/test_product.py
```

### Test Coverage
- All core layers (1-4) functionality
- Enhanced XAI with Ollama integration
- Scenario simulator and digital twin
- Performance benchmarks
- Reliability and fallback mechanisms
- End-to-end pipeline integration

### Test Categories
1. **Prerequisites**: File availability and model loading
2. **Core Layers**: Data collection, feature engineering, forecasting, RL agent
3. **Enhanced XAI**: Ollama integration, caching, fallbacks
4. **Scenario Analysis**: What-if scenarios and digital twin
5. **Performance**: Response time and throughput benchmarks
6. **Reliability**: Error handling and graceful degradation

## Deployment

### Production Setup
1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Start Ollama service**
```bash
ollama serve
```

4. **Run health check**
```bash
python tests/test_product.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3.1 model
RUN ollama pull llama3.1:8b

# Copy application
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Start services
CMD ["ollama", "serve"] & python src/main.py
```

## Monitoring & Maintenance

### Health Checks
```python
from src.xai import create_enhanced_xai

xai = create_enhanced_xai()
health = xai.health_check()
print(f"System Status: {health['status']}")
```

### Performance Monitoring
```python
stats = xai.get_performance_stats()
print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
print(f"Enhancement Rate: {stats['enhancement_rate']:.1%}")
```

### Cache Management
```python
# Clear cache
xai.clear_cache()

# Save cache state
xai.save_state()

# Export cache for backup
xai.cache_manager.export_cache("backup_cache.json")
```

## Troubleshooting

### Common Issues

#### Ollama Connection Failed
```bash
# Check if Ollama is running
ollama list

# Start Ollama service
ollama serve

# Check model availability
ollama pull llama3.1:8b
```

#### Model Files Missing
```bash
# Ensure model files are in models/ directory
ls models/

# Expected files:
# - demand_forecast_model.pkl
# - demand_scaler.pkl  
# - supply_chain_agent.pkl
```

#### Performance Issues
```python
# Check cache performance
stats = xai.get_performance_stats()
print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1%}")

# Clear cache if needed
xai.clear_cache()
```

### Error Codes
- **E001**: Ollama service unavailable
- **E002**: Model file not found
- **E003**: Invalid state parameters
- **E004**: Cache write error
- **E005**: Pipeline integration failure

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests with coverage
python -m pytest tests/ --cov=src

# Run linting
flake8 src/
black src/
```

### Code Structure
```
src/
  core/           # Core intelligence layers (1-4)
  xai/            # Enhanced explainable AI
  scenarios/      # Scenario simulator and digital twin
  utils/          # Utilities and pipeline orchestration
models/           # Trained model files
data/            # Data files
tests/           # Comprehensive test suite
docs/            # Documentation
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the test suite for usage examples

## Version History

### v1.0.0 (Current)
- Complete 5-layer supply chain intelligence system
- Enhanced XAI with Ollama Llama 3.1 integration
- Digital Twin scenario simulator
- Comprehensive caching and fallback mechanisms
- End-to-end testing and documentation

---

**Built with**: Python, XGBoost, Q-Learning, Ollama Llama 3.1, NewsData.io, scikit-learn, pandas, numpy
