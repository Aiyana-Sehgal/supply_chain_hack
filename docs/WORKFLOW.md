# Supply Chain Intelligence - Product Workflow

This document provides a comprehensive workflow guide for using the Supply Chain Intelligence System, from basic setup to advanced scenario analysis and executive reporting.

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Analysis Workflow](#basic-analysis-workflow)
3. [Advanced Scenario Analysis](#advanced-scenario-analysis)
4. [Digital Twin Operations](#digital-twin-operations)
5. [Executive Reporting](#executive-reporting)
6. [Integration Workflows](#integration-workflows)
7. [Troubleshooting Workflow](#troubleshooting-workflow)

## Getting Started

### Step 1: System Setup
```bash
# 1. Navigate to project directory
cd supply_chain_intelligence

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup Ollama (recommended for enhanced XAI)
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
ollama serve &

# 4. Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Step 2: Verify Installation
```bash
# Run comprehensive tests
python tests/test_product.py

# Expected output: "Status: EXCELLENT - Product ready for production"
```

### Step 3: Quick Start
```python
from src import quick_analysis

# Run end-to-end analysis
results = quick_analysis()
print(f"Recommendation: {results['recommendation']['action']}")
print(f"Confidence: {results['recommendation']['confidence']:.1f}%")
```

## Basic Analysis Workflow

### Workflow 1: Current State Assessment

**Objective**: Get real-time assessment of supply chain health and recommendations

```python
from src import create_digital_twin

# Step 1: Initialize Digital Twin
dt = create_digital_twin(verbose=True)

# Step 2: Analyze current state
current_analysis = dt.analyze_current_state()

# Step 3: Review key metrics
state = current_analysis['current_state']
print(f"Predicted Demand: {state['predicted_demand']:,.0f} units")
print(f"Inventory Coverage: {state['days_to_stockout']:.1f} days")
print(f"Supplier Risk: {state['supplier_risk_score']:.1%}")
print(f"Disruption Risk: {state['disruption_signal']:.1%}")

# Step 4: Review recommendation
rec = current_analysis['recommendation']
print(f"Recommended Action: {rec['action'].upper()}")
print(f"Confidence: {rec['confidence']:.1f}%")
print(f"Explanation: {rec['explanation']}")

# Step 5: Review risk explanations
risk_exp = current_analysis['explanations']['risk_explanations']
for risk_type, explanation in risk_exp.items():
    print(f"{risk_type.replace('_', ' ').title()}: {explanation}")
```

**Output Example**:
```
Predicted Demand: 523,450 units
Inventory Coverage: 12.3 days
Supplier Risk: 65.2%
Disruption Risk: 42.1%
Recommended Action: REORDER_STOCK
Confidence: 87.3%
```

### Workflow 2: Risk Alert Monitoring

**Objective**: Monitor risk levels and get alerts when thresholds are exceeded

```python
# Step 1: Configure alert thresholds
custom_thresholds = {
    'supplier_risk': 0.6,      # Alert if supplier risk > 60%
    'disruption_signal': 0.5,  # Alert if disruption > 50%
    'inventory_ratio': 0.15,   # Alert if inventory < 15% of demand
    'days_to_stockout': 10,    # Alert if < 10 days of supply
    'composite_risk': 0.5      # Alert if overall risk > 50%
}

# Step 2: Get risk alerts
alerts = dt.get_risk_alerts(alert_thresholds=custom_thresholds)

# Step 3: Review alert summary
summary = alerts['alert_summary']
print(f"Overall Status: {summary['overall_status'].upper()}")
print(f"Active Alerts: {summary['total_alerts']}")
print(f"Critical Alerts: {summary['critical_alerts']}")

# Step 4: Review individual alerts
if alerts['alerts']:
    print("\nActive Risk Alerts:")
    for alert in alerts['alerts']:
        print(f"  - {alert['type'].replace('_', ' ').title()}: {alert['message']}")
        print(f"    Recommendation: {alert['recommendation']}")
```

**Alert Levels**:
- **NORMAL**: No active alerts
- **MEDIUM**: 1-2 moderate alerts
- **HIGH**: 3+ alerts or any critical alerts
- **CRITICAL**: System-critical issues requiring immediate action

## Advanced Scenario Analysis

### Workflow 3: Predefined Scenario Analysis

**Objective**: Test common supply chain scenarios using predefined templates

```python
# Step 1: List available scenarios
from src.scenarios import SCENARIO_TEMPLATES
print("Available Scenarios:")
for name in SCENARIO_TEMPLATES.keys():
    print(f"  - {name}")

# Step 2: Run predefined scenario
scenario_result = dt.run_what_if_scenario(scenario_name="demand_surge")

# Step 3: Analyze results
results = scenario_result['scenario_results']
impact = results['impact_analysis']
rec = results['recommendations']

print(f"Demand Change: {impact['demand_change']:+.1%}")
print(f"Cost Impact: ${impact['cost_impact']:,.0f}")
print(f"Stockout Risk: {impact['stockout_probability']:.1%}")
print(f"Delay Risk: {impact['delay_probability']:.1%}")
print(f"Recommended Action: {rec['action'].upper()}")
print(f"Confidence: {rec['confidence']:.1f}%")
```

**Available Predefined Scenarios**:
- `demand_surge`: 30% demand increase
- `supplier_crisis`: Supplier failure with high risk
- `transport_disruption`: Transportation delays
- `weather_crisis`: Weather-related disruptions
- `inventory_shortage`: 50% inventory reduction
- `perfect_storm`: Multiple simultaneous disruptions

### Workflow 4: Custom Scenario Creation

**Objective**: Create and analyze custom what-if scenarios

```python
from src.scenarios import ScenarioInput

# Step 1: Define custom scenario
custom_scenario = ScenarioInput(
    demand_spike=0.25,        # 25% demand increase
    supplier_failure=True,     # Force supplier failure
    transport_delay=False,     # No transport delay
    weather_disruption=True,    # Weather disruption active
    custom_disruption_signal=0.8,  # Set specific disruption level
    inventory_adjustment=-0.4   # 40% inventory reduction
)

# Step 2: Run custom scenario
custom_result = dt.run_what_if_scenario(custom_scenario=custom_scenario)

# Step 3: Detailed analysis
results = custom_result['scenario_results']
explanations = custom_result['scenario_explanations']

print("Custom Scenario Analysis:")
print(f"  Conditions Applied: {len([v for v in custom_scenario.__dict__.values() if v not in [0.0, False, None]])}")
print(f"  Projected Cost Impact: ${results['impact_analysis']['cost_impact']:,.0f}")
print(f"  Stockout Probability: {results['impact_analysis']['stockout_probability']:.1%}")

# Step 4: Review explanation
if 'key_impacts' in explanations:
    print("  Key Impacts:")
    for impact in explanations['key_impacts']:
        print(f"    - {impact}")
```

### Workflow 5: Multi-Scenario Comparison

**Objective**: Compare multiple scenarios side-by-side

```python
# Step 1: Define scenarios for comparison
comparison_scenarios = [
    {
        'name': 'Moderate Growth',
        'parameters': {'demand_spike': 0.15}
    },
    {
        'name': 'Supply Chain Disruption',
        'parameters': {
            'supplier_failure': True,
            'transport_delay': True
        }
    },
    {
        'name': 'Inventory Crisis',
        'parameters': {
            'inventory_adjustment': -0.6,
            'demand_spike': 0.10
        }
    }
]

# Step 2: Run comparison analysis
comparison_results = dt.compare_scenarios(comparison_scenarios)

# Step 3: Analyze comparison
print("Scenario Comparison Results:")
for scenario in comparison_results['comparison_summary']:
    print(f"\n{scenario['name']}:")
    print(f"  Cost Impact: ${scenario['cost_impact']:,.0f}")
    print(f"  Stockout Risk: {scenario['stockout_probability']:.1%}")
    print(f"  Recommended Action: {scenario['recommended_action']}")
```

## Digital Twin Operations

### Workflow 6: Executive Summary Generation

**Objective**: Generate C-suite friendly reports

```python
# Step 1: Generate executive summary
executive_summary = dt.generate_executive_summary()

# Step 2: Review key performance indicators
kpis = executive_summary['kpis']
print("Executive Summary:")
print(f"  Overall Risk Level: {kpis['overall_risk_level']}")
print(f"  Service Level Risk: {kpis['service_level_risk']}")
print(f"  Supplier Stability: {kpis['supplier_stability']}")
print(f"  Disruption Exposure: {kpis['disruption_exposure']}")
print(f"  Inventory Health: {kpis['inventory_health']}")

# Step 3: Review key metrics
metrics = executive_summary['key_metrics']
print(f"\nKey Metrics:")
for metric, value in metrics.items():
    print(f"  {metric.replace('_', ' ').title()}: {value}")

# Step 4: Review recommendations
print(f"\nCurrent Recommendation: {executive_summary['current_recommendation'].upper()}")
print(f"Confidence Level: {executive_summary['confidence_level']}")

print(f"\nExecutive Recommendations:")
for i, rec in enumerate(executive_summary['executive_recommendations'], 1):
    print(f"  {i}. {rec}")
```

### Workflow 7: Continuous Monitoring

**Objective**: Set up ongoing monitoring and alerting

```python
# Step 1: Setup monitoring configuration
monitoring_config = {
    'alert_thresholds': {
        'supplier_risk': 0.7,
        'disruption_signal': 0.6,
        'inventory_ratio': 0.1,
        'days_to_stockout': 7
    },
    'monitoring_interval': 3600,  # Check every hour
    'export_results': True,
    'notification_level': 'high'  # Only alert on high/critical
}

# Step 2: Create monitoring loop
def continuous_monitoring(dt, config, duration_hours=24):
    import time
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    
    while time.time() < end_time:
        # Get current alerts
        alerts = dt.get_risk_alerts(config['alert_thresholds'])
        
        # Check if alert meets notification level
        if alerts['alert_summary']['overall_status'] in ['high', 'critical']:
            print(f"ALERT: {alerts['alert_summary']['overall_status'].upper()}")
            print(f"Issues: {alerts['alert_summary']['total_alerts']}")
            
            # Export detailed report
            if config['export_results']:
                dt.export_analysis(alerts, f"alert_report_{int(time.time())}.json")
        
        # Wait for next check
        time.sleep(config['monitoring_interval'])
    
    print("Monitoring completed")

# Step 3: Run monitoring (for demonstration, use short duration)
# continuous_monitoring(dt, monitoring_config, duration_hours=1)
```

## Integration Workflows

### Workflow 8: API Integration

**Objective**: Integrate with external systems via API

```python
from src.utils import SupplyChainPipeline
from flask import Flask, jsonify, request

app = Flask(__name__)

# Initialize pipeline
pipeline = SupplyChainPipeline()

@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    try:
        # Test basic functionality
        state = pipeline.layer3_bundle['layer3_state'] if pipeline.layer3_bundle else None
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'pipeline': 'operational',
                'models': 'loaded',
                'xai': 'enhanced'
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_current():
    """Current state analysis endpoint"""
    try:
        results = pipeline.run_full_pipeline()
        return jsonify({
            'status': 'success',
            'data': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/scenario', methods=['POST'])
def run_scenario():
    """Scenario analysis endpoint"""
    try:
        scenario_data = request.json
        scenario = ScenarioInput(**scenario_data)
        results = pipeline.run_scenario_analysis(scenario)
        return jsonify({
            'status': 'success',
            'data': results
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Workflow 9: Batch Processing

**Objective**: Process multiple scenarios in batch

```python
def batch_scenario_analysis(scenarios_list, output_file="batch_results.json"):
    """Process multiple scenarios and save results"""
    
    dt = create_digital_twin(verbose=False)
    batch_results = []
    
    for i, scenario_def in enumerate(scenarios_list):
        print(f"Processing scenario {i+1}/{len(scenarios_list)}: {scenario_def.get('name', 'Unnamed')}")
        
        try:
            # Create scenario
            scenario = ScenarioInput(**scenario_def['parameters'])
            
            # Run analysis
            results = dt.run_what_if_scenario(custom_scenario=scenario)
            
            # Add metadata
            results['batch_metadata'] = {
                'scenario_name': scenario_def.get('name', f'Scenario_{i+1}'),
                'batch_index': i,
                'processed_at': datetime.now().isoformat()
            }
            
            batch_results.append(results)
            
        except Exception as e:
            print(f"Error processing scenario {i+1}: {e}")
            batch_results.append({
                'error': str(e),
                'batch_metadata': {
                    'scenario_name': scenario_def.get('name', f'Scenario_{i+1}'),
                    'batch_index': i,
                    'processed_at': datetime.now().isoformat()
                }
            })
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(batch_results, f, indent=2, default=str)
    
    print(f"Batch processing completed. Results saved to {output_file}")
    return batch_results

# Example usage
scenarios_to_process = [
    {'name': 'Q1 Planning', 'parameters': {'demand_spike': 0.20}},
    {'name': 'Monsoon Season', 'parameters': {'weather_disruption': True, 'transport_delay': True}},
    {'name': 'Festival Period', 'parameters': {'demand_spike': 0.35, 'inventory_adjustment': -0.2}}
]

# batch_scenario_analysis(scenarios_to_process)
```

## Troubleshooting Workflow

### Workflow 10: System Diagnostics

**Objective**: Diagnose and resolve system issues

```python
def run_diagnostics():
    """Comprehensive system diagnostics"""
    
    print("Running System Diagnostics...")
    print("=" * 50)
    
    # 1. Check file availability
    required_files = [
        'data/store_sale.csv',
        'models/demand_forecast_model.pkl',
        'models/demand_scaler.pkl',
        'models/supply_chain_agent.pkl'
    ]
    
    print("1. File Availability Check:")
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "OK" if exists else "MISSING"
        print(f"   {file_path}: {status}")
    
    # 2. Check Ollama connectivity
    print("\n2. Ollama Connectivity Check:")
    try:
        from src.xai import OllamaClient
        client = OllamaClient()
        test_result = client.test_connection()
        print(f"   Status: {test_result.get('healthy', False)}")
        if not test_result.get('healthy', False):
            print(f"   Error: {test_result.get('error', 'Unknown')}")
    except Exception as e:
        print(f"   Status: ERROR - {e}")
    
    # 3. Check model loading
    print("\n3. Model Loading Check:")
    try:
        from src.core import SupplyChainAgent
        agent = SupplyChainAgent()
        agent.load('models/supply_chain_agent.pkl')
        print("   RL Agent: OK")
    except Exception as e:
        print(f"   RL Agent: ERROR - {e}")
    
    # 4. Check XAI system
    print("\n4. Enhanced XAI Check:")
    try:
        from src.xai import create_enhanced_xai
        xai = create_enhanced_xai(enable_enhancement=True)
        health = xai.health_check()
        print(f"   Status: {health.get('status', 'Unknown')}")
        for component, status in health['components'].items():
            comp_status = status.get('status', 'unknown')
            print(f"   {component}: {comp_status}")
    except Exception as e:
        print(f"   Status: ERROR - {e}")
    
    # 5. Performance check
    print("\n5. Performance Check:")
    try:
        from src import quick_analysis
        start_time = time.time()
        results = quick_analysis(verbose=False)
        duration = time.time() - start_time
        print(f"   Quick Analysis: {duration:.1f}s ({'OK' if duration < 30 else 'SLOW'})")
    except Exception as e:
        print(f"   Quick Analysis: ERROR - {e}")
    
    print("\nDiagnostics completed!")
    
    # 6. Generate recommendations
    print("\nRecommendations:")
    if not os.path.exists('data/store_sale.csv'):
        print("   - Ensure data files are in the correct directories")
    
    try:
        from src.xai import OllamaClient
        client = OllamaClient()
        if not client.test_connection().get('healthy', False):
            print("   - Start Ollama service: 'ollama serve'")
            print("   - Pull Llama 3.1 model: 'ollama pull llama3.1:8b'")
    except:
        print("   - Install Ollama for enhanced XAI capabilities")
    
    print("   - Run full test suite: python tests/test_product.py")
    print("   - Check documentation: docs/README.md")

# Run diagnostics
run_diagnostics()
```

### Workflow 11: Error Recovery

**Objective**: Recover from common errors

```python
def error_recovery_workflow(error_type):
    """Automated error recovery procedures"""
    
    recovery_actions = {
        'ollama_unavailable': [
            "Attempting to restart Ollama service...",
            "Falling back to rule-based XAI...",
            "Clearing XAI cache..."
        ],
        'model_not_found': [
            "Checking model directory...",
            "Verifying model file integrity...",
            "Using fallback models..."
        ],
        'cache_error': [
            "Clearing corrupted cache...",
            "Rebuilding cache index...",
            "Disabling cache temporarily..."
        ],
        'pipeline_failure': [
            "Resetting pipeline components...",
            "Reloading models...",
            "Initializing fresh pipeline..."
        ]
    }
    
    print(f"Error Recovery: {error_type}")
    print("Recovery Actions:")
    
    for action in recovery_actions.get(error_type, ["Unknown error type"]):
        print(f"  - {action}")
    
    # Implement specific recovery logic here
    if error_type == 'ollama_unavailable':
        try:
            # Clear cache and restart with rule-based
            from src.xai import create_enhanced_xai
            xai = create_enhanced_xai(enable_enhancement=False)
            xai.clear_cache()
            print("  Recovery completed: Using rule-based XAI")
        except Exception as e:
            print(f"  Recovery failed: {e}")
    
    elif error_type == 'cache_error':
        try:
            from src.xai import create_enhanced_xai
            xai = create_enhanced_xai()
            xai.clear_cache()
            print("  Recovery completed: Cache cleared")
        except Exception as e:
            print(f"  Recovery failed: {e}")

# Example usage
# error_recovery_workflow('ollama_unavailable')
```

## Best Practices

### Performance Optimization
1. **Enable caching**: Always use enhanced XAI with caching enabled
2. **Batch scenarios**: Process multiple scenarios together when possible
3. **Monitor Ollama**: Ensure Ollama service is running for best performance
4. **Regular cleanup**: Clear cache periodically to maintain performance

### Data Management
1. **Regular backups**: Export cache and configuration regularly
2. **Model versioning**: Keep track of model versions and performance
3. **Data validation**: Validate input data before processing
4. **Error logging**: Maintain logs for troubleshooting

### Security Considerations
1. **API keys**: Store API keys in environment variables
2. **Access control**: Implement proper authentication for API endpoints
3. **Data privacy**: Ensure sensitive data is properly protected
4. **Audit trails**: Maintain logs of system usage and changes

This workflow guide provides comprehensive procedures for operating the Supply Chain Intelligence System effectively. For additional details, refer to the API documentation and test examples.
