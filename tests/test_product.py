"""
Comprehensive Product Test Suite

Tests the entire supply chain intelligence system end-to-end including:
- All core layers (1-4)
- Enhanced XAI with Ollama integration
- Scenario simulator and digital twin
- Pipeline integration
- Performance and reliability
"""

import sys
import os
import time
import json
import traceback
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all components
from core import build_layer1_dataset, build_layer2_dataset, build_layer3_dataset, SupplyChainAgent
from xai import create_enhanced_xai, OllamaClient, CacheManager
from scenarios import create_digital_twin, ScenarioInput
from utils import SupplyChainPipeline, quick_analysis


class ProductTestSuite:
    """
    Comprehensive test suite for the supply chain intelligence product.
    Tests all functionality from data collection to AI-enhanced explanations.
    """
    
    def __init__(self, verbose=True):
        """Initialize test suite"""
        self.verbose = verbose
        self.test_results = []
        self.start_time = time.time()
        
        # Test configuration
        self.sales_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'store_sale.csv')
        self.models_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        print("=" * 80)
        print("SUPPLY CHAIN INTELLIGENCE - COMPREHENSIVE PRODUCT TEST")
        print("=" * 80)
        print(f"Test Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Sales Data Path: {self.sales_path}")
        print(f"Models Path: {self.models_path}")
    
    def log_test(self, test_name: str, passed: bool, duration: float, details: str = ""):
        """Log test result"""
        result = {
            'test_name': test_name,
            'passed': passed,
            'duration': duration,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status} ({duration:.2f}s)")
        if details and self.verbose:
            print(f"    Details: {details}")
    
    def test_prerequisites(self) -> bool:
        """Test if all required files are available"""
        print("\n[TEST 1] Prerequisites Check")
        print("-" * 50)
        
        start_time = time.time()
        missing_files = []
        
        # Check required files
        required_files = [
            self.sales_path,
            os.path.join(self.models_path, 'demand_forecast_model.pkl'),
            os.path.join(self.models_path, 'demand_scaler.pkl'),
            os.path.join(self.models_path, 'supply_chain_agent.pkl')
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        passed = len(missing_files) == 0
        details = f"Missing files: {missing_files}" if missing_files else "All required files present"
        
        self.log_test("Prerequisites Check", passed, time.time() - start_time, details)
        return passed
    
    def test_layer1_data_collection(self) -> bool:
        """Test Layer 1: Data Collection"""
        print("\n[TEST 2] Layer 1 - Data Collection")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            layer1_bundle = build_layer1_dataset(sales_path=self.sales_path, verbose=False)
            
            # Validate structure
            required_keys = ['monthly_sales', 'inventory_data', 'layer1_df']
            missing_keys = [key for key in required_keys if key not in layer1_bundle]
            
            if missing_keys:
                raise ValueError(f"Missing keys in layer1_bundle: {missing_keys}")
            
            # Validate data
            monthly_sales = layer1_bundle['monthly_sales']
            if len(monthly_sales) == 0:
                raise ValueError("No monthly sales data")
            
            details = f"Generated {len(monthly_sales)} months of data"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Layer 1 Data Collection", passed, time.time() - start_time, details)
        return passed
    
    def test_layer2_feature_engineering(self) -> bool:
        """Test Layer 2: Feature Engineering"""
        print("\n[TEST 3] Layer 2 - Feature Engineering")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # First get Layer 1 data
            layer1_bundle = build_layer1_dataset(sales_path=self.sales_path, verbose=False)
            
            # Process through Layer 2
            layer2_bundle = build_layer2_dataset(layer1_bundle, verbose=False)
            
            # Validate structure
            required_keys = ['consolidated_df', 'feature_columns', 'scaler']
            missing_keys = [key for key in required_keys if key not in layer2_bundle]
            
            if missing_keys:
                raise ValueError(f"Missing keys in layer2_bundle: {missing_keys}")
            
            # Validate features
            consolidated_df = layer2_bundle['consolidated_df']
            if len(consolidated_df) == 0:
                raise ValueError("No consolidated data")
            
            details = f"Generated {len(consolidated_df)} records with {len(layer2_bundle['feature_columns'])} features"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Layer 2 Feature Engineering", passed, time.time() - start_time, details)
        return passed
    
    def test_layer3_integration(self) -> bool:
        """Test Layer 3: Integration & Forecasting"""
        print("\n[TEST 4] Layer 3 - Integration & Forecasting")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            layer3_bundle = build_layer3_dataset(sales_path=self.sales_path, verbose=False)
            
            # Validate structure
            required_keys = ['layer3_state', 'layer3_dataframe', 'forecast_df']
            missing_keys = [key for key in required_keys if key not in layer3_bundle]
            
            if missing_keys:
                raise ValueError(f"Missing keys in layer3_bundle: {missing_keys}")
            
            # Validate state
            state = layer3_bundle['layer3_state']
            required_state_keys = ['predicted_demand', 'current_inventory', 'supplier_risk_score']
            missing_state_keys = [key for key in required_state_keys if key not in state]
            
            if missing_state_keys:
                raise ValueError(f"Missing state keys: {missing_state_keys}")
            
            details = f"Generated state with demand forecast: {state['predicted_demand']:,.0f} units"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Layer 3 Integration & Forecasting", passed, time.time() - start_time, details)
        return passed
    
    def test_layer4_rl_agent(self) -> bool:
        """Test Layer 4: RL Decision Engine"""
        print("\n[TEST 5] Layer 4 - RL Decision Engine")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Load RL agent
            agent = SupplyChainAgent()
            agent.load(os.path.join(self.models_path, 'supply_chain_agent.pkl'))
            
            # Create test state
            test_state = {
                'predicted_demand': 500000,
                'current_inventory': 25000,
                'supplier_risk_score': 0.75,
                'disruption_signal': 0.65,
                'days_to_stockout': 5,
                'composite_risk_score': 0.7
            }
            
            # Get recommendation
            action_id = agent.choose_action(test_state, training=False)
            action_name = agent.actions[action_id]
            
            details = f"Recommended action: {action_name} (ID: {action_id})"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Layer 4 RL Decision Engine", passed, time.time() - start_time, details)
        return passed
    
    def test_ollama_integration(self) -> bool:
        """Test Ollama Llama 3.1 integration"""
        print("\n[TEST 6] Ollama Integration")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Test Ollama client
            ollama_client = OllamaClient()
            connection_test = ollama_client.test_connection()
            
            if connection_test.get('healthy', False):
                details = f"Ollama healthy, model: {connection_test.get('model', 'unknown')}"
                passed = True
            else:
                details = f"Ollama unavailable: {connection_test.get('error', 'unknown')}"
                passed = False  # This is expected if Ollama is not running
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Ollama Integration", passed, time.time() - start_time, details)
        return passed
    
    def test_enhanced_xai(self) -> bool:
        """Test Enhanced XAI system"""
        print("\n[TEST 7] Enhanced XAI System")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Create enhanced XAI
            enhanced_xai = create_enhanced_xai(enable_enhancement=True)
            
            # Test state
            test_state = {
                'predicted_demand': 500000,
                'current_inventory': 25000,
                'supplier_risk_score': 0.75,
                'disruption_signal': 0.65,
                'days_to_stockout': 5
            }
            
            # Test risk explanation
            risk_exp = enhanced_xai.explain_risk_score(test_state)
            
            # Test recommendation explanation
            rec_exp = enhanced_xai.explain_recommendation(test_state, "switch_supplier", 78.5)
            
            # Test performance stats
            stats = enhanced_xai.get_performance_stats()
            
            details = f"Generated explanations for {len(risk_exp)} risk factors, cache hit rate: {stats.get('cache_hit_rate', 0):.1%}"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Enhanced XAI System", passed, time.time() - start_time, details)
        return passed
    
    def test_scenario_simulator(self) -> bool:
        """Test Scenario Simulator"""
        print("\n[TEST 8] Scenario Simulator")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            from scenarios import ScenarioSimulator, ScenarioInput
            
            simulator = ScenarioSimulator()
            
            # Test scenario
            scenario = ScenarioInput(
                demand_spike=0.20,
                supplier_failure=True,
                inventory_adjustment=-0.3
            )
            
            # Run scenario (this might take time)
            results = simulator.run_scenario_simulation(scenario)
            
            # Validate results
            required_keys = ['scenario_results', 'impact_analysis', 'recommendations']
            missing_keys = [key for key in required_keys if key not in results]
            
            if missing_keys:
                raise ValueError(f"Missing keys in results: {missing_keys}")
            
            impact = results['impact_analysis']
            details = f"Scenario completed with cost impact: ${impact.get('cost_impact', 0):,.0f}"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Scenario Simulator", passed, time.time() - start_time, details)
        return passed
    
    def test_digital_twin(self) -> bool:
        """Test Digital Twin interface"""
        print("\n[TEST 9] Digital Twin Interface")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Create digital twin
            digital_twin = create_digital_twin(sales_path=self.sales_path, verbose=False)
            
            # Test current state analysis
            analysis = digital_twin.analyze_current_state()
            
            # Test risk alerts
            alerts = digital_twin.get_risk_alerts()
            
            # Test executive summary
            executive = digital_twin.generate_executive_summary()
            
            details = f"Digital twin operational, risk status: {alerts['alert_summary']['overall_status']}"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Digital Twin Interface", passed, time.time() - start_time, details)
        return passed
    
    def test_unified_pipeline(self) -> bool:
        """Test Unified Pipeline"""
        print("\n[TEST 10] Unified Pipeline")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Create pipeline
            pipeline = SupplyChainPipeline(sales_path=self.sales_path, verbose=False)
            
            # Run full pipeline
            results = pipeline.run_full_pipeline()
            
            # Validate results
            if results.get('pipeline_status') != 'success':
                raise ValueError(f"Pipeline failed: {results.get('error_message', 'unknown')}")
            
            # Test scenario analysis
            from scenarios import ScenarioInput
            scenario = ScenarioInput(demand_spike=0.15)
            scenario_results = pipeline.run_scenario_analysis(scenario)
            
            details = f"Pipeline completed successfully, recommendation: {results.get('recommendation', {}).get('action', 'unknown')}"
            passed = True
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Unified Pipeline", passed, time.time() - start_time, details)
        return passed
    
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks"""
        print("\n[TEST 11] Performance Benchmarks")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Test quick analysis performance
            quick_start = time.time()
            quick_result = quick_analysis(sales_path=self.sales_path, verbose=False)
            quick_duration = time.time() - quick_start
            
            # Test XAI performance
            enhanced_xai = create_enhanced_xai(enable_enhancement=True)
            test_state = {
                'predicted_demand': 500000,
                'current_inventory': 25000,
                'supplier_risk_score': 0.75
            }
            
            xai_start = time.time()
            risk_exp = enhanced_xai.explain_risk_score(test_state)
            xai_duration = time.time() - xai_start
            
            # Performance criteria
            quick_acceptable = quick_duration < 30.0  # 30 seconds
            xai_acceptable = xai_duration < 5.0      # 5 seconds
            
            passed = quick_acceptable and xai_acceptable
            
            details = f"Quick analysis: {quick_duration:.1f}s, XAI: {xai_duration:.1f}s"
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Performance Benchmarks", passed, time.time() - start_time, details)
        return passed
    
    def test_reliability_fallbacks(self) -> bool:
        """Test reliability and fallback mechanisms"""
        print("\n[TEST 12] Reliability & Fallbacks")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # Test XAI fallback
            enhanced_xai = create_enhanced_xai(enable_enhancement=True)
            
            # Test health check
            health = enhanced_xai.health_check()
            
            # Test with invalid state (should handle gracefully)
            try:
                invalid_state = {'invalid': 'data'}
                risk_exp = enhanced_xai.explain_risk_score(invalid_state)
                fallback_works = True  # If it doesn't crash, fallback works
            except:
                fallback_works = False
            
            # Test cache reliability
            enhanced_xai.clear_cache()
            cache_stats = enhanced_xai.get_performance_stats().get('cache_stats', {})
            
            passed = health.get('status') in ['healthy', 'degraded'] and fallback_works
            details = f"Health status: {health.get('status', 'unknown')}, fallback works: {fallback_works}"
            
        except Exception as e:
            details = f"Error: {str(e)}"
            passed = False
        
        self.log_test("Reliability & Fallbacks", passed, time.time() - start_time, details)
        return passed
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        print(f"\n{'='*80}")
        print("RUNNING ALL TESTS")
        print(f"{'='*80}")
        
        # Test sequence
        tests = [
            self.test_prerequisites,
            self.test_layer1_data_collection,
            self.test_layer2_feature_engineering,
            self.test_layer3_integration,
            self.test_layer4_rl_agent,
            self.test_ollama_integration,
            self.test_enhanced_xai,
            self.test_scenario_simulator,
            self.test_digital_twin,
            self.test_unified_pipeline,
            self.test_performance_benchmarks,
            self.test_reliability_fallbacks
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func in tests:
            try:
                if test_func():
                    passed_tests += 1
            except Exception as e:
                print(f"  Test {test_func.__name__} crashed: {e}")
                self.log_test(test_func.__name__, False, 0.0, f"Crash: {str(e)}")
        
        # Generate summary
        total_duration = time.time() - self.start_time
        pass_rate = (passed_tests / total_tests) * 100
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': pass_rate,
            'total_duration': total_duration,
            'test_results': self.test_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Print summary
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Total Duration: {total_duration:.1f} seconds")
        
        if pass_rate >= 90:
            print("Status: EXCELLENT - Product ready for production")
        elif pass_rate >= 75:
            print("Status: GOOD - Product mostly functional")
        elif pass_rate >= 50:
            print("Status: FAIR - Product has significant issues")
        else:
            print("Status: POOR - Product needs major fixes")
        
        print(f"{'='*80}")
        
        return summary
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to file"""
        if filename is None:
            filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Test results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """Main test runner"""
    # Create test suite
    test_suite = ProductTestSuite(verbose=True)
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Save results
    test_suite.save_results(results)
    
    # Return appropriate exit code
    return 0 if results['pass_rate'] >= 75 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
