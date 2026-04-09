"""
Enhanced Explainable AI System

Combines rule-based explanations with Ollama Llama 3.1 enhancements to provide
contextual, business-focused explanations with caching and fallback mechanisms.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time

from .ollama_client import OllamaClient, OllamaUnavailable, OllamaConfig
from .cache_manager import CacheManager, CacheConfig
from .xai_explainer import XAIExplainer as RuleBasedExplainer

logger = logging.getLogger(__name__)


@dataclass
class ExplanationResult:
    """Result of explanation generation"""
    explanation: str
    source: str  # 'rule_based', 'ollama_enhanced', 'cached'
    confidence: float
    processing_time: float
    cache_hit: bool = False
    fallback_used: bool = False


@dataclass
class EnhancedXAIConfig:
    """Configuration for enhanced XAI system"""
    # Ollama configuration
    ollama_config: OllamaConfig = None
    
    # Cache configuration
    cache_config: CacheConfig = None
    
    # Enhancement settings
    enable_enhancement: bool = True
    fallback_on_error: bool = True
    max_enhancement_time: float = 5.0  # seconds
    
    # Quality thresholds
    min_enhancement_confidence: float = 0.5
    cache_ttl: float = 86400  # 24 hours


class EnhancedXAI:
    """
    Enhanced Explainable AI system that combines rule-based explanations
    with Ollama Llama 3.1 enhancements for business context and executive-friendly language.
    
    Features:
    - Hybrid approach: Rule-based + AI enhancement
    - Intelligent caching for performance
    - Graceful fallback to rule-based system
    - Performance monitoring and optimization
    - Business context enhancement
    """
    
    def __init__(self, config: EnhancedXAIConfig = None):
        """Initialize enhanced XAI system"""
        self.config = config or EnhancedXAIConfig()
        
        # Initialize components
        self.ollama_client = OllamaClient(self.config.ollama_config) if self.config.enable_enhancement else None
        self.cache_manager = CacheManager(self.config.cache_config)
        
        # Import rule-based explainer
        try:
            self.rule_explainer = RuleBasedExplainer()
            logger.info("Rule-based explainer loaded successfully")
        except ImportError as e:
            logger.error(f"Could not import rule-based explainer: {e}")
            self.rule_explainer = None
        
        # Performance tracking
        self._stats = {
            'total_explanations': 0,
            'enhanced_explanations': 0,
            'rule_based_explanations': 0,
            'cache_hits': 0,
            'fallbacks_used': 0,
            'errors': 0
        }
        
        logger.info("Enhanced XAI system initialized")
    
    def explain_risk_score(self, state: Dict[str, Any]) -> Dict[str, str]:
        """Generate enhanced risk score explanations"""
        if not self.rule_explainer:
            return {"error": "Rule-based explainer not available"}
        
        # Get rule-based explanations
        rule_explanations = self.rule_explainer.explain_risk_score(state)
        
        if not self.config.enable_enhancement:
            return rule_explanations
        
        enhanced_explanations = {}
        
        for risk_type, base_explanation in rule_explanations.items():
            try:
                enhanced_result = self._enhance_explanation(
                    base_explanation, state, "risk_analysis", 0.8
                )
                enhanced_explanations[risk_type] = enhanced_result.explanation
                
            except Exception as e:
                logger.warning(f"Failed to enhance {risk_type} explanation: {e}")
                enhanced_explanations[risk_type] = base_explanation
        
        return enhanced_explanations
    
    def explain_recommendation(self, state: Dict[str, Any], action: str, confidence: float) -> Dict[str, Any]:
        """Generate enhanced recommendation explanation"""
        if not self.rule_explainer:
            return {"error": "Rule-based explainer not available"}
        
        # Get rule-based explanation
        rule_explanation = self.rule_explainer.explain_recommendation(state, action, confidence)
        
        if not self.config.enable_enhancement:
            return rule_explanation
        
        try:
            # Enhance the rationale
            enhanced_result = self._enhance_explanation(
                rule_explanation['rationale'], state, action, confidence
            )
            
            # Update explanation with enhanced rationale
            enhanced_explanation = rule_explanation.copy()
            enhanced_explanation['rationale'] = enhanced_result.explanation
            enhanced_explanation['enhanced'] = True
            enhanced_explanation['enhancement_source'] = enhanced_result.source
            
            return enhanced_explanation
            
        except Exception as e:
            logger.warning(f"Failed to enhance recommendation explanation: {e}")
            return rule_explanation
    
    def explain_scenario_impact(self, scenario_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate enhanced scenario impact explanations"""
        if not self.rule_explainer:
            return {"error": "Rule-based explainer not available"}
        
        # Get rule-based explanation
        rule_explanation = self.rule_explainer.explain_scenario_impact(scenario_results)
        
        if not self.config.enable_enhancement:
            return rule_explanation
        
        try:
            # Create a summary for enhancement
            scenario_summary = rule_explanation.get('scenario_summary', '')
            state = scenario_results.get('state_details', {})
            
            enhanced_result = self._enhance_explanation(
                scenario_summary, state, "scenario_analysis", 0.7
            )
            
            # Update explanation
            enhanced_explanation = rule_explanation.copy()
            enhanced_explanation['scenario_summary'] = enhanced_result.explanation
            enhanced_explanation['enhanced'] = True
            enhanced_explanation['enhancement_source'] = enhanced_result.source
            
            return enhanced_explanation
            
        except Exception as e:
            logger.warning(f"Failed to enhance scenario explanation: {e}")
            return rule_explanation
    
    def _enhance_explanation(self, base_explanation: str, state: Dict[str, Any], 
                           action: str, confidence: float) -> ExplanationResult:
        """
        Enhance explanation using Ollama with caching and fallback
        
        Parameters:
        -----------
        base_explanation : str
            Rule-based explanation to enhance
        state : Dict[str, Any]
            Current state for context
        action : str
            Action or analysis type
        confidence : float
            Confidence level
        
        Returns:
        --------
        ExplanationResult
            Enhanced explanation result with metadata
        """
        start_time = time.time()
        self._stats['total_explanations'] += 1
        
        # Check cache first
        cached_result = self.cache_manager.get(state, action, confidence)
        if cached_result:
            self._stats['cache_hits'] += 1
            processing_time = time.time() - start_time
            
            return ExplanationResult(
                explanation=cached_result.enhanced_explanation,
                source='cached',
                confidence=confidence,
                processing_time=processing_time,
                cache_hit=True
            )
        
        # Try to enhance with Ollama
        if self.ollama_client and self.config.enable_enhancement:
            try:
                enhanced_text = self.ollama_client.generate_enhanced_explanation(
                    base_explanation, state, action, confidence
                )
                
                processing_time = time.time() - start_time
                
                # Cache the enhanced explanation
                self.cache_manager.put(
                    state, action, confidence,
                    base_explanation, enhanced_text, 'ollama'
                )
                
                self._stats['enhanced_explanations'] += 1
                
                return ExplanationResult(
                    explanation=enhanced_text,
                    source='ollama_enhanced',
                    confidence=confidence,
                    processing_time=processing_time,
                    cache_hit=False,
                    fallback_used=False
                )
                
            except OllamaUnavailable as e:
                logger.warning(f"Ollama unavailable, using rule-based: {e}")
                self._stats['fallbacks_used'] += 1
                
            except Exception as e:
                logger.error(f"Error enhancing explanation: {e}")
                self._stats['errors'] += 1
        
        # Fallback to rule-based
        if self.config.fallback_on_error:
            self._stats['rule_based_explanations'] += 1
            processing_time = time.time() - start_time
            
            # Cache rule-based explanation for future use
            self.cache_manager.put(
                state, action, confidence,
                base_explanation, base_explanation, 'rule_based'
            )
            
            return ExplanationResult(
                explanation=base_explanation,
                source='rule_based',
                confidence=confidence,
                processing_time=processing_time,
                cache_hit=False,
                fallback_used=True
            )
        
        raise RuntimeError("Failed to generate explanation and fallback disabled")
    
    def generate_comprehensive_explanation(self, state: Dict[str, Any], action: str, 
                                         confidence: float, scenario_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive explanation package with all components"""
        explanation_package = {
            'timestamp': time.time(),
            'state_summary': state,
            'recommendation': action,
            'confidence': confidence
        }
        
        # Risk explanations
        risk_explanations = self.explain_risk_score(state)
        explanation_package['risk_explanations'] = risk_explanations
        
        # Recommendation explanation
        rec_explanation = self.explain_recommendation(state, action, confidence)
        explanation_package['recommendation_explanation'] = rec_explanation
        
        # Scenario explanation (if provided)
        if scenario_results:
            scenario_explanation = self.explain_scenario_impact(scenario_results)
            explanation_package['scenario_explanation'] = scenario_explanation
        
        # System performance
        explanation_package['system_performance'] = self.get_performance_stats()
        
        return explanation_package
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total = self._stats['total_explanations']
        
        if total > 0:
            stats = {
                **self._stats,
                'enhancement_rate': self._stats['enhanced_explanations'] / total,
                'cache_hit_rate': self._stats['cache_hits'] / total,
                'fallback_rate': self._stats['fallbacks_used'] / total,
                'error_rate': self._stats['errors'] / total
            }
        else:
            stats = self._stats.copy()
            stats.update({
                'enhancement_rate': 0.0,
                'cache_hit_rate': 0.0,
                'fallback_rate': 0.0,
                'error_rate': 0.0
            })
        
        # Add cache stats
        cache_stats = self.cache_manager.get_stats()
        stats['cache_stats'] = cache_stats
        
        # Add Ollama status
        if self.ollama_client:
            ollama_test = self.ollama_client.test_connection()
            stats['ollama_status'] = ollama_test
        else:
            stats['ollama_status'] = {'healthy': False, 'message': 'Ollama client disabled'}
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health = {
            'timestamp': time.time(),
            'status': 'healthy',
            'components': {}
        }
        
        # Check rule-based explainer
        if self.rule_explainer:
            health['components']['rule_based'] = {'status': 'healthy'}
        else:
            health['components']['rule_based'] = {'status': 'unhealthy', 'message': 'Not available'}
            health['status'] = 'degraded'
        
        # Check Ollama
        if self.ollama_client:
            ollama_test = self.ollama_client.test_connection()
            health['components']['ollama'] = ollama_test
            if not ollama_test.get('healthy', False):
                health['status'] = 'degraded'
        else:
            health['components']['ollama'] = {'status': 'disabled', 'message': 'Enhancement disabled'}
        
        # Check cache
        cache_stats = self.cache_manager.get_stats()
        health['components']['cache'] = {
            'status': 'healthy',
            'entries': cache_stats['cache_entries'],
            'hit_rate': cache_stats['hit_rate']
        }
        
        return health
    
    def save_state(self):
        """Save cache and stats"""
        self.cache_manager.save()
        logger.info("Enhanced XAI state saved")
    
    def clear_cache(self):
        """Clear all cached explanations"""
        self.cache_manager.clear_cache()
        logger.info("Enhanced XAI cache cleared")
    
    def reset_stats(self):
        """Reset performance statistics"""
        self._stats = {
            'total_explanations': 0,
            'enhanced_explanations': 0,
            'rule_based_explanations': 0,
            'cache_hits': 0,
            'fallbacks_used': 0,
            'errors': 0
        }
        logger.info("Enhanced XAI statistics reset")


# Factory function
def create_enhanced_xai(ollama_host: str = "localhost", ollama_port: int = 11434,
                       ollama_model: str = "llama3.1:latest", cache_dir: str = "cache/xai",
                       enable_enhancement: bool = True) -> EnhancedXAI:
    """Create enhanced XAI system with default configuration"""
    
    ollama_config = OllamaConfig(host=ollama_host, port=ollama_port, model=ollama_model)
    cache_config = CacheConfig(cache_dir=cache_dir)
    xai_config = EnhancedXAIConfig(
        ollama_config=ollama_config,
        cache_config=cache_config,
        enable_enhancement=enable_enhancement
    )
    
    return EnhancedXAI(xai_config)


# Test function
def test_enhanced_xai():
    """Test enhanced XAI system"""
    print("Testing enhanced XAI system...")
    
    # Create enhanced XAI
    xai = create_enhanced_xai(enable_enhancement=True)
    
    # Test state
    test_state = {
        'predicted_demand': 500000,
        'current_inventory': 25000,
        'supplier_risk_score': 0.75,
        'disruption_signal': 0.65,
        'days_to_stockout': 5
    }
    
    # Test risk explanation
    print("\nTesting risk explanation...")
    risk_exp = xai.explain_risk_score(test_state)
    print(f"Risk explanations: {len(risk_exp)} categories")
    
    # Test recommendation explanation
    print("\nTesting recommendation explanation...")
    rec_exp = xai.explain_recommendation(test_state, "switch_supplier", 78.5)
    print(f"Recommendation: {rec_exp.get('rationale', 'N/A')}")
    
    # Test performance stats
    print("\nPerformance stats:")
    stats = xai.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")
    
    # Test health check
    print("\nHealth check:")
    health = xai.health_check()
    print(f"  Status: {health['status']}")
    for component, status in health['components'].items():
        print(f"  {component}: {status.get('status', 'unknown')}")
    
    # Save state
    xai.save_state()
    print("\nEnhanced XAI test completed!")


if __name__ == "__main__":
    test_enhanced_xai()
