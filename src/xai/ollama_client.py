"""
Ollama Client for Enhanced Explainable AI

Manages connection to Ollama's Llama 3.1 model for enhancing rule-based explanations
with contextual business insights and executive-friendly language.
"""

import json
import time
import hashlib
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client"""
    host: str = "localhost"
    port: int = 11434
    model: str = "llama3.1:8b"  # Default to 8B parameter model
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class OllamaUnavailable(Exception):
    """Raised when Ollama service is unavailable"""
    pass


class OllamaClient:
    """
    Client for interacting with Ollama's Llama 3.1 model to enhance explanations.
    
    Features:
    - Connection management with retry logic
    - Health checks and service monitoring
    - Optimized prompts for supply chain explanations
    - Response parsing and validation
    """
    
    def __init__(self, config: OllamaConfig = None):
        """Initialize Ollama client with configuration"""
        self.config = config or OllamaConfig()
        self.base_url = f"http://{self.config.host}:{self.config.port}"
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        
        # Track service health
        self._last_health_check = 0
        self._is_healthy = False
        self._health_check_interval = 60  # seconds
        
        logger.info(f"Ollama client initialized for model: {self.config.model}")
    
    def check_service_health(self) -> bool:
        """Check if Ollama service is running and model is available"""
        current_time = time.time()
        
        # Cache health check result
        if current_time - self._last_health_check < self._health_check_interval:
            return self._is_healthy
        
        try:
            # Check if service is running
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                # Check if our model is available
                if self.config.model in model_names:
                    self._is_healthy = True
                    logger.info(f"Ollama service healthy, model {self.config.model} available")
                else:
                    self._is_healthy = False
                    logger.warning(f"Model {self.config.model} not found. Available: {model_names}")
            else:
                self._is_healthy = False
                logger.warning(f"Ollama service returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            self._is_healthy = False
            logger.warning(f"Ollama service unavailable: {e}")
        
        self._last_health_check = current_time
        return self._is_healthy
    
    def ensure_service_running(self) -> bool:
        """Attempt to ensure Ollama service is running"""
        if self.check_service_health():
            return True
        
        logger.info("Attempting to start Ollama service...")
        
        # Note: In production, you might want to use subprocess to start Ollama
        # For now, we'll just check if it becomes available
        for attempt in range(5):
            time.sleep(2)
            if self.check_service_health():
                logger.info("Ollama service is now running")
                return True
        
        logger.error("Could not start Ollama service")
        return False
    
    def _make_request_with_retry(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to Ollama with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}{endpoint}",
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Request failed with status {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        raise OllamaUnavailable(f"Failed to complete request after {self.config.max_retries} attempts")
    
    def generate_enhanced_explanation(self, base_explanation: str, state: Dict[str, Any], 
                                    action: str, confidence: float) -> str:
        """
        Generate enhanced explanation using Llama 3.1
        
        Parameters:
        -----------
        base_explanation : str
            Rule-based explanation to enhance
        state : Dict[str, Any]
            Current supply chain state
        action : str
            Recommended action
        confidence : float
            Confidence level of recommendation
        
        Returns:
        --------
        str
            Enhanced explanation with business context
        """
        if not self.ensure_service_running():
            raise OllamaUnavailable("Ollama service not available")
        
        # Create optimized prompt
        prompt = self._create_enhancement_prompt(base_explanation, state, action, confidence)
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for consistent explanations
                "top_p": 0.9,
                "max_tokens": 500
            }
        }
        
        try:
            response = self._make_request_with_retry("/api/generate", payload)
            enhanced_text = response.get('response', '').strip()
            
            # Validate and clean response
            if enhanced_text:
                logger.info(f"Generated enhanced explanation ({len(enhanced_text)} chars)")
                return self._clean_response(enhanced_text)
            else:
                logger.warning("Empty response from Ollama")
                return base_explanation
                
        except Exception as e:
            logger.error(f"Error generating enhanced explanation: {e}")
            raise OllamaUnavailable(f"Failed to generate explanation: {e}")
    
    def _create_enhancement_prompt(self, base_explanation: str, state: Dict[str, Any], 
                                 action: str, confidence: float) -> str:
        """Create optimized prompt for explanation enhancement"""
        
        # Extract key metrics for context
        demand = state.get('predicted_demand', 0)
        inventory = state.get('current_inventory', 0)
        supplier_risk = state.get('supplier_risk_score', 0)
        disruption_signal = state.get('disruption_signal', 0)
        days_to_stockout = state.get('days_to_stockout', 0)
        
        # Calculate inventory ratio
        inventory_ratio = inventory / max(demand, 1) if demand > 0 else 0
        
        prompt = f"""You are an expert supply chain analyst providing executive-level insights for a retail business. 

ENHANCE the following supply chain explanation to make it more business-focused and actionable for C-suite executives:

BASE EXPLANATION: {base_explanation}

CURRENT STATE:
- Predicted Demand: {demand:,.0f} units
- Current Inventory: {inventory:,.0f} units ({inventory_ratio:.1%} of demand)
- Days of Supply: {days_to_stockout:.1f} days
- Supplier Risk Score: {supplier_risk:.1%}
- Disruption Signal: {disruption_signal:.1%}
- Recommended Action: {action}
- Confidence: {confidence:.1f}%

TASK: Enhance this explanation with:
1. Business impact language (revenue, customer service, market position)
2. Industry-specific terminology and best practices
3. Executive-friendly framing (bottom-line impact)
4. Actionable next steps with clear ownership
5. Risk communication appropriate for senior leadership

Keep the response concise (2-3 sentences max) but comprehensive. Focus on business outcomes and strategic implications.

ENHANCED EXPLANATION:"""

        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean and validate Ollama response"""
        # Remove common artifacts
        cleaned = response.strip()
        
        # Remove any remaining prompt artifacts
        if "ENHANCED EXPLANATION:" in cleaned:
            cleaned = cleaned.split("ENHANCED EXPLANATION:")[-1].strip()
        
        # Ensure it's not too long or too short
        if len(cleaned) < 20:
            return "Enhanced explanation unavailable - using standard analysis"
        elif len(cleaned) > 600:
            cleaned = cleaned[:597] + "..."
        
        return cleaned
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to Ollama service"""
        try:
            start_time = time.time()
            health = self.check_service_health()
            response_time = time.time() - start_time
            
            model_info = self.get_model_info() if health else {"error": "Service unavailable"}
            
            return {
                "healthy": health,
                "response_time": response_time,
                "model": self.config.model,
                "model_available": self.config.model in [m.get('name', '') for m in model_info.get('models', [])],
                "service_url": self.base_url,
                "models": model_info
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "service_url": self.base_url
            }


# Factory function for easy instantiation
def create_ollama_client(host: str = "localhost", port: int = 11434, 
                        model: str = "llama3.1:8b") -> OllamaClient:
    """Create Ollama client with default configuration"""
    config = OllamaConfig(host=host, port=port, model=model)
    return OllamaClient(config)


# Test function
def test_ollama_integration():
    """Test Ollama integration with sample data"""
    print("Testing Ollama integration...")
    
    client = create_ollama_client()
    
    # Test connection
    connection_test = client.test_connection()
    print(f"Connection test: {connection_test}")
    
    if not connection_test.get('healthy', False):
        print("Ollama service not available - skipping explanation test")
        return
    
    # Test explanation enhancement
    sample_state = {
        'predicted_demand': 500000,
        'current_inventory': 25000,
        'supplier_risk_score': 0.75,
        'disruption_signal': 0.65,
        'days_to_stockout': 5
    }
    
    base_explanation = "High supplier risk detected - recommend switching supplier"
    
    try:
        enhanced = client.generate_enhanced_explanation(
            base_explanation, sample_state, "switch_supplier", 78.5
        )
        print(f"Enhanced explanation: {enhanced}")
        print("Ollama integration test successful!")
        
    except Exception as e:
        print(f"Explanation enhancement failed: {e}")


if __name__ == "__main__":
    test_ollama_integration()
