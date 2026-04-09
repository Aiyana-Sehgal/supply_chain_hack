"""
Enhanced Explainable AI Package

Provides enhanced explanations using rule-based logic combined with Ollama Llama 3.1
for business context and executive-friendly language.
"""

from .ollama_client import OllamaClient, OllamaConfig, create_ollama_client
from .cache_manager import CacheManager, CacheConfig, create_cache_manager
from .enhanced_explainer import EnhancedXAI, ExplanationResult, EnhancedXAIConfig, create_enhanced_xai

__all__ = [
    'OllamaClient',
    'OllamaConfig', 
    'create_ollama_client',
    'CacheManager',
    'CacheConfig',
    'create_cache_manager',
    'EnhancedXAI',
    'ExplanationResult',
    'EnhancedXAIConfig',
    'create_enhanced_xai'
]
