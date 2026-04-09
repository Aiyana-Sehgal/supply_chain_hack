"""
Cache Manager for Enhanced XAI

Provides caching mechanism for Ollama responses to improve performance
and ensure availability when Ollama service is temporarily unavailable.
"""

import json
import time
import hashlib
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached explanation"""
    cache_key: str
    base_explanation: str
    enhanced_explanation: str
    state_hash: str
    action: str
    confidence: float
    timestamp: float
    ttl: float  # Time to live in seconds
    source: str  # 'ollama' or 'rule_based'
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > (self.timestamp + self.ttl)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class CacheConfig:
    """Configuration for cache manager"""
    cache_dir: str = "cache/xai"
    default_ttl: float = 86400  # 24 hours
    max_cache_size: int = 1000  # Maximum number of entries
    cleanup_interval: float = 3600  # 1 hour
    compression: bool = False  # Enable compression for large caches


class CacheManager:
    """
    Manages caching of enhanced explanations for performance and reliability.
    
    Features:
    - File-based JSON storage (easily upgradeable to Redis)
    - TTL management with automatic cleanup
    - Cache size limits with LRU eviction
    - Hash-based key generation
    - Cache statistics and monitoring
    """
    
    def __init__(self, config: CacheConfig = None):
        """Initialize cache manager with configuration"""
        self.config = config or CacheConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_file = self.cache_dir / "explanations.json"
        self.stats_file = self.cache_dir / "stats.json"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self._cache = self._load_cache()
        self._stats = self._load_stats()
        
        # Track last cleanup
        self._last_cleanup = time.time()
        
        logger.info(f"Cache manager initialized with {len(self._cache)} entries")
    
    def _load_cache(self) -> Dict[str, CacheEntry]:
        """Load cache from file"""
        if not self.cache_file.exists():
            return {}
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cache = {}
            for key, entry_data in data.items():
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired():
                    cache[key] = entry
            
            logger.info(f"Loaded {len(cache)} valid entries from cache")
            return cache
            
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load cache statistics"""
        default_stats = {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'cache_size': 0,
            'last_cleanup': time.time()
        }
        
        if not self.stats_file.exists():
            return default_stats
        
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading stats: {e}")
            return default_stats
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            data = {key: entry.to_dict() for key, entry in self._cache.items()}
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def _save_stats(self):
        """Save statistics to file"""
        try:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self._stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def generate_cache_key(self, state: Dict[str, Any], action: str, confidence: float) -> str:
        """Generate cache key from input parameters"""
        # Create a deterministic representation of the state
        state_items = sorted(state.items())
        state_str = json.dumps(state_items, sort_keys=True, separators=(',', ':'))
        
        # Create hash
        hash_input = f"{state_str}_{action}_{round(confidence, 2)}"
        cache_key = hashlib.md5(hash_input.encode()).hexdigest()
        
        return f"xai_{cache_key}"
    
    def generate_state_hash(self, state: Dict[str, Any]) -> str:
        """Generate hash for state comparison"""
        state_items = sorted(state.items())
        state_str = json.dumps(state_items, sort_keys=True, separators=(',', ':'))
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def get(self, state: Dict[str, Any], action: str, confidence: float) -> Optional[CacheEntry]:
        """Get cached explanation if available and not expired"""
        cache_key = self.generate_cache_key(state, action, confidence)
        
        # Update stats
        self._stats['total_requests'] += 1
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            if not entry.is_expired():
                self._stats['hits'] += 1
                logger.debug(f"Cache hit for key: {cache_key}")
                return entry
            else:
                # Remove expired entry
                del self._cache[cache_key]
                logger.debug(f"Cache entry expired: {cache_key}")
        
        self._stats['misses'] += 1
        logger.debug(f"Cache miss for key: {cache_key}")
        return None
    
    def put(self, state: Dict[str, Any], action: str, confidence: float,
            base_explanation: str, enhanced_explanation: str, source: str = 'ollama'):
        """Store explanation in cache"""
        cache_key = self.generate_cache_key(state, action, confidence)
        state_hash = self.generate_state_hash(state)
        
        entry = CacheEntry(
            cache_key=cache_key,
            base_explanation=base_explanation,
            enhanced_explanation=enhanced_explanation,
            state_hash=state_hash,
            action=action,
            confidence=confidence,
            timestamp=time.time(),
            ttl=self.config.default_ttl,
            source=source
        )
        
        # Check cache size limit
        if len(self._cache) >= self.config.max_cache_size:
            self._evict_lru()
        
        self._cache[cache_key] = entry
        self._stats['cache_size'] = len(self._cache)
        
        logger.debug(f"Cached explanation for key: {cache_key}")
        
        # Periodic cleanup
        self._maybe_cleanup()
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        if not self._cache:
            return
        
        # Sort by timestamp (oldest first)
        oldest_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(self._cache) // 10)
        
        for i in range(num_to_remove):
            key, _ = oldest_entries[i]
            del self._cache[key]
        
        logger.info(f"Evicted {num_to_remove} oldest cache entries")
    
    def _maybe_cleanup(self):
        """Perform cleanup if enough time has passed"""
        current_time = time.time()
        
        if current_time - self._last_cleanup > self.config.cleanup_interval:
            self.cleanup_expired()
            self._last_cleanup = current_time
    
    def cleanup_expired(self):
        """Remove expired entries from cache"""
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            self._stats['cache_size'] = len(self._cache)
    
    def clear_cache(self):
        """Clear all cache entries"""
        self._cache.clear()
        self._stats['cache_size'] = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self._stats['total_requests'] > 0:
            hit_rate = self._stats['hits'] / self._stats['total_requests']
        
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'cache_entries': len(self._cache),
            'cache_dir': str(self.cache_dir)
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information"""
        if not self._cache:
            return {
                'total_entries': 0,
                'sources': {},
                'actions': {},
                'age_distribution': {}
            }
        
        # Analyze cache contents
        sources = {}
        actions = {}
        ages = []
        current_time = time.time()
        
        for entry in self._cache.values():
            # Count sources
            source = entry.source
            sources[source] = sources.get(source, 0) + 1
            
            # Count actions
            action = entry.action
            actions[action] = actions.get(action, 0) + 1
            
            # Calculate age
            age = current_time - entry.timestamp
            ages.append(age)
        
        # Age distribution
        if ages:
            ages.sort()
            age_dist = {
                'min_age': ages[0],
                'max_age': ages[-1],
                'avg_age': sum(ages) / len(ages),
                'median_age': ages[len(ages) // 2]
            }
        else:
            age_dist = {}
        
        return {
            'total_entries': len(self._cache),
            'sources': sources,
            'actions': actions,
            'age_distribution': age_dist,
            'cache_size_mb': self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
        }
    
    def save(self):
        """Save cache and stats to disk"""
        self._save_cache()
        self._save_stats()
    
    def export_cache(self, export_path: str):
        """Export cache to specified file"""
        try:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'cache': {key: entry.to_dict() for key, entry in self._cache.items()},
                'stats': self._stats,
                'export_timestamp': time.time(),
                'config': asdict(self.config)
            }
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cache exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Error exporting cache: {e}")
    
    def import_cache(self, import_path: str):
        """Import cache from specified file"""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                raise FileNotFoundError(f"Import file not found: {import_path}")
            
            with open(import_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Import cache entries
            imported_cache = data.get('cache', {})
            for key, entry_data in imported_cache.items():
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired():
                    self._cache[key] = entry
            
            # Update stats
            self._stats.update(data.get('stats', {}))
            self._stats['cache_size'] = len(self._cache)
            
            logger.info(f"Imported {len(imported_cache)} cache entries from {import_path}")
            
        except Exception as e:
            logger.error(f"Error importing cache: {e}")


# Factory function
def create_cache_manager(cache_dir: str = "cache/xai", ttl: float = 86400) -> CacheManager:
    """Create cache manager with default configuration"""
    config = CacheConfig(cache_dir=cache_dir, default_ttl=ttl)
    return CacheManager(config)


# Test function
def test_cache_manager():
    """Test cache manager functionality"""
    print("Testing cache manager...")
    
    cache = create_cache_manager("cache/test")
    
    # Test data
    test_state = {
        'predicted_demand': 500000,
        'current_inventory': 25000,
        'supplier_risk_score': 0.75
    }
    
    # Test cache miss
    result = cache.get(test_state, "switch_supplier", 78.5)
    print(f"Cache miss result: {result}")
    
    # Test cache put
    cache.put(
        test_state, "switch_supplier", 78.5,
        "Base explanation", "Enhanced explanation", "ollama"
    )
    
    # Test cache hit
    result = cache.get(test_state, "switch_supplier", 78.5)
    print(f"Cache hit result: {result.enhanced_explanation if result else 'None'}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Test cleanup
    cache.save()
    print("Cache manager test completed!")


if __name__ == "__main__":
    test_cache_manager()
