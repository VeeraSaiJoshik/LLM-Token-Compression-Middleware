"""
Cache Manager
Manages both semantic cache and deduplication cache
"""
import logging
from typing import Dict, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from optimizers.semantic_cache import SemanticCache
from optimizers.deduplicator import RequestDeduplicator

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Unified cache manager

    Manages both:
    1. Semantic cache (embedding-based similarity)
    2. Deduplication cache (exact hash matching)
    """

    def __init__(
        self,
        use_semantic: bool = True,
        use_dedup: bool = True,
        semantic_threshold: float = 0.85,
        dedup_ttl: int = 3600
    ):
        """
        Initialize cache manager

        Args:
            use_semantic: Enable semantic caching
            use_dedup: Enable deduplication
            semantic_threshold: Similarity threshold for semantic cache
            dedup_ttl: TTL for dedup cache in seconds
        """
        self.use_semantic = use_semantic
        self.use_dedup = use_dedup

        if use_semantic:
            try:
                self.semantic_cache = SemanticCache(similarity_threshold=semantic_threshold)
                logger.info("Semantic cache initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize semantic cache: {str(e)}")
                self.use_semantic = False

        if use_dedup:
            self.dedup_cache = RequestDeduplicator(ttl=dedup_ttl)
            logger.info("Deduplication cache initialized")

    def check_cache(self, prompt: str, model: str) -> Optional[Dict]:
        """
        Check both caches for a match

        Priority: Dedup (exact) > Semantic (similar)

        Args:
            prompt: Input prompt
            model: Model name

        Returns:
            Cached result if found, None otherwise
        """
        # First check dedup (exact match)
        if self.use_dedup:
            result = self.dedup_cache.check_duplicate(prompt, model)
            if result:
                logger.info("Cache HIT (deduplication)")
                result["cache_type"] = "dedup"
                return result

        # Then check semantic (similar match)
        if self.use_semantic:
            result = self.semantic_cache.check_cache(prompt)
            if result:
                logger.info("Cache HIT (semantic)")
                result["cache_type"] = "semantic"
                return result

        logger.debug("Cache MISS")
        return None

    def add_to_cache(self, prompt: str, model: str, response: str, tokens: Dict):
        """
        Add to both caches

        Args:
            prompt: Input prompt
            model: Model name
            response: LLM response
            tokens: Token usage dict
        """
        if self.use_dedup:
            self.dedup_cache.add_request(prompt, model, response, tokens)

        if self.use_semantic:
            self.semantic_cache.add_to_cache(prompt, response, tokens)

        logger.debug("Added to cache")

    def clear_all_caches(self):
        """Clear all caches"""
        if self.use_dedup:
            self.dedup_cache.clear_cache()

        if self.use_semantic:
            self.semantic_cache.clear_cache()

        logger.info("All caches cleared")

    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics

        Returns:
            Dict with cache stats
        """
        stats = {
            "semantic_enabled": self.use_semantic,
            "dedup_enabled": self.use_dedup,
        }

        if self.use_semantic:
            stats["semantic_entries"] = self.semantic_cache.cache_size()

        if self.use_dedup:
            stats["dedup_entries"] = self.dedup_cache.cache_size()

        return stats
