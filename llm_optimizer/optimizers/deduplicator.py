"""
Request Deduplication
Detects and handles duplicate requests to avoid redundant API calls
"""
import hashlib
import logging
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)


class RequestDeduplicator:
    """
    Deduplicates identical requests using hash-based caching

    Unlike semantic cache which uses embeddings, this uses exact matching
    via hashing. Useful for detecting identical repeated requests.
    """

    def __init__(self, ttl: int = 3600):
        """
        Initialize deduplicator

        Args:
            ttl: Time-to-live for cached entries in seconds (default 1 hour)
        """
        self.cache = {}  # hash -> (response, tokens, timestamp)
        self.ttl = ttl

    def _hash_prompt(self, prompt: str, model: str) -> str:
        """
        Create hash of prompt and model

        Args:
            prompt: Input prompt
            model: Model name

        Returns:
            SHA256 hash string
        """
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if cache entry has expired"""
        return (time.time() - timestamp) > self.ttl

    def check_duplicate(self, prompt: str, model: str) -> Optional[Dict]:
        """
        Check if this exact request has been made before

        Args:
            prompt: Input prompt
            model: Model name

        Returns:
            Cached result if found and not expired, None otherwise
        """
        prompt_hash = self._hash_prompt(prompt, model)

        if prompt_hash in self.cache:
            cached_data, tokens, timestamp = self.cache[prompt_hash]

            if self._is_expired(timestamp):
                # Remove expired entry
                del self.cache[prompt_hash]
                logger.debug("Cache entry expired")
                return None

            logger.info(f"Duplicate request detected - Hash: {prompt_hash[:16]}...")
            return {
                "hit": True,
                "response": cached_data,
                "tokens_saved": tokens,
                "cached_at": timestamp,
            }

        return None

    def add_request(self, prompt: str, model: str, response: str, tokens: Dict):
        """
        Add request to deduplication cache

        Args:
            prompt: Input prompt
            model: Model name
            response: LLM response
            tokens: Token usage dict
        """
        prompt_hash = self._hash_prompt(prompt, model)
        timestamp = time.time()

        self.cache[prompt_hash] = (response, tokens, timestamp)
        logger.debug(f"Added to dedup cache - Hash: {prompt_hash[:16]}...")

    def clear_cache(self):
        """Clear all cached entries"""
        self.cache.clear()
        logger.info("Deduplication cache cleared")

    def cleanup_expired(self):
        """Remove expired entries from cache"""
        expired_keys = [
            key for key, (_, _, timestamp) in self.cache.items()
            if self._is_expired(timestamp)
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")

    def cache_size(self) -> int:
        """Get number of cached entries"""
        return len(self.cache)
