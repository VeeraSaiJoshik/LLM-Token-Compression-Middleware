"""
Semantic Caching using embeddings
Detects similar prompts and returns cached responses
"""
import os
import logging
from typing import Optional, Dict
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic cache using embedding similarity

    Stores prompt embeddings and responses. When a new prompt comes in,
    computes its embedding and checks for similar cached prompts.
    """

    def __init__(self, similarity_threshold: float = 0.85, embedding_model: str = "text-embedding-3-small"):
        """
        Initialize semantic cache

        Args:
            similarity_threshold: Minimum cosine similarity to consider a cache hit (0-1)
            embedding_model: OpenAI embedding model to use
        """
        self.threshold = similarity_threshold
        self.embedding_model = embedding_model
        self.cache = {}  # embedding_tuple -> cache_entry
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text

        Args:
            text: Input text

        Returns:
            Numpy array of embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            raise

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            a: First vector
            b: Second vector

        Returns:
            Similarity score (0-1)
        """
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def check_cache(self, prompt: str) -> Optional[Dict]:
        """
        Check if similar prompt exists in cache

        Args:
            prompt: Input prompt to check

        Returns:
            Cached result if found (with similarity score), None otherwise
        """
        if not self.cache:
            logger.debug("Cache is empty")
            return None

        try:
            embedding = self.get_embedding(prompt)

            best_match = None
            best_similarity = 0.0

            for cached_emb_tuple, cached_data in self.cache.items():
                cached_emb = np.array(cached_emb_tuple)
                similarity = self.cosine_similarity(embedding, cached_emb)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_data

            if best_similarity >= self.threshold:
                logger.info(
                    f"Cache HIT - Similarity: {best_similarity:.3f} "
                    f"(threshold: {self.threshold})"
                )
                return {
                    "hit": True,
                    "response": best_match["response"],
                    "tokens_saved": best_match["tokens"],
                    "similarity": best_similarity,
                    "original_prompt": best_match["prompt"],
                }

            logger.debug(
                f"Cache MISS - Best similarity: {best_similarity:.3f} "
                f"(threshold: {self.threshold})"
            )
            return None

        except Exception as e:
            logger.error(f"Cache check failed: {str(e)}")
            return None

    def add_to_cache(self, prompt: str, response: str, tokens: Dict):
        """
        Add prompt and response to cache

        Args:
            prompt: Input prompt
            response: LLM response
            tokens: Token usage dict from API response
        """
        try:
            embedding = self.get_embedding(prompt)
            embedding_tuple = tuple(embedding)

            self.cache[embedding_tuple] = {
                "prompt": prompt,
                "response": response,
                "tokens": tokens,
            }

            logger.info(f"Added to cache - Total entries: {len(self.cache)}")

        except Exception as e:
            logger.error(f"Failed to add to cache: {str(e)}")

    def clear_cache(self):
        """Clear all cached entries"""
        self.cache.clear()
        logger.info("Cache cleared")

    def cache_size(self) -> int:
        """Get number of cached entries"""
        return len(self.cache)
