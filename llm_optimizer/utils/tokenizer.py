"""
Token estimation using tiktoken
NOTE: These are ESTIMATES only - actual token counts come from API responses
"""
import logging
from typing import Optional
import tiktoken

logger = logging.getLogger(__name__)

# Default encodings for different providers
PROVIDER_ENCODINGS = {
    "openai": "cl100k_base",  # Used by GPT-4, GPT-3.5-turbo
    "anthropic": "cl100k_base",  # Similar tokenization
    "google": "cl100k_base",  # Approximate
}

# Model-specific encodings
MODEL_ENCODINGS = {
    "gpt-4": "cl100k_base",
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-3.5-turbo": "cl100k_base",
}


def get_encoding_for_model(model: str) -> str:
    """
    Get the appropriate encoding for a model

    Args:
        model: Model name or ID

    Returns:
        Encoding name
    """
    # Check for exact model match
    for model_prefix, encoding in MODEL_ENCODINGS.items():
        if model.startswith(model_prefix):
            return encoding

    # Default to cl100k_base
    return "cl100k_base"


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Estimate token count for text

    IMPORTANT: This is an ESTIMATE only. Actual token counts must come
    from the API response usage object.

    Args:
        text: Text to estimate tokens for
        model: Optional model name for accurate encoding

    Returns:
        Estimated token count
    """
    try:
        if model:
            encoding_name = get_encoding_for_model(model)
        else:
            encoding_name = "cl100k_base"

        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        count = len(tokens)

        logger.debug(f"Estimated {count} tokens using {encoding_name} encoding")
        return count

    except Exception as e:
        logger.warning(f"Token estimation failed: {str(e)}, using character approximation")
        # Fallback: rough approximation
        return len(text) // 4


def estimate_tokens_for_messages(messages: list, model: Optional[str] = None) -> int:
    """
    Estimate tokens for a list of messages (chat format)

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Optional model name

    Returns:
        Estimated token count
    """
    # This is a simplified estimation
    # Real API adds overhead for message formatting
    total = 0

    for message in messages:
        # Add tokens for role and content
        total += estimate_tokens(message.get("role", ""), model)
        total += estimate_tokens(message.get("content", ""), model)
        # Add overhead (approximate)
        total += 4  # Tokens per message for formatting

    # Add overhead for the messages wrapper
    total += 3

    return total


def compare_token_estimates(
    text1: str,
    text2: str,
    model: Optional[str] = None
) -> dict:
    """
    Compare token estimates for two texts

    Args:
        text1: First text
        text2: Second text
        model: Optional model name

    Returns:
        Dict with comparison data
    """
    tokens1 = estimate_tokens(text1, model)
    tokens2 = estimate_tokens(text2, model)
    difference = tokens1 - tokens2
    percent_change = (difference / tokens1 * 100) if tokens1 > 0 else 0

    return {
        "original_tokens": tokens1,
        "optimized_tokens": tokens2,
        "tokens_saved": difference,
        "percent_saved": percent_change,
    }
