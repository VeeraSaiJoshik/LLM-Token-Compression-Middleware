"""
Cost Calculator
Calculates costs using ACTUAL token counts from API responses
"""
import logging
from typing import Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PRICING
from providers import TokenCost

logger = logging.getLogger(__name__)


def calculate_cost(model: str, actual_tokens: TokenCost) -> Dict:
    """
    Calculate costs using actual token counts from API response

    CRITICAL: This uses actual token counts from the API, not estimates

    Args:
        model: Model name (e.g., 'gpt-4o', 'claude-sonnet-4-5')
        actual_tokens: TokenCost object with token counts

    Returns:
        Dict with cost breakdown
    """
    if model not in PRICING:
        logger.warning(f"Model {model} not found in pricing config")
        return {
            "input": 0.0,
            "output": 0.0,
            "cache_write": 0.0,
            "cache_read": 0.0,
            "total": 0.0,
            "error": f"Model {model} not in pricing config"
        }

    pricing = PRICING[model]
    provider = pricing["provider"]

    costs = {
        "input": 0.0,
        "output": 0.0,
        "cache_write": 0.0,
        "cache_read": 0.0,
        "total": 0.0
    }

    try:
        if provider == "openai":
            costs = _calculate_openai_cost(pricing, actual_tokens)
        elif provider == "anthropic":
            costs = _calculate_anthropic_cost(pricing, actual_tokens)
        elif provider == "google":
            costs = _calculate_google_cost(pricing, actual_tokens)
        else:
            logger.error(f"Unknown provider: {provider}")

        costs["total"] = sum([
            costs["input"],
            costs["output"],
            costs["cache_write"],
            costs["cache_read"]
        ])

        logger.debug(
            f"Cost calculated for {model}: "
            f"${costs['total']:.6f} "
            f"(input: ${costs['input']:.6f}, output: ${costs['output']:.6f})"
        )

    except Exception as e:
        logger.error(f"Cost calculation failed: {str(e)}")
        costs["error"] = str(e)

    return costs


def _calculate_openai_cost(pricing: Dict, tokens: TokenCost) -> Dict:
    """Calculate cost for OpenAI models"""
    costs = {
        "input": 0.0,
        "output": 0.0,
        "cache_write": 0.0,
        "cache_read": 0.0,
    }

    prompt_tokens = tokens.prompt_tokens
    completion_tokens = tokens.completion_tokens
    cached_tokens = tokens.cached_tokens

    # Regular input tokens (excluding cached)
    uncached_input = prompt_tokens - cached_tokens
    costs["input"] = (uncached_input / 1_000_000) * pricing["input"]

    # Cached tokens (if supported)
    if cached_tokens > 0 and pricing.get("cached_input"):
        costs["cache_read"] = (cached_tokens / 1_000_000) * pricing["cached_input"]

    # Output tokens
    costs["output"] = (completion_tokens / 1_000_000) * pricing["output"]

    return costs


def _calculate_anthropic_cost(pricing: Dict, tokens: TokenCost) -> Dict:
    """Calculate cost for Anthropic models"""
    costs = {
        "input": 0.0,
        "output": 0.0,
        "cache_write": 0.0,
        "cache_read": 0.0,
    }

    input_tokens = tokens.prompt_tokens
    output_tokens = tokens.completion_tokens
    cache_creation = getattr(tokens, 'cache_creation_tokens', 0)
    cache_read = tokens.cached_tokens

    # Regular input tokens (not from cache)
    regular_input = input_tokens - cache_read
    costs["input"] = (regular_input / 1_000_000) * pricing["input"]

    # Cache write (creation)
    if cache_creation > 0 and pricing.get("cache_write_5m"):
        costs["cache_write"] = (cache_creation / 1_000_000) * pricing["cache_write_5m"]

    # Cache read
    if cache_read > 0 and pricing.get("cache_read"):
        costs["cache_read"] = (cache_read / 1_000_000) * pricing["cache_read"]

    # Output tokens
    costs["output"] = (output_tokens / 1_000_000) * pricing["output"]

    return costs


def _calculate_google_cost(pricing: Dict, tokens: TokenCost) -> Dict:
    """Calculate cost for Google Gemini models"""
    costs = {
        "input": 0.0,
        "output": 0.0,
        "cache_write": 0.0,
        "cache_read": 0.0,
    }

    prompt_tokens = tokens.prompt_tokens
    completion_tokens = tokens.completion_tokens
    cached_tokens = tokens.cached_tokens

    # Check if over 200K threshold (for Gemini 2.5 Pro)
    if "threshold_200k" in pricing and prompt_tokens > pricing["threshold_200k"]:
        input_price = pricing["input_over_200k"]
        output_price = pricing["output_over_200k"]
        cached_price = pricing.get("cached_input_over_200k", pricing.get("cached_input", 0))
    else:
        input_price = pricing["input"]
        output_price = pricing["output"]
        cached_price = pricing.get("cached_input", 0)

    # Regular input tokens (not from cache)
    uncached_input = prompt_tokens - cached_tokens
    costs["input"] = (uncached_input / 1_000_000) * input_price

    # Cached tokens
    if cached_tokens > 0:
        costs["cache_read"] = (cached_tokens / 1_000_000) * cached_price

    # Output tokens
    costs["output"] = (completion_tokens / 1_000_000) * output_price

    return costs


def compare_costs(baseline_cost: Dict, optimized_cost: Dict) -> Dict:
    """
    Compare baseline and optimized costs

    Args:
        baseline_cost: Cost dict from baseline run
        optimized_cost: Cost dict from optimized run

    Returns:
        Comparison dict with savings
    """
    baseline_total = baseline_cost.get("total", 0)
    optimized_total = optimized_cost.get("total", 0)

    savings = baseline_total - optimized_total
    savings_percent = (savings / baseline_total * 100) if baseline_total > 0 else 0

    return {
        "baseline_cost": baseline_total,
        "optimized_cost": optimized_total,
        "savings": savings,
        "savings_percent": savings_percent,
    }
