"""
Model Router
Routes prompts to appropriate models based on complexity analysis
"""
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)

# Keywords indicating complex tasks
COMPLEXITY_KEYWORDS = {
    "high": [
        "code", "function", "algorithm", "implement", "class", "refactor",
        "analyze", "compare", "explain in detail", "comprehensive",
        "debug", "optimize", "architecture", "design pattern",
        "complex", "advanced", "sophisticated", "intricate"
    ],
    "medium": [
        "write", "create", "develop", "build", "make",
        "explain", "describe", "summarize", "list"
    ],
    "low": [
        "what is", "define", "translate", "convert",
        "simple", "quick", "basic", "brief"
    ]
}


def calculate_complexity_score(prompt: str) -> float:
    """
    Calculate complexity score (0-1) based on heuristics

    Factors considered:
    - Length of prompt
    - Presence of complexity keywords
    - Number of questions/requirements
    - Technical terms
    - Code blocks

    Args:
        prompt: Input prompt to analyze

    Returns:
        Complexity score from 0 (simple) to 1 (complex)
    """
    score = 0.0
    prompt_lower = prompt.lower()

    # 1. Length factor (up to 0.3)
    words = len(prompt.split())
    if words > 500:
        score += 0.3
    elif words > 200:
        score += 0.2
    elif words > 100:
        score += 0.1

    # 2. Complexity keywords (up to 0.4)
    high_count = sum(1 for kw in COMPLEXITY_KEYWORDS["high"] if kw in prompt_lower)
    medium_count = sum(1 for kw in COMPLEXITY_KEYWORDS["medium"] if kw in prompt_lower)
    low_count = sum(1 for kw in COMPLEXITY_KEYWORDS["low"] if kw in prompt_lower)

    if high_count >= 3:
        score += 0.4
    elif high_count >= 1:
        score += 0.3
    elif medium_count >= 2:
        score += 0.2
    elif low_count >= 2:
        score += 0.0  # Indicates simplicity

    # 3. Multiple questions/requirements (up to 0.2)
    question_count = prompt.count("?")
    bullet_points = len(re.findall(r'^[-*]\s', prompt, re.MULTILINE))
    numbered_items = len(re.findall(r'^\d+[\.)]\s', prompt, re.MULTILINE))

    requirements = question_count + bullet_points + numbered_items
    if requirements > 5:
        score += 0.2
    elif requirements > 2:
        score += 0.1

    # 4. Code blocks or technical syntax (up to 0.1)
    has_code_block = bool(re.search(r'```|`[^`]+`', prompt))
    has_technical = bool(re.search(r'\b(API|JSON|SQL|HTTP|REST)\b', prompt, re.IGNORECASE))

    if has_code_block:
        score += 0.1
    elif has_technical:
        score += 0.05

    return min(score, 1.0)


def route_prompt(prompt: str, provider: str = "openai") -> Dict:
    """
    Determine which model to use based on complexity

    Args:
        prompt: Input prompt to route
        provider: Provider to route within (openai, anthropic, google)

    Returns:
        Dict with model selection and routing info
    """
    complexity = calculate_complexity_score(prompt)

    # Model routing tables by provider
    routes = {
        "openai": {
            "simple": "gpt-4o-mini",
            "complex": "gpt-4o"
        },
        "anthropic": {
            "simple": "claude-3-5-haiku",
            "complex": "claude-sonnet-4-5"
        },
        "google": {
            "simple": "gemini-2-0-flash",
            "complex": "gemini-2-5-flash"
        }
    }

    # Determine category
    # Simple: 0.0 - 0.4
    # Complex: 0.5+
    category = "complex" if complexity >= 0.5 else "simple"
    model = routes.get(provider, routes["openai"])[category]

    reasoning = (
        f"Complexity score: {complexity:.2f} -> {category} model. "
        f"Factors: {len(prompt.split())} words"
    )

    logger.info(f"Model routing: {model} (complexity={complexity:.2f})")

    return {
        "model": model,
        "complexity": complexity,
        "category": category,
        "reasoning": reasoning,
        "provider": provider
    }


def should_use_simple_model(prompt: str, threshold: float = 0.5) -> bool:
    """
    Quick check if a simple model is sufficient

    Args:
        prompt: Input prompt
        threshold: Complexity threshold (default 0.5)

    Returns:
        True if simple model should be used
    """
    return calculate_complexity_score(prompt) < threshold
