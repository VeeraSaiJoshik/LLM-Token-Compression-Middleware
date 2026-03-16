"""
Whitespace Optimizer - Removes excessive whitespace without affecting accuracy.

This is a zero-impact optimization that normalizes whitespace to reduce token count
while preserving all meaningful content.
"""

import re
from typing import Dict


def optimize_whitespace(prompt: str) -> Dict[str, any]:
    """
    Normalize whitespace in prompt to reduce token count.

    Args:
        prompt: Original prompt text

    Returns:
        Dict with optimized prompt and token savings
    """
    original_length = len(prompt)

    # Replace multiple spaces with single space
    optimized = re.sub(r' {2,}', ' ', prompt)

    # Replace multiple newlines with max 2 (preserve paragraph breaks)
    optimized = re.sub(r'\n{3,}', '\n\n', optimized)

    # Remove trailing whitespace from each line
    optimized = '\n'.join(line.rstrip() for line in optimized.split('\n'))

    # Remove leading/trailing whitespace from entire prompt
    optimized = optimized.strip()

    # Remove spaces before punctuation
    optimized = re.sub(r'\s+([.,;:!?])', r'\1', optimized)

    # Normalize spaces around special characters (but preserve code blocks)
    # Only do this outside of code blocks
    if '```' not in optimized:
        optimized = re.sub(r'\s*([<>{}[\]()])\s*', r'\1', optimized)

    new_length = len(optimized)
    chars_saved = original_length - new_length

    return {
        "optimized_prompt": optimized,
        "chars_saved": chars_saved,
        "estimated_tokens_saved": chars_saved // 4,  # Rough estimate
        "description": f"Normalized whitespace (saved ~{chars_saved} chars)"
    }
