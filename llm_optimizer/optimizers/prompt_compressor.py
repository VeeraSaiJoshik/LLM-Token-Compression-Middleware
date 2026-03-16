"""
Deterministic Prompt Compression
Removes verbose patterns and unnecessary words without changing meaning
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Patterns to remove or simplify (pattern, replacement)
VERBOSE_PATTERNS = [
    # Remove polite filler words
    (r'\b(please|kindly)\b', ''),
    (r'\b(could you|can you|would you)\b', ''),
    (r'\b(I would like you to|I want you to|I need you to)\b', ''),

    # Simplify common phrases
    (r'\b(in order to)\b', 'to'),
    (r'\b(due to the fact that)\b', 'because'),
    (r'\b(at this point in time)\b', 'now'),
    (r'\b(for the purpose of)\b', 'for'),

    # Remove redundant words
    (r'\b(very|really|quite|just|actually|basically|simply)\b', ''),

    # Multiple spaces to single
    (r'\s+', ' '),

    # Multiple newlines to double (preserve paragraph breaks)
    (r'\n\s*\n\s*\n+', '\n\n'),
]

# Aggressive compression patterns (use with care)
AGGRESSIVE_PATTERNS = [
    # Remove articles where safe
    (r'\ba\s+', ''),
    (r'\ban\s+', ''),
    (r'\bthe\s+', ''),

    # Contract common verbs
    (r'\bis not\b', "isn't"),
    (r'\bare not\b', "aren't"),
    (r'\bdo not\b', "don't"),
    (r'\bdoes not\b', "doesn't"),
    (r'\bwill not\b', "won't"),
    (r'\bcannot\b', "can't"),
]


def compress_prompt(prompt: str, aggressive: bool = False) -> Tuple[str, int]:
    """
    Apply deterministic compression rules

    Args:
        prompt: Input prompt to compress
        aggressive: If True, apply more aggressive compression (may affect meaning)

    Returns:
        Tuple of (compressed_prompt, estimated_tokens_saved)
    """
    original_length = len(prompt)
    compressed = prompt

    # Apply standard patterns
    for pattern, replacement in VERBOSE_PATTERNS:
        compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

    # Apply aggressive patterns if requested
    if aggressive:
        for pattern, replacement in AGGRESSIVE_PATTERNS:
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

    # Clean up extra whitespace
    compressed = compressed.strip()
    compressed = re.sub(r'\s+', ' ', compressed)
    compressed = re.sub(r'\n\s*\n\s*\n+', '\n\n', compressed)

    # Estimate tokens saved (rough: character difference / 4)
    chars_saved = original_length - len(compressed)
    tokens_saved = max(0, chars_saved // 4)

    logger.info(
        f"Prompt compression: {original_length} chars -> {len(compressed)} chars "
        f"(~{tokens_saved} tokens saved estimate)"
    )

    return compressed, tokens_saved


def compress_prompt_conservative(prompt: str) -> Tuple[str, int]:
    """
    Conservative compression that preserves meaning

    Returns:
        Tuple of (compressed_prompt, estimated_tokens_saved)
    """
    return compress_prompt(prompt, aggressive=False)


def compress_prompt_aggressive(prompt: str) -> Tuple[str, int]:
    """
    Aggressive compression that may slightly affect readability but preserves core meaning

    Returns:
        Tuple of (compressed_prompt, estimated_tokens_saved)
    """
    return compress_prompt(prompt, aggressive=True)
