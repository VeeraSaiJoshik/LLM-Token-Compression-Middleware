"""
Intelligent Strategy Selector - Automatically chooses optimal optimization strategies
based on prompt characteristics to maximize savings while preserving accuracy.
"""

import re
import json
from typing import List, Dict, Optional


def analyze_prompt(prompt: str, json_data: Optional[str] = None) -> Dict[str, any]:
    """
    Analyze prompt characteristics to determine optimal strategies.

    Args:
        prompt: The prompt text
        json_data: Optional JSON data attached to prompt

    Returns:
        Dict containing prompt analysis metrics
    """
    analysis = {
        "has_json": json_data is not None and len(json_data) > 0,
        "has_excessive_whitespace": bool(re.search(r'  +|\n{3,}', prompt)),
        "is_verbose": False,
        "is_short_answer": False,
        "is_code_task": False,
        "is_data_extraction": False,
        "word_count": len(prompt.split()),
        "has_politeness": False,
        "json_size": len(json_data) if json_data else 0,
    }

    # Check for verbose language patterns
    verbose_patterns = [
        r'\bplease\b', r'\bkindly\b', r'\bcould you\b', r'\bwould you\b',
        r'\bI would like\b', r'\bI would appreciate\b', r'\bif possible\b',
        r'\bthank you\b', r'\bthanks\b'
    ]
    analysis["has_politeness"] = any(
        re.search(pattern, prompt, re.IGNORECASE) for pattern in verbose_patterns
    )
    analysis["is_verbose"] = analysis["has_politeness"] or analysis["word_count"] > 100

    # Detect short-answer questions
    question_words = ['what', 'when', 'where', 'who', 'which', 'how many']
    is_simple_question = any(prompt.lower().startswith(word) for word in question_words)
    analysis["is_short_answer"] = (
        is_simple_question and
        analysis["word_count"] < 20 and
        '?' in prompt
    )

    # Detect code-related tasks
    code_keywords = ['function', 'code', 'implement', 'class', 'method', 'algorithm', 'program']
    analysis["is_code_task"] = any(keyword in prompt.lower() for keyword in code_keywords)

    # Detect data extraction tasks
    extraction_keywords = ['extract', 'parse', 'analyze', 'summarize', 'find', 'identify']
    analysis["is_data_extraction"] = (
        any(keyword in prompt.lower() for keyword in extraction_keywords) and
        (analysis["has_json"] or 'data' in prompt.lower())
    )

    return analysis


def select_strategies(
    prompt: str,
    json_data: Optional[str] = None,
    available_strategies: List[str] = None
) -> Dict[str, any]:
    """
    Select optimal optimization strategies based on prompt analysis.

    Args:
        prompt: The prompt text
        json_data: Optional JSON data
        available_strategies: List of available strategy names

    Returns:
        Dict containing selected strategies and reasoning
    """
    if available_strategies is None:
        available_strategies = ['toon', 'compression', 'cache', 'whitespace']

    analysis = analyze_prompt(prompt, json_data)
    selected = []
    reasoning = []

    # Whitespace optimization - ALWAYS apply if there's excessive whitespace
    if 'whitespace' in available_strategies and analysis["has_excessive_whitespace"]:
        selected.append('whitespace')
        reasoning.append("Whitespace: Excessive spacing detected (zero accuracy impact)")

    # TOON conversion - Apply if JSON data exists
    if 'toon' in available_strategies and analysis["has_json"]:
        selected.append('toon')
        reasoning.append(f"TOON: JSON data present ({analysis['json_size']} chars, expect 30-50% savings)")

    # Compression - Apply if verbose or has politeness markers
    if 'compression' in available_strategies:
        if analysis["has_politeness"]:
            selected.append('compression')
            reasoning.append("Compression: Politeness markers detected (safe to remove)")
        elif analysis["is_verbose"] and not analysis["is_code_task"]:
            selected.append('compression')
            reasoning.append("Compression: Verbose prompt detected (careful reduction)")

    # Cache - ALWAYS available for semantic matching
    if 'cache' in available_strategies:
        selected.append('cache')
        reasoning.append("Cache: Checking for semantically similar previous requests")

    # Deduplication - ALWAYS check
    if 'dedup' in available_strategies:
        selected.append('dedup')
        reasoning.append("Dedup: Checking for exact duplicate requests")

    return {
        "selected_strategies": selected,
        "reasoning": reasoning,
        "analysis": analysis,
        "expected_impact": estimate_impact(selected, analysis)
    }


def estimate_impact(strategies: List[str], analysis: Dict) -> str:
    """
    Estimate the expected token savings from selected strategies.

    Args:
        strategies: List of selected strategy names
        analysis: Prompt analysis dict

    Returns:
        Human-readable impact estimation
    """
    savings_estimate = []

    if 'toon' in strategies:
        savings_estimate.append("30-50% on JSON data")

    if 'compression' in strategies:
        if analysis["has_politeness"]:
            savings_estimate.append("10-20% from politeness removal")
        else:
            savings_estimate.append("5-15% from compression")

    if 'whitespace' in strategies:
        savings_estimate.append("2-10% from whitespace")

    if 'cache' in strategies or 'dedup' in strategies:
        savings_estimate.append("100% if cache hit")

    if not savings_estimate:
        return "Minimal expected savings"

    return ", ".join(savings_estimate)


def should_use_conservative_compression(analysis: Dict) -> bool:
    """
    Determine if conservative compression should be used.

    Conservative mode only removes politeness and obvious redundancy,
    preserving all technical content and context.

    Args:
        analysis: Prompt analysis dict

    Returns:
        True if conservative mode should be used
    """
    # Use conservative compression for:
    # - Code tasks (preserve technical language)
    # - Short prompts (little to gain, risk losing context)
    # - Data extraction (preserve specific instructions)
    return (
        analysis["is_code_task"] or
        analysis["word_count"] < 30 or
        analysis["is_data_extraction"]
    )
