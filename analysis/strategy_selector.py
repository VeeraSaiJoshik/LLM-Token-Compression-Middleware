"""
Deterministic Strategy Selector for Token Optimization

Based on comprehensive test results analysis, this module provides
deterministic logic for selecting optimization strategies.
"""

import re
import json
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class CompressionLevel(Enum):
    """Compression aggressiveness levels."""
    NONE = "none"
    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"


@dataclass
class StrategyRecommendation:
    """Recommendation for a specific optimization strategy."""
    applicable: bool
    estimated_token_savings: int
    estimated_percentage_savings: float
    reason: str
    risk_level: str
    priority: int


@dataclass
class OptimizationPlan:
    """Complete optimization plan with all strategy recommendations."""
    cache: StrategyRecommendation
    toon: StrategyRecommendation
    whitespace: StrategyRecommendation
    compression: StrategyRecommendation
    total_estimated_savings: int
    total_estimated_percentage: float
    recommended_strategies: List[str]


class StrategySelector:
    """Deterministic strategy selector based on test data analysis."""

    # Politeness markers to detect
    POLITENESS_MARKERS = [
        "please", "kindly", "could you", "would you",
        "i would appreciate", "thank you", "thanks",
        "if possible", "i'd appreciate", "i would be grateful"
    ]

    # Code/technical indicators
    CODE_INDICATORS = [
        "function", "class", "implement", "algorithm",
        "code", "programming", "script", "method",
        "api", "endpoint", "authentication", "oauth"
    ]

    TECHNICAL_INDICATORS = [
        "technical", "documentation", "specification",
        "architecture", "protocol", "RFC"
    ]

    def __init__(self):
        """Initialize the strategy selector."""
        pass

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token estimation (words * 1.3 for English).
        This is a simple heuristic; actual tokenization may vary.
        """
        words = len(text.split())
        return int(words * 1.3)

    def contains_json(self, text: str) -> bool:
        """Check if text contains JSON data."""
        # Look for JSON patterns
        json_patterns = [
            r'\{[\s\S]*"[\w_]+"[\s\S]*:[\s\S]*\}',  # Object pattern
            r'\[[\s\S]*\{[\s\S]*\}[\s\S]*\]',  # Array of objects
        ]

        for pattern in json_patterns:
            if re.search(pattern, text):
                return True

        # Try to parse as JSON
        try:
            # Extract potential JSON blocks
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                potential_json = text[start_idx:end_idx + 1]
                json.loads(potential_json)
                return True
        except (json.JSONDecodeError, ValueError):
            pass

        return False

    def calculate_json_size(self, text: str) -> int:
        """Calculate the size of JSON content in the text."""
        try:
            # Extract JSON block
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = text[start_idx:end_idx + 1]
                return len(json_text)
        except Exception:
            pass
        return 0

    def has_excessive_whitespace(self, text: str) -> bool:
        """Detect excessive whitespace that can be normalized."""
        # Multiple consecutive spaces
        if text.count('  ') > 5:
            return True

        # Triple or more newlines
        if '\n\n\n' in text:
            return True

        # Many tabs
        if text.count('\t') > 10:
            return True

        # Lines with only whitespace
        lines = text.split('\n')
        empty_lines = sum(1 for line in lines if line.strip() == '')
        if empty_lines > len(lines) * 0.3:  # >30% empty lines
            return True

        return False

    def estimate_whitespace_savings(self, text: str) -> int:
        """Estimate tokens that can be saved by normalizing whitespace."""
        if not self.has_excessive_whitespace(text):
            return 0

        # Count excessive whitespace elements
        savings = 0
        savings += text.count('  ') * 0.5  # Each double space ~0.5 tokens
        savings += text.count('\n\n\n') * 2  # Triple newlines
        savings += text.count('\t') * 0.2  # Tabs

        # Estimate based on test data: average 15 tokens for excessive cases
        return min(int(savings), 20)  # Cap at 20 tokens

    def has_politeness_markers(self, text: str) -> bool:
        """Check if text contains politeness markers."""
        text_lower = text.lower()
        return any(marker in text_lower for marker in self.POLITENESS_MARKERS)

    def is_code_generation_task(self, text: str) -> bool:
        """Determine if this is a code generation task."""
        text_lower = text.lower()

        # Strong indicators
        strong_code_patterns = [
            r'\bwrite\s+(?:a\s+)?(?:function|class|method|script)',
            r'\bimplement\s+(?:a\s+)?(?:function|class|algorithm)',
            r'\bcreate\s+(?:a\s+)?(?:function|class|method)',
            r'\bgenerate\s+(?:a\s+)?(?:function|class|code)',
        ]

        for pattern in strong_code_patterns:
            if re.search(pattern, text_lower):
                return True

        # Weak indicators (need multiple)
        weak_indicators = sum(
            1 for indicator in self.CODE_INDICATORS
            if indicator in text_lower
        )

        return weak_indicators >= 2

    def is_technical_documentation(self, text: str) -> bool:
        """Determine if this is technical documentation query."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.TECHNICAL_INDICATORS)

    def should_apply_toon(self, text: str, baseline_tokens: int) -> StrategyRecommendation:
        """
        Determine if TOON conversion should be applied.

        Test data shows:
        - Large JSON (>300 chars): 50-60% reduction
        - Medium JSON (50-300 chars): 35-50% reduction
        - Small JSON (<50 chars): marginal benefit
        """
        if not self.contains_json(text):
            return StrategyRecommendation(
                applicable=False,
                estimated_token_savings=0,
                estimated_percentage_savings=0.0,
                reason="No JSON detected",
                risk_level="none",
                priority=0
            )

        json_size = self.calculate_json_size(text)

        if json_size > 300:
            # Large JSON - high value
            savings_pct = 0.55  # 55% average from tests
            return StrategyRecommendation(
                applicable=True,
                estimated_token_savings=int(baseline_tokens * savings_pct),
                estimated_percentage_savings=savings_pct,
                reason=f"Large JSON ({json_size} chars) detected - expect 50-60% savings",
                risk_level="none",
                priority=2  # High priority
            )

        elif json_size > 50:
            # Medium JSON - medium value
            savings_pct = 0.40  # 40% average from tests
            return StrategyRecommendation(
                applicable=True,
                estimated_token_savings=int(baseline_tokens * savings_pct),
                estimated_percentage_savings=savings_pct,
                reason=f"Medium JSON ({json_size} chars) detected - expect 35-50% savings",
                risk_level="none",
                priority=2
            )

        else:
            # Small JSON - marginal benefit
            return StrategyRecommendation(
                applicable=False,
                estimated_token_savings=0,
                estimated_percentage_savings=0.0,
                reason=f"Small JSON ({json_size} chars) - marginal benefit, skip to reduce complexity",
                risk_level="none",
                priority=0
            )

    def should_apply_whitespace(self, text: str, baseline_tokens: int) -> StrategyRecommendation:
        """
        Determine if whitespace normalization should be applied.

        Test data shows:
        - 10-20 token savings when excessive whitespace present
        - Zero risk - no accuracy impact
        """
        if not self.has_excessive_whitespace(text):
            return StrategyRecommendation(
                applicable=False,
                estimated_token_savings=0,
                estimated_percentage_savings=0.0,
                reason="Normal whitespace - no optimization needed",
                risk_level="none",
                priority=0
            )

        estimated_savings = self.estimate_whitespace_savings(text)
        savings_pct = estimated_savings / baseline_tokens if baseline_tokens > 0 else 0

        return StrategyRecommendation(
            applicable=True,
            estimated_token_savings=estimated_savings,
            estimated_percentage_savings=savings_pct,
            reason=f"Excessive whitespace detected - expect {estimated_savings} token savings",
            risk_level="none",  # Zero risk
            priority=3  # Apply early (doesn't affect other strategies)
        )

    def should_apply_compression(self, text: str, baseline_tokens: int) -> StrategyRecommendation:
        """
        Determine compression level based on content type.

        Test data shows:
        - General queries: 5-20% reduction (safe)
        - Code generation: minimal compression (1-5% reduction)
        - Technical docs: minimal compression
        """
        if not self.has_politeness_markers(text):
            return StrategyRecommendation(
                applicable=False,
                estimated_token_savings=0,
                estimated_percentage_savings=0.0,
                reason="No politeness markers detected",
                risk_level="none",
                priority=0
            )

        # Determine compression level based on content type
        is_code = self.is_code_generation_task(text)
        is_technical = self.is_technical_documentation(text)

        if is_code or is_technical:
            # Conservative compression
            savings_pct = 0.05  # 5% average
            level = CompressionLevel.MINIMAL
            risk = "low"
            reason = "Code/technical content - conservative compression (5% expected)"
        else:
            # Standard compression
            savings_pct = 0.12  # 12% average from tests
            level = CompressionLevel.STANDARD
            risk = "none"
            reason = "General query - standard compression (10-15% expected)"

        return StrategyRecommendation(
            applicable=True,
            estimated_token_savings=int(baseline_tokens * savings_pct),
            estimated_percentage_savings=savings_pct,
            reason=reason,
            risk_level=risk,
            priority=1  # Apply last (after structural changes)
        )

    def create_optimization_plan(self, text: str) -> OptimizationPlan:
        """
        Create a complete optimization plan for the given text.

        Returns recommendations for all strategies with estimated savings.
        """
        baseline_tokens = self.estimate_tokens(text)

        # Evaluate each strategy
        cache_rec = StrategyRecommendation(
            applicable=True,  # Always check cache
            estimated_token_savings=baseline_tokens,
            estimated_percentage_savings=1.0,
            reason="Always check cache first - 100% savings on hit",
            risk_level="none",
            priority=4  # Highest priority
        )

        whitespace_rec = self.should_apply_whitespace(text, baseline_tokens)
        toon_rec = self.should_apply_toon(text, baseline_tokens)
        compression_rec = self.should_apply_compression(text, baseline_tokens)

        # Calculate total estimated savings (excluding cache)
        # Note: Savings don't simply add up - they compound
        total_savings = 0
        remaining_tokens = baseline_tokens

        # Apply in priority order (excluding cache)
        strategies_by_priority = [
            ("whitespace", whitespace_rec),
            ("toon", toon_rec),
            ("compression", compression_rec),
        ]
        strategies_by_priority.sort(key=lambda x: x[1].priority, reverse=True)

        recommended = []
        for name, rec in strategies_by_priority:
            if rec.applicable:
                # Apply percentage to remaining tokens
                strategy_savings = int(remaining_tokens * rec.estimated_percentage_savings)
                total_savings += strategy_savings
                remaining_tokens -= strategy_savings
                recommended.append(name)

        total_pct = total_savings / baseline_tokens if baseline_tokens > 0 else 0

        return OptimizationPlan(
            cache=cache_rec,
            toon=toon_rec,
            whitespace=whitespace_rec,
            compression=compression_rec,
            total_estimated_savings=total_savings,
            total_estimated_percentage=total_pct,
            recommended_strategies=["cache"] + recommended  # Cache always first
        )

    def explain_plan(self, plan: OptimizationPlan, verbose: bool = True) -> str:
        """Generate human-readable explanation of the optimization plan."""
        lines = []

        lines.append("=== OPTIMIZATION PLAN ===\n")

        # Summary
        lines.append(f"Estimated Total Savings: {plan.total_estimated_savings} tokens ({plan.total_estimated_percentage:.1%})")
        lines.append(f"Recommended Strategies: {', '.join(plan.recommended_strategies)}\n")

        if verbose:
            # Detailed breakdown
            lines.append("Strategy Details:")
            lines.append("-" * 60)

            strategies = [
                ("Cache", plan.cache),
                ("Whitespace Normalization", plan.whitespace),
                ("TOON Conversion", plan.toon),
                ("Compression", plan.compression),
            ]

            for name, rec in strategies:
                status = "✓ APPLY" if rec.applicable else "✗ SKIP"
                lines.append(f"\n{name}: {status}")
                lines.append(f"  Reason: {rec.reason}")
                if rec.applicable:
                    lines.append(f"  Estimated Savings: {rec.estimated_token_savings} tokens ({rec.estimated_percentage_savings:.1%})")
                    lines.append(f"  Risk Level: {rec.risk_level}")
                    lines.append(f"  Priority: {rec.priority}")

        return "\n".join(lines)


def analyze_test_case(text: str) -> None:
    """Analyze a test case and print the optimization plan."""
    selector = StrategySelector()
    plan = selector.create_optimization_plan(text)
    print(selector.explain_plan(plan, verbose=True))
    print("\n")


if __name__ == "__main__":
    # Test with examples from the test data

    print("=" * 70)
    print("STRATEGY SELECTOR - TEST CASES")
    print("=" * 70)
    print()

    # Test 1: Simple question (cache only)
    print("TEST 1: Simple Fact Question")
    print("-" * 70)
    analyze_test_case("What is the capital of Japan?")

    # Test 2: Polite verbose question
    print("TEST 2: Polite Verbose Question")
    print("-" * 70)
    analyze_test_case(
        "Could you please kindly explain to me how machine learning works? "
        "I would really appreciate it if you could provide a simple explanation. "
        "Thank you so much for your time and assistance!"
    )

    # Test 3: Excessive whitespace
    print("TEST 3: Excessive Whitespace")
    print("-" * 70)
    analyze_test_case(
        "What are the main differences    between relational and     document databases?\n\n\n\n"
        "Please    explain."
    )

    # Test 4: JSON data
    print("TEST 4: JSON Data")
    print("-" * 70)
    analyze_test_case('''
Analyze the sales performance:

{
  "quarterly_sales": [
    {
      "quarter": "Q1",
      "revenue": 125000,
      "units": 450,
      "region": "North America"
    },
    {
      "quarter": "Q2",
      "revenue": 138000,
      "units": 520,
      "region": "North America"
    }
  ]
}
''')

    # Test 5: Code generation (conservative)
    print("TEST 5: Code Generation Task")
    print("-" * 70)
    analyze_test_case(
        "Write a Python function that implements a binary search algorithm "
        "with proper error handling"
    )

    # Test 6: Mixed content
    print("TEST 6: Mixed Content (Polite + JSON)")
    print("-" * 70)
    analyze_test_case('''
Could you please kindly review this configuration and let me know if there are any issues?
I would really appreciate your help!

{
  "config": {
    "timeout": 30,
    "retries": 3,
    "endpoints": ["api.example.com", "backup.example.com"]
  }
}
''')
