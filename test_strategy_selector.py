#!/usr/bin/env python3
"""
Quick test of the strategy selector without requiring API calls
"""
import sys
import json
sys.path.insert(0, '/home/user/Ostia-LLM-Optimization-Claude/llm-optimizer')

from optimizers.strategy_selector import select_strategies, analyze_prompt
from optimizers.whitespace_optimizer import optimize_whitespace

# Test cases
test_cases = [
    {
        "name": "Simple question",
        "prompt": "What is the capital of France?",
        "json_data": None
    },
    {
        "name": "Verbose with politeness",
        "prompt": "Could you please kindly help me understand how this works? I would really appreciate it!",
        "json_data": None
    },
    {
        "name": "Whitespace heavy",
        "prompt": "What  are   the    main     differences?",
        "json_data": None
    },
    {
        "name": "JSON data",
        "prompt": "Analyze this data:",
        "json_data": json.dumps({"items": [{"id": 1, "name": "Test"}]})
    },
    {
        "name": "Code task",
        "prompt": "Write a function to implement binary search",
        "json_data": None
    },
    {
        "name": "All strategies",
        "prompt": "Could  you   please   analyze   this   data   kindly?",
        "json_data": json.dumps({"data": [1, 2, 3]})
    }
]

print("=" * 80)
print("STRATEGY SELECTOR TEST")
print("=" * 80)

for test in test_cases:
    print(f"\n{test['name']}")
    print("-" * 80)
    print(f"Prompt: {test['prompt'][:60]}...")
    print(f"Has JSON: {test['json_data'] is not None}")

    # Analyze prompt
    analysis = analyze_prompt(test['prompt'], test['json_data'])
    print(f"\nAnalysis:")
    print(f"  - Has JSON: {analysis['has_json']}")
    print(f"  - Has excessive whitespace: {analysis['has_excessive_whitespace']}")
    print(f"  - Is verbose: {analysis['is_verbose']}")
    print(f"  - Has politeness: {analysis['has_politeness']}")
    print(f"  - Is code task: {analysis['is_code_task']}")
    print(f"  - Word count: {analysis['word_count']}")

    # Select strategies
    selection = select_strategies(test['prompt'], test['json_data'])
    print(f"\nSelected Strategies: {', '.join(selection['selected_strategies'])}")
    print(f"Expected Impact: {selection['expected_impact']}")
    print(f"\nReasoning:")
    for reason in selection['reasoning']:
        print(f"  • {reason}")

print("\n" + "=" * 80)
print("WHITESPACE OPTIMIZER TEST")
print("=" * 80)

whitespace_test = "What  are   the    main     differences     between      SQL?"
print(f"\nOriginal: '{whitespace_test}'")
result = optimize_whitespace(whitespace_test)
print(f"Optimized: '{result['optimized_prompt']}'")
print(f"Chars saved: {result['chars_saved']}")
print(f"Estimated tokens saved: {result['estimated_tokens_saved']}")

print("\n✓ All tests completed successfully!")
