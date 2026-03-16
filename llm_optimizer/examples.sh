#!/bin/bash
# Example commands for LLM Cost Optimization Tool

echo "=================================="
echo "LLM Cost Optimization Examples"
echo "=================================="
echo ""

echo "1. List all available models:"
echo "   python main.py list-models"
echo ""

echo "2. Simple test with default model:"
echo "   python main.py test 'What is machine learning?'"
echo ""

echo "3. Multi-model comparison:"
echo "   python main.py test 'Explain quantum computing' --models gpt-4o-mini,claude-3-5-haiku,gemini-2-0-flash"
echo ""

echo "4. Test with JSON data (TOON optimization):"
echo "   python main.py test 'Analyze this sales data' --json-data '{\"sales\":[{\"product\":\"Widget\",\"amount\":100}]}'"
echo ""

echo "5. Run batch tests:"
echo "   python main.py batch tests/sample_prompts.json"
echo ""

echo "6. View statistics from batch results:"
echo "   python main.py stats --results-file results.json"
echo ""

echo "7. Test specific optimization strategies:"
echo "   python main.py test 'Write a sorting function' --strategies toon,compression"
echo ""

echo "=================================="
echo "Run any of these commands to try the tool!"
echo "=================================="
