#!/usr/bin/env python3
"""
LLM Cost Optimization Testing Tool
Main CLI interface
"""
import click
import logging
import json
import sys
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from typing import Dict, List, Optional
import time

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PRICING
from providers.openai_client import OpenAIClient, call_openai
from providers.anthropic_client import AnthropicClient, call_anthropic
from providers.google_client import GoogleClient, call_gemini
from optimizers.toon_converter import convert_prompt_to_toon, add_json_to_prompt
from optimizers.prompt_compressor import compress_prompt
from optimizers.whitespace_optimizer import optimize_whitespace
from optimizers.strategy_selector import select_strategies, should_use_conservative_compression
from utils.tokenizer import estimate_tokens
from utils.cost_calculator import calculate_cost, compare_costs
from utils.cache_manager import CacheManager
from optimizers import runCompressionAlgorithm, CompressionAlgorithms
from providers import LLMComparisionResult, LLMOutput
from json import JSONEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_optimizer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

console = Console()

# Global cache manager
cache_manager = CacheManager(use_semantic=True, use_dedup=True)


def call_llm(prompt: str, model_key: str, use_cache: bool = False) -> LLMOutput:
    """
    Make LLM API call with actual token extraction

    Args:
        prompt: Input prompt
        model_key: Model key from PRICING config
        use_cache: Whether to use prompt caching (provider-specific)

    Returns:
        LLMOutput object with response and actual token counts
    """
    if model_key not in PRICING:
        raise ValueError(f"Model {model_key} not found in pricing config")

    pricing = PRICING[model_key]
    model_id = pricing["model_id"]
    provider = pricing["provider"]

    logger.info(f"Calling {provider} with model {model_id}")

    try:
        if provider == "openai":
            result = call_openai(prompt, model_id)
        elif provider == "anthropic":
            result = call_anthropic(prompt, model_id, use_cache=use_cache)
        elif provider == "google":
            result = call_gemini(prompt, model_id)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return result

    except Exception as e:
        logger.error(f"LLM call failed: {str(e)}")
        raise


def run_baseline(prompt: str, model_key: str, json_data: Optional[str] = None) -> Dict:
    """
    Run baseline (unoptimized) test

    Args:
        prompt: Original prompt
        model_key: Model to test
        json_data: Optional JSON data to include in prompt

    Returns:
        Dict with results
    """
    # Add JSON data if provided (in standard JSON format for baseline)
    full_prompt = prompt
    if json_data:
        full_prompt = add_json_to_prompt(prompt, json_data)

    # Estimate tokens (for comparison)
    estimated_tokens = estimate_tokens(full_prompt, model_key)

    # Make actual API call
    start_time = time.time()
    result = call_llm(full_prompt, model_key)
    elapsed = time.time() - start_time

    # Calculate cost using ACTUAL tokens
    cost = calculate_cost(model_key, result.actual_tokens)

    return {
        "prompt": full_prompt,
        "prompt_length": len(full_prompt),
        "estimated_tokens": estimated_tokens,
        "actual_tokens": result.actual_tokens,
        "cost": cost,
        "response": result.response,
        "response_preview": result.response[:200] + "..." if len(result.response) > 200 else result.response,
        "model": model_key,
        "elapsed_time": elapsed,
    }


def run_optimized(
    prompt: str,
    model_key: str,
    strategies: List[str],
    json_data: Optional[str] = None
) -> Dict:
    """
    Run optimized test with selected strategies

    Args:
        prompt: Original prompt
        model_key: Model to test
        strategies: List of optimization strategies to apply ('auto' for intelligent selection)
        json_data: Optional JSON data to include

    Returns:
        Dict with results and optimization details
    """
    optimized_prompt = prompt
    optimization_log = []
    strategy_reasoning = []

    # Intelligent strategy selection
    if 'auto' in strategies or 'all' in strategies:
        selection = select_strategies(prompt, json_data, ['toon', 'compression', 'cache', 'whitespace'])
        strategies = selection['selected_strategies']
        strategy_reasoning = selection['reasoning']
        logger.info(f"Auto-selected strategies: {strategies}")
        logger.info(f"Expected impact: {selection['expected_impact']}")

    # 1. Whitespace optimization (apply first, zero impact)
    if "whitespace" in strategies:
        result = optimize_whitespace(optimized_prompt)
        if result["chars_saved"] > 0:
            optimized_prompt = result["optimized_prompt"]
            optimization_log.append(("Whitespace Normalization", result["estimated_tokens_saved"]))

    # 2. Add JSON data if provided
    if json_data and "toon" in strategies:
        try:
            json_obj = json.loads(json_data) if isinstance(json_data, str) else json_data
            optimized_prompt = add_json_to_prompt(optimized_prompt, json_obj)
            optimization_log.append(("JSON Added", 0))
        except Exception as e:
            logger.warning(f"Failed to add JSON: {str(e)}")

    # 3. TOON conversion
    if "toon" in strategies:
        optimized_prompt, tokens_saved = convert_prompt_to_toon(optimized_prompt)
        if tokens_saved > 0:
            optimization_log.append(("TOON Conversion", tokens_saved))

    # 4. Prompt compression
    if "compression" in strategies:
        optimized_prompt, tokens_saved = compress_prompt(optimized_prompt)
        if tokens_saved > 0:
            optimization_log.append(("Compression", tokens_saved))

    # 5. Check cache
    cache_hit = None
    if "cache" in strategies:
        cache_result = cache_manager.check_cache(optimized_prompt, model_key)
        if cache_result:
            cache_hit = cache_result
            optimization_log.append((f"Cache HIT ({cache_result['cache_type']})", 0))

    # Make API call (unless cache hit)
    if cache_hit:
        # Use cached response - create a mock LLMOutput-like structure
        class CachedResult:
            def __init__(self, response, tokens):
                self.response = response
                self.actual_tokens = tokens
        
        result = CachedResult(cache_hit["response"], cache_hit["tokens_saved"])
        cost = {"total": 0.0, "input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_write": 0.0}
        elapsed = 0.0
    else:
        # Estimate tokens
        estimated_tokens = estimate_tokens(optimized_prompt, model_key)

        # Make actual API call
        start_time = time.time()
        result = call_llm(optimized_prompt, model_key)
        elapsed = time.time() - start_time

        # Calculate cost using ACTUAL tokens
        cost = calculate_cost(model_key, result.actual_tokens)

        # Add to cache
        if "cache" in strategies:
            cache_manager.add_to_cache(
                optimized_prompt,
                model_key,
                result.response,
                result.actual_tokens
            )

    return {
        "prompt": optimized_prompt,
        "prompt_length": len(optimized_prompt),
        "actual_tokens": result.actual_tokens,
        "cost": cost,
        "response": result.response,
        "response_preview": result.response[:200] + "..." if len(result.response) > 200 else result.response,
        "model": model_key,
        "elapsed_time": elapsed,
        "optimization_log": optimization_log,
        "cache_hit": cache_hit is not None,
        "strategy_reasoning": strategy_reasoning,
        "selected_strategies": strategies,
    }


def display_baseline_results(result: Dict):
    """Display baseline results"""
    table = Table(title="Baseline Results (No Optimization)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Prompt Length", f"{result['prompt_length']} chars")
    table.add_row("Estimated Tokens", f"~{result['estimated_tokens']}")

    # Actual tokens (using TokenCost model)
    actual = result["actual_tokens"]
    table.add_row(
        "Actual Tokens (API)",
        f"{actual.total_tokens} (input: {actual.prompt_tokens}, output: {actual.completion_tokens})"
    )

    cost = result["cost"]
    table.add_row("Total Cost", f"${cost['total']:.6f}")
    table.add_row("  - Input Cost", f"${cost['input']:.6f}")
    table.add_row("  - Output Cost", f"${cost['output']:.6f}")
    table.add_row("Response Time", f"{result['elapsed_time']:.2f}s")

    console.print(table)
    console.print(f"\n[dim]Response preview:[/dim] {result['response_preview']}\n")


def display_comparison(baseline: Dict, optimized: Dict):
    """Display comparison between baseline and optimized"""
    table = Table(title="Optimization Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Baseline", style="yellow")
    table.add_column("Optimized", style="green")
    table.add_column("Savings", style="magenta", justify="right")

    # Prompt length
    chars_saved = baseline["prompt_length"] - optimized["prompt_length"]
    table.add_row(
        "Prompt Length",
        f"{baseline['prompt_length']} chars",
        f"{optimized['prompt_length']} chars",
        f"-{chars_saved} chars"
    )

    # Tokens (using TokenCost model)
    baseline_tokens = baseline["actual_tokens"].total_tokens
    optimized_tokens = optimized["actual_tokens"].total_tokens if not optimized["cache_hit"] else 0

    tokens_saved = baseline_tokens - optimized_tokens
    tokens_pct = (tokens_saved / baseline_tokens * 100) if baseline_tokens > 0 else 0

    table.add_row(
        "Total Tokens (API)",
        str(baseline_tokens),
        str(optimized_tokens) if not optimized["cache_hit"] else "0 (cached)",
        f"-{tokens_saved} ({tokens_pct:.1f}%)"
    )

    # Cost
    cost_saved = baseline["cost"]["total"] - optimized["cost"]["total"]
    cost_pct = (cost_saved / baseline["cost"]["total"] * 100) if baseline["cost"]["total"] > 0 else 0

    table.add_row(
        "Total Cost",
        f"${baseline['cost']['total']:.6f}",
        f"${optimized['cost']['total']:.6f}",
        f"-${cost_saved:.6f} ({cost_pct:.1f}%)"
    )

    console.print(table)

    # Strategy selection reasoning (if auto-selected)
    if optimized.get("strategy_reasoning"):
        console.print("\n[bold]Strategy Selection (Auto):[/bold]")
        for reason in optimized["strategy_reasoning"]:
            console.print(f"  • {reason}")

    # Optimization breakdown
    if optimized["optimization_log"]:
        console.print("\n[bold]Applied Optimizations:[/bold]")
        for strategy, est_tokens in optimized["optimization_log"]:
            if est_tokens > 0:
                console.print(f"  ✓ {strategy}: ~{est_tokens} tokens saved (estimated)")
            else:
                console.print(f"  ℹ {strategy}")

    # Show selected strategies
    if optimized.get("selected_strategies"):
        console.print(f"\n[dim]Strategies used: {', '.join(optimized['selected_strategies'])}[/dim]")

    console.print()


@click.group()
def cli():
    """LLM Cost Optimization Testing Tool"""
    pass

@cli.command()
@click.argument('prompt')
@click.option('--models', type=click.STRING, multiple=True, help='enter multiple model keys (e.g., gpt-4o,claude-sonnet-4-5)')
@click.option('--strategies', type=click.Choice([algo.value[1] for algo in CompressionAlgorithms]), multiple=True)
def compare_output(prompt, models, strategies):
    def use_llm_model(prompt, system_prompt = "", output_format: Optional[BaseModel] = None, max_tokens: int = 4096): 
        if "gpt" in model:
            return call_openai(prompt, model, system_prompt=system_prompt, output_format=output_format, max_tokens=max_tokens)
        elif "claude" in model:
            return call_anthropic(prompt, model, system_prompt=system_prompt, output_format=output_format, max_tokens=max_tokens)
        
        return call_gemini(prompt, model, system_prompt=system_prompt, output_format=output_format, max_tokens=max_tokens)

    initial_prompt = prompt
    final_prompt, compression_time = runCompressionAlgorithm(prompt, list(strategies))
    
    model = models[0] if models else "gpt-4o-mini"

    original_response: LLMOutput = use_llm_model(initial_prompt)
    optimized_response: LLMOutput = use_llm_model(final_prompt)

    time_table = Table(title="Optimization Time")
    time_table.add_column("Method", style="cyan")
    time_table.add_column("Time", style="red")

    for time in compression_time:
        time_table.add_row(time[0], time[1])
    
    console.print(time_table)

    # Show token and cost comparison
    original_cost = calculate_cost(model, original_response.actual_tokens)
    optimized_cost = calculate_cost(model, optimized_response.actual_tokens)
    
    table = Table(title="Compression Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Original", style="yellow")
    table.add_column("Compressed", style="green")
    table.add_column("Savings", style="magenta")
    
    table.add_row("Prompt Length", f"{len(initial_prompt)} chars", f"{len(final_prompt)} chars", f"{len(final_prompt) - len(initial_prompt)} chars")
    table.add_row("Input Tokens", str(original_response.actual_tokens.prompt_tokens), str(optimized_response.actual_tokens.prompt_tokens), f"{optimized_response.actual_tokens.prompt_tokens - original_response.actual_tokens.prompt_tokens}")
    table.add_row("Output Tokens", str(original_response.actual_tokens.completion_tokens), str(optimized_response.actual_tokens.completion_tokens), f"{optimized_response.actual_tokens.completion_tokens - original_response.actual_tokens.completion_tokens}")
    table.add_row("Total Tokens", str(original_response.actual_tokens.total_tokens), str(optimized_response.actual_tokens.total_tokens), f"{optimized_response.actual_tokens.total_tokens - original_response.actual_tokens.total_tokens}")
    table.add_row("Cost", f"${original_cost['total']:.6f}", f"${optimized_cost['total']:.6f}", f"${optimized_cost['total'] - original_cost['total']:.6f}")
    
    console.print(table)
    
    # Analzying and comparing output
    checklist: LLMOutput = use_llm_model(JSONEncoder().encode({
        "Original Response": original_response.response,
        "Optimized Response": optimized_response.response
    }), system_prompt="""You are a expert text analyst benchmarking bot. Your purpose is to analyze the origional response and optimized response
                        to see how the responses differ semantically. You will generate 'topic_checklists' for each of the 2 responses which are essentially
                        outlines regarding what core information is provided in each response. Each 'topic_checklist' will either have 0 or more associated sentences
                        from both the origional and optimized response. It is your responsibility to identify what 'topic_checklists' the original and
                        optimized responses have in common and how they differ. Remember, every sentence in both responses should be assigned to 1 and only 1
                        topic checklist

                        DO NOT WRITE ABOUT SPECIFIC COMPONENTS WHEN CREATING 'TOPIC_CHECKLISTS' EACH TOPIC CHECKLIST SHOULD BE 5 WORDS MAX, AND SHOULD ONLY HAVE OVERARCHING IDEAS AND NOT SPECIIFCS

                        FOR EXAMPLE IF ONE RESPONSE HAS AN EXAMPLE ABOUT CATS AND ANOTHER HAS AN EXAMPLE ABOUT DOGS BOTH TRYING TO CONVEY TOPIC X, 'TOPIC_CHECKLIST' SHOULD BE 'EXAMPLE ABOUT TOPIC X'

                        Each 'topic_checklist' should be generalized in the sense that if the origional response and optimized response give an example of what
                        an AI model is but they use different examples (one is about a cat and oanother is about a dog), they should both just appear as 'example about AI'

                        So each topic checklist should convey the outline and purpose of a piece of text rather than low level detail about how that purpose is conveyed
    """, output_format=LLMComparisionResult, max_tokens=8192)

    print(checklist.response)
    checklist_response: LLMComparisionResult = LLMComparisionResult.model_validate_json(checklist.response)

    # Highly distinct colors for side-by-side comparison
    colors = [
        "bright_red",
        "bright_green", 
        "bright_blue",
        "bright_yellow",
        "bright_magenta",
        "bright_cyan",
        "orange1",
        "purple",
        "white",
        "black",
        "gold1",
        "deep_pink1"
    ]

    topic_colors = {}
    for i, topic in enumerate(checklist_response.checklists):
        topic_colors[topic.criteria_name] = colors[i % len(colors)]

    semantic_comp_table = Table(title="Semantic Comparison")
    semantic_comp_table.add_column("Sub Topic", style="cyan")
    semantic_comp_table.add_column("Original")
    semantic_comp_table.add_column("Compressed")
    semantic_comp_table.add_column("Delta")

    for topic in checklist_response.checklists:
        original_length = sum(len(sentence) for sentence in topic.original_response)
        optimized_length = sum(len(sentence) for sentence in topic.optimized_response)

        # Get the color for this topic
        topic_color = topic_colors[topic.criteria_name]

        # Determine color for Original vs Compressed
        if original_length == 0:
            original_str = "[green]x[/green]"
        else:
            if original_length > optimized_length:
                original_str = f"[red]{original_length}[/red]"
            elif original_length < optimized_length:
                original_str = f"[green]{original_length}[/green]"
            else:
                original_str = str(original_length)

        if optimized_length == 0:
            optimized_str = "[green]x[/green]"
        else:
            if optimized_length > original_length:
                optimized_str = f"[red]{optimized_length}[/red]"
            elif optimized_length < original_length:
                optimized_str = f"[green]{optimized_length}[/green]"
            else:
                optimized_str = str(optimized_length)

        # Delta styling (green when negative, red when positive)
        delta = optimized_length - original_length
        if delta < 0:
            delta_str = f"[green]{delta}[/green]"
        elif delta > 0:
            delta_str = f"[red]{delta}[/red]"
        else:
            delta_str = str(delta)

        semantic_comp_table.add_row(
            f"[{topic_color}]{topic.criteria_name}[/{topic_color}]",
            original_str,
            optimized_str,
            delta_str
        )
    
    console.print(semantic_comp_table)
    
    # Create side-by-side view with highlighted text
    side_by_side_table = Table(title="Side-by-Side Response Comparison (Color-Coded by Topic)")
    side_by_side_table.add_column("Original Response", style="white", width=80)
    side_by_side_table.add_column("Optimized Response", style="white", width=80)
    
    def highlight_text_by_topic(text: str, response_sentences: list[str], topic_name: str, color: str) -> str:
        highlighted = text
        for sentence in response_sentences:
            if sentence.strip() in text:
                highlighted = highlighted.replace(sentence, f"[{color}]{sentence}[/{color}]")
        return highlighted
    
    # Get the full responses
    original_text = original_response.response
    optimized_text = optimized_response.response
    
    # Apply highlighting for each topic
    for topic in checklist_response.checklists:
        color = topic_colors[topic.criteria_name]
        original_text = highlight_text_by_topic(original_text, topic.original_response, topic.criteria_name, color)
        optimized_text = highlight_text_by_topic(optimized_text, topic.optimized_response, topic.criteria_name, color)
    
    side_by_side_table.add_row(original_text, optimized_text)
    console.print(side_by_side_table)

    side_by_side_table = Table(title="Side-by-Side Prompt Comparison")
    side_by_side_table.add_column("Original Response", style="white", width=80)
    side_by_side_table.add_column("Optimized Response", style="white", width=80)

    side_by_side_table.add_row(initial_prompt, final_prompt)

    console.print(side_by_side_table)

@cli.command()
@click.argument('prompt')
@click.option('--models', default='gpt-4o-mini', help='Comma-separated model keys (e.g., gpt-4o,claude-sonnet-4-5)')
@click.option('--strategies', default='auto', help='Strategies: auto (intelligent), all (force all), or comma-separated: toon,compression,cache,whitespace')
@click.option('--json-file', type=click.Path(exists=True), help='JSON file to include in prompt')
@click.option('--json-data', help='JSON data as string')
@click.option('--no-baseline', is_flag=True, help='Skip baseline run')
def test(prompt, models, strategies, json_file, json_data, no_baseline):
    """Test optimization strategies on a prompt"""

    # Parse models
    model_list = [m.strip() for m in models.split(',')]

    # Parse strategies
    if strategies == 'all':
        strategy_list = ['toon', 'compression', 'cache', 'whitespace']
    elif strategies == 'auto':
        strategy_list = ['auto']
    else:
        strategy_list = [s.strip() for s in strategies.split(',')]

    # Load JSON data if provided
    json_content = None
    if json_file:
        with open(json_file, 'r') as f:
            json_content = f.read()
    elif json_data:
        json_content = json_data

    # Display header
    console.print(f"\n[bold cyan]{'=' * 70}[/bold cyan]")
    console.print("[bold]LLM COST OPTIMIZATION TEST[/bold]", justify="center")
    console.print(f"[bold cyan]{'=' * 70}[/bold cyan]\n")

    console.print(f"[yellow]Prompt:[/yellow] {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    console.print(f"[yellow]Models:[/yellow] {', '.join(model_list)}")
    console.print(f"[yellow]Strategies:[/yellow] {', '.join(strategy_list)}\n")

    # Test each model
    for model_key in model_list:
        if model_key not in PRICING:
            console.print(f"[red]Error: Model '{model_key}' not found in pricing config[/red]")
            continue

        test_single_model(prompt, model_key, strategy_list, json_content, no_baseline)


def test_single_model(
    prompt: str,
    model_key: str,
    strategies: List[str],
    json_data: Optional[str],
    skip_baseline: bool
):
    """Test optimizations for a single model"""

    console.print(f"\n[bold green]{'─' * 70}[/bold green]")
    console.print(f"[bold green]Testing Model: {model_key}[/bold green]")
    console.print(f"[bold green]{'─' * 70}[/bold green]\n")

    try:
        # Run baseline
        if not skip_baseline:
            console.print("[bold]BASELINE (No Optimization)[/bold]")
            baseline_result = run_baseline(prompt, model_key, json_data)
            display_baseline_results(baseline_result)

        # Run optimized
        console.print("[bold]WITH OPTIMIZATIONS[/bold]")
        optimized_result = run_optimized(prompt, model_key, strategies, json_data)

        # Display comparison
        if not skip_baseline:
            display_comparison(baseline_result, optimized_result)
        else:
            console.print(f"[yellow]Optimized cost: ${optimized_result['cost']['total']:.6f}[/yellow]")

    except Exception as e:
        console.print(f"[red]Error testing {model_key}: {str(e)}[/red]")
        logger.exception(f"Error in test_single_model for {model_key}")


def get_token_counts(token_cost) -> tuple:
    """
    Extract input/output token counts from TokenCost model

    Returns: (input_tokens, output_tokens, total_tokens)
    """
    if hasattr(token_cost, 'prompt_tokens'):
        # TokenCost object
        return (
            token_cost.prompt_tokens,
            token_cost.completion_tokens,
            token_cost.total_tokens
        )
    else:
        # Legacy dictionary format (fallback)
        if "prompt_tokens" in token_cost:
            return (
                token_cost["prompt_tokens"],
                token_cost["completion_tokens"],
                token_cost["total_tokens"]
            )
        elif "input_tokens" in token_cost:
            return (
                token_cost["input_tokens"],
                token_cost["output_tokens"],
                token_cost["input_tokens"] + token_cost["output_tokens"]
            )
        elif "prompt_token_count" in token_cost:
            return (
                token_cost["prompt_token_count"],
                token_cost["candidates_token_count"],
                token_cost["total_token_count"]
            )
        else:
            return (0, 0, 0)


def display_batch_summary(results: list):
    """Display comprehensive summary table of all batch test results"""

    console.print("\n" + "=" * 160)
    console.print("[bold cyan]BATCH TEST SUMMARY - DETAILED COMPARISON[/bold cyan]", justify="center")
    console.print("=" * 160 + "\n")

    for test in results:
        test_name = test["test_name"]

        # Create table for this test
        table = Table(title=f"{test_name}", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", width=18)
        table.add_column("Type", style="yellow", width=10)
        table.add_column("Input\nTokens", justify="right", style="green", width=10)
        table.add_column("Output\nTokens", justify="right", style="green", width=10)
        table.add_column("Total\nTokens", justify="right", style="green", width=10)
        table.add_column("Cost", justify="right", style="blue", width=12)
        table.add_column("Savings", justify="right", style="magenta", width=18)
        table.add_column("Strategies Used", style="white", width=25)

        for result in test.get("results", []):
            model = result["model"]
            baseline = result["baseline"]
            optimized = result["optimized"]

            # Get token counts
            base_in, base_out, base_total = get_token_counts(baseline["actual_tokens"])
            opt_in, opt_out, opt_total = get_token_counts(optimized["actual_tokens"])

            # Calculate savings
            tokens_saved = base_total - opt_total
            tokens_pct = (tokens_saved / base_total * 100) if base_total > 0 else 0
            cost_saved = baseline["cost"]["total"] - optimized["cost"]["total"]
            cost_pct = (cost_saved / baseline["cost"]["total"] * 100) if baseline["cost"]["total"] > 0 else 0

            # Get strategies used
            strategies_used = ", ".join(optimized.get("selected_strategies", ["manual"]))

            # Add baseline row
            table.add_row(
                model,
                "Baseline",
                str(base_in),
                str(base_out),
                str(base_total),
                f"${baseline['cost']['total']:.6f}",
                "-",
                "-"
            )

            # Add optimized row
            table.add_row(
                "",
                "Optimized",
                str(opt_in),
                str(opt_out),
                str(opt_total),
                f"${optimized['cost']['total']:.6f}",
                f"-{tokens_saved} tokens\n-${cost_saved:.6f}\n({tokens_pct:.1f}%)",
                strategies_used
            )

            # Add separator between models
            table.add_row("", "", "", "", "", "", "", "")

        console.print(table)
        console.print()


@cli.command()
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--output', default='results.json', help='Output file for results')
def batch(test_file, output):
    """Run batch tests from JSON file"""

    with open(test_file, 'r') as f:
        test_cases = json.load(f)

    console.print(f"\n[bold cyan]Running {len(test_cases)} batch tests...[/bold cyan]\n")

    results = []

    for i, test_case in enumerate(test_cases, 1):
        console.print(f"\n[bold]Test {i}/{len(test_cases)}: {test_case['name']}[/bold]")

        prompt = test_case["prompt"]
        models = test_case.get("models", ["gpt-4o-mini"])
        json_data = json.dumps(test_case.get("json_data")) if "json_data" in test_case else None

        test_results = []
        for model_key in models:
            if model_key not in PRICING:
                console.print(f"[red]Skipping unknown model: {model_key}[/red]")
                continue

            try:
                baseline = run_baseline(prompt, model_key, json_data)
                # Use 'auto' to intelligently select strategies
                optimized = run_optimized(prompt, model_key, ["auto"], json_data)

                test_results.append({
                    "model": model_key,
                    "baseline": baseline,
                    "optimized": optimized,
                })

                # Brief output with strategies used
                savings = baseline["cost"]["total"] - optimized["cost"]["total"]
                strategies_str = ", ".join(optimized.get("selected_strategies", []))
                console.print(f"  ✓ {model_key}: ${savings:.6f} saved (strategies: {strategies_str})")

            except Exception as e:
                console.print(f"  ✗ {model_key}: {str(e)}")
                logger.exception(f"Error in batch test for {model_key}")

        results.append({
            "test_name": test_case["name"],
            "results": test_results
        })

    # Save results
    with open(output, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"\n[green]✓ Results saved to {output}[/green]")

    # Display summary table
    display_batch_summary(results)


@cli.command()
@click.option('--results-file', default='results.json', type=click.Path(exists=True), help='Results file to analyze')
def stats(results_file):
    """Display optimization statistics from results file"""

    with open(results_file, 'r') as f:
        results = json.load(f)

    total_baseline_cost = 0.0
    total_optimized_cost = 0.0
    total_tests = 0

    for test in results:
        for result in test.get("results", []):
            if "baseline" in result and "optimized" in result:
                total_baseline_cost += result["baseline"]["cost"]["total"]
                total_optimized_cost += result["optimized"]["cost"]["total"]
                total_tests += 1

    total_savings = total_baseline_cost - total_optimized_cost
    savings_pct = (total_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0

    # Display summary
    table = Table(title="Aggregate Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Tests", str(total_tests))
    table.add_row("Total Baseline Cost", f"${total_baseline_cost:.4f}")
    table.add_row("Total Optimized Cost", f"${total_optimized_cost:.4f}")
    table.add_row("Total Savings", f"${total_savings:.4f} ({savings_pct:.1f}%)")

    console.print(table)

    # Cache stats
    cache_stats = cache_manager.get_cache_stats()
    console.print(f"\n[bold]Cache Statistics:[/bold]")
    console.print(f"  Semantic cache: {cache_stats.get('semantic_entries', 0)} entries")
    console.print(f"  Dedup cache: {cache_stats.get('dedup_entries', 0)} entries")


@cli.command()
def list_models():
    """List all available models and their pricing"""

    table = Table(title="Available Models")
    table.add_column("Model Key", style="cyan")
    table.add_column("Provider", style="yellow")
    table.add_column("Input ($/1M)", style="green", justify="right")
    table.add_column("Output ($/1M)", style="green", justify="right")
    table.add_column("Cached ($/1M)", style="blue", justify="right")

    for model_key, pricing in PRICING.items():
        cached_price = pricing.get("cached_input") or pricing.get("cache_read") or "N/A"
        table.add_row(
            model_key,
            pricing["provider"],
            f"${pricing['input']:.2f}",
            f"${pricing['output']:.2f}",
            f"${cached_price:.2f}" if cached_price != "N/A" else "N/A"
        )

    console.print(table)


if __name__ == '__main__':
    cli()
