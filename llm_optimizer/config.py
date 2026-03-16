"""
Pricing data as of November 20, 2025
All prices are per million tokens
Sources: Official provider documentation
"""

PRICING = {
    # OpenAI Models
    "gpt-4o": {
        "model_id": "gpt-4o-2024-11-20",
        "provider": "openai",
        "input": 2.50,
        "output": 10.00,
        "cached_input": 1.25,
        "batch_input": 1.25,
        "batch_output": 5.00,
        "supports_caching": True,
        "cache_threshold": 1024,  # tokens
        "context_window": 128000,
    },
    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini-2024-07-18",
        "provider": "openai",
        "input": 0.15,
        "output": 0.60,
        "cached_input": 0.075,
        "batch_input": 0.075,
        "batch_output": 0.30,
        "supports_caching": True,
        "cache_threshold": 1024,
        "context_window": 128000,
    },
    "gpt-4-turbo": {
        "model_id": "gpt-4-turbo-2024-04-09",
        "provider": "openai",
        "input": 10.00,
        "output": 30.00,
        "cached_input": None,
        "batch_input": 5.00,
        "batch_output": 15.00,
        "supports_caching": False,
        "context_window": 128000,
    },

    # Anthropic Claude Models
    "claude-sonnet-4-5": {
        "model_id": "claude-sonnet-4-5-20250929",
        "provider": "anthropic",
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_read": 0.30,
        "batch_input": 1.50,
        "batch_output": 7.50,
        "supports_caching": True,
        "cache_threshold": 2048,
        "context_window": 200000,
    },
    "claude-3-5-sonnet": {
        "model_id": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "input": 3.00,
        "output": 15.00,
        "cache_write_5m": 3.75,
        "cache_read": 0.30,
        "batch_input": 1.50,
        "batch_output": 7.50,
        "supports_caching": True,
        "cache_threshold": 2048,
        "context_window": 200000,
    },
    "claude-haiku-4-5": {
        "model_id": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "input": 1.00,
        "output": 5.00,
        "cache_write_5m": 1.25,
        "cache_read": 0.10,
        "batch_input": 0.50,
        "batch_output": 2.50,
        "supports_caching": True,
        "cache_threshold": 2048,
        "context_window": 200000,
    },
    "claude-3-5-haiku": {
        "model_id": "claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "input": 0.80,
        "output": 4.00,
        "cache_write_5m": 1.00,
        "cache_read": 0.08,
        "batch_input": 0.40,
        "batch_output": 2.00,
        "supports_caching": True,
        "cache_threshold": 2048,
        "context_window": 200000,
    },

    # Google Gemini Models
    "gemini-2-0-flash": {
        "model_id": "gemini-2.0-flash",
        "provider": "google",
        "input": 0.10,
        "output": 0.40,
        "cached_input": 0.025,
        "batch_input": 0.05,
        "batch_output": 0.20,
        "supports_caching": True,
        "context_window": 1000000,
    },
    "gemini-2-5-flash": {
        "model_id": "gemini-2.5-flash",
        "provider": "google",
        "input": 0.30,
        "output": 2.50,
        "cached_input": 0.03,
        "batch_input": 0.15,
        "batch_output": 1.25,
        "supports_caching": True,
        "context_window": 1000000,
    },
    "gemini-2-5-pro": {
        "model_id": "gemini-2.5-pro",
        "provider": "google",
        "input": 1.25,
        "output": 10.00,
        "input_over_200k": 2.50,
        "output_over_200k": 15.00,
        "cached_input": 0.125,
        "cached_input_over_200k": 0.25,
        "batch_input": 0.625,
        "batch_output": 5.00,
        "supports_caching": True,
        "context_window": 2000000,
        "threshold_200k": 200000,
    },
}

# Embedding model pricing
EMBEDDING_PRICING = {
    "text-embedding-3-small": {
        "provider": "openai",
        "input": 0.02,
    }
}
