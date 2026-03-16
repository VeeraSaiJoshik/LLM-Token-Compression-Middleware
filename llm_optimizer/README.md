# LLM Optimizer — Caveman Algorithm

A Python toolkit for reducing LLM API costs through prompt compression. The core of this project is a custom **Caveman Algorithm** — a semantic compression engine that strips prompts down to their most essential words, reducing token usage while preserving meaning.

---

## The Caveman Algorithm

The Caveman Algorithm answers a simple question: *which words in a prompt actually matter?* It compresses prompts the way a caveman speaks — cutting out filler and keeping only the semantically load-bearing words.

### How It Works

**Step 1 — POS Chunking**

The prompt is tokenized and each word is tagged with its part of speech (NLTK). Words are chunked by semantic role:
- Nouns, adjectives, and possessives are grouped into chunks
- Proper nouns are tracked separately and always preserved
- Verbs, conjunctions, prepositions, and filler words become candidates for removal

**Step 2 — Generate Candidates**

For each word chunk in the prompt, a candidate sentence is constructed by removing that chunk. This yields `N` candidate prompts, one per removable chunk.

**Step 3 — Semantic Similarity Scoring**

All candidates and the original prompt are encoded using `sentence-transformers` (`paraphrase-MiniLM-L12-v2`). Cosine similarity is computed between each candidate and the original, producing a similarity score per removed chunk — a measure of how much meaning was lost by removing it.

**Step 4 — Outlier Filtering**

Raw similarity scores contain noise. A **Modified Z-Score** (using MAD rather than standard deviation) filters statistical outliers — words whose removal causes an anomalous drop in similarity are flagged as important.

**Step 5 — Derivative Analysis (Cut-Point Detection)**

The filtered, sorted similarity scores are differentiated. Outliers in the *derivative* signal inflection points — these are the natural "cut points" in the similarity curve where removing more words starts meaningfully degrading the prompt. If no derivative outliers exist, local maxima are used instead.

**Step 6 — Reconstruct**

Words above the cut-point are dropped. Proper nouns are re-inserted at their original positions regardless of score. The remaining words are joined back into a compressed prompt.

### Example

```
Input:  "Could you please kindly analyze this sales data and provide a very
         detailed comprehensive explanation of the results for the team?"

Output: "analyze sales data provide detailed explanation results"
```

The algorithm identifies that "Could you please kindly", "a very", "comprehensive", "for the team" contribute minimal semantic weight and removes them. Proper nouns (e.g., team names, product names) are always kept.

### Key Design Decisions

- **Proper noun preservation** — Proper nouns carry critical specificity and are never removed, even if their similarity score is low.
- **MAD-based outlier detection** — Standard deviation is sensitive to extreme values; MAD is more robust for small, noisy similarity distributions.
- **Derivative cut-points** — Rather than applying a fixed similarity threshold, the algorithm finds the natural "elbow" in the curve, making it adaptive to each prompt's structure.
- **Chunk-level, not word-level** — Noun phrases and adjective clusters are treated as atomic units so multi-word concepts ("machine learning", "red balloon") aren't split.

### Near-Neighbor Variant

A second mode (`near_neighbor_vector_optimization`) scores each word by its **weighted similarity to its neighbors**, using a Gaussian kernel over positional distance. This surfaces words that are semantically isolated from their context — good candidates for removal in dense technical text.

---

## Other Optimization Strategies

The Caveman Algorithm handles the hard problem of semantic compression. The toolkit also includes lighter-weight strategies that run quickly with no model inference:

### TOON Conversion

Converts JSON payloads embedded in prompts to a compact tabular notation. Uniform arrays of objects are collapsed into a header row + value rows, removing redundant key names.

```
Before (JSON):
{"sales": [{"date": "2025-01-01", "product": "Widget A", "amount": 1250.50},
           {"date": "2025-01-02", "product": "Widget B", "amount": 890.25}]}

After (TOON):
sales:
  [2]{date,product,amount}:
  2025-01-01,Widget A,1250.50
  2025-01-02,Widget B,890.25
```

Typical savings: **30–50%** on structured data.

### Prompt Compression

Regex-based removal of politeness markers, redundant qualifiers, and filler phrases ("Could you please kindly", "I would like you to", "very detailed comprehensive").

Typical savings: **10–20%** on verbose prompts.

### Semantic Caching

Embeds prompts and checks cosine similarity against a cache of previous requests (threshold: 0.85). Cache hits skip the API call entirely.

Savings: **100%** for near-duplicate requests.

### Model Routing

Analyzes prompt complexity (length, technical keywords, code blocks) and routes to the cheapest model capable of handling it.

Example savings: routing a simple question to `gpt-4o-mini` vs `gpt-4o` is a **94% cost reduction** ($0.15 vs $2.50 per 1M tokens).

### Intelligent Strategy Selection

`strategy_selector.py` analyzes each prompt and automatically picks which strategies to apply, avoiding unnecessary overhead:
- Always applies whitespace normalization if excess spacing is detected
- Only runs TOON if JSON is present
- Only runs compression if politeness markers or high word count is detected
- Always checks cache

---

## Installation

```bash
pip install -r requirements.txt
```

Download NLTK data (required for the Caveman Algorithm):

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
```

Set up API keys:

```bash
cp .env.example .env
# Add your OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY
```

---

## Usage

### Caveman Algorithm (direct)

```bash
python optimizers/vector_prompt_optimizer.py single-prompt "Could you please explain how neural networks learn from data in detail?"

python optimizers/vector_prompt_optimizer.py time-graph  # benchmark across sample prompts
```

### Full Pipeline (with cost measurement)

```bash
# Single prompt, compare baseline vs optimized
python main.py test "Explain quantum computing" --models gpt-4o-mini,claude-3-5-haiku

# With JSON data (triggers TOON conversion)
python main.py test "Analyze this sales data" --json-file data.json

# Choose specific strategies
python main.py test "Write a sort function" --strategies toon,compression,cache

# Batch test
python main.py batch tests/sample_prompts.json --output results.json

# View aggregated stats
python main.py stats --results-file results.json
```

Available strategies: `toon`, `compression`, `cache`, `routing`, `all`

---

## Cost Measurement

All cost calculations use **actual token counts** from API responses, not estimates. The `tiktoken` pre-flight estimate is displayed for reference only.

```
BASELINE                        OPTIMIZED
Prompt Length  487 chars    →   298 chars  (-189)
Total Tokens   487          →   362        (-25.7%)
Total Cost     $0.001698    →   $0.001345  (-20.8%)
```

### Pricing (March 2026)

| Model | Input | Output | Cached Input |
|-------|-------|--------|--------------|
| gpt-4o | $2.50 | $10.00 | $1.25 |
| gpt-4o-mini | $0.15 | $0.60 | $0.075 |
| claude-sonnet-4-5 | $3.00 | $15.00 | $0.30 |
| claude-3-5-haiku | $0.80 | $4.00 | $0.08 |
| gemini-2-0-flash | $0.10 | $0.40 | $0.025 |
| gemini-2-5-pro | $1.25 | $10.00 | $0.125 |

---

## Project Structure

```
llm-optimizer/
├── main.py                          # CLI — cost benchmarking pipeline
├── config.py                        # Model pricing
├── optimizers/
│   ├── vector_prompt_optimizer.py   # Caveman Algorithm (core)
│   ├── toon_converter.py            # JSON → TOON compact notation
│   ├── prompt_compressor.py         # Regex-based filler removal
│   ├── semantic_cache.py            # Embedding-based response cache
│   ├── model_router.py              # Complexity-based model selection
│   ├── deduplicator.py              # SHA256 exact-duplicate cache
│   ├── whitespace_optimizer.py      # Whitespace normalization
│   └── strategy_selector.py        # Automatic strategy selection
├── providers/
│   ├── openai_client.py
│   ├── anthropic_client.py
│   └── google_client.py
├── utils/
│   ├── tokenizer.py                 # tiktoken pre-flight estimates
│   ├── cost_calculator.py           # Cost from actual API token counts
│   └── cache_manager.py
└── tests/
    └── sample_prompts.json
```

---

## Supported Models

**OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`

**Anthropic**: `claude-sonnet-4-5`, `claude-3-5-sonnet`, `claude-haiku-4-5`, `claude-3-5-haiku`

**Google**: `gemini-2-5-pro`, `gemini-2-5-flash`, `gemini-2-0-flash`

```bash
python main.py list-models
```

---

## License

MIT
