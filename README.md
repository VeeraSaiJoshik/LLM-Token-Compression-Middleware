# Vector Prompt Compression

A prompt compression algorithm that uses sentence embeddings and leave-one-out semantic similarity to identify and remove redundant words from LLM prompts while preserving meaning.

## How It Works

The core idea: encode both the original prompt and a version with each word removed, then measure how much the meaning changes. Words whose removal barely changes the meaning are redundant and can be dropped.

### Pipeline

```
prompt → POS tagging → leave-one-out chunks → sentence embeddings
       → cosine similarity → outlier filtering → elbow detection → compressed prompt
```

### Step 1: POS Tagging & Chunking

NLTK tags each word with its part of speech. The chunker groups consecutive **nouns**, **adjectives**, and **possessives** into semantic units, then generates a leave-one-out list — one version of the prompt per semantic unit, with that unit removed.

**Proper nouns are handled separately**: they are extracted with fractional index positions and always reinserted into the final output, never considered for removal.

```python
# Input: "Explain the concept of recursion in simple terms"
# Chunks (one per removable unit):
#   "the concept of recursion in simple terms"   ← "Explain" removed
#   "Explain of recursion in simple terms"        ← "the concept" removed
#   ...
```

### Step 2: Sentence Embeddings

Each chunk and the original prompt are encoded using `sentence-transformers/paraphrase-MiniLM-L12-v2` via async batching:

```python
tasks = [get_sentence_transformation(prompt, model)] + [
    get_sentence_transformation(chunk, model) for chunk in chunks
]
encoding_results = await asyncio.gather(*tasks)
```

### Step 3: Cosine Similarity Scoring

Each chunk's embedding is compared to the original using cosine similarity. A **high similarity score** means removing that word had little effect on meaning — the word is redundant.

```python
sim = util.pytorch_cos_sim(original_embedding, chunk_embedding)
```

### Step 4: Outlier Filtering (Modified Z-Score / MAD)

Similarity scores are filtered using the **Median Absolute Deviation** method, which is robust to non-normal distributions. Extreme outliers (words that are clearly critical to meaning) are removed from consideration before ranking.

```python
def remove_outliers(data, threshold=3.5):
    median = sorted(data)[len(data) // 2]
    mad = sorted([abs(x - median) for x in data])[len(data) // 2]
    return [x for x in data if abs(0.6745 * (x - median) / mad) <= threshold]
```

### Step 5: Elbow Detection

After sorting the remaining words by similarity, the algorithm computes the **first derivative** of the similarity curve and finds outliers in that derivative — the point where similarity drops sharply. Words below this elbow (i.e., whose removal significantly changes meaning) are kept; words above it are dropped.

If no derivative outliers exist, local maxima of the derivative are used as the cutoff instead.

```python
sim_der = [sim_norm[i+1] - sim_norm[i] for i in range(len(sim_norm) - 1)]
sim_der_outliers = get_outliers(sim_der)
if len(sim_der_outliers) == 0:
    sim_der_outliers = find_maximas(sim_der)
```

### Step 6: Reconstruction

Kept words and proper nouns (pinned to their fractional indexes) are sorted back into their original order and joined to form the compressed prompt.

---

## Example

```
Input:  "Give a detailed comparison between solar power and wind power, including
         cost, efficiency, and environmental impact considerations."

Output: "comparison solar power wind power cost efficiency environmental impact"
```

Compression: ~45% token reduction

---

## Usage

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

```bash
# Compress a single prompt
python optimizers/vector_prompt_optimizer.py single-prompt "Your prompt here"

# Run benchmark across 10 sample prompts with time + compression graphs
python optimizers/vector_prompt_optimizer.py time-graph
```

---

## Also Includes: Near-Neighbor Vector Scoring

A secondary scoring function (`near_neighbor_vector_optimization`) encodes each individual word and computes a **Gaussian-weighted neighborhood similarity** — how semantically similar each word is to its surrounding words.

```python
weight = math.exp(-(distance**2) / (2 * sigma**2))  # sigma=2.0
score = sum(weight * cos_sim(word_i, word_j) for j != i) / sum(weights)
```

This scores words by how much they fit their local context, useful for a different style of redundancy detection.

---

## Dependencies

- `sentence-transformers` — MiniLM-L12-v2 for semantic embeddings
- `nltk` — POS tagging and tokenization
- `numpy` — vector operations
- `matplotlib` — similarity curve and time graphs
- `click`, `rich` — CLI and output formatting
- `tqdm` — progress bars
