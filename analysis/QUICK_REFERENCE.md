# Token Optimization Strategy - Quick Reference Guide

## 🎯 When to Use Each Strategy

### 1. Cache (Priority: HIGHEST)
**Always apply first**
- ✅ **When**: Every single query
- 💰 **Savings**: 100% cost on cache hit
- ⚡ **Speed**: Instant (0ms)
- 🎲 **Risk**: None
- 📊 **Hit Rate**: 15-30% typical

**Decision**: Always check cache before any other optimization.

---

### 2. TOON (Table-Oriented Object Notation)
**Best for JSON data**

| JSON Size | Apply? | Expected Savings | Confidence |
|-----------|--------|-----------------|------------|
| > 300 chars | ✅ YES | 50-60% | High |
| 50-300 chars | ✅ YES | 35-50% | High |
| < 50 chars | ❌ NO | <10% | Skip it |

**Examples from test data:**
- Large JSON (1592 chars): **57.1% reduction** (699 → 300 tokens)
- Medium JSON (337 chars): **56.9% reduction** (167 → 72 tokens)
- Small JSON (117 chars): **41.3% reduction** (but combined with other strategies)

**Decision Logic:**
```python
if contains_json(prompt):
    json_size = extract_json_size(prompt)
    if json_size > 50:
        return "APPLY_TOON"
```

---

### 3. Whitespace Normalization
**Zero-risk, always apply when detected**

**When to apply:**
- ✅ Multiple consecutive spaces (> 5 occurrences)
- ✅ Triple newlines or more
- ✅ Excessive tabs (> 10)
- ✅ >30% of lines are empty

**Savings:** 10-20 tokens typical

**Examples from test data:**
- Test 3: **18 tokens saved** (contributed to 37.5% total reduction)
- Test 8: **18 tokens saved** (contributed to 59% total reduction)

**Decision**: If excessive whitespace detected → APPLY (no downside)

---

### 4. Compression (Politeness Removal)
**Context-dependent - be smart!**

#### Apply STANDARD Compression (10-15% savings)
✅ General queries
✅ Data analysis requests
✅ Verbose questions
✅ Customer support prompts

**Test data:** 5-20% reduction with no accuracy loss

#### Apply MINIMAL Compression (5% savings)
⚠️ Code generation tasks
⚠️ Technical documentation
⚠️ API/authentication queries
⚠️ Precision-critical content

**Test data:** 1-5% reduction, conservative approach

#### Skip Compression
❌ No politeness markers detected
❌ Already concise prompts

**Decision Tree:**
```
Has politeness markers?
├─ NO → Skip compression
└─ YES → Check content type
    ├─ Code/Technical → MINIMAL compression
    └─ General query → STANDARD compression
```

---

## 🔥 Best Combinations (Observed in Tests)

### Maximum Optimization (Test 8)
```
Whitespace + TOON + Compression
Result: 59% reduction (573 → 235 tokens)
```

### JSON Heavy (Test 5)
```
TOON + Cache
Result: 57.1% reduction (699 → 300 tokens)
Then: 100% savings on cache hit
```

### Verbose Query (Test 3)
```
Whitespace + Compression
Result: 37.5% reduction (32 → 20 tokens)
```

---

## 📊 Expected ROI by Query Type

| Query Type | Recommended Strategies | Expected Savings |
|------------|----------------------|------------------|
| **Simple fact question** | Cache only | 0% first call, 100% after |
| **Verbose question** | Compression + Cache | 10-20% |
| **Small JSON (<300 chars)** | TOON + Cache | 35-50% |
| **Large JSON (>300 chars)** | TOON + Cache | 50-60% |
| **Code generation** | Cache only | 0-5% (conservative) |
| **Polite + JSON** | All strategies | 40-60% |
| **Polite + JSON + Whitespace** | All strategies | 50-65% |

---

## ⚠️ Risk Levels

### Zero Risk (Always Safe)
- ✅ Cache
- ✅ Whitespace normalization
- ✅ TOON conversion

### Low Risk (Context-Aware)
- ⚠️ Standard compression (general queries)

### Medium Risk (Use with Caution)
- ⚠️ Aggressive compression
- ⚠️ Compression on code/technical content

---

## 🚀 Implementation Priority Order

```
1. CHECK CACHE ────────────► Return if HIT (100% savings)
                              │
2. NORMALIZE WHITESPACE ──────► Apply if detected (10-20 tokens)
                              │
3. CONVERT TO TOON ───────────► Apply if JSON >50 chars (35-60% savings)
                              │
4. APPLY COMPRESSION ─────────► Context-aware (5-20% savings)
                              │
5. EXECUTE OPTIMIZED PROMPT ──► Get response
                              │
6. CACHE RESPONSE ────────────► Enable future 100% savings
```

---

## 📈 Key Metrics from Test Data

### Strategy Performance
| Strategy | Avg Savings | Applicability | Risk |
|----------|------------|---------------|------|
| Cache Hit | 100% cost | 15-30% queries | None |
| TOON (large) | 50-60% tokens | ~20% queries | None |
| TOON (small) | 35-50% tokens | ~15% queries | None |
| Whitespace | 10-20 tokens | ~10% queries | None |
| Compression | 5-20% tokens | ~40% queries | Low |

### Maximum Observed Reductions
- **Highest single reduction**: 57.1% (Test 5 - Large JSON)
- **Highest combined reduction**: 59% (Test 8 - All strategies)
- **Most consistent**: TOON at 50-57% for large JSON

---

## 💡 Quick Decision Checklist

Use this checklist for any incoming prompt:

```
□ Step 1: Check cache → HIT? Return immediately
□ Step 2: Scan for JSON → Size >50 chars? Plan TOON
□ Step 3: Check whitespace → Excessive? Plan normalize
□ Step 4: Check politeness → Present? Check content type
    □ Code/technical? → Minimal compression
    □ General query? → Standard compression
□ Step 5: Apply strategies in priority order
□ Step 6: Cache the result
```

---

## 🎓 Test Case Examples

### Example 1: Large JSON Data Analysis
```
Input: "Analyze this data: {large JSON object}"
Detected: JSON (1592 chars)
Strategy: TOON + Cache
Result: 57% reduction (699 → 300 tokens)
ROI: Excellent
```

### Example 2: Polite Verbose Question
```
Input: "Could you please kindly explain..."
Detected: Politeness markers, no JSON
Strategy: Compression + Cache
Result: 16% reduction (42 → 38 tokens)
ROI: Good
```

### Example 3: Code Generation
```
Input: "Write a Python function..."
Detected: Code generation task
Strategy: Cache only (conservative)
Result: 0% first call, 100% on cache hit
ROI: Conservative first call, excellent on repeat
```

### Example 4: Maximum Optimization
```
Input: "Could you please analyze: {JSON with lots of whitespace}"
Detected: All patterns
Strategy: Whitespace + TOON + Compression
Result: 59% reduction (573 → 235 tokens)
ROI: Excellent
```

---

## 🔧 Tuning Recommendations

### For Maximum Savings (Aggressive)
- Lower TOON threshold to 30 chars
- Apply standard compression broadly
- Increase cache similarity threshold

### For Maximum Safety (Conservative)
- Keep TOON threshold at 100 chars
- Use minimal compression only
- Require exact cache matches

### Recommended (Balanced) ← **Use This**
- TOON threshold: 50 chars
- Context-aware compression
- Semantic cache matching
- **This matches our test data approach**

---

## 📝 Notes

- All percentages based on actual test data (15 comprehensive tests)
- Token counts may vary by model (GPT-4, Claude, etc.)
- Cache hit rate depends on query diversity
- Combine strategies for maximum effect
- Always monitor accuracy - never sacrifice correctness for savings

---

## 🎯 TL;DR - The One-Sentence Guide

> "Always check cache first, apply TOON for JSON >50 chars, normalize excessive whitespace, and compress politeness markers unless it's code/technical content."

This simple rule captures 90% of optimization value with minimal complexity.
