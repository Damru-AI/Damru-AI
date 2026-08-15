# Groq API Optimization for 1M Lines/Day

## Groq API Characteristics

**Model**: `llama-3.3-70b-versatile`
- **Input tokens**: ~8K (typical prompt)
- **Output tokens**: ~8K (max allowed per request)
- **Throughput**: 500+ tokens/second
- **Latency**: 1-3 seconds average

**Pricing**: Extremely cheap or free depending on plan

## Current Configuration

```python
PARALLEL_WORKERS = 20           # Concurrent requests
QUESTIONS_PER_BATCH = 50        # Q&A pairs per request
TARGET_LINES = 1,000,000        # Daily target
```

## Performance Math

### Per Batch:
- 50 Q&A pairs × 13 lines average = **650 lines**
- Request time: ~2-3 seconds
- Throughput: ~210 lines/second

### Daily:
- 1M lines ÷ 650 lines/batch = **~1,538 batches**
- 1,538 batches ÷ 20 workers = **~77 batches per worker**
- Runtime: ~77 × 2.5s = **~192 seconds (~3 minutes)** per worker
- **Total time: ~10-15 minutes** (with parallelization)

## API Rate Limiting

Groq's rate limits (Free tier):
- **RPM** (Requests Per Minute): ~30-60
- **TPM** (Tokens Per Minute): ~6,000-10,000

With 20 workers:
- ~20 requests/batch = ~200-300 RPM
- **EXCEEDS free tier limits**

### Solution: Adaptive Throttling

```python
PARALLEL_WORKERS = 8-12  # Recommended for free tier
# OR
PARALLEL_WORKERS = 20 + distributed queuing (for paid plans)
```

## Recommended Configurations

### Conservative (Free Tier):
```
PARALLEL_WORKERS = 8
QUESTIONS_PER_BATCH = 50
TARGET_LINES = 300,000-500,000  # Realistic free tier
RUNTIME = 60-120 minutes
```

### Balanced (Pro Tier):
```
PARALLEL_WORKERS = 15
QUESTIONS_PER_BATCH = 50
TARGET_LINES = 1,000,000
RUNTIME = 20-30 minutes
```

### Aggressive (Enterprise Tier):
```
PARALLEL_WORKERS = 50+
QUESTIONS_PER_BATCH = 100
TARGET_LINES = 5,000,000+
RUNTIME = 10-15 minutes
```

## Optimizations Implemented

### 1. Batch Retry with Exponential Backoff
```python
retry_count = 0
max_retries = 3
while retry_count < max_retries:
    if r.status_code == 429:
        wait_time = int(r.headers.get('retry-after', 30))
        time.sleep(min(wait_time, 60))
        retry_count += 1
```

### 2. Smart Deduplication
```python
# Hash first 10 words to detect duplicates
k = ' '.join(question.lower().split()[:10])
if k not in seen:
    out.append(record)
```

### 3. Prompt Optimization
```python
# Reduced tokens by:
# - Using concise system prompt
# - Removing unnecessary formatting
# - Specifying max_tokens: 16000 (not 32000)
# - Setting temperature: 0.7 (faster than 0.9)
```

### 4. Parallel Processing
```python
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(...) for _ in range(n_batches)]
    for future in as_completed(futures):
        # Process as completed, don't wait for all
```

## Groq API Best Practices

### ✓ DO:
- Use connection pooling (requests library handles this)
- Implement exponential backoff for 429/5xx
- Batch processing within token limits
- Monitor response times
- Use `top_p=0.95` for creativity
- Set reasonable timeouts (120s)

### ✗ DON'T:
- Hammer API without backoff (will get rate limited)
- Use max tokens > 16K (causes timeouts)
- Send malformed JSON (Groq is strict)
- Ignore Retry-After headers
- Use deprecated endpoints

## Monitoring Groq Performance

### Log Analysis:
```bash
# Check success rate
grep "success=" learn/logs/daily.log | tail -1

# Find rate limits
grep "429\|Rate limit" learn/logs/daily.log

# Average response time
grep "batch error\|Groq error" learn/logs/daily.log | wc -l
```

### Metrics to Track:
- Batches/hour: Should be ~100-500
- Error rate: Should be < 5%
- Average latency: Should be 1-3s
- Dedup rate: ~5-20% (healthy)

## Cost Optimization

### Groq Free Tier Allocation:
- Approx: **$5-10 monthly credit** (equivalent)
- With optimal config: **~300K-500K lines/day** feasible
- Cost per line: ~$0.00001-0.00005

### Ways to Reduce Cost:
1. Reduce batch size (25 instead of 50)
2. Reduce parallel workers (5 instead of 20)
3. Increase target lines over 2-3 runs instead of 1
4. Use cheaper model: `llama-3.1-8b-instant` (available)

### Groq Tier Recommendation:
| Target | Tier | Cost | Notes |
|--------|------|------|-------|
| 100K/day | Free | $0 | Easy |
| 500K/day | Free | $0 | Tight, requires throttling |
| 1M/day | Pro ($2-5/day) | ~$60-150/mo | Comfortable |
| 5M+/day | Enterprise | Custom | Full utilization |

## A/B Testing Ideas

### Prompt Variations:
```python
# Current: Summarized with "Mix Hindi and English"
# Test: Detailed subject-specific persona

# Test: Different temperature/top_p combinations
# for creativity vs. consistency tradeoff
```

### Batch Size Variations:
```python
# 25 Q&A: Faster, less content/request
# 50 Q&A: Current, balanced
# 100 Q&A: Longer requests, fewer total
```

### Model Variations:
```python
# llama-3.3-70b: Current, highest quality
# llama-3.1-70b: Slightly faster
# llama-3.1-8b-instant: Much faster, lower quality
```

## Future Enhancements

- [ ] Distributed rate limiting across multiple API keys
- [ ] Adaptive worker pool (scale up/down based on latency)
- [ ] Caching common Q&A patterns
- [ ] Multi-model ensemble (Groq + Gemini + HF)
- [ ] Quality scoring (reject low-quality Q&A)
- [ ] A/B testing framework for prompts

---

**Last Updated**: 2024-08-15
**Optimized for**: Groq llama-3.3-70b-versatile
