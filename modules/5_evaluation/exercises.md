# Module 5 Exercises: Evaluation & Metrics

Complete these exercises after studying `demo.py`. Solutions are in `solutions.py`.

---

## Easy Exercises (Start Here!)

### Exercise 1: Calculate Precision, Recall, F1

**Task**: Implement the core retrieval metrics.

**Fill in this function:**
```python
def calculate_metrics(retrieved_ids, relevant_ids, k=3):
    """
    Precision@k = (relevant docs in top-k) / k
    Recall@k = (relevant docs in top-k) / (total relevant docs)
    F1@k = harmonic mean of Precision and Recall
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # True positives: relevant docs that were retrieved
    tp = len(retrieved_set & relevant_set)
    
    # TODO: Calculate precision, recall, f1
    precision = ???
    recall = ???
    f1 = ???
    
    return {'precision': precision, 'recall': recall, 'f1': f1}
```

**Test with**:
```python
retrieved = ['TICK-001', 'TICK-002', 'TICK-003']
relevant = ['TICK-001', 'TICK-003']
# Expected: precision=0.67, recall=1.0, f1=0.80
```

---

### Exercise 2: Evaluate All Queries

**Task**: Run the metrics on all evaluation queries and compute averages.

**In demo.py, find the evaluation loop and add your own print statements:**
```python
all_metrics = []
for query in eval_queries:
    docs = vector_store.similarity_search(query['question'], k=3)
    retrieved = [doc.metadata['ticket_id'] for doc in docs]
    metrics = calculate_metrics(retrieved, query['relevant_ticket_ids'])
    all_metrics.append(metrics)

# Compute averages
avg_precision = np.mean([m['precision'] for m in all_metrics])
avg_recall = np.mean([m['recall'] for m in all_metrics])
avg_f1 = np.mean([m['f1'] for m in all_metrics])

print(f"Precision@3: {avg_precision:.4f}")
print(f"Recall@3:    {avg_recall:.4f}")
print(f"F1@3:        {avg_f1:.4f}")
```

**Questions**:
- What's your system's precision and recall?
- Which metric is higher? Why?

---

### Exercise 3: Compare Different k Values

**Task**: See how precision and recall change with different k values.

```python
for k in [1, 3, 5, 10]:
    metrics = []
    for query in eval_queries:
        docs = vector_store.similarity_search(query['question'], k=k)
        retrieved = [doc.metadata['ticket_id'] for doc in docs]
        m = calculate_metrics(retrieved, query['relevant_ticket_ids'], k=k)
        metrics.append(m)
    
    print(f"k={k}: Precision={np.mean([m['precision'] for m in metrics]):.2f}, "
          f"Recall={np.mean([m['recall'] for m in metrics]):.2f}")
```

**Questions**:
- As k increases, what happens to precision?
- As k increases, what happens to recall?
- What's the best k for your use case?

---

### Exercise 4: Calculate Average Precision

**Task**: Implement Average Precision (AP), which considers ranking order.

```python
def average_precision(retrieved_ids, relevant_ids):
    """
    AP = average of precision@k for all k where a relevant doc is found
    Higher score = relevant docs appear earlier in results
    """
    precisions = []
    relevant_count = 0
    
    for k, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_at_k = relevant_count / k
            precisions.append(precision_at_k)
    
    return np.mean(precisions) if precisions else 0.0
```

**Test**:
- `retrieved=['A', 'B', 'C']`, `relevant=['A', 'C']` → AP = (1.0 + 0.67) / 2 = 0.83
- `retrieved=['B', 'A', 'C']`, `relevant=['A', 'C']` → AP = (0.5 + 0.67) / 2 = 0.58

---

## Medium Exercises

### Exercise 5: LLM-as-Judge for Groundedness

**Task**: Use an LLM to check if answers are supported by context (no hallucinations).

```python
def evaluate_groundedness(answer, context_docs):
    """Check if answer is supported by context"""
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Rate groundedness (0-10):
- 10: Every claim is supported
- 5: Some claims unsupported
- 0: Completely hallucinated

Format: Score: X / Reason: <explanation>"""

    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content
```

**Test on your RAG system's answers. Are they grounded?**

---

### Exercise 6: LLM-as-Judge for Completeness

**Task**: Check if answers fully address the question.

```python
def evaluate_completeness(question, answer, reference_answer=None):
    """Check if answer fully addresses the question"""
    prompt = f"""Evaluate if the ANSWER fully addresses the QUESTION.

QUESTION: {question}
ANSWER: {answer}

Rate completeness (0-10):
- 10: Fully answers all aspects
- 5: Partial answer
- 0: Does not answer

Format: Score: X / Reason: <explanation>"""

    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content
```

---

### Exercise 7: Failure Analysis

**Task**: Find queries where retrieval performs poorly.

```python
from collections import defaultdict

def analyze_failures(eval_queries, vector_store, threshold=0.5):
    """Find queries with Precision < threshold"""
    failures = []
    
    for query in eval_queries:
        docs = vector_store.similarity_search(query['question'], k=3)
        retrieved = [doc.metadata['ticket_id'] for doc in docs]
        metrics = calculate_metrics(retrieved, query['relevant_ticket_ids'])
        
        if metrics['precision'] < threshold:
            failures.append({
                'query': query['question'],
                'precision': metrics['precision'],
                'retrieved': retrieved,
                'expected': query['relevant_ticket_ids'],
                'category': query.get('category', 'Unknown')
            })
    
    # Group by category to find patterns
    by_category = defaultdict(list)
    for f in failures:
        by_category[f['category']].append(f)
    
    print(f"Found {len(failures)} failures")
    for category, items in by_category.items():
        print(f"  {category}: {len(items)} failures")
    
    return failures
```

**Questions**:
- Which categories have the most failures?
- What patterns do you see?
- How would you fix them?

---

### Exercise 8: Track Latency and Cost

**Task**: Measure operational metrics for production readiness.

```python
import time

class RAGMetrics:
    def __init__(self):
        self.query_times = []
        self.token_counts = []
    
    def track_query(self, query, result, elapsed_time):
        self.query_times.append(elapsed_time)
        tokens = len(result) / 4  # Rough estimate
        self.token_counts.append(tokens)
    
    def report(self):
        print(f"Total queries: {len(self.query_times)}")
        print(f"Avg latency: {np.mean(self.query_times):.3f}s")
        print(f"p95 latency: {np.percentile(self.query_times, 95):.3f}s")
        
        total_tokens = sum(self.token_counts)
        cost = (total_tokens / 1000) * 0.002  # GPT-4o-mini estimate
        print(f"Est. cost: ${cost:.4f}")

# Usage
metrics = RAGMetrics()
for query in eval_queries[:10]:
    start = time.time()
    result = generate_answer(query['question'])
    elapsed = time.time() - start
    
    metrics.track_query(query['question'], result['answer'], elapsed)

metrics.report()
```

---

## Bonus Challenge

### Bonus: Create Comprehensive Evaluation Report

**Task**: Combine all metrics into a single report card.

```python
def evaluation_report(eval_queries, vector_store, generate_fn):
    """Generate comprehensive evaluation report"""
    retrieval_scores = []
    generation_scores = []
    
    for query in eval_queries[:5]:  # Limit for demo
        # Retrieval metrics
        docs = vector_store.similarity_search(query['question'], k=3)
        retrieved = [doc.metadata['ticket_id'] for doc in docs]
        r_metrics = calculate_metrics(retrieved, query['relevant_ticket_ids'])
        retrieval_scores.append(r_metrics)
        
        # Generation metrics
        result = generate_fn(query['question'])
        groundedness = evaluate_groundedness(result['answer'], docs)
        generation_scores.append(groundedness)
    
    print("=" * 60)
    print("RAG SYSTEM EVALUATION REPORT")
    print("=" * 60)
    print(f"\nRETRIEVAL:")
    print(f"  Precision@3: {np.mean([s['precision'] for s in retrieval_scores]):.4f}")
    print(f"  Recall@3:    {np.mean([s['recall'] for s in retrieval_scores]):.4f}")
    print(f"\nGENERATION:")
    print(f"  Groundedness: (check individual scores)")
    print("=" * 60)
```

**Targets for production:**
- Precision@3 > 0.80
- Recall@3 > 0.70
- Groundedness > 0.80
- Completeness > 0.75

---

## Key Concepts Summary

| Metric | What It Measures | Target |
|--------|------------------|--------|
| **Precision@k** | % of retrieved docs that are relevant | > 0.80 |
| **Recall@k** | % of relevant docs that were retrieved | > 0.70 |
| **F1@k** | Harmonic mean of precision & recall | > 0.75 |
| **Groundedness** | Is answer supported by context? | > 0.80 |
| **Completeness** | Does answer fully address question? | > 0.75 |

---

## 🎉 Congratulations!

You've completed the RAG Evaluation module! You now know how to:

1. Calculate retrieval metrics (Precision, Recall, F1)
2. Use LLM-as-judge for generation quality
3. Identify and analyze failures
4. Track operational metrics (latency, cost)
5. Build comprehensive evaluation reports

---

**Need help?** Check `solutions.py` or ask the instructor!
