# Module 5: RAG Evaluation

## Introduction

Evaluation is critical for building reliable RAG systems. Without proper metrics, you can't measure improvement, compare approaches, or ensure quality. This module covers comprehensive two-layer evaluation: **Retrieval** (finding the right documents) and **Generation** (producing quality answers).

## Why Evaluation Matters

### The Challenge
- How do you know if your RAG system is working well?
- Which indexing strategy performs best?
- Are answers accurate and grounded?
- How do changes impact quality?

### Without Evaluation
❌ Guessing which approach is better  
❌ Deploying broken systems  
❌ No way to measure improvement  
❌ Can't justify technical decisions  

### With Evaluation
✅ Data-driven decisions  
✅ Catch regressions early  
✅ Optimize systematically  
✅ Build confidence in system  

## Two-Layer Evaluation Approach

```
┌─────────────────────────────────────────────┐
│           RAG SYSTEM EVALUATION             │
└─────────────────────────────────────────────┘

Layer 1: RETRIEVAL EVALUATION
  ├─ Precision@K: How many retrieved docs are relevant?
  ├─ Recall@K: How many relevant docs were retrieved?
  └─ F1 Score: Harmonic mean of precision and recall

Layer 2: GENERATION EVALUATION
  ├─ Groundedness: Is answer supported by context?
  └─ Completeness: Does answer address the question fully?
```

### Why Two Layers?

**Retrieval Failure** → Wrong documents → Bad answer
**Generation Failure** → Right documents → Still bad answer

You must evaluate **both** to diagnose problems correctly.

## Layer 1: Retrieval Evaluation

### Creating Ground Truth

```python
evaluation_queries = [
    {
        "query_id": "Q1",
        "question": "How do I fix authentication failures?",
        "relevant_ticket_ids": ["TICK-001", "TICK-011", "TICK-014"],
        "category": "Authentication"
    },
    {
        "query_id": "Q2",
        "question": "What causes database timeouts?",
        "relevant_ticket_ids": ["TICK-002"],
        "category": "Database"
    }
]
```

**How to Create Ground Truth:**
1. Sample real user queries
2. Manually identify relevant documents
3. Have multiple annotators (inter-rater agreement)
4. Start with 15-50 queries
5. Expand as needed

### Precision@K

**Definition:** Of the K documents retrieved, how many are relevant?

```python
def calculate_precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_k) & set(relevant_ids)
    precision = len(relevant_retrieved) / k
    return precision
```

**Example:**
```
Query: "Fix authentication"
Retrieved (k=5): [TICK-001, TICK-003, TICK-011, TICK-020, TICK-014]
Relevant: [TICK-001, TICK-011, TICK-014]

Relevant in top 5: {TICK-001, TICK-011, TICK-014} = 3
Precision@5 = 3 / 5 = 0.60 (60%)
```

**Interpretation:**
- **High precision** (>0.8): Most results are relevant
- **Low precision** (<0.5): Lots of noise in results

**When to Optimize:**
- Users complain about irrelevant results
- They only look at top results
- False positives are costly

### Recall@K

**Definition:** Of all relevant documents, how many did we retrieve?

```python
def calculate_recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant_retrieved = set(retrieved_k) & set(relevant_ids)
    recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0
    return recall
```

**Example:**
```
Query: "Fix authentication"
Retrieved (k=5): [TICK-001, TICK-003, TICK-011, TICK-020, TICK-014]
Relevant: [TICK-001, TICK-011, TICK-014, TICK-016]

Found: {TICK-001, TICK-011, TICK-014} = 3
Total relevant: 4
Recall@5 = 3 / 4 = 0.75 (75%)
```

**Interpretation:**
- **High recall** (>0.8): Capturing most relevant docs
- **Low recall** (<0.5): Missing important information

**When to Optimize:**
- Users say "there should be more results"
- Missing critical documents
- False negatives are costly

### F1 Score

**Definition:** Harmonic mean of precision and recall.

```python
def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

**Formula:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Example:**
```
Precision@5 = 0.60
Recall@5 = 0.75
F1@5 = 2 × (0.60 × 0.75) / (0.60 + 0.75)
     = 2 × 0.45 / 1.35
     = 0.67
```

**Interpretation:**
- **High F1** (>0.75): Good balance
- **Low F1** (<0.5): Poor overall performance

**Why F1?**
- Balances precision and recall
- Single metric for comparison
- Standard in IR research

### Complete Retrieval Metrics Function

```python
def calculate_retrieval_metrics(retrieved_ids, relevant_ids, k):
    """Calculate P, R, F1 for retrieval evaluation"""
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_k)
    
    # True positives
    true_positives = len(relevant_set & retrieved_set)
    
    # Precision@K
    precision = true_positives / k if k > 0 else 0
    
    # Recall@K
    recall = true_positives / len(relevant_set) if relevant_set else 0
    
    # F1 Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives
    }
```

## Layer 2: Generation Evaluation

### The Challenge
How do you evaluate if an answer is "good"?
- Traditional metrics (ROUGE, BLEU) don't capture semantic quality
- Need to check **groundedness** and **completeness**

### Using LLM-as-Judge

**Idea:** Use a powerful LLM to evaluate answer quality.

**Advantages:**
✅ Captures semantic similarity  
✅ Understands nuance  
✅ Cost-effective vs human annotation  
✅ Scales to large datasets  

**Disadvantages:**
❌ LLM bias  
❌ API costs  
❌ Not always accurate  

### Groundedness (Faithfulness)

**Definition:** Is the answer supported by the retrieved context?

**Prompt:**
```python
def evaluate_groundedness(answer, context_docs):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Evaluate if the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Task: Rate groundedness on scale 0-1:
- 1.0: Answer is fully supported by context, no hallucinations
- 0.5: Answer is partially supported, some unsupported claims
- 0.0: Answer is not supported by context or contradicts it

Output only the score (0.0 to 1.0):"""
    
    response = llm.invoke(prompt)
    score = float(response.strip())
    return score
```

**Example:**

```
Context: "TICK-001: Users unable to login after password reset."

Answer: "The issue was resolved by clearing sessions."
Score: 0.0 (Not in context - hallucination!)

Answer: "Users can't login after resetting passwords."
Score: 1.0 (Fully supported)
```

**Improved Prompt (Chain-of-Thought):**
```python
prompt = f"""Evaluate groundedness step-by-step.

Context:
{context}

Answer:
{answer}

Step 1: List all claims in the answer
Step 2: For each claim, check if it appears in the context
Step 3: Calculate: (supported claims) / (total claims)

Output format:
Claims: [list]
Supported: [list]
Score: [0.0 to 1.0]"""
```

### Completeness

**Definition:** Does the answer fully address the question?

**Prompt:**
```python
def evaluate_completeness(question, answer, reference_answer):
    prompt = f"""Evaluate if the answer completely addresses the question.

Question:
{question}

Reference Answer (what a complete answer should cover):
{reference_answer}

Actual Answer:
{answer}

Task: Rate completeness on scale 0-1:
- 1.0: Answer covers all aspects of the reference
- 0.5: Answer is partially complete, missing some points
- 0.0: Answer misses most key points

Output only the score (0.0 to 1.0):"""
    
    response = llm.invoke(prompt)
    score = float(response.strip())
    return score
```

**Example:**

```
Question: "How to fix authentication issues?"
Reference: "Clear sessions, update SAML config, fix NTP for 2FA"

Answer: "Clear active sessions."
Score: 0.3 (Incomplete - missing 2 of 3 points)

Answer: "Clear sessions and update SAML to SHA-256."
Score: 0.7 (Mostly complete - missing NTP)
```

### Relevance

**Definition:** Does the answer address the question?

```python
def evaluate_relevance(question, answer):
    prompt = f"""Rate how relevant the answer is to the question.

Question: {question}
Answer: {answer}

Score (0.0 to 1.0):
- 1.0: Directly answers the question
- 0.5: Partially relevant
- 0.0: Off-topic or unrelated

Output only the score:"""
    
    response = llm.invoke(prompt)
    return float(response.strip())
```

### Combining Metrics

```python
def comprehensive_evaluation(question, answer, context_docs, reference):
    metrics = {
        'groundedness': evaluate_groundedness(answer, context_docs),
        'completeness': evaluate_completeness(question, answer, reference),
        'relevance': evaluate_relevance(question, answer)
    }
    
    # Overall score (weighted average)
    metrics['overall'] = (
        0.5 * metrics['groundedness'] +
        0.3 * metrics['completeness'] +
        0.2 * metrics['relevance']
    )
    
    return metrics
```

## Production Targets

### Recommended Thresholds

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Precision@3 | >0.80 | <0.70 | <0.50 |
| Recall@3 | >0.70 | <0.60 | <0.40 |
| F1@3 | >0.75 | <0.65 | <0.45 |
| Groundedness | >0.85 | <0.75 | <0.60 |
| Completeness | >0.75 | <0.65 | <0.50 |

### Interpreting Results

**High Precision, Low Recall:**
```
Problem: Too conservative, missing relevant docs
Solution: 
- Increase k (retrieve more docs)
- Improve chunking (smaller chunks)
- Enhance query (add synonyms)
```

**Low Precision, High Recall:**
```
Problem: Too many irrelevant docs
Solution:
- Use MMR for diversity
- Add metadata filters
- Improve embeddings
- Use hybrid search
```

**High Retrieval, Low Generation:**
```
Problem: Right docs, wrong answer
Solution:
- Improve prompt engineering
- Use better LLM
- Add few-shot examples
- Implement verification step
```

**Low Groundedness:**
```
Problem: Hallucinations
Solution:
- Stronger grounding prompts
- Lower temperature (0)
- Add source citation requirement
- Implement verification
```

## A/B Testing Different Configurations

### Setup

```python
def compare_configurations(eval_queries, configs):
    results = {}
    
    for name, config in configs.items():
        print(f"Testing {name}...")
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'groundedness': [],
            'completeness': []
        }
        
        for query in eval_queries:
            # Retrieve with this config
            docs = config['retriever'].get_relevant_documents(
                query['question'], 
                k=config['k']
            )
            
            # Retrieval metrics
            retrieved_ids = [d.metadata['id'] for d in docs]
            retrieval = calculate_retrieval_metrics(
                retrieved_ids,
                query['relevant_ids'],
                k=config['k']
            )
            metrics['precision'].append(retrieval['precision'])
            metrics['recall'].append(retrieval['recall'])
            metrics['f1'].append(retrieval['f1'])
            
            # Generation metrics (if reference available)
            if 'reference' in query:
                answer = config['chain'].invoke(query['question'])
                metrics['groundedness'].append(
                    evaluate_groundedness(answer, docs)
                )
                metrics['completeness'].append(
                    evaluate_completeness(query['question'], answer, query['reference'])
                )
        
        # Average metrics
        results[name] = {
            metric: np.mean(values)
            for metric, values in metrics.items()
        }
    
    return results
```

### Example Comparison

```python
configs = {
    'baseline': {
        'retriever': vector_store.as_retriever(search_kwargs={'k': 3}),
        'k': 3,
        'chain': basic_chain
    },
    'hybrid': {
        'retriever': ensemble_retriever,
        'k': 5,
        'chain': basic_chain
    },
    'optimized': {
        'retriever': mmr_retriever,
        'k': 5,
        'chain': advanced_chain_with_examples
    }
}

results = compare_configurations(eval_queries, configs)

# Print comparison
print(f"{'Config':<15} {'P@K':<8} {'R@K':<8} {'F1':<8} {'Ground':<8} {'Complete':<8}")
print("-" * 60)
for name, metrics in results.items():
    print(f"{name:<15} "
          f"{metrics['precision']:<8.3f} "
          f"{metrics['recall']:<8.3f} "
          f"{metrics['f1']:<8.3f} "
          f"{metrics['groundedness']:<8.3f} "
          f"{metrics['completeness']:<8.3f}")
```

**Output:**
```
Config          P@K      R@K      F1       Ground   Complete
------------------------------------------------------------
baseline        0.600    0.750    0.667    0.750    0.650   
hybrid          0.667    0.833    0.741    0.780    0.680   
optimized       0.733    0.867    0.793    0.850    0.750   
```

## Automated Evaluation Pipeline

```python
class RAGEvaluator:
    def __init__(self, rag_system, eval_queries):
        self.rag_system = rag_system
        self.eval_queries = eval_queries
    
    def run_evaluation(self):
        results = []
        
        for query_data in self.eval_queries:
            # Get answer and sources
            response = self.rag_system.query(query_data['question'])
            
            # Retrieval evaluation
            retrieval_metrics = calculate_retrieval_metrics(
                response['source_ids'],
                query_data['relevant_ids'],
                k=3
            )
            
            # Generation evaluation
            generation_metrics = {
                'groundedness': evaluate_groundedness(
                    response['answer'],
                    response['source_docs']
                ),
                'completeness': evaluate_completeness(
                    query_data['question'],
                    response['answer'],
                    query_data['reference']
                )
            }
            
            # Combine
            result = {
                'query_id': query_data['query_id'],
                'question': query_data['question'],
                **retrieval_metrics,
                **generation_metrics
            }
            results.append(result)
        
        return self.summarize_results(results)
    
    def summarize_results(self, results):
        summary = {}
        metrics = ['precision', 'recall', 'f1', 'groundedness', 'completeness']
        
        for metric in metrics:
            values = [r[metric] for r in results]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary, results
```

## Human Evaluation

While LLM-as-judge is useful, human evaluation remains the gold standard.

### Annotation Guidelines

```python
evaluation_form = {
    'query_id': 'Q1',
    'question': "How do I fix authentication issues?",
    'answer': "...",
    'sources': ["TICK-001", "TICK-011"],
    
    'ratings': {
        'retrieval_quality': 1-5,  # Were right docs retrieved?
        'answer_accuracy': 1-5,  # Is answer correct?
        'answer_completeness': 1-5,  # Is answer complete?
        'clarity': 1-5,  # Is answer clear?
        'groundedness': 1-5  # Is answer supported?
    },
    
    'feedback': "Optional comments..."
}
```

### Inter-Rater Agreement

```python
from sklearn.metrics import cohen_kappa_score

# Two annotators rate same samples
rater1 = [5, 4, 3, 5, 4]
rater2 = [5, 4, 4, 5, 3]

# Cohen's Kappa (agreement beyond chance)
kappa = cohen_kappa_score(rater1, rater2)
# > 0.8: Excellent agreement
# 0.6-0.8: Good agreement
# < 0.6: Poor agreement
```

## Continuous Evaluation

### Monitoring Dashboard

```python
class RAGMonitor:
    def __init__(self):
        self.metrics_history = []
    
    def log_query(self, query, response, retrieval_metrics, generation_metrics):
        entry = {
            'timestamp': datetime.now(),
            'query': query,
            'latency': response['latency'],
            **retrieval_metrics,
            **generation_metrics
        }
        self.metrics_history.append(entry)
    
    def get_daily_summary(self, date):
        day_metrics = [
            m for m in self.metrics_history
            if m['timestamp'].date() == date
        ]
        
        return {
            'total_queries': len(day_metrics),
            'avg_precision': np.mean([m['precision'] for m in day_metrics]),
            'avg_recall': np.mean([m['recall'] for m in day_metrics]),
            'avg_groundedness': np.mean([m['groundedness'] for m in day_metrics]),
            'avg_latency': np.mean([m['latency'] for m in day_metrics])
        }
```

### Alerts

```python
def check_thresholds(metrics):
    alerts = []
    
    if metrics['precision'] < 0.7:
        alerts.append("WARNING: Precision below threshold")
    
    if metrics['groundedness'] < 0.75:
        alerts.append("CRITICAL: High hallucination risk")
    
    if metrics['latency'] > 5.0:
        alerts.append("WARNING: Slow response time")
    
    return alerts
```

## Best Practices

### 1. Start Small
✅ Begin with 15-20 evaluation queries  
✅ Expand to 50-100 for production  
✅ Update regularly with new query types  

### 2. Balanced Dataset
✅ Mix of easy and hard questions  
✅ Cover all categories/topics  
✅ Include edge cases  

### 3. Multiple Annotators
✅ 2-3 people for ground truth  
✅ Resolve disagreements  
✅ Measure inter-rater agreement  

### 4. Iterate
✅ Evaluate → Analyze → Improve → Repeat  
✅ Track changes over time  
✅ A/B test improvements  

## Common Pitfalls

### 1. No Ground Truth
❌ Testing without labeled data  
✅ Create evaluation dataset first  

### 2. Overfitting to Eval Set
❌ Optimizing for specific queries  
✅ Keep eval set diverse and representative  

### 3. Only Testing Generation
❌ Ignoring retrieval quality  
✅ Evaluate both layers separately  

### 4. Trusting LLM-as-Judge Blindly
❌ Accepting scores without validation  
✅ Spot-check with human evaluation  

## Tools and Frameworks

### RAGAS (Recommended)
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

results = evaluate(
    dataset=eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
```

### LangChain Evaluation
```python
from langchain.evaluation import load_evaluator

evaluator = load_evaluator("qa", llm=llm)
result = evaluator.evaluate_strings(
    prediction=answer,
    reference=reference,
    input=question
)
```

## References

- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Survey](https://arxiv.org/abs/2312.10997)
- [Information Retrieval Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))

## Conclusion

Evaluation is not optional—it's essential. By systematically measuring retrieval and generation quality, you can:
- Build confidence in your system
- Make data-driven improvements
- Catch issues before users do
- Demonstrate business value

Start with basic metrics (P/R/F1, Groundedness, Completeness), then expand as needed. Remember: **what gets measured gets improved**.
