# Hour 4 Exercises: Evaluation & Metrics

## Exercise 1: Calculate Custom Metrics (Easy)

**Task**: Implement additional retrieval metrics.

**Metrics to implement**:

1. **Average Precision (AP)**
```python
def average_precision(retrieved_ids, relevant_ids):
    """
    AP = average of precision@k for all relevant documents
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

2. **Normalized Discounted Cumulative Gain (NDCG)**
```python
def ndcg_at_k(retrieved_ids, relevant_ids, k):
    """
    NDCG considers ranking order (earlier = better)
    """
    # TODO: Implement NDCG
    # Hint: DCG = sum(rel_i / log2(i+1))
    pass
```

**Test**: Calculate these for all evaluation queries and compare with Precision@K.

---

## Exercise 2: Build an Evaluation Dashboard (Medium)

**Task**: Create a visual dashboard for evaluation metrics.

**Requirements**:
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics_comparison(results):
    """
    Create bar chart comparing different metrics
    """
    metrics = ['precision@1', 'precision@3', 'recall@3', 'f1@3']
    values = [np.mean([r[m] for r in results]) for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.ylabel('Score')
    plt.title('Retrieval Metrics Overview')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_dashboard.png')
    plt.show()
```

**Additional plots**:
- Precision@K vs K (1, 3, 5, 10)
- Per-category performance
- Distribution of MRR scores
- Confusion matrix for retrieval

---

## Exercise 3: Create Failure Analysis (Medium)

**Task**: Identify and analyze queries with poor retrieval performance.

**Implementation**:
```python
def analyze_failures(retrieval_results, threshold=0.5):
    """
    Find queries where Precision@3 < threshold
    """
    failures = []
    
    for result in retrieval_results:
        if result['precision@3'] < threshold:
            failures.append({
                'query': result['question'],
                'precision@3': result['precision@3'],
                'retrieved': result['retrieved'][:3],
                'expected': result['relevant'],
                'category': result.get('category', 'Unknown')
            })
    
    # Group by category
    by_category = defaultdict(list)
    for f in failures:
        by_category[f['category']].append(f)
    
    # Print analysis
    print(f"\nFound {len(failures)} queries with Precision@3 < {threshold}")
    print("\nFailures by category:")
    for category, items in by_category.items():
        print(f"  {category}: {len(items)} failures")
    
    return failures
```

**Analysis questions**:
- Which categories have most failures?
- Are failures due to wrong embeddings or lack of relevant docs?
- Do certain query patterns fail more often?

---

## Exercise 4: Implement RAGAS Framework (Hard)

**Task**: Use the RAGAS library for comprehensive RAG evaluation.

**Setup**:
```bash
pip install ragas
```

**Implementation**:
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from datasets import Dataset

def evaluate_with_ragas(questions, answers, contexts, ground_truths):
    """
    Comprehensive evaluation using RAGAS
    """
    data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    
    dataset = Dataset.from_dict(data)
    
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ]
    )
    
    return result

# Prepare evaluation data
questions = [q['question'] for q in eval_queries[:5]]
ground_truths = [q['reference_answer'] for q in eval_queries[:5]]

# Get RAG system answers
answers = []
contexts = []

for query in questions:
    result = qa_chain({"query": query})
    answers.append(result['result'])
    contexts.append([doc.page_content for doc in result['source_documents']])

# Evaluate
scores = evaluate_with_ragas(questions, answers, contexts, ground_truths)
print(scores)
```

**Metrics**:
- **Faithfulness**: Is answer faithful to context?
- **Answer Relevancy**: Is answer relevant to question?
- **Context Recall**: Does context contain info from ground truth?
- **Context Precision**: Are retrieved contexts relevant?

---

## Exercise 5: Benchmark Different Embedding Models (Medium)

**Task**: Compare retrieval performance across different OpenAI embedding models.

**Models to test**:
```python
models = [
    'text-embedding-3-small',   # 1536 dim, fastest, most cost-effective
    'text-embedding-3-large',   # 3072 dim, highest quality
    'text-embedding-ada-002',   # 1536 dim, legacy model
]
```

**Implementation**:
```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import numpy as np

def benchmark_embeddings(models, eval_queries, documents):
    """
    Compare retrieval performance across OpenAI models
    """
    results = {}
    
    for model_name in models:
        print(f"\nTesting {model_name}...")
        
        # Build vector store with this model
        embeddings = OpenAIEmbeddings(model=model_name)
        store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=model_name.replace('-', '_')
        )
        
        # Evaluate
        metrics = []
        for query in eval_queries:
            docs = store.similarity_search(query['question'], k=3)
            retrieved = [doc.metadata['ticket_id'] for doc in docs]
            p3 = precision_at_k(retrieved, query['relevant_ticket_ids'], 3)
            metrics.append(p3)
        
        results[model_name] = {
            'avg_precision': np.mean(metrics),
            'cost_multiplier': 1.0 if 'small' in model_name else (2.0 if 'large' in model_name else 1.0)
        }
    
    return results

# Run benchmark
results = benchmark_embeddings(models, eval_queries, chunks)

# Display results
for model, metrics in results.items():
    print(f"\n{model}:")
    print(f"  Avg Precision@3: {metrics['avg_precision']:.3f}")
    print(f"  Relative Cost: {metrics['cost_multiplier']}x")
```

**Visualize**:
```python
import matplotlib.pyplot as plt

models_list = list(results.keys())
precisions = [results[m]['avg_precision'] for m in models_list]
costs = [results[m]['cost_multiplier'] for m in models_list]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Precision comparison
ax1.barh(models_list, precisions, color='#1f77b4')
ax1.set_xlabel('Average Precision@3')
ax1.set_title('Retrieval Quality')

# Cost vs Quality
ax2.scatter(costs, precisions, s=200)
for i, model in enumerate(models_list):
    ax2.annotate(model.split('-')[-1], (costs[i], precisions[i]))
ax2.set_xlabel('Relative Cost')
ax2.set_ylabel('Precision@3')
ax2.set_title('Cost vs Quality Trade-off')

plt.tight_layout()
plt.show()
```

**Questions**:
- Is text-embedding-3-large worth the extra cost?
- How much does model choice affect retrieval?
- Which model provides the best cost/performance balance?

---

## Exercise 6: Human-in-the-Loop Evaluation (Medium)

**Task**: Build a tool for manual answer quality assessment.

**Implementation**:
```python
def manual_evaluation_tool(qa_chain, eval_queries):
    """
    Interactive tool for human evaluation
    """
    ratings = []
    
    for i, query_data in enumerate(eval_queries, 1):
        print("\n" + "="*80)
        print(f"Query {i}/{len(eval_queries)}")
        print("="*80)
        
        query = query_data['question']
        print(f"\nQuestion: {query}")
        
        # Get RAG answer
        result = qa_chain({"query": query})
        answer = result['result']
        sources = result['source_documents']
        
        print(f"\nAnswer:\n{answer}")
        print(f"\nSources: {[doc.metadata['ticket_id'] for doc in sources]}")
        
        print(f"\nReference Answer:\n{query_data['reference_answer']}")
        
        # Human rating
        print("\nRate this answer:")
        print("1 - Poor (wrong or irrelevant)")
        print("2 - Fair (partially correct)")
        print("3 - Good (mostly correct)")
        print("4 - Excellent (perfect)")
        
        rating = input("Rating (1-4): ")
        
        hallucination = input("Contains hallucination? (y/n): ").lower() == 'y'
        
        ratings.append({
            'query': query,
            'rating': int(rating),
            'hallucination': hallucination,
            'answer': answer
        })
        
        # Option to stop early
        if input("\nContinue? (y/n): ").lower() != 'y':
            break
    
    # Summary
    avg_rating = np.mean([r['rating'] for r in ratings])
    hallucination_rate = sum(r['hallucination'] for r in ratings) / len(ratings)
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Average rating: {avg_rating:.2f}/4.0")
    print(f"Hallucination rate: {hallucination_rate*100:.1f}%")
    
    return ratings
```

---

## Exercise 7: Cost & Latency Tracking (Medium)

**Task**: Measure operational metrics for production readiness.

**Metrics to track**:
- Query latency
- Embedding generation time
- LLM token usage & cost
- Cache hit rate

**Implementation**:
```python
import time
from functools import wraps

class RAGMetrics:
    def __init__(self):
        self.query_times = []
        self.token_counts = []
        self.cache_hits = 0
        self.cache_misses = 0
    
    def track_latency(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            self.query_times.append(elapsed)
            return result
        return wrapper
    
    def track_tokens(self, prompt, response):
        # Rough estimate: 4 chars ≈ 1 token
        tokens = (len(prompt) + len(response)) / 4
        self.token_counts.append(tokens)
    
    def report(self):
        print("\n" + "="*80)
        print("OPERATIONAL METRICS")
        print("="*80)
        print(f"Total queries: {len(self.query_times)}")
        print(f"Avg latency: {np.mean(self.query_times):.3f}s")
        print(f"p95 latency: {np.percentile(self.query_times, 95):.3f}s")
        print(f"Avg tokens: {np.mean(self.token_counts):.0f}")
        
        # Cost estimate (GPT-3.5-turbo: $0.002/1K tokens)
        total_tokens = sum(self.token_counts)
        cost = (total_tokens / 1000) * 0.002
        print(f"Estimated cost: ${cost:.4f}")

# Usage
metrics = RAGMetrics()

@metrics.track_latency
def query_rag(question):
    return qa_chain({"query": question})

for query in eval_queries:
    result = query_rag(query['question'])
    metrics.track_tokens(query['question'], result['result'])

metrics.report()
```

---

## Exercise 8: Build a Regression Test Suite (Hard)

**Task**: Create automated tests to catch performance regressions.

**Implementation**:
```python
import pytest
import json

class RAGRegressionTests:
    def __init__(self, vector_store, qa_chain):
        self.vector_store = vector_store
        self.qa_chain = qa_chain
        
        # Load baseline metrics
        try:
            with open('baseline_metrics.json', 'r') as f:
                self.baseline = json.load(f)
        except FileNotFoundError:
            self.baseline = None
    
    def test_retrieval_precision(self, eval_queries, min_precision=0.7):
        """Ensure retrieval precision meets threshold"""
        precisions = []
        
        for query in eval_queries:
            docs = self.vector_store.similarity_search(query['question'], k=3)
            retrieved = [doc.metadata['ticket_id'] for doc in docs]
            p = precision_at_k(retrieved, query['relevant_ticket_ids'], 3)
            precisions.append(p)
        
        avg_precision = np.mean(precisions)
        assert avg_precision >= min_precision, \
            f"Precision@3 ({avg_precision:.3f}) below threshold ({min_precision})"
    
    def test_no_regression(self, current_metrics, tolerance=0.05):
        """Ensure no regression from baseline"""
        if not self.baseline:
            print("No baseline found, saving current metrics...")
            with open('baseline_metrics.json', 'w') as f:
                json.dump(current_metrics, f)
            return
        
        for metric in ['precision@3', 'recall@3', 'f1@3']:
            baseline_val = self.baseline[metric]
            current_val = current_metrics[metric]
            
            assert current_val >= baseline_val - tolerance, \
                f"{metric} regressed: {current_val:.3f} < {baseline_val:.3f}"
    
    def test_response_time(self, query, max_latency=5.0):
        """Ensure queries complete within time limit"""
        start = time.time()
        result = self.qa_chain({"query": query})
        elapsed = time.time() - start
        
        assert elapsed <= max_latency, \
            f"Query took {elapsed:.2f}s (max: {max_latency}s)"

# Run tests
tests = RAGRegressionTests(vector_store, qa_chain)
tests.test_retrieval_precision(eval_queries)
tests.test_response_time("How to fix database timeouts?")
```

---

## Exercise 9: Cross-Validation for RAG (Advanced)

**Task**: Implement k-fold cross-validation for RAG evaluation.

**Why**: Get more robust performance estimates.

**Implementation**:
```python
from sklearn.model_selection import KFold

def cross_validate_rag(documents, eval_queries, k=5):
    """
    K-fold cross-validation for RAG
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(documents), 1):
        print(f"\nFold {fold}/{k}")
        
        # Build vector store with training docs
        train_docs = [documents[i] for i in train_idx]
        store = Chroma.from_documents(train_docs, embeddings)
        
        # Test on eval queries
        metrics = []
        for query in eval_queries:
            docs = store.similarity_search(query['question'], k=3)
            retrieved = [doc.metadata['ticket_id'] for doc in docs]
            p = precision_at_k(retrieved, query['relevant_ticket_ids'], 3)
            metrics.append(p)
        
        avg = np.mean(metrics)
        fold_results.append(avg)
        print(f"  Precision@3: {avg:.4f}")
    
    print(f"\nCross-validation Precision@3: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    return fold_results
```

---

## Bonus: Production Monitoring Dashboard

Create a real-time monitoring dashboard:

```python
from flask import Flask, render_template, jsonify
import threading
import time

app = Flask(__name__)

class RAGMonitor:
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'avg_latency': 0,
            'error_rate': 0,
            'cache_hit_rate': 0
        }
    
    def update(self, latency, error=False, cache_hit=False):
        self.metrics['total_queries'] += 1
        # Update rolling averages...

monitor = RAGMonitor()

@app.route('/metrics')
def get_metrics():
    return jsonify(monitor.metrics)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Final Challenge: Achieve 90% Precision@3

Can you tune your RAG system to achieve 90%+ Precision@3?

**Strategies**:
- Fine-tune embeddings on your data
- Implement re-ranking
- Use hybrid search
- Improve prompt engineering
- Add query expansion
- Filter low-quality retrievals

**Document your approach and results!**

---

## Solutions

Full solutions provided after the workshop. Good luck!

---

🎉 **Congratulations on completing the workshop!** You now have the skills to build, deploy, and evaluate production-ready RAG systems.
