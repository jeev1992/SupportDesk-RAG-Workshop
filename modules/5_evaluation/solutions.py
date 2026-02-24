# -*- coding: utf-8 -*-
"""
Module 5 Solutions: Evaluation & Metrics
========================================

Solutions for all exercises in exercises.md
"""

import json
import os
import time
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

# ============================================================================
# Setup: Load data and build RAG system
# ============================================================================
print("Loading data...")
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)

with open('evaluation_queries.json', 'r', encoding='utf-8') as f:
    eval_queries = json.load(f)

documents = []
for ticket in tickets:
    content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}"""
    
    doc = Document(
        page_content=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} documents, {len(eval_queries)} eval queries")

# Initialize embeddings and OpenAI client
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
chat_model = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

# Build vector store
print("Building vector store...")
vector_store = FAISS.from_documents(documents, embeddings)
print("✓ Vector store ready")

# RAG function
def generate_answer(query, k=3):
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Answer the question using ONLY the provided context. Cite ticket IDs.

Context:
{context}

Question: {query}

Answer:"""
    
    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return {'answer': response.choices[0].message.content, 'source_documents': docs}


# ============================================================================
# Exercise 1: Calculate Precision, Recall, F1 (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 1: Calculate Precision, Recall, F1")
print("=" * 80)

def calculate_metrics(retrieved_ids, relevant_ids, k=3):
    """
    Calculate Precision, Recall, F1 at k
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # True positives: relevant docs that were retrieved
    tp = len(retrieved_set & relevant_set)
    
    # Precision: what % of retrieved docs are relevant?
    precision = tp / k if k > 0 else 0
    
    # Recall: what % of relevant docs were retrieved?
    recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0
    
    # F1: harmonic mean
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

# Test on first few queries
for i, query in enumerate(eval_queries[:3], 1):
    docs = vector_store.similarity_search(query['question'], k=3)
    retrieved = [doc.metadata['ticket_id'] for doc in docs]
    relevant = query['relevant_ticket_ids']
    
    metrics = calculate_metrics(retrieved, relevant)
    
    print(f"\nQuery {i}: {query['question'][:50]}...")
    print(f"  Retrieved: {retrieved}")
    print(f"  Relevant:  {relevant}")
    print(f"  Precision: {metrics['precision']:.2f}, Recall: {metrics['recall']:.2f}, F1: {metrics['f1']:.2f}")


# ============================================================================
# Exercise 2: Evaluate All Queries (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Evaluate All Queries")
print("=" * 80)

all_metrics = []
for query in eval_queries:
    docs = vector_store.similarity_search(query['question'], k=3)
    retrieved = [doc.metadata['ticket_id'] for doc in docs]
    metrics = calculate_metrics(retrieved, query['relevant_ticket_ids'])
    all_metrics.append(metrics)

avg_precision = np.mean([m['precision'] for m in all_metrics])
avg_recall = np.mean([m['recall'] for m in all_metrics])
avg_f1 = np.mean([m['f1'] for m in all_metrics])

print(f"\nAveraged over {len(eval_queries)} queries:")
print(f"  Precision@3: {avg_precision:.4f}")
print(f"  Recall@3:    {avg_recall:.4f}")
print(f"  F1@3:        {avg_f1:.4f}")


# ============================================================================
# Exercise 3: Compare Different k Values (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Compare Different k Values")
print("=" * 80)

for k in [1, 3, 5, 10]:
    metrics = []
    for query in eval_queries:
        docs = vector_store.similarity_search(query['question'], k=k)
        retrieved = [doc.metadata['ticket_id'] for doc in docs]
        m = calculate_metrics(retrieved, query['relevant_ticket_ids'], k=k)
        metrics.append(m)
    
    print(f"\nk={k}:")
    print(f"  Precision@{k}: {np.mean([m['precision'] for m in metrics]):.4f}")
    print(f"  Recall@{k}:    {np.mean([m['recall'] for m in metrics]):.4f}")


# ============================================================================
# Exercise 4: LLM-as-Judge for Groundedness (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: LLM-as-Judge for Groundedness")
print("=" * 80)

def evaluate_groundedness(answer, context_docs):
    """Check if answer is supported by context (no hallucinations)"""
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Rate groundedness (0-10):
- 10: Every claim is supported by context
- 5: Some claims unsupported
- 0: Completely hallucinated

Format: Score: X / Reason: <explanation>"""

    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    output = response.choices[0].message.content
    try:
        score = float(output.split('Score:')[1].split('/')[0].strip()) / 10
    except:
        score = 0.5
    
    return {'score': score, 'output': output}

# Test on first 2 queries
print("\nEvaluating groundedness (LLM-as-judge)...")
for i, query in enumerate(eval_queries[:2], 1):
    print(f"\n--- Query {i}: {query['question'][:50]}... ---")
    
    result = generate_answer(query['question'])
    groundedness = evaluate_groundedness(result['answer'], result['source_documents'])
    
    print(f"Answer: {result['answer'][:150]}...")
    print(f"Groundedness score: {groundedness['score']:.2f}")
    print(f"Verdict: {'GROUNDED' if groundedness['score'] >= 0.7 else 'HALLUCINATED'}")


# ============================================================================
# Exercise 5: LLM-as-Judge for Completeness (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: LLM-as-Judge for Completeness")
print("=" * 80)

def evaluate_completeness(question, answer, reference_answer=None):
    """Check if answer fully addresses the question"""
    ref_section = f"\nREFERENCE ANSWER:\n{reference_answer}" if reference_answer else ""
    
    prompt = f"""Evaluate if the ANSWER fully addresses the QUESTION.
{ref_section}

QUESTION:
{question}

ANSWER:
{answer}

Rate completeness (0-10):
- 10: Fully answers all aspects
- 5: Partial answer
- 0: Does not answer

Format: Score: X / Reason: <explanation>"""

    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    output = response.choices[0].message.content
    try:
        score = float(output.split('Score:')[1].split('/')[0].strip()) / 10
    except:
        score = 0.5
    
    return {'score': score, 'output': output}

# Test
for i, query in enumerate(eval_queries[:2], 1):
    print(f"\n--- Query {i} ---")
    result = generate_answer(query['question'])
    completeness = evaluate_completeness(
        query['question'], 
        result['answer'],
        query.get('reference_answer')
    )
    print(f"Completeness score: {completeness['score']:.2f}")


# ============================================================================
# Exercise 6: Failure Analysis (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Failure Analysis")
print("=" * 80)

def analyze_failures(eval_queries, vector_store, threshold=0.5):
    """Find and analyze queries with poor retrieval"""
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
    
    # Group by category
    by_category = defaultdict(list)
    for f in failures:
        by_category[f['category']].append(f)
    
    print(f"\nFound {len(failures)} queries with Precision < {threshold}")
    print("\nFailures by category:")
    for category, items in by_category.items():
        print(f"  {category}: {len(items)} failures")
    
    return failures

failures = analyze_failures(eval_queries, vector_store)

if failures:
    print("\n\nExample failure:")
    f = failures[0]
    print(f"  Query: {f['query']}")
    print(f"  Expected: {f['expected']}")
    print(f"  Retrieved: {f['retrieved']}")


# ============================================================================
# Exercise 7: Track Latency and Cost (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Track Latency and Cost")
print("=" * 80)

class RAGMetrics:
    def __init__(self):
        self.query_times = []
        self.token_counts = []
    
    def track_query(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        self.query_times.append(elapsed)
        
        # Estimate tokens (rough: 4 chars ≈ 1 token)
        tokens = len(str(result.get('answer', ''))) / 4
        self.token_counts.append(tokens)
        
        return result
    
    def report(self):
        print(f"\nTotal queries: {len(self.query_times)}")
        print(f"Avg latency: {np.mean(self.query_times):.3f}s")
        print(f"p95 latency: {np.percentile(self.query_times, 95):.3f}s")
        
        total_tokens = sum(self.token_counts)
        cost = (total_tokens / 1000) * 0.002  # GPT-4o-mini pricing estimate
        print(f"Est. tokens: {total_tokens:.0f}")
        print(f"Est. cost: ${cost:.4f}")

metrics = RAGMetrics()

# Run a few queries with tracking
for query in eval_queries[:5]:
    metrics.track_query(generate_answer, query['question'])

metrics.report()


# ============================================================================
# Bonus: Comprehensive Evaluation Report
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Comprehensive Evaluation Report")
print("=" * 80)

# Collect all metrics
retrieval_metrics = []
generation_metrics = []

print("\nRunning comprehensive evaluation (this may take a minute)...")

for i, query in enumerate(eval_queries[:5], 1):
    print(f"  Evaluating query {i}/5...", end='\r')
    
    # Retrieval metrics
    docs = vector_store.similarity_search(query['question'], k=3)
    retrieved = [doc.metadata['ticket_id'] for doc in docs]
    r_metrics = calculate_metrics(retrieved, query['relevant_ticket_ids'])
    retrieval_metrics.append(r_metrics)
    
    # Generation metrics
    result = generate_answer(query['question'])
    groundedness = evaluate_groundedness(result['answer'], result['source_documents'])
    completeness = evaluate_completeness(query['question'], result['answer'])
    generation_metrics.append({
        'groundedness': groundedness['score'],
        'completeness': completeness['score']
    })

# Print report
print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM EVALUATION                    │
├─────────────────────────────────────────────────────────────┤
│ RETRIEVAL LAYER                                             │
│   • Precision@3:  {np.mean([m['precision'] for m in retrieval_metrics]):.4f}                                       │
│   • Recall@3:     {np.mean([m['recall'] for m in retrieval_metrics]):.4f}                                       │
│   • F1 Score@3:   {np.mean([m['f1'] for m in retrieval_metrics]):.4f}                                       │
├─────────────────────────────────────────────────────────────┤
│ GENERATION LAYER                                            │
│   • Groundedness: {np.mean([m['groundedness'] for m in generation_metrics]):.4f}                                       │
│   • Completeness: {np.mean([m['completeness'] for m in generation_metrics]):.4f}                                       │
└─────────────────────────────────────────────────────────────┘
""")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1. Two-Layer Evaluation:
   - Retrieval: Precision, Recall, F1
   - Generation: Groundedness, Completeness

2. Metrics Meaning:
   - Precision: % of retrieved docs that are relevant
   - Recall: % of relevant docs that we found
   - Groundedness: Is the answer faithful to context?
   - Completeness: Does the answer fully address the question?

3. LLM-as-Judge:
   - Use GPT-4 to evaluate generation quality
   - More scalable than human evaluation

4. Production Targets:
   ✓ Precision@3 > 0.80
   ✓ Recall@3 > 0.70
   ✓ Groundedness > 0.80
   ✓ Completeness > 0.75

5. Failure Analysis:
   - Identify weak areas for targeted improvement
   - Group by category to find patterns

Congratulations on completing the workshop!
""")
