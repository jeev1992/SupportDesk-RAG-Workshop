# -*- coding: utf-8 -*-
"""
RAG Evaluation Demo
===================

This demo teaches:
1. Two-layer evaluation approach for RAG systems
2. RETRIEVAL LAYER: Precision, Recall, F1 Score
3. GENERATION LAYER: Groundedness (Faithfulness), Response Completeness
4. Using LLM-as-judge for generation metrics
5. Creating comprehensive evaluation reports
"""

import json
import os
import numpy as np
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("="*80)
print("RAG EVALUATION: TWO-LAYER APPROACH")
print("="*80)
print("\nLayer 1: RETRIEVAL (Precision, Recall, F1)")
print("Layer 2: GENERATION (Groundedness, Completeness)")

# ============================================================================
# PART 1: Setup - Load Data and Build RAG System
# ============================================================================
print("\n" + "="*80)
print("PART 1: Setup RAG System for Evaluation")
print("="*80)

# Load tickets
with open('../../data/synthetic_tickets.json', 'r', encoding='utf-8') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

# Load evaluation queries with ground truth
with open('evaluation_queries.json', 'r', encoding='utf-8') as f:
    eval_queries = json.load(f)
print(f"✓ Loaded {len(eval_queries)} evaluation queries with ground truth")

# Build documents
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

# Initialize embeddings and LLM with timeout
embeddings = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    request_timeout=30
)

# Use direct OpenAI client instead of LangChain ChatOpenAI (fixes connection pooling issue)
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=30.0, max_retries=2)
chat_model = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

print("✓ OpenAI models initialized (30s timeout)")

# Test API connection
print("\nTesting OpenAI API connection...")
try:
    test_response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Say 'OK'"}],
        max_tokens=5
    )
    print("✓ API connection successful")
except Exception as e:
    print(f"✗ API connection failed: {str(e)[:100]}")
    print("\nPlease check:")
    print("  1. OPENAI_API_KEY is set correctly in .env")
    print("  2. You have internet connectivity")
    print("  3. Your API key is valid and has credits")
    exit(1)

# Build vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="eval_supportdesk"
)
print("✓ Vector store built with Chroma")

# Create simple RAG function
def generate_answer(query, k=3):
    """Retrieve context and generate answer"""
    # Retrieve documents
    docs = vector_store.similarity_search(query, k=k)
    
    # Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = f"""You are a technical support assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {query}

Answer (cite ticket IDs):"""
    
    # Generate answer using direct OpenAI client
    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = response.choices[0].message.content
    
    return {'answer': answer, 'source_documents': docs}

print("✓ RAG function ready")

# ============================================================================
# PART 2: RETRIEVAL LAYER EVALUATION
# ============================================================================
print("\n" + "="*80)
print("PART 2: RETRIEVAL LAYER EVALUATION")
print("="*80)
print("\nMetrics: Precision, Recall, F1 Score")
print("Measures: Are we retrieving the right documents?")

def calculate_retrieval_metrics(retrieved_ids, relevant_ids, k=3):
    """
    Calculate Precision, Recall, and F1 at k
    
    Precision@k = (relevant docs in top-k) / k
    Recall@k = (relevant docs in top-k) / (total relevant docs)
    F1@k = harmonic mean of Precision and Recall
    """
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # True positives: relevant docs that were retrieved
    tp = len(retrieved_set & relevant_set)
    
    # Precision: what % of retrieved docs are relevant?
    precision = tp / k if k > 0 else 0
    
    # Recall: what % of relevant docs were retrieved?
    recall = tp / len(relevant_set) if len(relevant_set) > 0 else 0
    
    # F1: harmonic mean balancing precision and recall
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': tp,
        'retrieved_count': k,
        'relevant_count': len(relevant_set)
    }

# Evaluate retrieval for all queries
print("\nEvaluating retrieval for all queries...")
retrieval_results = []

for idx, eval_query in enumerate(eval_queries, 1):
    query = eval_query['question']
    relevant_ids = eval_query['relevant_ticket_ids']
    
    print(f"  [{idx}/{len(eval_queries)}] Processing query...", end='\r')
    
    try:
        # Retrieve documents
        results = vector_store.similarity_search(query, k=5)
        retrieved_ids = [doc.metadata['ticket_id'] for doc in results]
        
        # Calculate metrics at k=3
        metrics = calculate_retrieval_metrics(retrieved_ids, relevant_ids, k=3)
        metrics['query_id'] = eval_query['query_id']
        metrics['question'] = query
        metrics['retrieved'] = retrieved_ids[:3]
        metrics['relevant'] = relevant_ids
        
        retrieval_results.append(metrics)
    except Exception as e:
        print(f"\n  ⚠ Error on query {idx}: {str(e)[:50]}")
        continue

print(f"\n✓ Completed retrieval evaluation for {len(retrieval_results)} queries")

# Aggregate results
print("\n" + "-"*80)
print("RETRIEVAL METRICS (Averaged across all queries)")
print("-"*80)

avg_precision = np.mean([r['precision'] for r in retrieval_results])
avg_recall = np.mean([r['recall'] for r in retrieval_results])
avg_f1 = np.mean([r['f1'] for r in retrieval_results])

print(f"\nPrecision@3: {avg_precision:.4f}")
print(f"  → What % of retrieved documents are actually relevant?")
print(f"  → Target: > 0.80 for production")

print(f"\nRecall@3:    {avg_recall:.4f}")
print(f"  → What % of all relevant documents did we find?")
print(f"  → Target: > 0.70 for production")

print(f"\nF1 Score@3:  {avg_f1:.4f}")
print(f"  → Balanced measure of retrieval quality")
print(f"  → Target: > 0.75 for production")

# Show per-query breakdown
print("\n" + "-"*80)
print("PER-QUERY RETRIEVAL RESULTS (Sample)")
print("-"*80)

for i in range(min(3, len(retrieval_results))):
    result = retrieval_results[i]
    print(f"\n{result['query_id']}: {result['question'][:60]}...")
    print(f"  Relevant:  {result['relevant']}")
    print(f"  Retrieved: {result['retrieved']}")
    print(f"  Precision: {result['precision']:.2f} | Recall: {result['recall']:.2f} | F1: {result['f1']:.2f}")

# ============================================================================
# PART 3: GENERATION LAYER EVALUATION
# ============================================================================
print("\n" + "="*80)
print("PART 3: GENERATION LAYER EVALUATION")
print("="*80)
print("\nMetrics: Groundedness, Response Completeness")
print("Measures: Is the generated answer faithful and complete?")

# Initialize OpenAI client for LLM-as-judge (reuse the same client)
client = openai_client

def evaluate_groundedness(answer, context_docs):
    """
    Groundedness (Faithfulness): Is the answer supported by the retrieved context?
    Uses LLM-as-judge to check if answer contains hallucinations
    
    Returns: score 0.0-1.0 (higher = more grounded)
    """
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    prompt = f"""Evaluate if the ANSWER is fully supported by the CONTEXT. Check for hallucinations or unsupported claims.

CONTEXT:
{context}

ANSWER:
{answer}

Rate the groundedness from 0 to 10:
- 10: Every claim in the answer is directly supported by the context
- 7-9: Most claims supported, minor unsupported details
- 4-6: Some claims supported, some speculation or external knowledge
- 1-3: Little support from context, mostly hallucinated
- 0: Completely hallucinated, no relation to context

Provide:
1. Score (0-10)
2. Reasoning (one sentence)

Format:
Score: X
Reasoning: <explanation>"""

    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30
        )
        
        output = response.choices[0].message.content
        
        # Parse score
        try:
            score_line = [line for line in output.split('\n') if line.startswith('Score:')][0]
            score = float(score_line.split(':')[1].strip()) / 10.0  # Normalize to 0-1
        except:
            score = 0.5  # Default if parsing fails
        
        return {
            'score': score,
            'verdict': 'GROUNDED' if score >= 0.7 else 'PARTIAL' if score >= 0.4 else 'HALLUCINATED',
            'explanation': output
        }
    except Exception as e:
        print(f"\n    ⚠ Error evaluating groundedness: {str(e)[:50]}")
        return {'score': 0.5, 'verdict': 'ERROR', 'explanation': str(e)}

def evaluate_completeness(question, answer, reference_answer=None):
    """
    Response Completeness: Does the answer fully address the question?
    Uses LLM-as-judge to check if all aspects of the question are answered
    
    Returns: score 0.0-1.0 (higher = more complete)
    """
    if reference_answer:
        prompt = f"""Evaluate if the ANSWER fully addresses the QUESTION compared to the REFERENCE ANSWER.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

GENERATED ANSWER:
{answer}

Rate the completeness from 0 to 10:
- 10: Fully answers all aspects of the question (as good as reference)
- 7-9: Answers most parts, minor gaps
- 4-6: Partial answer, missing key information
- 1-3: Minimal answer, major gaps
- 0: Does not answer the question

Provide:
1. Score (0-10)
2. Reasoning (one sentence)

Format:
Score: X
Reasoning: <explanation>"""
    else:
        prompt = f"""Evaluate if the ANSWER fully addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Rate the completeness from 0 to 10:
- 10: Fully answers all aspects of the question
- 7-9: Answers most parts, minor gaps
- 4-6: Partial answer, missing key information
- 1-3: Minimal answer, major gaps
- 0: Does not answer the question

Provide:
1. Score (0-10)
2. Reasoning (one sentence)

Format:
Score: X
Reasoning: <explanation>"""

    try:
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=30
        )
        
        output = response.choices[0].message.content
        
        # Parse score
        try:
            score_line = [line for line in output.split('\n') if line.startswith('Score:')][0]
            score = float(score_line.split(':')[1].strip()) / 10.0  # Normalize to 0-1
        except:
            score = 0.5  # Default if parsing fails
        
        return {
            'score': score,
            'verdict': 'COMPLETE' if score >= 0.7 else 'PARTIAL' if score >= 0.4 else 'INCOMPLETE',
            'explanation': output
        }
    except Exception as e:
        print(f"\n    ⚠ Error evaluating completeness: {str(e)[:50]}")
        return {'score': 0.5, 'verdict': 'ERROR', 'explanation': str(e)}

# Evaluate generation for sample queries
print("\nEvaluating generation quality (sample queries)...")
print("Note: This uses LLM-as-judge (GPT-4o-mini) to evaluate quality")
print("This may take 30-60 seconds per query...\n")

generation_results = []

for idx, eval_query in enumerate(eval_queries[:2], 1):  # Evaluate first 2 to save time
    query = eval_query['question']
    reference = eval_query.get('reference_answer', None)
    
    print("-"*80)
    print(f"\n[{idx}/2] Query: {query}")
    
    try:
        # Generate answer using RAG
        print("  → Generating answer...")
        result = generate_answer(query)
        answer = result['answer']
        source_docs = result['source_documents']
        
        print(f"\nGenerated Answer:\n{answer}")
        
        # Evaluate groundedness
        print("\n  → Evaluating Groundedness...")
        groundedness = evaluate_groundedness(answer, source_docs)
        print(f"    Score: {groundedness['score']:.2f} - {groundedness['verdict']}")
        
        # Evaluate completeness
        print("\n  → Evaluating Completeness...")
        completeness = evaluate_completeness(query, answer, reference)
        print(f"    Score: {completeness['score']:.2f} - {completeness['verdict']}")
        
        generation_results.append({
            'query_id': eval_query['query_id'],
            'question': query,
            'answer': answer,
            'groundedness_score': groundedness['score'],
            'completeness_score': completeness['score']
        })
    except Exception as e:
        print(f"\n  ✗ Error processing query: {str(e)[:80]}")
        continue

print(f"\n✓ Completed generation evaluation for {len(generation_results)} queries")

# Aggregate generation metrics
print("\n" + "="*80)
print("GENERATION METRICS (Averaged)")
print("="*80)

avg_groundedness = np.mean([r['groundedness_score'] for r in generation_results])
avg_completeness = np.mean([r['completeness_score'] for r in generation_results])

print(f"\nGroundedness:  {avg_groundedness:.4f}")
print(f"  → Are answers supported by retrieved context?")
print(f"  → Target: > 0.80 (minimize hallucinations)")

print(f"\nCompleteness:  {avg_completeness:.4f}")
print(f"  → Do answers fully address the questions?")
print(f"  → Target: > 0.75 (comprehensive responses)")

# ============================================================================
# PART 4: Comprehensive Evaluation Report
# ============================================================================
print("\n" + "="*80)
print("PART 4: COMPREHENSIVE EVALUATION REPORT")
print("="*80)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM EVALUATION                    │
├─────────────────────────────────────────────────────────────┤
│ Dataset: {len(eval_queries)} evaluation queries                           │
├─────────────────────────────────────────────────────────────┤
│ RETRIEVAL LAYER                                             │
│   • Precision@3:  {avg_precision:.4f}  {'✓' if avg_precision >= 0.80 else '⚠' if avg_precision >= 0.70 else '✗'}                              │
│   • Recall@3:     {avg_recall:.4f}  {'✓' if avg_recall >= 0.70 else '⚠' if avg_recall >= 0.60 else '✗'}                              │
│   • F1 Score@3:   {avg_f1:.4f}  {'✓' if avg_f1 >= 0.75 else '⚠' if avg_f1 >= 0.65 else '✗'}                              │
├─────────────────────────────────────────────────────────────┤
│ GENERATION LAYER                                            │
│   • Groundedness: {avg_groundedness:.4f}  {'✓' if avg_groundedness >= 0.80 else '⚠' if avg_groundedness >= 0.70 else '✗'}                              │
│   • Completeness: {avg_completeness:.4f}  {'✓' if avg_completeness >= 0.75 else '⚠' if avg_completeness >= 0.65 else '✗'}                              │
└─────────────────────────────────────────────────────────────┘

INTERPRETATION:
""")

# Provide diagnostic feedback
if avg_precision < 0.70:
    print("⚠ Low Precision → Too much noise in retrieval")
    print("  Fix: Improve chunking, use metadata filtering, or increase similarity threshold")

if avg_recall < 0.60:
    print("⚠ Low Recall → Missing relevant documents")
    print("  Fix: Increase k, improve embedding model, or use hybrid retrieval")

if avg_f1 < 0.65:
    print("⚠ Low F1 → Overall retrieval quality needs improvement")
    print("  Fix: Balance precision and recall by tuning retrieval parameters")

if avg_groundedness < 0.70:
    print("⚠ Low Groundedness → System is hallucinating")
    print("  Fix: Strengthen prompt instructions, use stricter grounding, or improve context quality")

if avg_completeness < 0.65:
    print("⚠ Low Completeness → Answers are incomplete")
    print("  Fix: Retrieve more context (increase k), improve prompt, or use better LLM")

if all([avg_precision >= 0.80, avg_recall >= 0.70, avg_f1 >= 0.75, 
        avg_groundedness >= 0.80, avg_completeness >= 0.75]):
    print("✓ All metrics in target range - System is production-ready!")

# ============================================================================
# PART 5: Comparing Different RAG Configurations
# ============================================================================
print("\n" + "="*80)
print("PART 5: A/B Testing Framework")
print("="*80)

print("\nExample: Compare retrieval with k=3 vs k=5")

def compare_configurations(queries, k_values):
    """Compare different retrieval configurations"""
    results = {}
    
    for k in k_values:
        print(f"  Testing k={k}...")
        retrievals = []
        for query in queries[:3]:  # Test subset
            try:
                docs = vector_store.similarity_search(query['question'], k=k)
                retrieved_ids = [doc.metadata['ticket_id'] for doc in docs]
                metrics = calculate_retrieval_metrics(retrieved_ids, query['relevant_ticket_ids'], k=k)
                retrievals.append(metrics)
            except Exception as e:
                print(f"\n    ⚠ Error: {str(e)[:50]}")
                continue
        
        if retrievals:
            results[f'k={k}'] = {
                'precision': np.mean([r['precision'] for r in retrievals]),
                'recall': np.mean([r['recall'] for r in retrievals]),
                'f1': np.mean([r['f1'] for r in retrievals])
            }
    
    return results

comparison = compare_configurations(eval_queries, [3, 5])

print("\nConfiguration Comparison:")
print(f"{'Config':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 46)
for config, metrics in comparison.items():
    print(f"{config:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)

print("""
Key Takeaways:
──────────────
1. Two-Layer Evaluation is Essential
   → Separately measure retrieval and generation quality
   
2. Retrieval Metrics (Precision, Recall, F1)
   → Diagnose if you're finding the right documents
   
3. Generation Metrics (Groundedness, Completeness)
   → Ensure faithful and comprehensive answers
   
4. Use LLM-as-Judge for Generation Metrics
   → Automated evaluation using GPT-4/Claude
   
5. Always Create Evaluation Datasets
   → Ground truth enables systematic improvement
   
6. A/B Test Different Configurations
   → Measure impact of changes quantitatively

Production Targets:
───────────────────
✓ Precision@3 > 0.80
✓ Recall@3 > 0.70
✓ F1@3 > 0.75
✓ Groundedness > 0.80
✓ Completeness > 0.75
""")
