"""
Hour 4: RAG Evaluation & Metrics Demo
======================================

This demo teaches:
1. Retrieval metrics: Precision@K, Recall@K, F1, MRR
2. Generation metrics: ROUGE-L, BLEU
3. Creating evaluation datasets
4. Systematic testing and improvement
"""

import json
import numpy as np
from collections import defaultdict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

print("="*80)
print("HOUR 4: RAG EVALUATION & METRICS")
print("="*80)

# ============================================================================
# PART 1: Load Data and Build System
# ============================================================================
print("\n" + "="*80)
print("PART 1: Setup Evaluation Environment")
print("="*80)

# Load tickets
with open('../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

# Load evaluation queries
with open('evaluation_queries.json', 'r') as f:
    eval_queries = json.load(f)
print(f"✓ Loaded {len(eval_queries)} evaluation queries")

# Build vector store
documents = []
for ticket in tickets:
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
    """.strip()
    
    doc = Document(
        page_content=content,
        metadata={'ticket_id': ticket['ticket_id']}
    )
    documents.append(doc)

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="eval_store"
)
print("✓ Vector store built")

# ============================================================================
# PART 2: Retrieval Metrics
# ============================================================================
print("\n" + "="*80)
print("PART 2: Evaluating Retrieval Quality")
print("="*80)

def precision_at_k(retrieved_ids, relevant_ids, k):
    """
    Precision@K = (# relevant retrieved in top-k) / k
    """
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / k

def recall_at_k(retrieved_ids, relevant_ids, k):
    """
    Recall@K = (# relevant retrieved in top-k) / (total relevant)
    """
    retrieved_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(retrieved_k & relevant) / len(relevant) if relevant else 0

def f1_at_k(retrieved_ids, relevant_ids, k):
    """
    F1@K = harmonic mean of Precision@K and Recall@K
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def mean_reciprocal_rank(retrieved_ids, relevant_ids):
    """
    MRR = 1 / rank of first relevant document
    """
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

# Evaluate retrieval for all queries
retrieval_results = []

for eval_query in eval_queries:
    query = eval_query['question']
    relevant_ids = eval_query['relevant_ticket_ids']
    
    # Retrieve documents
    results = vector_store.similarity_search(query, k=5)
    retrieved_ids = [doc.metadata['ticket_id'] for doc in results]
    
    # Compute metrics
    metrics = {
        'query_id': eval_query['query_id'],
        'question': query,
        'retrieved': retrieved_ids,
        'relevant': relevant_ids,
        'precision@1': precision_at_k(retrieved_ids, relevant_ids, 1),
        'precision@3': precision_at_k(retrieved_ids, relevant_ids, 3),
        'precision@5': precision_at_k(retrieved_ids, relevant_ids, 5),
        'recall@1': recall_at_k(retrieved_ids, relevant_ids, 1),
        'recall@3': recall_at_k(retrieved_ids, relevant_ids, 3),
        'recall@5': recall_at_k(retrieved_ids, relevant_ids, 5),
        'f1@3': f1_at_k(retrieved_ids, relevant_ids, 3),
        'mrr': mean_reciprocal_rank(retrieved_ids, relevant_ids)
    }
    retrieval_results.append(metrics)

# Aggregate metrics
print("\n" + "-"*80)
print("RETRIEVAL METRICS (Averaged across all queries)")
print("-"*80)

avg_metrics = {
    'Precision@1': np.mean([r['precision@1'] for r in retrieval_results]),
    'Precision@3': np.mean([r['precision@3'] for r in retrieval_results]),
    'Precision@5': np.mean([r['precision@5'] for r in retrieval_results]),
    'Recall@1': np.mean([r['recall@1'] for r in retrieval_results]),
    'Recall@3': np.mean([r['recall@3'] for r in retrieval_results]),
    'Recall@5': np.mean([r['recall@5'] for r in retrieval_results]),
    'F1@3': np.mean([r['f1@3'] for r in retrieval_results]),
    'MRR': np.mean([r['mrr'] for r in retrieval_results])
}

for metric, value in avg_metrics.items():
    print(f"{metric:15} : {value:.4f}")

# Show detailed results for a few queries
print("\n" + "-"*80)
print("DETAILED RESULTS (Sample Queries)")
print("-"*80)

for i in range(min(3, len(retrieval_results))):
    result = retrieval_results[i]
    print(f"\n{result['query_id']}: {result['question']}")
    print(f"  Relevant: {result['relevant']}")
    print(f"  Retrieved: {result['retrieved'][:3]}")
    print(f"  Precision@3: {result['precision@3']:.2f}")
    print(f"  Recall@3: {result['recall@3']:.2f}")
    print(f"  F1@3: {result['f1@3']:.2f}")

# ============================================================================
# PART 3: Generation Metrics (ROUGE & BLEU)
# ============================================================================
print("\n" + "="*80)
print("PART 3: Evaluating Generated Answers")
print("="*80)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def evaluate_answer(generated, reference):
    """
    Evaluate generated answer against reference answer
    """
    # ROUGE scores
    rouge_scores = scorer.score(reference, generated)
    
    # BLEU score
    reference_tokens = reference.lower().split()
    generated_tokens = generated.lower().split()
    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
    
    return {
        'rouge1_f': rouge_scores['rouge1'].fmeasure,
        'rouge2_f': rouge_scores['rouge2'].fmeasure,
        'rougeL_f': rouge_scores['rougeL'].fmeasure,
        'bleu': bleu
    }

# Example: Simulate generated answers (in real scenario, these come from LLM)
print("\nExample Answer Evaluation:")
print("-"*80)

example_reference = eval_queries[0]['reference_answer']
print(f"\nReference Answer:\n{example_reference}")

# Good answer (high overlap)
good_answer = "Clear all active sessions and implement automatic session cleanup when users change passwords to prevent stale tokens from causing authentication failures."
print(f"\nGenerated Answer (Good):\n{good_answer}")

good_scores = evaluate_answer(good_answer, example_reference)
print(f"\nScores:")
print(f"  ROUGE-1: {good_scores['rouge1_f']:.4f}")
print(f"  ROUGE-2: {good_scores['rouge2_f']:.4f}")
print(f"  ROUGE-L: {good_scores['rougeL_f']:.4f}")
print(f"  BLEU: {good_scores['bleu']:.4f}")

# Bad answer (low overlap)
bad_answer = "Try restarting the server and checking the logs for errors."
print(f"\nGenerated Answer (Bad):\n{bad_answer}")

bad_scores = evaluate_answer(bad_answer, example_reference)
print(f"\nScores:")
print(f"  ROUGE-1: {bad_scores['rouge1_f']:.4f}")
print(f"  ROUGE-2: {bad_scores['rouge2_f']:.4f}")
print(f"  ROUGE-L: {bad_scores['rougeL_f']:.4f}")
print(f"  BLEU: {bad_scores['bleu']:.4f}")

# ============================================================================
# PART 4: Semantic Similarity Evaluation
# ============================================================================
print("\n" + "="*80)
print("PART 4: Semantic Similarity Metrics")
print("="*80)

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(text1, text2):
    """
    Compute cosine similarity between two texts
    """
    emb1 = model.encode([text1])
    emb2 = model.encode([text2])
    return cosine_similarity(emb1, emb2)[0][0]

print("\nSemantic Similarity Examples:")
print("-"*80)

# Good paraphrase
ref = "Clear all sessions and force re-authentication"
para = "Force users to log in again and remove active sessions"
sim = semantic_similarity(ref, para)
print(f"\nReference: {ref}")
print(f"Paraphrase: {para}")
print(f"Similarity: {sim:.4f} ✓ (Good paraphrase)")

# Different meaning
diff = "Increase database connection pool size"
sim2 = semantic_similarity(ref, diff)
print(f"\nDifferent: {diff}")
print(f"Similarity: {sim2:.4f} ✗ (Different topic)")

# ============================================================================
# PART 5: Hallucination Detection
# ============================================================================
print("\n" + "="*80)
print("PART 5: Hallucination Detection")
print("="*80)

def detect_hallucination(answer, source_documents, threshold=0.5):
    """
    Check if answer is grounded in source documents
    """
    # Combine all source content
    source_text = " ".join([doc.page_content for doc in source_documents])
    
    # Compute semantic similarity
    similarity = semantic_similarity(answer, source_text)
    
    # Check grounding
    is_grounded = similarity >= threshold
    
    return {
        'is_grounded': is_grounded,
        'similarity': similarity,
        'verdict': 'GROUNDED' if is_grounded else 'POSSIBLE HALLUCINATION'
    }

print("\nHallucination Detection Examples:")
print("-"*80)

query = "How to fix authentication failures?"
docs = vector_store.similarity_search(query, k=3)

# Grounded answer
grounded = "Authentication failures after password reset are caused by stale session tokens. Clear all active sessions to fix this issue."
result1 = detect_hallucination(grounded, docs)
print(f"\nAnswer: {grounded}")
print(f"Verdict: {result1['verdict']} (similarity: {result1['similarity']:.4f})")

# Hallucinated answer
hallucinated = "You need to upgrade to Python 3.11 and install the latest security patches for Windows 11."
result2 = detect_hallucination(hallucinated, docs)
print(f"\nAnswer: {hallucinated}")
print(f"Verdict: {result2['verdict']} (similarity: {result2['similarity']:.4f})")

# ============================================================================
# PART 6: Create Evaluation Report
# ============================================================================
print("\n" + "="*80)
print("PART 6: Comprehensive Evaluation Report")
print("="*80)

def create_evaluation_report(retrieval_metrics, generation_metrics=None):
    """
    Generate comprehensive evaluation report
    """
    report = {
        'retrieval': {
            'precision@1': np.mean([m['precision@1'] for m in retrieval_metrics]),
            'precision@3': np.mean([m['precision@3'] for m in retrieval_metrics]),
            'precision@5': np.mean([m['precision@5'] for m in retrieval_metrics]),
            'recall@3': np.mean([m['recall@3'] for m in retrieval_metrics]),
            'recall@5': np.mean([m['recall@5'] for m in retrieval_metrics]),
            'f1@3': np.mean([m['f1@3'] for m in retrieval_metrics]),
            'mrr': np.mean([m['mrr'] for m in retrieval_metrics]),
        },
        'total_queries': len(retrieval_metrics),
        'perfect_retrievals': sum(1 for m in retrieval_metrics if m['precision@1'] == 1.0)
    }
    
    return report

report = create_evaluation_report(retrieval_results)

print("\n" + "="*80)
print("EVALUATION REPORT")
print("="*80)
print(f"\nDataset: {report['total_queries']} evaluation queries")
print(f"Perfect top-1 retrievals: {report['perfect_retrievals']}/{report['total_queries']} ({report['perfect_retrievals']/report['total_queries']*100:.1f}%)")

print("\n--- Retrieval Metrics ---")
for metric, value in report['retrieval'].items():
    status = "✓" if value >= 0.7 else "⚠" if value >= 0.5 else "✗"
    print(f"{status} {metric.upper():15} : {value:.4f}")

# ============================================================================
# PART 7: A/B Testing Framework
# ============================================================================
print("\n" + "="*80)
print("PART 7: A/B Testing Different Configurations")
print("="*80)

def compare_configurations(queries, config_a, config_b):
    """
    Compare two RAG configurations
    """
    print(f"\nComparing:")
    print(f"  Config A: {config_a['name']}")
    print(f"  Config B: {config_b['name']}")
    
    results = {'A': [], 'B': []}
    
    # Test Config A
    for query in queries[:3]:  # Test on subset
        docs_a = vector_store.similarity_search(
            query['question'],
            k=config_a['k']
        )
        retrieved_a = [doc.metadata['ticket_id'] for doc in docs_a]
        p_a = precision_at_k(retrieved_a, query['relevant_ticket_ids'], 3)
        results['A'].append(p_a)
        
        # For demo, Config B just uses different k
        docs_b = vector_store.similarity_search(
            query['question'],
            k=config_b['k']
        )
        retrieved_b = [doc.metadata['ticket_id'] for doc in docs_b]
        p_b = precision_at_k(retrieved_b, query['relevant_ticket_ids'], 3)
        results['B'].append(p_b)
    
    avg_a = np.mean(results['A'])
    avg_b = np.mean(results['B'])
    
    print(f"\nResults:")
    print(f"  Config A Precision@3: {avg_a:.4f}")
    print(f"  Config B Precision@3: {avg_b:.4f}")
    
    winner = "A" if avg_a > avg_b else "B" if avg_b > avg_a else "Tie"
    print(f"\n  Winner: Config {winner}")

config_a = {'name': 'k=3', 'k': 3}
config_b = {'name': 'k=5', 'k': 5}

compare_configurations(eval_queries, config_a, config_b)

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. Retrieval: Measure Precision@K, Recall@K, F1, MRR")
print("2. Generation: Use ROUGE, BLEU, semantic similarity")
print("3. Always check for hallucinations")
print("4. Create evaluation datasets with ground truth")
print("5. A/B test different configurations systematically")
print("6. Aim for Precision@3 > 0.8 for production systems")
print("\n🎉 Congratulations! You've built and evaluated a complete RAG system!")
