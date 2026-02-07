# -*- coding: utf-8 -*-
"""
Hour 1 Solutions: Embeddings & Similarity Search
=================================================

Solutions for all exercises in exercises.md
Run each section independently or the whole file.
"""

import json
import time
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

# Load data (used by multiple exercises)
print("Loading data...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

texts = [f"{t['title']}. {t['description']}" for t in tickets]
response = client.embeddings.create(input=texts, model=model)
embeddings = np.array([data.embedding for data in response.data])
print(f"✓ Loaded {len(tickets)} tickets\n")


# ============================================================================
# Exercise 1: Change the Search Query (Easy)
# ============================================================================
print("=" * 80)
print("EXERCISE 1: Change the Search Query")
print("=" * 80)

# Solution: Just change the query string
query = "Database is running very slowly"  # Changed from original

query_response = client.embeddings.create(input=[query], model=model)
query_embedding = np.array([query_response.data[0].embedding])
similarities = cosine_similarity(query_embedding, embeddings)[0]

top_indices = np.argsort(similarities)[::-1][:5]

print(f"\nQuery: '{query}'")
print("-" * 50)
for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]
    print(f"#{rank} [{score:.4f}] {ticket['title']}")
    print(f"    Category: {ticket['category']}")


# ============================================================================
# Exercise 2: Adjust the Number of Results (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust the Number of Results")
print("=" * 80)

# Solution: Change top_k from 5 to 10
top_k = 10  # Changed from 5

query = "Users can't login after changing password"
query_response = client.embeddings.create(input=[query], model=model)
query_embedding = np.array([query_response.data[0].embedding])
similarities = cosine_similarity(query_embedding, embeddings)[0]

top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\nQuery: '{query}'")
print(f"Showing top {top_k} results:")
print("-" * 50)

below_threshold_rank = None
for rank, idx in enumerate(top_indices, 1):
    score = similarities[idx]
    ticket = tickets[idx]
    print(f"#{rank} [{score:.4f}] {ticket['title']}")
    
    if below_threshold_rank is None and score < 0.5:
        below_threshold_rank = rank

print(f"\n→ Score drops below 0.5 at rank: {below_threshold_rank or 'Never (all above 0.5)'}")


# ============================================================================
# Exercise 3: Add a Similarity Threshold (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Add a Similarity Threshold")
print("=" * 80)

# Solution: Add "if score < 0.5: continue" in the loop

def search_with_threshold(query_text, threshold=0.5):
    query_response = client.embeddings.create(input=[query_text], model=model)
    query_embedding = np.array([query_response.data[0].embedding])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = np.argsort(similarities)[::-1][:10]
    
    print(f"\nQuery: '{query_text}' (threshold: {threshold})")
    print("-" * 50)
    
    count = 0
    for rank, idx in enumerate(top_indices, 1):
        ticket = tickets[idx]
        score = similarities[idx]
        
        # THE SOLUTION: Skip results below threshold
        if score < threshold:
            continue
        
        count += 1
        print(f"#{rank} [{score:.4f}] {ticket['title']}")
    
    if count == 0:
        print("No relevant tickets found above threshold.")
    else:
        print(f"\n→ {count} tickets above threshold")

# Test with relevant query
search_with_threshold("login authentication problem")

# Test with irrelevant query
search_with_threshold("How to make pizza")


# ============================================================================
# Exercise 4: Compare Two Queries (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Compare Two Queries")
print("=" * 80)

# Solution: Just add this code to compare queries
query1 = "Login authentication failed"
query2 = "Slow database performance"

for q in [query1, query2]:
    response = client.embeddings.create(input=[q], model=model)
    q_emb = np.array([response.data[0].embedding])
    sims = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{q}'")
    print(f"  Best match: {tickets[top_idx]['title']}")
    print(f"  Category: {tickets[top_idx]['category']}")
    print(f"  Score: {sims[top_idx]:.4f}")


# ============================================================================
# Exercise 5: Test Semantic Understanding (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Test Semantic Understanding")
print("=" * 80)

# These mean the SAME thing but use DIFFERENT words
test_texts = [
    "User authentication failed",      # Original
    "Login credentials rejected",       # Same meaning, different words
    "Cannot sign in to account",        # Same meaning, different words
    "Database connection timeout",      # DIFFERENT topic
]

# Generate embeddings
response = client.embeddings.create(input=test_texts, model=model)
test_embeddings = np.array([data.embedding for data in response.data])

# Calculate all pairwise similarities
similarity_matrix = cosine_similarity(test_embeddings)

# Print results
print("\nSimilarity Matrix Analysis:")
print("-" * 60)
for i, text1 in enumerate(test_texts):
    for j, text2 in enumerate(test_texts):
        if i < j:  # Only print upper triangle
            sim = similarity_matrix[i][j]
            relation = "SIMILAR" if sim > 0.7 else "DIFFERENT"
            print(f"{sim:.3f} [{relation}] '{text1[:25]}...' vs '{text2[:25]}...'")

print("\n→ Notice: Auth/Login texts have HIGH similarity despite different words")
print("→ Notice: Database text has LOW similarity - different topic entirely")


# ============================================================================
# Exercise 6: Filter by Category (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Filter by Category")
print("=" * 80)

def search_with_category(query, category_filter=None, top_k=5):
    """Search tickets, optionally filtering by category"""
    # Get query embedding
    response = client.embeddings.create(input=[query], model=model)
    query_emb = np.array([response.data[0].embedding])
    
    # Calculate similarities
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # Get results with category filter
    results = []
    for idx in np.argsort(similarities)[::-1]:
        ticket = tickets[idx]
        
        # THE SOLUTION: Skip if category doesn't match filter
        if category_filter and ticket['category'] != category_filter:
            continue
        
        results.append((ticket, similarities[idx]))
        if len(results) >= top_k:
            break
    
    return results

# Test it
print("\nAll categories:")
for ticket, score in search_with_category("login problem"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")

print("\nOnly 'Authentication' category:")
for ticket, score in search_with_category("login problem", category_filter="Authentication"):
    print(f"  {score:.3f} [{ticket['category']}] {ticket['title']}")


# ============================================================================
# Exercise 7: Batch vs Single Embedding (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Batch vs Single Embedding")
print("=" * 80)

batch_texts = [
    "Password reset not working",
    "Database connection timeout", 
    "App crashes on startup",
    "Payment declined error",
    "Email notifications delayed",
]

# Method 1: SLOW - One API call per text
print("\nMethod 1: Single API calls...")
start = time.time()
for text in batch_texts:
    response = client.embeddings.create(input=[text], model=model)
time_slow = time.time() - start
print(f"  Time: {time_slow:.2f} seconds")

# Method 2: FAST - One API call for all texts
print("\nMethod 2: Batch API call...")
start = time.time()
response = client.embeddings.create(input=batch_texts, model=model)
time_fast = time.time() - start
print(f"  Time: {time_fast:.2f} seconds")

# Compare
speedup = time_slow / time_fast if time_fast > 0 else float('inf')
print(f"\n✓ Batch is {speedup:.1f}x faster!")
print("  Always batch your embeddings in production!")


# ============================================================================
# Bonus: Similarity Matrix Heatmap (Challenge)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Similarity Matrix Heatmap")
print("=" * 80)

# Use first 10 tickets
sample_tickets = tickets[:10]
sample_texts = [t['title'] for t in sample_tickets]

# Generate embeddings
response = client.embeddings.create(input=sample_texts, model=model)
sample_embeddings = np.array([data.embedding for data in response.data])

# Compute similarity matrix
sim_matrix = cosine_similarity(sample_embeddings)

# Create heatmap
plt.figure(figsize=(12, 10))
plt.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
plt.colorbar(label='Cosine Similarity')

# Add labels
labels = [f"{t['ticket_id']}\n({t['category'][:8]})" for t in sample_tickets]
plt.xticks(range(10), labels, rotation=45, ha='right', fontsize=8)
plt.yticks(range(10), labels, fontsize=8)

# Add values to cells
for i in range(10):
    for j in range(10):
        plt.text(j, i, f'{sim_matrix[i, j]:.2f}', 
                ha='center', va='center', fontsize=8,
                color='white' if sim_matrix[i, j] < 0.5 else 'black')

plt.title('Ticket Similarity Matrix\n(First 10 Tickets)')
plt.tight_layout()
plt.savefig('solution_similarity_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved as solution_similarity_heatmap.png")
plt.show()


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1. Embeddings capture semantic meaning, not just keywords
2. Cosine similarity measures how related two texts are
3. Threshold filtering removes irrelevant results
4. Category filtering narrows search to specific domains
5. Batch API calls are 5-10x faster than single calls
6. Similarity matrices reveal relationships between documents

Next: Move on to Module 2 - Chunking & Vector Stores!
""")
