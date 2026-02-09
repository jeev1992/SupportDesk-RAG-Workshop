# -*- coding: utf-8 -*-
"""
================================================================================
MODULE 1: Embeddings & Similarity Search Demo
================================================================================

WHY EMBEDDINGS MATTER:
━━━━━━━━━━━━━━━━━━━━━
Traditional search: "password reset" only finds docs with those exact words
Semantic search:    "password reset" also finds "forgot credentials", 
                    "can't log in after changing password", "auth issues"

This is the FOUNDATION of RAG - without good embeddings, retrieval fails!

WHAT YOU'LL LEARN:
━━━━━━━━━━━━━━━━━
1. How to generate embeddings from text using OpenAI
2. What embeddings actually look like (1536 numbers!)
3. Computing similarity scores (cosine similarity)
4. Finding most similar documents (semantic search)
5. Visualizing the relationships embeddings capture

THE BIG PICTURE:
━━━━━━━━━━━━━━━
    Text → [Embedding Model] → Vector (1536 floats) → [Compare] → Similarity Score
    
    "login fails"     →  [0.023, -0.041, ...]  ─┐
                                                 ├─→ 0.89 (similar!)
    "auth not working" → [0.019, -0.038, ...]  ─┘

LEARNING RESOURCES:
- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings
- Understanding Vector Embeddings: https://www.pinecone.io/learn/vector-embeddings/
- Cosine Similarity Explained: https://en.wikipedia.org/wiki/Cosine_similarity
"""

# =============================================================================
# IMPORTS
# =============================================================================
import json
import numpy as np              # For numerical operations on embedding vectors
import os
from openai import OpenAI       # OpenAI API client for generating embeddings
from sklearn.metrics.pairwise import cosine_similarity  # Measure similarity between vectors
import matplotlib.pyplot as plt # For visualizing embeddings
from dotenv import load_dotenv  # Load environment variables from .env file

# =============================================================================
# SETUP: Load Environment Variables
# =============================================================================
# Best practice: NEVER hardcode API keys in your code!
# Store them in a .env file and load with python-dotenv
#
# Your .env file should contain:
#   OPENAI_API_KEY=sk-...
#   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# =============================================================================
load_dotenv()

# =============================================================================
# INITIALIZE OPENAI CLIENT
# =============================================================================
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# =============================================================================
# EMBEDDING MODEL SELECTION
# =============================================================================
#
# OpenAI offers several embedding models (as of 2024):
#
# | Model                    | Dimensions | Cost          | Use Case            |
# |--------------------------|------------|---------------|---------------------|
# | text-embedding-3-small   | 1536       | $0.02/1M tok  | General purpose ✓   |
# | text-embedding-3-large   | 3072       | $0.13/1M tok  | Higher quality      |
# | text-embedding-ada-002   | 1536       | $0.10/1M tok  | Legacy (deprecated) |
#
# WE USE: text-embedding-3-small
#   - Good balance of quality and cost
#   - 1536 dimensions is plenty for most use cases
#   - ~6x cheaper than large model
#
# Reference: https://platform.openai.com/docs/guides/embeddings/embedding-models
# =============================================================================
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
embedding_dim = 1536  # Number of dimensions in the embedding vector
print(f"Using OpenAI model: {embedding_model}")
print(f"Embedding dimension: {embedding_dim}")

# =============================================================================
# LOAD DATA: Support Tickets
# =============================================================================
print("\nLoading support tickets...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} support tickets")

# Display sample ticket - understand what we're working with
print("\n" + "="*80)
print("SAMPLE TICKET:")
print("="*80)
sample = tickets[0]
print(f"ID: {sample['ticket_id']}")
print(f"Title: {sample['title']}")
print(f"Description: {sample['description'][:200]}...")
print(f"Category: {sample['category']}")
print(f"Priority: {sample['priority']}")

# ============================================================================
# PART 1: Generate Embeddings
# ============================================================================
#
# WHAT IS AN EMBEDDING?
# ━━━━━━━━━━━━━━━━━━━━
# A list of 1536 numbers that represents the "meaning" of text.
# Think of it as coordinates in a 1536-dimensional space where
# similar meanings are located close together.
#
# ANALOGY: If we could plot words in 2D...
#
#     "cat" ●         ● "dog"      (close = similar)
#           
#                     ● "python"   (far = different meaning)
#     
#     ● "car"         ● "truck"    (close = similar)
#
# But we need 1536 dimensions to capture all the nuances of language!
#
# ============================================================================
print("\n" + "="*80)
print("PART 1: Generating Embeddings")
print("="*80)

# -----------------------------------------------------------------------------
# Prepare text for embedding
# -----------------------------------------------------------------------------
# TIP: Combine relevant fields for richer context
# The more context, the better the embedding captures the meaning
# 
# BAD:  Just title → "Login issue"
# GOOD: Title + description → "Login issue. User reports authentication 
#       failure after password reset. Error code 401..."
# -----------------------------------------------------------------------------
ticket_texts = [
    f"{ticket['title']}. {ticket['description']}" 
    for ticket in tickets
]

# -----------------------------------------------------------------------------
# Generate embeddings via OpenAI API
# -----------------------------------------------------------------------------
# This is a BATCH call - we send all texts at once for efficiency
# The API returns one embedding per input text
#
# ⚠️ RATE LIMITS: Be aware of OpenAI's rate limits for production!
# ⚠️ COST: ~$0.02 per 1M tokens for text-embedding-3-small
# -----------------------------------------------------------------------------
print("\nGenerating embeddings for all tickets...")
response = client.embeddings.create(input=ticket_texts, model=embedding_model)

# Convert API response to NumPy array for mathematical operations
# Shape: (num_tickets, 1536) - each row is one ticket's embedding
embeddings = np.array([data.embedding for data in response.data])
print(f"✓ Generated embeddings with shape: {embeddings.shape}")
print(f"  ({len(tickets)} tickets × {embedding_dim} dimensions)")

# -----------------------------------------------------------------------------
# Inspect an embedding
# -----------------------------------------------------------------------------
# You can't interpret individual values, but together they encode meaning
# Similar texts will have embeddings that point in similar directions
# -----------------------------------------------------------------------------
print(f"\nFirst 10 values of embedding for ticket 1:")
print(embeddings[0][:10])
print("  (These 1536 numbers encode the semantic meaning of the text)")

# ============================================================================
# PART 2: Compute Similarity Scores
# ============================================================================
#
# HOW DO WE COMPARE EMBEDDINGS?
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# We need a way to measure "how similar" two vectors are.
# The most common method is COSINE SIMILARITY.
#
# COSINE SIMILARITY EXPLAINED:
# ────────────────────────────
# Measures the angle between two vectors (not the distance!)
#
#     Range: -1 to 1
#     1     = identical direction (very similar)
#     0     = perpendicular (unrelated)
#     -1    = opposite direction (contradictory)
#
# WHY COSINE? (vs Euclidean distance)
# ────────────────────────────────────
# • Invariant to vector magnitude - only cares about direction
# • Works well in high dimensions (Euclidean distance suffers from "curse of dimensionality")
# • Standard in NLP and embedding-based search
#
# FORMULA: cos(θ) = (A · B) / (||A|| × ||B||)
#          (dot product divided by product of magnitudes)
#
# ============================================================================
print("\n" + "="*80)
print("PART 2: Computing Similarity Scores")
print("="*80)

# Create a search query - this is what a user might type
query = "Users can't login after changing password"
print(f"\nSearch Query: '{query}'")

# -----------------------------------------------------------------------------
# Generate embedding for the query
# -----------------------------------------------------------------------------
# CRITICAL: Use the SAME model as documents!
# Different models produce incompatible vector spaces
# 
# text-embedding-3-small documents + text-embedding-3-large query = WRONG!
# -----------------------------------------------------------------------------
query_response = client.embeddings.create(input=[query], model=embedding_model)
query_embedding = np.array([query_response.data[0].embedding])
print(f"Query embedding shape: {query_embedding.shape}")  # (1, 1536)

# -----------------------------------------------------------------------------
# Compute cosine similarity between query and ALL tickets
# -----------------------------------------------------------------------------
# This is the core of semantic search!
# We compare the query vector against every document vector
# Result: one similarity score per document
# -----------------------------------------------------------------------------
similarities = cosine_similarity(query_embedding, embeddings)[0]
print(f"\nComputed similarity scores for {len(similarities)} tickets")
print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

# ============================================================================
# PART 3: Retrieve Most Similar Tickets
# ============================================================================
#
# THIS IS SEMANTIC SEARCH!
# ━━━━━━━━━━━━━━━━━━━━━━━━
# Instead of matching keywords, we find documents with similar MEANING.
#
# The process:
# 1. Embed the query → vector
# 2. Compare to all document vectors → similarity scores
# 3. Sort by similarity → ranked results
# 4. Return top-K → most relevant documents
#
# Notice: "password reset" in query will match "forgot credentials"
#         even though they share no words!
#
# ============================================================================
print("\n" + "="*80)
print("PART 3: Finding Most Similar Tickets")
print("="*80)

# Get top-5 most similar tickets
top_k = 5

# np.argsort() returns indices that would sort the array (ascending)
# [::-1] reverses to get descending order (highest similarity first)
# [:top_k] takes only the top K results
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\nTop {top_k} most similar tickets to query: '{query}'")
print("-" * 80)

for rank, idx in enumerate(top_indices, 1):
    ticket = tickets[idx]
    score = similarities[idx]
    
    print(f"\n#{rank} - Similarity: {score:.4f}")
    print(f"Ticket ID: {ticket['ticket_id']}")
    print(f"Title: {ticket['title']}")
    print(f"Category: {ticket['category']} | Priority: {ticket['priority']}")
    print(f"Description: {ticket['description'][:150]}...")

# ============================================================================
# PART 4: Visualize Embedding Relationships
# ============================================================================
#
# THE CHALLENGE WITH 1536 DIMENSIONS:
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# We can't visualize 1536D space directly (humans can only see 2D/3D)
#
# OPTIONS FOR VISUALIZATION:
# ──────────────────────────
# 1. Similarity heatmap - show pairwise similarities (what we do here)
# 2. t-SNE/UMAP - project to 2D (LOSES information, can be misleading!)
# 3. PCA - linear projection (also loses information)
#
# WE CHOOSE: Similarity heatmap
# WHY? Shows TRUE relationships without distortion
#
# TEACHING POINT:
# Don't trust dimensionality reduction plots!
# They can make distant points look close or vice versa.
# Similarity scores are the GROUND TRUTH.
#
# ============================================================================
print("\n" + "="*80)
print("PART 4: Visualizing Similarity Relationships")
print("="*80)

print("\nEmbeddings capture semantic relationships through similarity scores.")
print("Let's visualize these relationships using exact similarity measurements.\n")

# Create similarity heatmap for top tickets
print("Creating similarity heatmap...")

# Select top matches and a few random others for comparison
# This lets us see: high-similarity pairs vs. unrelated pairs
selected_indices = list(top_indices[:5]) + list(np.random.choice(
    [i for i in range(len(tickets)) if i not in top_indices[:5]], 
    size=min(5, len(tickets) - 5), 
    replace=False
))

# Compute similarity matrix for selected tickets
# This creates a 10x10 matrix of pairwise similarities
selected_embeddings = embeddings[selected_indices]
similarity_matrix = cosine_similarity(selected_embeddings)

# -----------------------------------------------------------------------------
# Create the visualization
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# LEFT PLOT: Similarity heatmap
# Shows pairwise similarity between all selected tickets
# Green = high similarity, Red = low similarity
im = ax1.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
ax1.set_xticks(range(len(selected_indices)))
ax1.set_yticks(range(len(selected_indices)))

# Label with ticket IDs and categories
labels = [f"{tickets[i]['ticket_id']}\n({tickets[i]['category']})" 
          for i in selected_indices]
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax1.set_yticklabels(labels, fontsize=8)

# Add similarity values to cells (so students can read exact numbers)
for i in range(len(selected_indices)):
    for j in range(len(selected_indices)):
        text = ax1.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

ax1.set_title('Similarity Heatmap: What Embeddings Actually Measure\n' + 
             '(Top 5 matches + random others)', fontweight='bold', fontsize=11)
plt.colorbar(im, ax=ax1, label='Cosine Similarity')

# RIGHT PLOT: Query similarities bar chart
# Shows how similar each ticket is to the original query
query_similarities = [similarities[i] for i in selected_indices]
colors_bar = ['green' if i < 5 else 'gray' for i in range(len(selected_indices))]

ax2.barh(range(len(selected_indices)), query_similarities, color=colors_bar, alpha=0.7)
ax2.set_yticks(range(len(selected_indices)))
ax2.set_yticklabels([f"{tickets[i]['ticket_id']}" for i in selected_indices], fontsize=9)
ax2.set_xlabel('Similarity to Query', fontweight='bold')
ax2.set_title(f'Similarity Scores for Query:\n"{query}"\n(Green = Top 5 matches)', 
             fontweight='bold', fontsize=11)
ax2.set_xlim(0, 1)
ax2.grid(axis='x', alpha=0.3)

# Add score labels on bars
for i, score in enumerate(query_similarities):
    ax2.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('embeddings_similarity_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Visualization saved as 'embeddings_similarity_analysis.png'")
print("\nKEY INSIGHTS FROM THIS VISUALIZATION:")
print("  • Left heatmap: Shows TRUE pairwise similarities in 1536D space")
print("  • Right chart: Query similarity scores (what drives retrieval)")
print("  • High similarity (green) = semantically similar content")
print("  • Low similarity (red) = different topics/meanings")
print("  • These scores are EXACT - they show true relationships in 1536D space!")
plt.show()

# ============================================================================
# PART 5: Experiment with Different Queries
# ============================================================================
#
# TEACHING POINT: Show that semantic search "understands" meaning
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 
# Notice how queries match tickets even without shared keywords:
# 
# Query: "Database is timing out"
# Match: "Connection pool exhausted" (same concept, different words!)
#
# Query: "App crashes on iPhone"
# Match: "iOS application terminates unexpectedly" (synonyms work!)
#
# This is the MAGIC of embeddings!
#
# ============================================================================
print("\n" + "="*80)
print("PART 5: Try Different Queries")
print("="*80)

test_queries = [
    "Database is timing out",
    "Payment not working for foreign customers",
    "App crashes on iPhone",
    "Emails are not being sent"
]

print("\nTesting semantic search with different queries:")
for test_query in test_queries:
    # Generate query embedding
    query_resp = client.embeddings.create(input=[test_query], model=embedding_model)
    query_emb = np.array([query_resp.data[0].embedding])
    
    # Compare to all tickets
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{test_query}'")
    print(f"  → Best match: {tickets[top_idx]['title']}")
    print(f"  → Similarity: {sims[top_idx]:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("""
KEY TAKEAWAYS:
━━━━━━━━━━━━━━
1. EMBEDDINGS convert text → 1536-dimensional vectors
   Each vector encodes the semantic "meaning" of the text

2. COSINE SIMILARITY measures how similar two vectors are
   Range: -1 to 1 (higher = more similar)

3. SEMANTIC SEARCH finds meaning, not just keywords
   "password reset" matches "forgot credentials"!

4. THE SAME MODEL must be used for queries and documents
   Different models = incompatible vector spaces

5. SIMILARITY SCORES are the truth
   Don't trust 2D visualizations - they lose information

CONNECTION TO RAG:
━━━━━━━━━━━━━━━━━
In RAG, we use embeddings to find relevant documents,
then pass those documents to an LLM for generation.

Query → [Embed] → [Search] → Top Documents → [LLM] → Answer

NEXT: Module 2 - Chunking & Vector Stores
""")
