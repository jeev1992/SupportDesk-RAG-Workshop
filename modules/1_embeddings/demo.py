# -*- coding: utf-8 -*-
"""
Hour 1: Embeddings & Similarity Search Demo
============================================

This demo teaches:
1. How to generate embeddings from text
2. Computing similarity scores
3. Finding most similar documents
4. Visualizing embeddings in 2D space
"""

import json
import numpy as np
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
print("Initializing OpenAI client...")
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
embedding_dim = 1536  # text-embedding-3-small dimension
print(f"Using OpenAI model: {embedding_model}")
print(f"Embedding dimension: {embedding_dim}")

# Load synthetic tickets
print("\nLoading support tickets...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"Loaded {len(tickets)} support tickets")

# Display sample ticket
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
print("\n" + "="*80)
print("PART 1: Generating Embeddings")
print("="*80)

# Combine title and description for richer context
ticket_texts = [
    f"{ticket['title']}. {ticket['description']}" 
    for ticket in tickets
]

print("\nGenerating embeddings for all tickets...")
response = client.embeddings.create(input=ticket_texts, model=embedding_model)
embeddings = np.array([data.embedding for data in response.data])
print(f"✓ Generated embeddings with shape: {embeddings.shape}")
print(f"  ({len(tickets)} tickets × {embedding_dim} dimensions)")

# Show what an embedding looks like
print(f"\nFirst 10 values of embedding for ticket 1:")
print(embeddings[0][:10])
print("  (These numbers encode the semantic meaning of the text)")

# ============================================================================
# PART 2: Compute Similarity Scores
# ============================================================================
print("\n" + "="*80)
print("PART 2: Computing Similarity Scores")
print("="*80)

# Create a search query
query = "Users can't authenticate after changing password"
print(f"\nSearch Query: '{query}'")

# Generate embedding for the query
query_response = client.embeddings.create(input=[query], model=embedding_model)
query_embedding = np.array([query_response.data[0].embedding])
print(f"Query embedding shape: {query_embedding.shape}")

# Compute cosine similarity between query and all tickets
similarities = cosine_similarity(query_embedding, embeddings)[0]
print(f"\nComputed similarity scores for {len(similarities)} tickets")
print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")

# ============================================================================
# PART 3: Retrieve Most Similar Tickets
# ============================================================================
print("\n" + "="*80)
print("PART 3: Finding Most Similar Tickets")
print("="*80)

# Get top-5 most similar tickets
top_k = 5
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
# PART 4: Visualize Embeddings in Semantic Space
# ============================================================================
print("\n" + "="*80)
print("PART 4: Visualizing Embeddings in Semantic Space")
print("="*80)

print(f"\nTo visualize {embedding_dim}-dimensional embeddings, we need to project them into 2D...")
print("Think of it like taking a photo of a 3D object - we lose some detail but can see relationships.")

# Use PCA to reduce dimensions (behind the scenes - students don't need to worry about this)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
query_2d = pca.transform(query_embedding)

print("✓ Embeddings projected to 2D for visualization")
print("\nWhat you'll see:")
print("  • Each dot = one support ticket")
print("  • Closer dots = more semantically similar tickets")
print("  • Red star = your search query")
print("  • Red circles = top-5 matches")

# Create the plot
plt.figure(figsize=(12, 8))

# Plot all tickets by category
categories = list(set(ticket['category'] for ticket in tickets))
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
category_to_color = dict(zip(categories, colors))

for i, ticket in enumerate(tickets):
    color = category_to_color[ticket['category']]
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
               c=[color], label=ticket['category'], s=100, alpha=0.6)

# Highlight top-5 matches with red circles
for idx in top_indices:
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
               s=300, facecolors='none', edgecolors='red', linewidths=2)

# Plot query as red star
plt.scatter(query_2d[0, 0], query_2d[0, 1], 
           c='red', marker='*', s=500, label='Query', edgecolors='black', linewidths=2)

# Remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best')

plt.title('Embeddings in Semantic Space: Similar Meanings Cluster Together', 
         fontsize=14, fontweight='bold')
plt.xlabel('Semantic Dimension 1')
plt.ylabel('Semantic Dimension 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('embeddings_visualization.png', dpi=150)
print("\n✓ Visualization saved as 'embeddings_visualization.png'")
plt.show()

# ============================================================================
# PART 5: Experiment with Different Queries
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
    query_resp = client.embeddings.create(input=[test_query], model=embedding_model)
    query_emb = np.array([query_resp.data[0].embedding])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = np.argmax(sims)
    
    print(f"\nQuery: '{test_query}'")
    print(f"  → Best match: {tickets[top_idx]['title']}")
    print(f"  → Similarity: {sims[top_idx]:.4f}")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. Embeddings convert text into numerical vectors")
print("2. Similar meanings → similar vectors (measured by cosine similarity)")
print("3. Semantic search finds meaning, not just keywords")
print("4. Embeddings can be visualized to understand relationships")
print("\nNext: Hour 2 - Chunking & Vector Stores")
