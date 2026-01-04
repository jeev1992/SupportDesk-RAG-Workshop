# Hour 1 Exercises: Embeddings & Similarity Search

## Exercise 1: Find Similar Tickets (Easy)

**Task**: Modify the demo to find the top-10 most similar tickets instead of top-5.

**Steps**:
1. Change the `top_k` variable in the demo code
2. Run the modified script
3. Observe how similarity scores decrease as you go down the rankings

**Questions**:
- At what rank does the similarity score drop below 0.5?
- Are all top-10 results still relevant to the query?

---

## Exercise 2: Experiment with Different Embedding Models (Medium)

**Task**: Compare results using different OpenAI embedding models.

**Models to try**:
```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Small and fast (1536 dimensions)
model = 'text-embedding-3-small'

# Larger and more accurate (3072 dimensions)
model = 'text-embedding-3-large'

# Legacy model (1536 dimensions)
model = 'text-embedding-ada-002'
```

**Steps**:
1. Modify the demo to use each model
2. Compare the top-5 results for the same query
3. Note differences in similarity scores and retrieval quality

**Questions**:
- Which model gives the most relevant results?
- How does the larger model (text-embedding-3-large) compare?
- Is the quality improvement worth the higher cost?
- What's the trade-off between model size, cost, and accuracy?

---

## Exercise 3: Custom Query Testing (Medium)

**Task**: Test the semantic search with your own queries.

**Suggested queries**:
- "Credentials rejected after password change"
- "Connection pool exhausted"  
- "iOS application force closes"
- "Credit card payment declined"
- "Memory usage increasing continuously"

**Steps**:
1. Create a function to search for similar tickets:
```python
def search_tickets(query, tickets, embeddings, client, model_name, top_k=5):
    # Generate query embedding
    response = client.embeddings.create(input=[query], model=model_name)
    query_emb = np.array([response.data[0].embedding])
    
    # Calculate similarities
    similarities = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'ticket': tickets[idx],
            'similarity': similarities[idx]
        })
    return results
```

2. Test with your queries
3. Analyze which types of queries work best

**Questions**:
- Does semantic search find relevant tickets even with different wording?
- What happens when you search for something not in the dataset?

---

## Exercise 4: Similarity Threshold (Medium)

**Task**: Implement a similarity threshold to filter irrelevant results.

**Requirements**:
- Only show results with similarity > 0.5
- If no results meet threshold, display "No relevant tickets found"
- Display the number of tickets that met the threshold

**Starter code**:
```python
def search_with_threshold(query, tickets, embeddings, client, model_name, threshold=0.5, top_k=5):
    # Generate query embedding
    response = client.embeddings.create(input=[query], model=model_name)
    query_emb = np.array([response.data[0].embedding])
    
    similarities = cosine_similarity(query_emb, embeddings)[0]
    
    # TODO: Filter by threshold
    # TODO: Sort by similarity
    # TODO: Return top_k results
    pass
```

**Test with**:
- A relevant query (should return results)
- An irrelevant query like "How to make pizza" (should return nothing)

---

## Exercise 5: Analyze Embedding Dimensions (Hard)

**Task**: Understand what information different dimensions capture.

**Steps**:
1. Generate embeddings for these texts:
```python
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

texts = [
    "User authentication failed",
    "Login error",
    "Database connection timeout",
    "Pizza recipe ingredients"
]
```

2. Calculate pairwise cosine similarities:
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Generate embeddings
response = client.embeddings.create(input=texts, model='text-embedding-3-small')
embeddings = np.array([data.embedding for data in response.data])

similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)
```

3. Create a heatmap:
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, xticklabels=texts, 
            yticklabels=texts, cmap='YlOrRd', vmin=0, vmax=1)
plt.title('Similarity Matrix')
plt.tight_layout()
plt.show()
```

**Questions**:
- Which texts are most similar? Does it match your intuition?
- What's the similarity between "authentication failed" and "login error"?
- How different is "Pizza recipe" from the technical queries?

---

## Exercise 6: Build a Simple Search Interface (Challenge)

**Task**: Create an interactive command-line search tool.

**Requirements**:
```python
def main():
    # Load tickets and generate embeddings once
    tickets, embeddings, model = load_data()
    
    print("SupportDesk Search (type 'quit' to exit)")
    while True:
        query = input("\nEnter search query: ")
        if query.lower() == 'quit':
            break
            
        results = search_tickets(query, tickets, embeddings, model)
        display_results(results)

if __name__ == "__main__":
    main()
```

**Bonus features**:
- Color-code results by priority (High = red, Medium = yellow, etc.)
- Show execution time for each search
- Allow filtering by category
- Display similarity score as a percentage

---

## Hints & Tips

### Debugging Similarity Scores
If all scores are very low (<0.3):
- Check that your embeddings are normalized
- Verify the model loaded correctly
- Ensure query and documents use the same model

### Performance Optimization
```python
# Normalize embeddings once for faster cosine similarity
from sklearn.preprocessing import normalize
normalized_embeddings = normalize(embeddings)
```

### Visualization Tips
```python
# Add annotations to your 2D plot
for i, ticket in enumerate(tickets):
    plt.annotate(ticket['ticket_id'], 
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                fontsize=8, alpha=0.7)
```

---

## Solutions

Solutions are available in `solutions.py` (will be provided after the workshop).

For now, try to solve these yourself! Learning happens through struggle. 💪

---

## Next Steps

Ready for more? Move on to **Hour 2: Chunking & Vector Stores** where we'll:
- Scale to thousands of documents
- Learn efficient storage and retrieval
- Build production-ready vector databases

---

**Questions?** Ask the instructor or refer back to the demo code!
