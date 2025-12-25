# Hour 3 Exercises: RAG Pipeline

## Exercise 1: Modify the Prompt Template (Easy)

**Task**: Experiment with different prompt templates and observe how they affect answers.

**Try these variations**:

**Version A: More concise**
```python
template = """Answer the question using only the ticket context below. Cite ticket IDs.

Context: {context}

Question: {question}

Answer:"""
```

**Version B: Step-by-step reasoning**
```python
template = """You are a support assistant. Answer using ONLY the context below.

Context: {context}

Question: {question}

Think step by step:
1. What tickets are relevant?
2. What information do they contain?
3. How does this answer the question?

Answer:"""
```

**Version C: Format with bullet points**
```python
template = """Answer using only the context. Format as bullet points with ticket citations.

Context: {context}

Question: {question}

Answer (bullet points with sources):"""
```

**Test with**: "How do I fix authentication issues?"

**Questions**:
- Which prompt gives the most useful answers?
- Which format is easiest to read?
- Does any version hallucinate more?

---

## Exercise 2: Adjust Retrieval Parameters (Medium)

**Task**: Experiment with different retrieval configurations.

**Parameters to tune**:
```python
# Number of documents
retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # vs k=5
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# Search type
retriever = vector_store.as_retriever(
    search_type="similarity",  # Default
    search_kwargs={"k": 3}
)

retriever = vector_store.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance (diverse results)
    search_kwargs={"k": 3, "fetch_k": 10}
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 3}
)
```

**Test queries**:
- "Payment processing failures"
- "Mobile app crashes"
- "Slow dashboard loading"

**Questions**:
- How does k affect answer quality?
- When is MMR better than similarity?
- What's a good similarity threshold?

---

## Exercise 3: Implement Citation Formatting (Medium)

**Task**: Make the assistant always include inline citations.

**Goal output**:
```
Authentication failures after password reset are caused by stale session tokens [TICK-001]. 
The solution is to clear all active sessions and force re-authentication [TICK-001]. 
Implement automatic session cleanup on password change to prevent this issue.
```

**Approach**:
```python
citation_prompt = """Answer the question using the context. Include inline citations [TICK-XXX] after each fact.

Example format:
"Database connection timeouts occur when the pool is undersized [TICK-002]. Increase max_connections and monitor usage [TICK-002]."

Context:
{context}

Question: {question}

Answer with inline citations:"""
```

**Bonus**: Create a function to extract and verify all cited tickets exist in context.

---

## Exercise 4: Build a Fallback System (Medium)

**Task**: Implement graceful fallbacks when retrieval fails or confidence is low.

**Requirements**:
```python
def smart_rag(query, qa_chain, vector_store):
    """
    RAG with intelligent fallbacks
    """
    # Step 1: Check retrieval quality
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    if not docs_with_scores:
        return "No relevant tickets found."
    
    best_score = docs_with_scores[0][1]
    
    # Step 2: Score-based responses
    if best_score < 0.3:  # Very relevant
        return qa_chain({"query": query})['result']
    
    elif best_score < 0.7:  # Somewhat relevant
        return f"Found possibly relevant tickets ({docs_with_scores[0][0].metadata['ticket_id']}), but confidence is moderate. Would you like me to show them?"
    
    else:  # Not relevant
        return "I don't have relevant ticket history for this question. Could you rephrase or ask about a different issue?"
```

**Test with**:
- High confidence: "authentication problems"
- Medium confidence: "system performance"
- Low confidence: "how to bake cookies"

---

## Exercise 5: Multi-Turn Conversation with Memory (Hard)

**Task**: Add conversation memory so the assistant remembers previous questions.

**Requirements**:
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

# Create conversational chain
conversational_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)
```

**Test conversation**:
```
User: "What causes authentication failures?"
Assistant: [answers about TICK-001]

User: "How do I fix it?"  # Should remember we're talking about auth
Assistant: [provides solution from TICK-001]

User: "What about database issues?"  # New topic
Assistant: [switches to database tickets]
```

**Challenge**: Clear memory when switching topics.

---

## Exercise 6: Compare Chain Types (Medium)

**Task**: LangChain supports different chain types for RAG. Compare them.

**Chain types**:

1. **"stuff"** - Concatenate all docs into one prompt
```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
```

2. **"map_reduce"** - Process each doc separately, then combine
```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=retriever
)
```

3. **"refine"** - Iteratively refine answer with each doc
```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="refine",
    retriever=retriever
)
```

**Compare**:
- Speed
- Answer quality
- Token usage
- Best use cases

---

## Exercise 7: Add Streaming Responses (Medium)

**Task**: Make the assistant stream responses word-by-word (better UX for long answers).

**Implementation**:
```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Now responses will stream!
result = qa_chain({"query": "How to fix database timeouts?"})
```

**Bonus**: Create a custom callback to count tokens as they stream.

---

## Exercise 8: Implement RAG with Local LLM (Challenge)

**Task**: Use Ollama or other local models instead of OpenAI.

**Setup Ollama**:
```bash
# Install Ollama from ollama.ai
ollama pull llama2
```

**Use in LangChain**:
```python
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama2",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)
```

**Compare**:
- Speed vs OpenAI
- Answer quality
- Cost (free!)
- Privacy (no data leaves your machine)

---

## Exercise 9: Build a RAG API (Hard)

**Task**: Create a FastAPI endpoint for your RAG system.

**Implementation**:
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    question: str
    k: int = 3

@app.post("/ask")
async def ask_question(query: Query):
    """RAG API endpoint"""
    # Update retriever with custom k
    custom_retriever = vector_store.as_retriever(
        search_kwargs={"k": query.k}
    )
    
    # Get answer
    result = qa_chain({"query": query.question})
    
    return {
        "answer": result['result'],
        "sources": [
            {
                "ticket_id": doc.metadata['ticket_id'],
                "title": doc.metadata['title']
            }
            for doc in result['source_documents']
        ]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Run**:
```bash
uvicorn api:app --reload
```

**Test**:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "How to fix auth issues?", "k": 5}'
```

---

## Exercise 10: Hallucination Detection (Advanced)

**Task**: Build a system to detect when the LLM hallucinates.

**Approach**:
```python
def detect_hallucination(query, answer, source_documents):
    """
    Check if answer contains information not in sources
    """
    # Extract all content from source documents
    source_text = " ".join([doc.page_content for doc in source_documents])
    
    # Embed both
    model = SentenceTransformer('all-MiniLM-L6-v2')
    answer_embedding = model.encode([answer])
    source_embedding = model.encode([source_text])
    
    # Check similarity
    similarity = cosine_similarity(answer_embedding, source_embedding)[0][0]
    
    # Low similarity = possible hallucination
    if similarity < 0.6:
        return True, f"Warning: Answer may contain information not in sources (similarity: {similarity:.2f})"
    
    return False, "Answer appears grounded in sources"

# Test
result = qa_chain({"query": "How to fix database timeouts?"})
is_hallucination, message = detect_hallucination(
    query="How to fix database timeouts?",
    answer=result['result'],
    source_documents=result['source_documents']
)
print(message)
```

**Bonus**: Use an LLM as a judge to verify grounding.

---

## Production Checklist

Before deploying RAG to production:

- [ ] Implement proper error handling
- [ ] Add rate limiting
- [ ] Set up monitoring and logging
- [ ] Cache common queries
- [ ] Implement authentication
- [ ] Add input sanitization
- [ ] Set token limits to control costs
- [ ] Create fallback for API failures
- [ ] Add response time tracking
- [ ] Implement A/B testing framework

---

## Next Steps

Ready for **Hour 4: Evaluation**? Learn how to systematically measure and improve your RAG system!

---

**Need help?** Check the demo code or ask the instructor!
