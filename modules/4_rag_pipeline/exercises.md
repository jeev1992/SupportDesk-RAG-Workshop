# Module 4 Exercises: RAG Pipeline

Complete these exercises after studying `demo.py`. Solutions are in `solutions.py`.

> ✅ **Exercise style for this workshop:** keep each solution to a **small edit** (usually 3–15 lines) in existing files. Avoid creating new modules.

---

## Easy Exercises (Start Here!)

### Exercise 1: Modify the Prompt Template

**Task**: Change the prompt template and observe how it affects answers.

**In demo.py, find this prompt template (around line 155) and try these variations:**

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

**Version C: Bullet point format**
```python
template = """Answer using only the context. Format as bullet points with ticket citations.

Context: {context}

Question: {question}

Answer (bullet points with sources):"""
```

**Test with**: `"How do I fix authentication issues?"`

**Questions to answer**:
- Which prompt gives the most useful answers?
- Which format is easiest to read?
- Does any version hallucinate more?

---

### Exercise 2: Adjust Retrieval Parameters

**Task**: Change the number of retrieved documents and search type.

**In demo.py, find the retriever setup (around line 120) and try:**

```python
# Change k from 3 to different values
retriever = vector_store.as_retriever(search_kwargs={"k": 1})   # Very focused
retriever = vector_store.as_retriever(search_kwargs={"k": 5})   # Broader
retriever = vector_store.as_retriever(search_kwargs={"k": 10})  # Maximum context

# Try MMR for diverse results
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
```

**Test queries**:
- "Payment processing failures"
- "Mobile app crashes"
- "Slow dashboard loading"

**Questions**:
- How does k affect answer quality?
- When is MMR better than similarity?

---

### Exercise 3: Implement Citation Formatting

**Task**: Make the assistant always include inline citations.

**Modify the prompt to require citations:**
```python
citation_prompt = """Answer the question using the context. Include inline citations [TICK-XXX] after each fact.

Example format:
"Database connection timeouts occur when the pool is undersized [TICK-002]. Increase max_connections [TICK-002]."

Context:
{context}

Question: {question}

Answer with inline citations:"""
```

**Goal output**:
```
Authentication failures after password reset are caused by stale session tokens [TICK-001]. 
The solution is to clear all active sessions and force re-authentication [TICK-001].
```

---

### Exercise 4: Build a Fallback System

**Task**: Tune fallback behavior with a tiny patch (no new function).

**In `demo.py`, update the existing validation path by changing only:**
1. `min_similarity_score` (try `0.5`, then `0.7`)
2. The fallback message text to be clearer for users

```python
# Existing function already present in demo.py
def rag_with_validation(query, retriever, llm, min_similarity_score=0.5):
    ...
```

**Goal:** Observe how stricter thresholds reduce risky answers.

**Test with**:
- High confidence: "authentication problems"
- Medium confidence: "system performance"
- Low confidence: "how to bake cookies"

---

## Medium Exercises

### Exercise 5: Compare Chain Types

**Task**: Compare strategies with very small edits (do not implement full map-reduce).

```python
# Keep chain structure unchanged.
# Only edit retriever config in place and compare outputs:

# Run 1: similarity
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Run 2: mmr
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
```

**Compare for query `"How do I fix database timeouts?"`**:
- Speed
- Answer quality  
- Best use cases

Keep this exercise to retriever-line changes only.

---

### Exercise 6: Add Metadata Filtering

**Task**: Filter results by ticket category or priority.

```python
# Without filter - returns any category
docs = vector_store.similarity_search(query, k=3)

# With category filter
docs = vector_store.similarity_search(
    query, 
    k=3,
    filter={"category": "Authentication"}
)

# With priority filter
docs = vector_store.similarity_search(
    query, 
    k=3,
    filter={"priority": "High"}
)
```

**Test query**: "system problem"
- Compare results with and without filters

---

### Exercise 7: Add Streaming Responses

**Task**: Stream responses word-by-word for better UX.

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Create streaming LLM
streaming_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# Use in your chain
chain = your_prompt | streaming_llm | StrOutputParser()
result = chain.invoke(query)  # Will print token by token!
```

---

### Exercise 8: Multi-Turn Conversation

**Task**: Add message history so the assistant remembers previous questions.

Wire the retrieval step *inside* the chain using `RunnablePassthrough.assign()` so that
the chain receives a single dict and produces everything the prompt needs without any
manual pre-processing outside the chain.

```python
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

# Store chat history
chat_history = []

# Create conversational prompt
# Note: variable_name must match the key in the input dict
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are SupportDesk AI. Answer using ONLY the context below.
Always cite ticket IDs. If the answer isn't in the context, say "I don't have that information."

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# TODO: Build the chain using RunnablePassthrough.assign() so that:
#   - "question" and "chat_history" are passed through unchanged from the input dict
#   - "context" is computed by extracting "question" from the dict, running it
#     through the retriever, and formatting the resulting documents
#
# Hint: itemgetter("question") | retriever | format_docs
conv_chain = (
    RunnablePassthrough.assign(
        context=# YOUR CODE HERE
    )
    | conv_prompt
    | llm
    | StrOutputParser()
)

def ask_with_history(question, history):
    answer = conv_chain.invoke({"question": question, "chat_history": history})
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    return answer

# Test — "How do I fix it?" should remember we were talking about auth failures
result1 = ask_with_history("What causes authentication failures?", chat_history)
result2 = ask_with_history("How do I fix it?", chat_history)  # "it" = auth failures
```

---

## Bonus Challenge

### Bonus: Hallucination Detection

**Task**: Add one guardrail with a tiny edit.

```python
def detect_hallucination(answer, source_documents, llm):
    """Use LLM-as-judge to verify grounding"""
    source_text = "\n\n".join([doc.page_content for doc in source_documents])
    
    prompt = f"""You are a fact-checker. Determine if the answer is fully grounded in sources.

SOURCE DOCUMENTS:
{source_text}

ANSWER TO CHECK:
{answer}

Is every claim supported by the sources?
Respond: "GROUNDED" or "HALLUCINATION" with brief explanation.

Response:"""
    
    verdict = llm.invoke(prompt).content
    return verdict
```

**Small-change requirement:**
- Add only one extra line that forces a conservative output when no source documents are passed.
- Keep the rest of the function unchanged.

**Test**: Run on one grounded answer and one intentionally unsupported answer.

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

---

## Next Steps

Ready for **Module 5: Evaluation**? Learn how to systematically measure and improve your RAG system!

---

**Need help?** Check `solutions.py` or ask the instructor!
