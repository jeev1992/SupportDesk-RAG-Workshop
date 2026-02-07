# Module 4 Exercises: RAG Pipeline

Complete these exercises after studying `demo.py`. Solutions are in `solutions.py`.

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

**Task**: Handle cases where retrieval confidence is low.

**Add this function to handle fallbacks:**
```python
def smart_rag(query, vector_store, qa_chain):
    """RAG with intelligent fallbacks"""
    # Get docs with scores (lower distance = better match)
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    if not docs_with_scores:
        return "No relevant tickets found."
    
    best_distance = docs_with_scores[0][1]
    
    if best_distance < 0.5:  # Very relevant
        return qa_chain.invoke(query)
    
    elif best_distance < 1.0:  # Somewhat relevant
        ticket_id = docs_with_scores[0][0].metadata['ticket_id']
        return f"Found possibly relevant ticket ({ticket_id}), but confidence is moderate."
    
    else:  # Not relevant
        return "I don't have relevant ticket history for this question."
```

**Test with**:
- High confidence: "authentication problems"
- Medium confidence: "system performance"
- Low confidence: "how to bake cookies"

---

## Medium Exercises

### Exercise 5: Compare Chain Types

**Task**: Implement different retrieval strategies and compare performance.

```python
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. "stuff" strategy - All docs in one prompt (default LCEL pattern)
stuff_prompt = ChatPromptTemplate.from_template(
    "Answer using context:\n\nContext: {context}\n\nQuestion: {question}"
)
stuff_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | stuff_prompt | llm | StrOutputParser()
)

# 2. "map_reduce" strategy - Process each doc, then combine
docs = retriever.invoke(query)
individual_answers = []
for doc in docs:
    single_chain = single_doc_prompt | llm | StrOutputParser()
    individual_answers.append(single_chain.invoke({"doc": doc.page_content}))
combined_result = combine_chain.invoke({"summaries": "\n".join(individual_answers)})
```

**Compare for query `"How do I fix database timeouts?"`**:
- Speed
- Answer quality  
- Token usage
- Best use cases

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
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store chat history
chat_history = []

# Create conversational prompt
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Build chain with history
def ask_with_history(question):
    context = format_docs(retriever.invoke(question))
    chain = conv_prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "history": chat_history, "question": question})
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response

# Test conversation - context carries over
result1 = ask_with_history("What causes authentication failures?")
result2 = ask_with_history("How do I fix it?")  # Remembers auth topic
```

---

## Bonus Challenge

### Bonus: Hallucination Detection

**Task**: Detect when the LLM makes up information not in sources.

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
    
    return llm.invoke(prompt).content
```

**Test**: Run this on your RAG answers and see if it catches any hallucinations.

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
