# Module 4: RAG Pipeline

## Introduction

Retrieval-Augmented Generation (RAG) combines the best of both worlds: the vast knowledge of large language models with the precision of retrieved, factual information. This module covers building a complete end-to-end RAG system.

## What is RAG?

### Definition
RAG is a technique that enhances LLM responses by:
1. **Retrieving** relevant documents from a knowledge base
2. **Augmenting** the prompt with retrieved context
3. **Generating** an answer grounded in those documents

### The Problem RAG Solves

**Pure LLM (without RAG):**
```
User: "What's ticket TICK-001 about?"
LLM: "I don't have access to your ticket system..."
```

**LLM with RAG:**
```
User: "What's ticket TICK-001 about?"
System: [Retrieves TICK-001 from knowledge base]
LLM: "TICK-001 reports users unable to log in after password reset.
      The issue was resolved by clearing active sessions..."
```

### Benefits of RAG

| Benefit | Description |
|---------|-------------|
| **Factual Accuracy** | Grounded in real documents, not hallucinations |
| **Up-to-date** | Knowledge updates without retraining |
| **Traceable** | Cite sources for verification |
| **Cost-effective** | No need to fine-tune massive models |
| **Domain-specific** | Works with your private data |

## RAG Architecture

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                        RAG SYSTEM ARCHITECTURE                      │
└────────────────────────────────────────────────────────────────────┘

OFFLINE PHASE (One-time Setup)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Documents  │      │   Chunking  │      │  Embedding  │
│             │─────▶│             │─────▶│   Model     │
│ tickets.json│      │ Size: 500   │      │ OpenAI-3    │
└─────────────┘      └─────────────┘      └──────┬──────┘
                                                  │
                                                  │ Vectors
                                                  │ [0.23, 0.11, ...]
                                                  ▼
                                          ┌───────────────┐
                                          │ Vector Store  │
                                          │               │
                                          │ ┌───┐ ┌───┐  │
                                          │ │Doc│ │Doc│  │
                                          │ │ 1 │ │ 2 │  │
                                          │ └───┘ └───┘  │
                                          └───────────────┘


ONLINE PHASE (Per Query)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User Query: "How do I fix authentication issues?"
     │
     │ Step 1: Query Embedding
     ▼
┌──────────────────┐
│ Embedding Model  │ 
│                  │ → Query Vector: [0.45, 0.22, 0.18, ...]
└────────┬─────────┘
         │
         │ Step 2: Similarity Search
         ▼
┌──────────────────────────────────────────┐
│         Vector Store Retrieval           │
│                                          │
│  Cosine Similarity:                      │
│  ┌──────┐  Score: 0.92  ✓ Top-K        │
│  │Doc 1 │ ───────────────────────────▶  │
│  └──────┘                                │
│  ┌──────┐  Score: 0.87  ✓ Top-K        │
│  │Doc 2 │ ───────────────────────────▶  │
│  └──────┘                                │
│  ┌──────┐  Score: 0.45  ✗ Below thresh │
│  │Doc 3 │                               │
│  └──────┘                                │
└────────┬─────────────────────────────────┘
         │
         │ Step 3: Retrieved Documents (Top-3)
         ▼
┌──────────────────────────────────────────┐
│        Prompt Augmentation               │
│                                          │
│  Template:                               │
│  ┌────────────────────────────────────┐ │
│  │ Context: [Doc 1, Doc 2, Doc 3]    │ │
│  │                                    │ │
│  │ Question: {user_query}             │ │
│  │                                    │ │
│  │ Instructions: Answer based on     │ │
│  │ context only...                    │ │
│  └────────────────────────────────────┘ │
└────────┬─────────────────────────────────┘
         │
         │ Step 4: Augmented Prompt
         ▼
┌──────────────────────────────────────────┐
│      Large Language Model (LLM)          │
│                                          │
│         GPT-4 / Claude / Llama           │
│                                          │
│  Input: Prompt with context              │
│  Output: Generated answer                │
└────────┬─────────────────────────────────┘
         │
         │ Step 5: Response + Sources
         ▼
┌──────────────────────────────────────────┐
│        Post-Processing                   │
│                                          │
│  • Format answer                         │
│  • Add source citations                  │
│  • Include confidence scores             │
│  • Highlight key points                  │
└────────┬─────────────────────────────────┘
         │
         ▼
    Final Response to User
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Answer: "To fix authentication issues..."
    Sources: [TICK-001, TICK-011, TICK-014]
    Confidence: High


DATA FLOW SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Documents ──[Chunk]──▶ Chunks ──[Embed]──▶ Vectors ──[Store]──▶ Vector DB
                                                                      │
User Query ──[Embed]──▶ Query Vector ──[Search]──────────────────────┘
                                             │
                                             ▼
                             Retrieved Docs + Query ──[Template]──▶ Prompt
                                                                      │
                                                                      ▼
                                             Answer ◀──[Generate]── LLM
```

### Complete Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    RAG PIPELINE                         │
└─────────────────────────────────────────────────────────┘

1. INDEXING (Offline)
   Documents → Chunking → Embedding → Vector Store
   
2. RETRIEVAL (Online)
   User Query → Embed Query → Similarity Search → Top-K Docs
   
3. AUGMENTATION (Online)
   Query + Retrieved Docs → Prompt Template
   
4. GENERATION (Online)
   Augmented Prompt → LLM → Final Answer
   
5. POST-PROCESSING (Online)
   Answer → Citation → Source Attribution → User
```

### Component Breakdown

#### 1. Document Ingestion
```python
def ingest_documents(file_paths):
    documents = []
    for path in file_paths:
        # Load document
        with open(path) as f:
            content = f.read()
        
        # Create document object
        doc = Document(
            page_content=content,
            metadata={'source': path, 'timestamp': datetime.now()}
        )
        documents.append(doc)
    
    return documents
```

#### 2. Chunking
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
```

#### 3. Embedding
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
```

#### 4. Vector Store
```python
from langchain_community.vectorstores import FAISS

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

#### 5. Retriever
```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

#### 6. Prompt Template
```python
from langchain_core.prompts import ChatPromptTemplate

template = """Answer the question based on the following context:

Context:
{context}

Question: {question}

Answer: Provide a detailed answer based on the context. If the answer 
is not in the context, say "I don't have enough information."
"""

prompt = ChatPromptTemplate.from_template(template)
```

#### 7. LLM
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0  # Deterministic for factual answers
)
```

#### 8. Chain Assembly
```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Use it
answer = chain.invoke("How do I reset my password?")
```

## Building the Pipeline

### Method 1: LCEL (Modern, Recommended)

**LCEL (LangChain Expression Language)** is the modern way to build chains using the pipe operator (`|`).

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Helper to format retrieved documents
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Build the chain using pipe operator
chain = (
    {
        "context": retriever | format_docs,  # Get docs and format them
        "question": RunnablePassthrough()     # Pass question through
    }
    | prompt      # Fill in the prompt template
    | llm         # Generate answer
    | StrOutputParser()  # Extract string from response
)

# Use the chain
answer = chain.invoke("What causes authentication failures?")
print(answer)
```

**Benefits of LCEL:**
- ✅ Composable: Chain components like Unix pipes
- ✅ Type-safe: Better error messages
- ✅ Streaming: Built-in support for streaming responses
- ✅ Async: Native async/await support
- ✅ Flexible: Easy to add custom logic
- ✅ Modern: Actively maintained

**Learn more:** https://python.langchain.com/docs/expression_language/

### Method 2: Legacy Approaches (Deprecated)

⚠️ **Note:** `RetrievalQA` and `ConversationalRetrievalChain` are deprecated in LangChain 0.3+. Use LCEL instead.

**Chain Types (conceptual - now implemented via LCEL):**
- `stuff`: Put all docs in one prompt (default LCEL pattern)
- `map_reduce`: Summarize each doc, then combine (implement via loop + combine)
- `refine`: Iteratively refine answer with each doc (implement via reduce)
- `map_rerank`: Score each doc's answer, return best

### Method 3: Conversation with History (Modern LCEL)

```python
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store chat history
chat_history = []

# Create conversational prompt
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using the context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

conv_chain = conv_prompt | llm | StrOutputParser()

def ask_with_history(question):
    context = format_docs(retriever.invoke(question))
    response = conv_chain.invoke({
        "context": context, "history": chat_history, "question": question
    })
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response

# Multi-turn conversation
ask_with_history("What's ticket TICK-001?")
ask_with_history("How was it resolved?")  # Remembers context
```

## Prompt Engineering for RAG

### Basic Template

```python
template = """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
```

### Advanced Template (Anti-Hallucination)

```python
template = """You are a helpful assistant for a support ticketing system.
Your job is to answer questions based ONLY on the provided context.

RULES:
1. Only use information from the context below
2. If the answer is not in the context, say "I don't have that information"
3. Cite the ticket ID when answering
4. Be concise and direct

Context:
{context}

Question: {question}

Answer (remember to cite sources):"""
```

### Few-Shot Examples

```python
template = """Answer questions based on the context provided.

Example 1:
Context: TICK-001: Users unable to login after password reset...
Question: What's TICK-001 about?
Answer: TICK-001 involves users having trouble logging in after resetting their password.

Example 2:
Context: TICK-005: Memory leak in worker process...
Question: What causes the memory leak?
Answer: According to TICK-005, the memory leak is caused by unclosed database cursors.

Now answer this question:

Context:
{context}

Question: {question}

Answer:"""
```

### Chain-of-Thought

```python
template = """Answer the question by thinking step by step.

Context:
{context}

Question: {question}

Let's solve this step by step:
1. First, identify relevant information from the context
2. Then, analyze how it relates to the question
3. Finally, provide a clear answer with citations

Answer:"""
```

## Retrieval Strategies

### 1. Similarity Search (Default)

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

**Best for:** Most queries

### 2. MMR (Maximal Marginal Relevance)

```python
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,  # Fetch 20, return diverse 5
        "lambda_mult": 0.5  # Balance relevance vs diversity
    }
)
```

**Best for:** Avoiding duplicate information
**lambda**: 0 = max diversity, 1 = max relevance

### 3. Similarity Score Threshold

```python
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.7,  # Only return if similarity > 0.7
        "k": 5
    }
)
```

**Best for:** High-precision applications

### 4. Metadata Filtering

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {"category": "authentication"}
    }
)
```

**Best for:** Scoped searches

### 5. Ensemble Retriever (Hybrid)

```python
from langchain.retrievers import EnsembleRetriever

# Combine multiple retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)
```

**Best for:** Maximum accuracy

## Anti-Hallucination Techniques

### 1. Explicit Grounding Instructions

```python
template = """CRITICAL: Only use information from the context below.
If you cannot find the answer in the context, respond with:
"I cannot find that information in the available documents."

DO NOT use your general knowledge.
DO NOT make assumptions.
DO NOT guess.

Context:
{context}

Question: {question}

Answer:"""
```

### 2. Source Attribution

```python
template = """Answer the question and cite your sources.

Context:
{context}

Question: {question}

Answer format:
[Your answer here]

Sources: [List ticket IDs or document names used]"""
```

### 3. Confidence Scoring

```python
template = """Answer the question and rate your confidence.

Context:
{context}

Question: {question}

Answer: [Your answer]
Confidence: [High/Medium/Low]
Reasoning: [Why this confidence level]"""
```

### 4. Two-Step Verification

```python
def answer_with_verification(question):
    # Step 1: Generate answer
    answer = qa_chain(question)
    
    # Step 2: Verify against sources
    verification_prompt = f"""
    Question: {question}
    Answer: {answer}
    Context: {context}
    
    Is the answer fully supported by the context? (Yes/No)
    If No, what's wrong?
    """
    
    verification = llm(verification_prompt)
    return answer, verification
```

### 5. Hallucination Detection

```python
def detect_hallucination(answer, context):
    prompt = f"""
    Does the answer contain information NOT in the context?
    
    Context: {context}
    Answer: {answer}
    
    Response format:
    Hallucinated: Yes/No
    Details: [What specific claims are not supported]
    """
    
    result = llm(prompt)
    return result
```

## Response Modes (LCEL Implementation)

### Stuff (Simple) - Default LCEL Pattern

```python
# All documents concatenated into one prompt
stuff_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**How it works:** Put all retrieved docs in one prompt
**Best for:** Few documents, short content
**Limitation:** Context window size

### Map-Reduce (LCEL)

```python
# Process each document separately, then combine
docs = retriever.invoke(query)

# Map: Answer for each doc
individual_answers = []
for doc in docs:
    single_prompt = ChatPromptTemplate.from_template("Summarize: {doc}")
    chain = single_prompt | llm | StrOutputParser()
    individual_answers.append(chain.invoke({"doc": doc.page_content}))

# Reduce: Combine answers
combine_prompt = ChatPromptTemplate.from_template(
    "Combine these summaries to answer: {question}\n\n{summaries}"
)
final = combine_prompt | llm | StrOutputParser()
result = final.invoke({"question": query, "summaries": "\n".join(individual_answers)})
```

**How it works:**
1. Map: Answer question for each doc separately
2. Reduce: Combine all answers into final answer

**Best for:** Many documents, need comprehensive answer
**Trade-off:** Multiple LLM calls = slower + more expensive

### Refine (LCEL)

```python
# Iteratively refine answer with each doc
docs = retriever.invoke(query)
refine_prompt = ChatPromptTemplate.from_template(
    "Current answer: {answer}\n\nRefine using: {doc}\n\nQuestion: {question}"
)
refine_chain = refine_prompt | llm | StrOutputParser()

answer = ""  # Start empty
for doc in docs:
    answer = refine_chain.invoke({"answer": answer, "doc": doc.page_content, "question": query})
```

**How it works:**
1. Answer with first doc
2. Refine answer with second doc
3. Continue refining with each doc

**Best for:** Highest quality answers
**Trade-off:** Slowest method

### Map-Rerank (LCEL)

```python
# Score each answer and return best
from langchain_core.output_parsers import JsonOutputParser

docs = retriever.invoke(query)

rank_prompt = ChatPromptTemplate.from_template(
    """Answer the question using ONLY this document.
    Document: {doc}
    Question: {question}
    
    Return JSON: {{"answer": "...", "confidence": 0-100}}"""
)
rank_chain = rank_prompt | llm | JsonOutputParser()

results = []
for doc in docs:
    result = rank_chain.invoke({"doc": doc.page_content, "question": query})
    results.append(result)

# Return highest confidence answer
best = max(results, key=lambda x: x.get("confidence", 0))
```

**How it works:**
1. Answer question for each doc
2. Score each answer's confidence
3. Return highest-scored answer

**Best for:** Diverse sources, need best single answer

## Streaming Responses

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

streaming_llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

streaming_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | streaming_llm
    | StrOutputParser()
)

# Streams to console as generated
streaming_chain.invoke("What causes performance issues?")
```
```

**Custom streaming:**

```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomStreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(f"Token: {token}")
        # Send to WebSocket, UI, etc.

llm = ChatOpenAI(
    streaming=True,
    callbacks=[CustomStreamHandler()]
)
```

## Error Handling

### Handle No Results

```python
def safe_query(question):
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return "I couldn't find any relevant information for your question."
    
    return chain.invoke(question)
```

### Handle API Errors

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10)
)
def query_with_retry(question):
    return chain.invoke(question)

try:
    answer = query_with_retry(question)
except Exception as e:
    answer = f"Sorry, I encountered an error: {str(e)}"
```

### Timeout Protection

```python
import asyncio

async def query_with_timeout(question, timeout=30):
    try:
        return await asyncio.wait_for(
            chain.ainvoke(question),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return "The query took too long. Please try a simpler question."
```

## Performance Optimization

### 1. Cache Embeddings

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(text):
    return embeddings.embed_query(text)
```

### 2. Batch Processing

```python
questions = ["Q1", "Q2", "Q3"]

# Bad: Sequential
for q in questions:
    answers.append(chain.invoke(q))

# Good: Parallel
answers = chain.batch(questions)
```

### 3. Async Operations

```python
async def async_query(question):
    return await chain.ainvoke(question)

# Process multiple queries concurrently
answers = await asyncio.gather(*[
    async_query(q) for q in questions
])
```

### 4. Reduce Retrieved Chunks

```python
# More chunks = better context but slower
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}  # Start with 3, increase if needed
)
```

## Best Practices

### 1. Prompt Design
✅ Be explicit about using only the context
✅ Request source citations
✅ Use few-shot examples
✅ Set the right temperature (0 for factual, 0.7 for creative)

### 2. Retrieval Tuning
✅ Test different k values (3, 5, 7)
✅ Use MMR for diversity
✅ Add metadata filters when possible
✅ Consider hybrid search (vector + keyword)

### 3. Error Handling
✅ Gracefully handle no results
✅ Implement retry logic
✅ Add timeouts
✅ Log errors for debugging

### 4. Monitoring
✅ Track response times
✅ Log retrieved documents
✅ Monitor costs (API calls)
✅ Collect user feedback

## Common Pitfalls

### 1. Context Overflow
❌ Retrieving too many documents
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 20})  # Too many!
```

✅ Start small, increase if needed
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

### 2. Weak Prompts
❌ Vague instructions
```python
prompt = "Answer: {question} Context: {context}"
```

✅ Clear, specific instructions
```python
prompt = "Based ONLY on the context below, answer the question..."
```

### 3. Ignoring Sources
❌ No attribution
```python
return answer
```

✅ Return sources
```python
return {
    'answer': answer,
    'sources': [doc.metadata['source'] for doc in docs]
}
```

### 4. Wrong Temperature
❌ High temperature for facts
```python
llm = ChatOpenAI(temperature=0.9)  # Too creative!
```

✅ Low temperature for facts
```python
llm = ChatOpenAI(temperature=0)  # Deterministic
```

## Testing RAG Systems

```python
def test_rag_pipeline():
    test_cases = [
        {
            'question': "What's TICK-001 about?",
            'expected_keywords': ['password', 'reset', 'login'],
            'expected_source': 'TICK-001'
        }
    ]
    
    for test in test_cases:
        result = qa_chain(test['question'])
        answer = result['result']
        sources = result['source_documents']
        
        # Check keywords present
        assert any(kw in answer.lower() for kw in test['expected_keywords'])
        
        # Check correct source
        assert any(test['expected_source'] in doc.metadata.get('ticket_id', '') 
                   for doc in sources)
        
        print(f"✓ Test passed: {test['question']}")
```

## References

- [LangChain RAG Tutorial](https://python.langchain.com/docs/use_cases/question_answering/)
- [RAG Paper (Facebook AI)](https://arxiv.org/abs/2005.11401)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## Next Steps

Now that you understand RAG pipelines, proceed to **Module 5: Evaluation** to learn how to measure and improve your system's performance.
