# LangChain Component Stack Guide (Module 4)

This guide explains the LangChain component stack in a clear, step-by-step format.

Read this document in the following order:
1. Start with one full chain.
2. See what goes in and what comes out at every step.
3. Use each building block only when it is needed.

---

## 1) The One Chain to Remember First

```python
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

### Plain-English meaning
You ask a question, LangChain finds relevant ticket text, adds that text into a prompt, asks the model, and returns a clean string answer.

### Type flow (super important)

```text
input question (str)
  -> retriever: List[Document]
  -> format_docs: context (str)
  -> prompt: ChatPromptValue
  -> llm: AIMessage
  -> StrOutputParser: final answer (str)
```

When chains break, it is usually because one step produced the wrong type.

---

## 2) Running Example (we will reuse this everywhere)

### User question
`"How do I fix authentication failures after password reset?"`

### Retrieved documents (example)
- TICK-001: stale sessions after password reset
- TICK-011: token invalidation issue
- TICK-014: forced re-authentication fix

### Final answer (expected style)
"Authentication failures after password reset are commonly caused by stale session tokens. Clear active sessions and force re-authentication (TICK-001, TICK-011)."

Keep this exact example in mind as you read each concept below.

---

## 3) The Stack in Four Layers

| Layer | Job | Typical Components |
|---|---|---|
| Data | Hold text + metadata | `Document`, text splitters |
| Embedding + Storage | Semantic retrieval | `OpenAIEmbeddings`, `Chroma`, retriever |
| Prompt + LLM | Grounded answer generation | `ChatPromptTemplate`, `ChatOpenAI` |
| Orchestration | Connect everything | LCEL `|`, `RunnablePassthrough`, output parsers |

If you remember one formula, remember this:
**RAG = Retrieve + Prompt + Generate + Parse**.

---

## 4) Components, Step by Step (with mini examples)

### 4.1 `Document` — the atomic unit

`Document` keeps the text and metadata together.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Users cannot log in after password reset. Resolution: clear sessions.",
    metadata={"ticket_id": "TICK-001", "category": "authentication", "priority": "high"}
)
```

Why this matters:
- Model reads `page_content`
- Your app uses `metadata` for citations/filtering

---

### 4.2 `RecursiveCharacterTextSplitter` — chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

Simple intuition:
- Big chunks -> more context, lower precision
- Small chunks -> better precision, may lose context
- Overlap helps preserve boundary information

Quick experiment:
- try `chunk_size=300`, then `chunk_size=800`
- compare retrieval quality on same question

---

### 4.3 `OpenAIEmbeddings` — semantic vectors

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

What this solves:
- "login issue after reset" and "auth failure after password change" can match, even with different words.

---

### 4.4 `Chroma` — store vectors and search nearest neighbors

```python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="supportdesk_rag",
    persist_directory="./vectorstore",
    collection_metadata={"hnsw:space": "cosine"}
)
```

Note:
- Indexing is done once (offline)
- Retrieval happens per question (online)

---

### 4.5 `.as_retriever()` — chain-friendly retrieval interface

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

Useful variants:
- `similarity`: closest chunks
- `mmr`: relevance + diversity
- `similarity_score_threshold`: return nothing if not confident

---

### 4.6 `format_docs` — convert `List[Document]` to one context string

```python
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
```

Better version for citations:

```python
def format_docs(docs):
    blocks = []
    for i, doc in enumerate(docs, 1):
        ticket_id = doc.metadata.get("ticket_id", f"DOC-{i}")
        blocks.append(f"[SOURCE {i}: {ticket_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)
```

This makes grounded answers easier to verify.

---

### 4.7 `ChatPromptTemplate` — build a clean, controlled prompt

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the context. Cite sources.\n\nContext:\n{context}"),
    ("human", "{question}")
])
```

Rule of thumb:
- Put policies in `system`
- Put user question in `human`

---

### 4.8 `ChatOpenAI` — call the model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

Tip:
- `temperature=0` for factual/support workflows

---

### 4.9 `StrOutputParser` / `JsonOutputParser` — normalize model output

```python
from langchain_core.output_parsers import StrOutputParser

# AIMessage -> str
```

Use `JsonOutputParser` when you need structured output like:

```json
{"answer": "...", "confidence": 87, "sources": ["TICK-001"]}
```

---

## 5) LCEL (`|`) as a Mental Model

LCEL pipe means: output of left step becomes input of right step.

```python
result = (step_a | step_b | step_c).invoke(input)
```

Why LCEL works well:
- easy to read left-to-right
- easy to test each step independently
- supports sync/async/batch/stream consistently

---

## 6) Why `RunnablePassthrough` Exists

In this pattern:

```python
{
  "context": retriever | format_docs,
  "question": RunnablePassthrough()
}
```

- `context` branch transforms input question into retrieved context
- `question` branch keeps original question unchanged

Without passthrough, prompt may not receive the raw question.

---

## 7) Runnable Interface (one contract, many components)

Most LangChain components support:
- `.invoke(input)`
- `.batch(inputs)`
- `.stream(input)`
- `.ainvoke(input)`

Example:

```python
answer = chain.invoke("How do I fix auth failures?")
answers = chain.batch(["Q1", "Q2", "Q3"])

for token in chain.stream("What caused the outage?"):
    print(token, end="")
```

---

## 8) Conversational RAG: `.assign(...)` + `itemgetter`

When input is already a dict (e.g., `question` + `chat_history`), use `.assign(...)`:

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

conv_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | conv_prompt
    | llm
    | StrOutputParser()
)
```

Why this is safer:
- keeps existing keys (`question`, `chat_history`)
- adds `context` without losing other inputs

---

## 9) Memory: `RunnableWithMessageHistory` (modern approach)

Use session-based history instead of manually appending lists.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}
def get_history(session_id: str):
    return store.setdefault(session_id, InMemoryChatMessageHistory())

chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="history"
)
```

This is cleaner for multi-turn demos and production code.

---

## 10) End-to-End Walkthrough (input/output snapshot)

### Input

```python
question = "How do I fix authentication failures after password reset?"
```

### Retrieval output (example)

```python
docs = retriever.invoke(question)
# -> [Document(TICK-001), Document(TICK-011), Document(TICK-014)]
```

### Context formatting output

```python
context = format_docs(docs)
# -> "[SOURCE 1: TICK-001]\n...\n\n---\n\n[SOURCE 2: TICK-011]\n..."
```

### Prompt payload

```python
payload = {"context": context, "question": question}
```

### Final result

```python
answer = chain.invoke(question)
# -> "Authentication failures after password reset are commonly caused by..."
```

This is exactly what the full LCEL chain automates.

---

## 11) Common Mistakes (and fixes)

### Mistake 1: Passing `List[Document]` directly into prompt
Symptom: template errors or weird prompt rendering.

Fix: always convert with `format_docs` first.

### Mistake 2: Losing question during fan-out
Symptom: prompt missing `{question}`.

Fix: include `"question": RunnablePassthrough()`.

### Mistake 3: No grounding rules
Symptom: confident hallucinations.

Fix: strict system prompt + citation requirement.

### Mistake 4: High temperature for support workflows
Symptom: inconsistent answers.

Fix: set `temperature=0`.

### Mistake 5: Wrong chunk settings
Symptom: low retrieval quality.

Fix: tune `chunk_size`, `chunk_overlap`, and `k` together.

---

## 12) Practice Exercises

### Exercise A: `k` tuning
Run same query with `k=1`, `k=3`, `k=5`. Compare precision and completeness.

### Exercise B: source-aware formatting
Add `[SOURCE n: ticket_id]` labels in `format_docs` and check if citations improve.

### Exercise C: parser swap
Replace `StrOutputParser` with `JsonOutputParser` and ask model for JSON response.

### Exercise D: conversation memory
Add `RunnableWithMessageHistory` and test follow-up:
- Q1: "What is TICK-001 about?"
- Q2: "How was it resolved?"

---

## 13) Quick Recap Card

- Full chain first, components second.
- Track data types across steps.
- Retrieval quality drives answer quality.
- Prompt design controls hallucination risk.
- LCEL composes runnables cleanly.
- Use passthrough/assign/history patterns for robust real workflows.

If this mental model is clear, LangChain will feel much simpler.
