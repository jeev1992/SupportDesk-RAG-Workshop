# LangChain Component Stack Guide (Module 4)

This guide teaches the LangChain component stack **top-down**:

1. See the full chain and watch it run.
2. See a conversational chain and watch it run.
3. Understand why every component can `.invoke()` — the Runnable contract.
4. Learn each component one by one.

---

## 1) The Simple RAG Chain — Definition AND Invocation

### 1.1 The prompt template

Before building the chain, define the prompt. This is the instruction sheet the LLM will see every time:

```python
from langchain_core.prompts import ChatPromptTemplate

prompt_template = """You are SupportDesk AI, a technical support assistant that helps engineers troubleshoot issues using historical support ticket data.

CRITICAL RULES:
1. Answer using ONLY information from the provided context.
2. If the question is broad or underspecified, provide the best matching known issue(s) from context and state any assumptions.
3. If context is partially relevant, still provide the most likely troubleshooting guidance from relevant tickets.
4. If the answer is truly not present in context, say "I don't have enough information in the ticket history to answer that question."
5. DO NOT make up information or use external knowledge.
6. Always cite ticket IDs for every issue/resolution you mention.
7. If multiple tickets are relevant, summarize each briefly.

Context from support tickets:
{context}

Question: {question}

Helpful Answer (with ticket citations):"""

prompt = ChatPromptTemplate.from_template(prompt_template)
```

Key observations:
- Uses `from_template()` — a single string with `{context}` and `{question}` placeholders.
- All grounding rules live in the template itself.
- The chain will fill `{context}` from retrieval and `{question}` from the user input.

### 1.2 Define the chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

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

### 1.3 Invoke the chain

```python
answer = chain.invoke("How do I fix authentication failures after password reset?")
print(answer)
# -> "Authentication failures after password reset are commonly caused by
#     stale session tokens. Clear active sessions and force
#     re-authentication (TICK-001, TICK-011)."
```

That single `.invoke()` call triggers **every** step internally:

```text
"How do I fix auth failures?"          ← you pass a str
  ↓
retriever.invoke(question)             ← returns List[Document]
  ↓
format_docs(docs)                      ← returns context str
  ↓
RunnablePassthrough()                  ← keeps original question str unchanged
  ↓
prompt.invoke({"context": ...,         ← returns ChatPromptValue
               "question": ...})
  ↓
llm.invoke(prompt_value)               ← returns AIMessage
  ↓
StrOutputParser().invoke(ai_msg)       ← returns plain str
```

When chains break, it is almost always because one step produced a type the next step did not expect. Keep this type flow pinned in your mind.

---

## 2) The Conversational RAG Chain — With Chat History

Real products rarely have single-turn conversations. The user says "What is TICK-001 about?" and then follows up with "How was it resolved?". The second question needs the first turn to make sense.

### 2.1 The conversational prompt template

The conversational prompt is different from the simple one in two important ways:
1. It uses **separate messages** (`system`, `human`) instead of a single string.
2. It includes a `MessagesPlaceholder` for chat history so the LLM can see previous turns.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

conv_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are SupportDesk AI. Answer using the ticket context below and the chat history.
Use chat history to resolve references like "that issue" or "that ticket".
For factual claims, prioritize the retrieved context.
If information is not available in context or history, say "I don't have that information."
Always cite ticket IDs when available.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
```

Compare the two prompt styles:

| | Simple RAG prompt | Conversational RAG prompt |
|---|---|---|
| **Created with** | `from_template()` (single string) | `from_messages()` (list of role-tagged messages) |
| **Placeholders** | `{context}`, `{question}` | `{context}`, `{question}`, `MessagesPlaceholder("chat_history")` |
| **Chat history** | Not supported | Injected between system and human messages |
| **Use when** | Single-turn Q&A | Multi-turn conversations |

### 2.2 Define the chain

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

conv_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | conv_prompt
    | llm
    | StrOutputParser()
)
```

### 2.3 Invoke the chain with a helper

```python
from langchain_core.messages import HumanMessage, AIMessage

def ask_with_history(question, history):
    """Ask a question and automatically append the turn to history."""
    answer = conv_chain.invoke({"question": question, "chat_history": history})
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    return answer
```

### 2.4 Watch it run (multi-turn)

```python
history = []

# Turn 1
print(ask_with_history("What is TICK-001 about?", history))
# -> "TICK-001 describes users being unable to log in after a password reset
#     due to stale session tokens."

# Turn 2 – follow-up; "it" refers to TICK-001
print(ask_with_history("How was it resolved?", history))
# -> "It was resolved by clearing all active sessions and forcing
#     re-authentication on next login."
```

Notice:
- The input is now a **dict** (`{"question": ..., "chat_history": ...}`), not a plain string.
- The chain must keep `question` and `chat_history` intact while **adding** `context`.

This is exactly the problem `RunnablePassthrough.assign()` solves.

---

## 3) Why `RunnablePassthrough` and `.assign()` Exist

### 3.1 Simple chain → `RunnablePassthrough()` for fan-out

In the simple chain the input is a single string. The prompt needs **two** keys: `context` and `question`. So we fan out:

```python
{
    "context": retriever | format_docs,   # transforms the string into context
    "question": RunnablePassthrough()     # keeps the original string untouched
}
```

Both branches receive the same input string. `RunnablePassthrough` simply passes it through — no transformation.

Without it, the prompt would receive `context` but not the raw `question`.

### 3.2 Conversational chain → `.assign()` for adding keys

When the input is already a dict (`{"question": ..., "chat_history": ...}`), we do **not** want to throw away `chat_history`. `.assign()` merges a new key into the existing dict:

```python
RunnablePassthrough.assign(
    context=itemgetter("question") | retriever | format_docs
)
```

Step by step:

```text
Input dict
  {"question": "How was it resolved?", "chat_history": [...]}
      ↓
itemgetter("question")          → "How was it resolved?"
      ↓
retriever.invoke(...)           → [Document(...), ...]
      ↓
format_docs(docs)               → "[SOURCE 1: TICK-001]\n..."
      ↓
.assign(context=...)            → {
                                      "question": "How was it resolved?",
                                      "chat_history": [...],
                                      "context": "[SOURCE 1: TICK-001]\n..."
                                   }
```

All three keys now flow into the prompt template.

### 3.3 When to use which?

| Input shape | Pattern | Why |
|---|---|---|
| Single string | `{"key": RunnablePassthrough()}` | Need to duplicate input into multiple prompt keys |
| Dict with existing keys | `RunnablePassthrough.assign(new_key=...)` | Need to add a computed key without losing existing keys |

---

## 4) The Runnable Interface — One Contract, Many Components

Here is the key insight: **every** building block in the chains above — the retriever, `format_docs`, the prompt, the LLM, the output parser — implements the same **Runnable** interface.

That interface guarantees four methods:

```python
component.invoke(input)       # single input  → single output
component.batch([inp1, inp2]) # list of inputs → list of outputs
component.stream(input)       # single input  → output token by token
component.ainvoke(input)      # async version of invoke
```

This is why the `|` pipe works — every component speaks the same protocol. LangChain just calls `.invoke()` on the left piece, takes the output, and feeds it as input to `.invoke()` on the right piece.

### Prove it yourself

```python
# Each component is independently invocable:

docs = retriever.invoke("auth failures after reset")
# -> [Document(TICK-001), Document(TICK-011), Document(TICK-014)]

context_str = format_docs(docs)
# -> "[SOURCE 1: TICK-001]\nUsers cannot log in after..."

prompt_value = prompt.invoke({"context": context_str, "question": "How do I fix auth failures?"})
# -> ChatPromptValue(messages=[SystemMessage(...), HumanMessage(...)])

ai_msg = llm.invoke(prompt_value)
# -> AIMessage(content="Authentication failures after password reset...")

answer = StrOutputParser().invoke(ai_msg)
# -> "Authentication failures after password reset..."
```

The full chain just automates calling these five `.invoke()` calls in sequence.

### Why this matters

- **Debugging**: test any component alone — if its output looks wrong, the bug is there.
- **Swapping**: replace `ChatOpenAI` with `ChatAnthropic` — same `.invoke()` contract.
- **Composition**: stack components with `|` — they are guaranteed to be compatible.

---

## 5) LCEL (`|`) as a Mental Model

LCEL is LangChain Expression Language. The pipe `|` means: *output of the left step becomes input of the right step*.

```python
result = (step_a | step_b | step_c).invoke(input)

# is the same as:
temp1 = step_a.invoke(input)
temp2 = step_b.invoke(temp1)
result = step_c.invoke(temp2)
```

The pipe version is cleaner to read left-to-right and also gives you `.batch()`, `.stream()`, and `.ainvoke()` for free on the whole chain.

---

## 6) Components, One by One

Now that you have seen the full chains running, here is every component in detail.

### 6.1 `Document` — the atomic unit

`Document` keeps text and metadata together.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Users cannot log in after password reset. Resolution: clear sessions.",
    metadata={"ticket_id": "TICK-001", "category": "authentication", "priority": "high"}
)
```

- The **model** reads `page_content`.
- Your **app** uses `metadata` for citations and filtering.

---

### 6.2 `RecursiveCharacterTextSplitter` — chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)
```

Simple intuition:
- Big chunks → more context, lower precision.
- Small chunks → better precision, may lose context.
- Overlap preserves information at chunk boundaries.

Quick experiment:
- Try `chunk_size=300` vs `chunk_size=800` and compare retrieval quality on the same question.

---

### 6.3 `OpenAIEmbeddings` — semantic vectors

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```

What this solves:
- "login issue after reset" and "auth failure after password change" **match**, even though the words are different.

Implements Runnable: `embeddings.invoke("some text")` → vector (list of floats).

---

### 6.4 `Chroma` — store vectors and search nearest neighbors

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
- **Indexing** is done once (offline).
- **Retrieval** happens per question (online).

---

### 6.5 `.as_retriever()` — chain-friendly retrieval interface

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
```

Now you can use `retriever.invoke(question)` and get `List[Document]`.

Useful variants:

| `search_type` | Behavior |
|---|---|
| `"similarity"` | Closest chunks by cosine distance |
| `"mmr"` | Relevance + diversity (reduces redundancy) |
| `"similarity_score_threshold"` | Returns nothing if confidence is below threshold |

---

### 6.6 `format_docs` — convert `List[Document]` to one context string

Basic version:

```python
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
```

Citation-aware version (recommended):

```python
def format_docs(docs):
    blocks = []
    for i, doc in enumerate(docs, 1):
        ticket_id = doc.metadata.get("ticket_id", f"DOC-{i}")
        blocks.append(f"[SOURCE {i}: {ticket_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)
```

This makes grounded answers easier to verify.

Note: `format_docs` is a plain Python function, not a LangChain class. When you use it inside an LCEL pipe (`retriever | format_docs`), LangChain automatically wraps it into a `RunnableLambda`, which gives it the full Runnable interface (`.invoke()`, `.batch()`, `.stream()`, `.ainvoke()`). So even plain functions become Runnables the moment they enter a pipe.

Implements Runnable (via auto-wrapping): `RunnableLambda(format_docs).invoke(docs)` → context `str`.

---

### 6.7 `ChatPromptTemplate` — build a clean, controlled prompt

You already saw both prompt templates in Sections 1 and 2. Here is a summary of the two construction styles:

**`from_template(string)`** — single string with `{placeholder}` variables. Good for simple, single-turn prompts.

```python
prompt = ChatPromptTemplate.from_template(prompt_template)
# prompt_template is the long string defined in Section 1.1
```

**`from_messages(list)`** — list of role-tagged messages. Required when you need separate system / human roles or chat history.

```python
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "...rules + {context}..."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
# Full version shown in Section 2.1
```

Rule of thumb:
- Grounding rules and policies go in `system`.
- Past turns go in `MessagesPlaceholder`.
- Current question goes in `human`.

Implements Runnable: `prompt.invoke({"context": ..., "question": ...})` → `ChatPromptValue`.

---

### 6.8 `ChatOpenAI` — call the model

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

- `temperature=0` is recommended for factual / support workflows.

Implements Runnable: `llm.invoke(prompt_value)` → `AIMessage`.

---

### 6.9 `StrOutputParser` / `JsonOutputParser` — normalize model output

The LLM returns an `AIMessage` object, not a plain string. Output parsers extract the part you actually need.

#### `StrOutputParser` — extract plain text

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage

parser = StrOutputParser()

# What the LLM actually returns:
ai_msg = AIMessage(content="Clear sessions and force re-auth (TICK-001).")
print(type(ai_msg))      # <class 'langchain_core.messages.ai.AIMessage'>
print(ai_msg.content)    # "Clear sessions and force re-auth (TICK-001)."

# What the parser does — pulls out .content as a plain str:
answer = parser.invoke(ai_msg)
print(type(answer))      # <class 'str'>
print(answer)            # "Clear sessions and force re-auth (TICK-001)."
```

Without the parser, your chain would return an `AIMessage` object instead of a string — which breaks downstream code that expects `str`.

#### `JsonOutputParser` — extract structured data

Use this when you need the model to return a predictable structure (e.g., answer + confidence + sources).

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Step 1: Define the schema you want
class TicketAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    confidence: int = Field(description="Confidence score 0-100")
    sources: list[str] = Field(description="List of ticket IDs used")

# Step 2: Create the parser
json_parser = JsonOutputParser(pydantic_object=TicketAnswer)

# Step 3: Get format instructions (tells the model what JSON to produce)
print(json_parser.get_format_instructions())
# -> "Return a JSON object with keys: answer (str), confidence (int), sources (list[str])..."
```

#### Using `JsonOutputParser` in a chain

The key: inject `format_instructions` into your prompt so the model knows what JSON shape to produce.

```python
from langchain_core.prompts import ChatPromptTemplate

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer using ONLY the context. Cite sources.

Context:
{context}

{format_instructions}"""),
    ("human", "{question}")
])

json_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "format_instructions": lambda _: json_parser.get_format_instructions()
    }
    | json_prompt
    | llm
    | json_parser
)

result = json_chain.invoke("How do I fix auth failures after password reset?")
print(type(result))  # <class 'dict'>
print(result)
# -> {
#      "answer": "Authentication failures after password reset are caused by stale session tokens. Clear active sessions and force re-authentication.",
#      "confidence": 92,
#      "sources": ["TICK-001", "TICK-011"]
#    }
```

#### When to use which?

| Parser | Output type | Use when |
|---|---|---|
| `StrOutputParser` | `str` | You just need the text answer (most common) |
| `JsonOutputParser` | `dict` | You need structured fields like confidence, sources, categories |

Both implement Runnable: `parser.invoke(ai_message)` → `str` or `dict`.

---

## 7) Memory: `RunnableWithMessageHistory` (modern approach)

For production multi-turn conversations, use session-based history instead of manually appending to a list.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_history(session_id: str):
    return store.setdefault(session_id, InMemoryChatMessageHistory())

chain_with_history = RunnableWithMessageHistory(
    conv_chain,
    get_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)
```

Now history is managed automatically per session:

```python
config = {"configurable": {"session_id": "user-42"}}
chain_with_history.invoke({"question": "What is TICK-001?"}, config=config)
chain_with_history.invoke({"question": "How was it resolved?"}, config=config)
```

This is cleaner than the manual `ask_with_history` helper — but the manual version is easier to understand first.

---

## 8) End-to-End Walkthrough (input → output at every step)

### Input

```python
question = "How do I fix authentication failures after password reset?"
```

### Step 1 — Retrieval

```python
docs = retriever.invoke(question)
# -> [Document(TICK-001), Document(TICK-011), Document(TICK-014)]
```

### Step 2 — Format context

```python
context = format_docs(docs)
# -> "[SOURCE 1: TICK-001]\nUsers cannot log in after password reset..."
#    "\n\n---\n\n"
#    "[SOURCE 2: TICK-011]\nToken invalidation issue..."
```

### Step 3 — Build prompt

```python
prompt_value = prompt.invoke({"context": context, "question": question})
# -> ChatPromptValue(messages=[
#        SystemMessage("Answer using ONLY the context..."),
#        HumanMessage("How do I fix authentication failures...")
#    ])
```

### Step 4 — Call LLM

```python
ai_msg = llm.invoke(prompt_value)
# -> AIMessage(content="Authentication failures after password reset
#     are commonly caused by stale session tokens. Clear active sessions
#     and force re-authentication (TICK-001, TICK-011).")
```

### Step 5 — Parse output

```python
answer = StrOutputParser().invoke(ai_msg)
# -> "Authentication failures after password reset are commonly caused by..."
```

The full LCEL chain automates all five steps in a single `.invoke()` call.

---

## 9) The Stack in Four Layers (Summary)

| Layer | Job | Components |
|---|---|---|
| **Data** | Hold text + metadata | `Document`, text splitters |
| **Embedding + Storage** | Semantic retrieval | `OpenAIEmbeddings`, `Chroma`, retriever |
| **Prompt + LLM** | Grounded answer generation | `ChatPromptTemplate`, `ChatOpenAI` |
| **Orchestration** | Connect everything | LCEL `\|`, `RunnablePassthrough`, output parsers |

One formula: **RAG = Retrieve + Prompt + Generate + Parse**.

---

## 10) Common Mistakes (and fixes)

| # | Mistake | Symptom | Fix |
|---|---|---|---|
| 1 | Pass `List[Document]` directly into prompt | Template errors or garbled prompt | Always convert with `format_docs` first |
| 2 | Lose question during fan-out | Prompt missing `{question}` value | Add `"question": RunnablePassthrough()` |
| 3 | No grounding rules in system prompt | Confident hallucinations | Add "Answer ONLY from context" + citation requirement |
| 4 | High temperature for factual workflows | Inconsistent answers | Set `temperature=0` |
| 5 | Wrong chunk settings | Low retrieval quality | Tune `chunk_size`, `chunk_overlap`, and `k` together |
| 6 | Use `RunnablePassthrough()` when input is already a dict | Lost keys (`chat_history` disappears) | Use `RunnablePassthrough.assign()` instead |

---

## 11) Practice Exercises

### Exercise A: `k` tuning
Run the same query with `k=1`, `k=3`, `k=5`. Compare precision and completeness.

### Exercise B: source-aware formatting
Add `[SOURCE n: ticket_id]` labels in `format_docs` and check if citations improve.

### Exercise C: parser swap
Replace `StrOutputParser` with `JsonOutputParser` and ask model for JSON response.

### Exercise D: conversation memory
Use `ask_with_history` and test follow-up:
1. "What is TICK-001 about?"
2. "How was it resolved?"

Then refactor to use `RunnableWithMessageHistory` and compare.

### Exercise E: step-by-step debugging
Call `.invoke()` on each component individually (`retriever`, `format_docs`, `prompt`, `llm`, `parser`) and print the output. Verify that each output type matches what the next component expects.

---

## 12) Quick Recap Card

1. **Start from the full chain** — see it run, then zoom in.
2. **Every component is a Runnable** — it has `.invoke()`, `.batch()`, `.stream()`, `.ainvoke()`.
3. **`|` pipe** chains Runnables: left output → right input.
4. **`RunnablePassthrough()`** — keeps the input unchanged (use for fan-out from a string).
5. **`RunnablePassthrough.assign()`** — adds a new key to an existing dict (use for conversational chains).
6. **Track types across steps** — when a chain breaks, check what type each step produced.
7. **Retrieval quality drives answer quality** — tune `chunk_size`, `k`, and `search_type`.
8. **Prompt design controls hallucination risk** — be strict in system message.

If these eight points are clear, LangChain will feel much simpler.
