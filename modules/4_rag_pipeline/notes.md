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
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="supportdesk_rag",
    persist_directory="./vectorstore",
    collection_metadata={"hnsw:space": "cosine"}  # cosine distance for semantic search
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

### The LangChain Component Stack

LangChain gives you pre-built blocks for each layer. LCEL (the `|` operator) is how you snap them together.

| Layer | Purpose | Classes Used |
|---|---|---|
| **Data Layer** | Load and chunk documents | `Document`, `RecursiveCharacterTextSplitter` |
| **Embedding + Storage** | Convert text to vectors, store and search | `OpenAIEmbeddings`, `Chroma` |
| **Prompt + LLM** | Format context, generate answer | `ChatPromptTemplate`, `ChatOpenAI` |
| **Orchestration** | Wire all pieces into a pipeline | LCEL `|` operator, `RunnablePassthrough`, `StrOutputParser` |

```
Data Layer → Embedding+Storage → Prompt+LLM
                    ↑                  ↑
              (offline index)    (online query)
                    └──── LCEL Orchestration ────┘
```

### LangChain Constructs: How They Fit the RAG Pipeline

Rather than a lookup table, this section tells the story of how each construct earns its place as a support ticket travels from raw JSON to a grounded answer.

---

#### Step 1 — Turning raw data into something LangChain understands: `Document`

Before anything else can happen, your support tickets need to be wrapped in a `Document`. A `Document` is LangChain's universal unit of information — it carries two things: the text the model will actually read (`page_content`) and a dict of structured facts about that text (`metadata`). In our pipeline the metadata holds the ticket ID, category, and priority. This separation matters because when a user asks a question, the model answers using `page_content`, but the application displays citations using `metadata["ticket_id"]`. The two travel together through the entire pipeline without ever getting mixed up.

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Users cannot log in after resetting their password. "
                 "Resolution: clear active sessions and force re-auth.",
    metadata={
        "ticket_id": "TICK-001",
        "category": "authentication",
        "priority": "high"
    }
)
# Later, after retrieval:
print(doc.page_content)          # → text the LLM reads
print(doc.metadata["ticket_id"]) # → "TICK-001" shown as citation
```

#### Step 2 — Breaking large documents into retrievable pieces: `RecursiveCharacterTextSplitter`

Some ticket descriptions are three lines. Others are three paragraphs. The embedding model and the LLM both have size limits, and you want each chunk to represent one coherent idea so that retrieval finds the right piece rather than a grab-bag of unrelated content. `RecursiveCharacterTextSplitter` does this carefully: it tries to cut on paragraph breaks first (`"\n\n"`), then line breaks (`"\n"`), then spaces. This means it will never split a sentence in the middle if it can avoid it — it always tries to preserve the most natural unit of meaning.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # max characters per chunk
    chunk_overlap=50   # overlap so context isn't lost at boundaries
)
chunks = splitter.split_documents(documents)
# A 2,000-char ticket description becomes ~4 overlapping chunks,
# each representing one coherent section of the ticket.
```

#### Step 3 — Giving text a location in meaning-space: `OpenAIEmbeddings`

You cannot search text by meaning using a keyword index. What you need is to project every document into a mathematical space where "authentication failure after password reset" and "login broken following credential change" land near each other — because they describe the same problem even though they share almost no words. `OpenAIEmbeddings` does this by sending each piece of text to OpenAI's embedding model, which returns a list of roughly 1,500 numbers (a vector). The direction and magnitude of that vector encodes semantic meaning. Documents about similar topics produce vectors that point in similar directions.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single query to see what a vector looks like
vector = embeddings.embed_query("authentication failure after password reset")
print(len(vector))   # → 1536 numbers
print(vector[:4])    # → [0.023, -0.117, 0.045, 0.201, ...]
# "login broken after credential change" would produce a vector
# pointing in nearly the same direction as this one.
```

#### Step 4 — Storing and searching those vectors: `Chroma`

Once you have vectors, you need somewhere to store them and a fast way to find the ones closest to a new query vector. `Chroma` is that store. You hand it your documents and an embedding function, it embeds everything and saves both the vectors and the original text to disk. At query time you give it a new question, it embeds the question and runs a nearest-neighbour search across all stored vectors, and it returns the top matching documents with their metadata intact. Because it persists to disk, you only pay the embedding cost once — not every time you restart your application.

The `collection_metadata={"hnsw:space": "cosine"}` argument tells Chroma to use cosine distance as its similarity metric. Cosine distance measures the angle between two vectors regardless of their magnitude — it is the standard choice for semantic text search. The scores returned by `similarity_search_with_score` are cosine distances: `0` means identical, `1` means orthogonal (unrelated), and values above `1` mean the vectors point in opposite directions. Lower is always more similar.

```python
from langchain_chroma import Chroma

# First run: embed all documents and save to disk
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="supportdesk_rag",
    persist_directory="./vectorstore",   # saved here after this call
    collection_metadata={"hnsw:space": "cosine"}  # use cosine distance
)

# Subsequent runs: load from disk — no re-embedding cost
vector_store = Chroma(
    collection_name="supportdesk_rag",
    embedding_function=embeddings,
    persist_directory="./vectorstore"
)
```

#### Step 5 — Making the vector store queryable inside a chain: `.as_retriever()`

`Chroma` by itself is a database. To use it inside an LCEL chain with the `|` operator, it needs to conform to LangChain's `Retriever` interface — a common contract that says "given a string query, return a list of Documents." Calling `.as_retriever()` creates that wrapper. The `search_type` argument is where retrieval strategy lives: `"similarity"` gives you the top-K closest vectors, `"mmr"` adds a diversity penalty so three nearly-identical tickets don't crowd out more varied results, and `"similarity_score_threshold"` lets the retriever return nothing at all if no document is close enough — which is exactly the behaviour you want when you'd rather say "I don't know" than return a barely-relevant result.

```python
# Default: top-3 most similar
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# MMR: top-3 but diverse — avoids returning near-duplicate tickets
mmr_retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7}
)

# Threshold: only return if score > 0.7, otherwise return nothing
strict_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5}
)

# All three respond the same way to .invoke() — that's the interface
docs = retriever.invoke("authentication failure after password reset")
```

#### Step 5b — Bridging the retriever and the prompt: `format_docs`

The retriever returns a `List[Document]`. The prompt template has a `{context}` placeholder that expects a plain string. These two types don't match — you can't pipe a list of objects into a template slot that expects text. `format_docs` is the bridge: a plain Python function (not a LangChain class) that joins the `page_content` of each document into a single formatted string.

It is short enough to write inline, but it is doing something important: it controls exactly what the model sees as its evidence. A naive join that just concatenates everything gives the model no structure. A well-formatted join that labels each source clearly — `[SOURCE 1: TICK-001]\n...` — makes citations easier and reduces the chance the model confuses one ticket's details with another's.

Because LCEL automatically wraps any callable into a `RunnableLambda`, `format_docs` can be used directly in a chain with `|` even though it is just a regular function.

```python
# Minimal version — just joins the text
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Labelled version — tells the model which ticket each section came from
def format_docs(docs):
    sections = []
    for i, doc in enumerate(docs, 1):
        ticket_id = doc.metadata.get("ticket_id", f"Doc {i}")
        sections.append(f"[SOURCE {i}: {ticket_id}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)

# Used in a chain — LCEL wraps format_docs as a RunnableLambda automatically
chain = retriever | format_docs | prompt | llm | StrOutputParser()
#       ↑ returns      ↑ converts to      ↑ receives
#   List[Document]     str (context)      {context: str, question: str}
```

Once each source is tagged with `[SOURCE N]`, you can instruct the model to cite sources by number, and groundedness validation becomes much easier to verify.

#### Step 6 — Telling the LLM how to use the retrieved context: `ChatPromptTemplate`

At this point in the pipeline you have a context string (from `format_docs`) and the original user question. You need to combine them into a complete instruction set for the model — one that tells it what role to play, what the rules are, and where to find the evidence. `ChatPromptTemplate` is a reusable, parameterised template that does exactly this.

##### What a prompt template actually is

A `ChatPromptTemplate` is not a string — it is a **factory for messages**. You define it once with `{placeholder}` slots, and every time the chain executes it fills in those slots with the actual values and produces a fully-formed list of chat messages ready to send to the model. The output type is called `ChatPromptValue`; it is what `ChatOpenAI` receives as its input.

The placeholders (`{context}`, `{question}`, `{chat_history}`) are validated at call time. If you invoke a chain and one of the required variables is missing from the input dict, LangChain raises an error immediately — before the model is even called — telling you exactly which variable is absent. This is much better than getting a confusing model response caused by an empty context.

##### `from_template()` — single-turn, flat string

`from_template()` wraps the entire string in a single human message. The model sees one block of text containing both instructions and the question. This is the simplest pattern and works fine for prototyping, but it has a limitation: every line of your instruction is in the same message as the user's question, which means the model has no structural signal about which part is a rule versus which part is an input.

##### `from_messages()` — structured multi-role prompt

Chat models are trained on conversations with distinct roles: `system`, `user` (human), and `assistant` (AI). The `system` role is treated as a privileged channel for instructions — the model is trained to follow system messages as standing rules that apply to the entire conversation, not as something the user said. Grounding rules ("answer ONLY from the context", "always cite ticket IDs", "say I don't know if unsure") belong in the system message because they are policy, not input.

`from_messages()` takes a list of `(role, text)` tuples — or message objects like `MessagesPlaceholder` — and builds a structured multi-turn prompt. The `{context}` placeholder typically lives in the system message (it is evidence the system is providing), while `{question}` lives in the human message (it is what the user typed).

##### How `{placeholders}` get filled at runtime

When the chain calls `prompt.invoke({"context": "...", "question": "..."})`, the template substitutes the values and returns a `ChatPromptValue` containing a list of fully-formed messages. The `ChatOpenAI` step then serialises those messages into the format the API expects and makes the call. You never manually format a string — the template handles it.

```python
from langchain_core.prompts import ChatPromptTemplate

# ── from_template: one user message, everything in one block ──────────
simple_prompt = ChatPromptTemplate.from_template("""
Answer using ONLY the context below. Cite ticket IDs.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
""")
# input_variables: ["context", "question"]
# Produces: one HumanMessage with the filled string

# ── from_messages: system + human, rules separated from input ─────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are SupportDesk AI. Use ONLY the sources below to answer.\n"
     "Rules:\n"
     "1. Cite [SOURCE N] after every claim\n"
     "2. If the answer isn't in the sources, say 'I don't have that information'\n"
     "3. Never use outside knowledge\n\n"
     "Sources:\n{context}"),
    ("human", "{question}"),
])
# input_variables: ["context", "question"]
# Produces: [SystemMessage("You are SupportDesk AI..."), HumanMessage("...")]

# At runtime the chain calls this internally:
filled = rag_prompt.invoke({"context": "TICK-001: ...", "question": "How do I fix auth failures?"})
print(type(filled))         # → ChatPromptValue
print(filled.to_messages()) # → [SystemMessage(...), HumanMessage(...)]
# This ChatPromptValue is passed directly to llm.invoke()
```

##### Why the system/human split matters for grounding

If your grounding rules are buried in a user message alongside the question, the model can "talk itself out of" the rules — it has been trained to weigh everything in a user message as part of the same conversational turn. When the rules are in a system message, they are treated as standing policy. In practice this means the model is significantly less likely to hallucinate when the instruction "answer ONLY from context" arrives as a system message rather than as a prefix in a user message.

#### Step 7 — Keeping track of the conversation: `MessagesPlaceholder`, `HumanMessage`, `AIMessage`

A single-turn RAG chain answers questions in isolation. If the user asks "How was it resolved?" as a follow-up, the chain has no idea what "it" refers to. To fix this, you maintain a list of past turns and inject it into every new prompt. `MessagesPlaceholder` is the slot in a `from_messages()` template that says "expand the entire history list here." At call time you pass in a list of `HumanMessage` and `AIMessage` objects — typed wrappers that carry the text of each turn along with a role label. LangChain uses those role labels to format the history correctly for the model: human turns become "user" messages and AI turns become "assistant" messages. The model then sees the full conversation and can resolve pronouns, follow-up questions, and references to earlier answers correctly.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are SupportDesk AI. Answer from context only.\n\nContext:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),  # ← history injected here
    ("human", "{question}"),
])

# Build up history across turns
history = []
history.append(HumanMessage(content="What is TICK-001 about?"))
history.append(AIMessage(content="TICK-001 is about login failures after password reset."))

# On the next call, the model sees both turns before the new question,
# so "it" in "How was it resolved?" correctly refers to TICK-001.
```

#### Step 8 — Generating the answer: `ChatOpenAI`

With the prompt assembled and filled, `ChatOpenAI` sends it to the model and gets back a response. The key parameter to understand is `temperature`. At `temperature=0` the model is deterministic — it will produce the same answer every time it sees the same prompt. For a support assistant this is almost always what you want: a user who asks the same question twice should get the same answer, not a different one depending on random sampling. The model returns an `AIMessage` object rather than a plain string, which is why the next step is always a parser.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,      # deterministic — same prompt → same answer every time
    timeout=120,
    max_retries=3,
)

# The raw return is an AIMessage, not a string
response = llm.invoke("Hello")
print(type(response))        # → <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)      # → "Hello! How can I help you today?"
# This is why StrOutputParser is always the last step.
```

#### Step 9 — Extracting a usable result: `StrOutputParser` and `JsonOutputParser`

`ChatOpenAI` returns an `AIMessage` object. Your application just wants a string. `StrOutputParser` unwraps the `.content` attribute and returns it — it is always the final step in a chain that produces plain text. When you instead need structured data — for example, a confidence score alongside an answer — you write a prompt that asks the model to respond in JSON and then use `JsonOutputParser` as the final step. It parses the model's string output into a Python dict that your code can work with directly.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Plain text chain — StrOutputParser unwraps AIMessage.content
chain = prompt | llm | StrOutputParser()
answer = chain.invoke({"context": "...", "question": "..."})
print(type(answer))   # → str

# Structured chain — prompt asks for JSON, parser converts the string to a dict
json_prompt = ChatPromptTemplate.from_template(
    'Answer the question. Return JSON: {{"answer": "...", "confidence": 0-100}}\n'
    'Context: {context}\nQuestion: {question}'
)
scored_chain = json_prompt | llm | JsonOutputParser()
result = scored_chain.invoke({"context": "...", "question": "..."})
print(result["confidence"])   # → 87  (an int, not a string)
```

#### Step 10 — Wiring everything together: the Runnable interface and LCEL

##### The Runnable contract

Every construct described above — retriever, prompt template, LLM, parser — implements the same `Runnable` interface. A `Runnable` is not a class you inherit from directly; it is a **contract** that guarantees four methods on every component:

- **`.invoke(input)`** — the default: one input, one output, executed synchronously. This is what you use in demos, scripts, and anywhere you just want an answer before continuing.
- **`.batch(inputs)`** — takes a list of inputs and runs them in parallel internally. If you need to evaluate your RAG system against 100 test questions, calling `.batch(questions)` is dramatically faster than looping over `.invoke()` one by one.
- **`.stream(input)`** — instead of waiting for the full response, it yields output tokens one by one as the model generates them. In a chat UI this makes the experience feel instant — the user sees words appearing rather than a blank screen followed by a wall of text.
- **`.ainvoke(input)`** — the async counterpart of `.invoke()`. Inside a FastAPI endpoint or any async web server, blocking the thread while waiting for a slow LLM call would prevent the server from handling other requests. `.ainvoke()` releases the thread while waiting and resumes when the result arrives.

```python
# All four methods work on any chain — they are inherited from Runnable
answer  = chain.invoke("How do I fix auth failures?")           # one answer
answers = chain.batch(["Q1", "Q2", "Q3"])                       # three answers, parallel
for token in chain.stream("What caused the outage?"):           # stream word-by-word
    print(token, end="", flush=True)
answer  = await chain.ainvoke("async — use inside FastAPI")     # non-blocking
```

##### How `|` actually works: `RunnableSequence` and `RunnableParallel`

When you write `a | b`, LCEL creates a `RunnableSequence` under the hood — an object that calls `a.invoke(input)`, takes the result, and passes it as input to `b.invoke(result)`. The full chain `retriever | format_docs | prompt | llm | StrOutputParser()` is one nested `RunnableSequence`. Because the sequence itself is also a `Runnable`, you can call `.invoke()`, `.batch()`, `.stream()`, and `.ainvoke()` on the whole thing, and LCEL handles threading each method through every link.

When you place a plain Python dict as a step in a chain — `{"context": retriever | format_docs, "question": RunnablePassthrough()}` — LCEL wraps it into a `RunnableParallel`. This is the fan-out step: it runs all the values in the dict simultaneously and collects the results into a new dict before passing it on. In this example, the retriever runs on one thread and `RunnablePassthrough` runs on another (though it does nothing), and the prompt receives `{"context": "...", "question": "..."}` as a single merged result.

```python
# What LCEL builds when you write this chain:
chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# Is equivalent to (conceptually):
# RunnableSequence(
#   RunnableParallel({
#       "context": RunnableSequence(retriever, format_docs),
#       "question": RunnablePassthrough()
#   }),
#   prompt,
#   llm,
#   StrOutputParser()
# )
```

##### The fan-out problem: why `RunnablePassthrough` exists

The pipeline input is a single string — the user's question. The prompt template needs two things: the retrieved context AND the original question. Once the string goes into the retriever, it gets transformed into a list of Documents. The question string is gone. `RunnablePassthrough` solves this by sitting alongside the retriever in the `RunnableParallel` dict and doing nothing — it just forwards the original input string unchanged so the prompt receives both values.

```python
# Without RunnablePassthrough, you would lose the question:
broken = retriever | format_docs | prompt | llm  # prompt never gets {question}

# With RunnablePassthrough, both values arrive at the prompt:
working = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt   # now has both {context} and {question}
    | llm
    | StrOutputParser()
)
```

##### Upgrading to `RunnablePassthrough.assign()` for multi-input chains

To understand why `.assign()` is needed, you first need to understand exactly what the `RunnableParallel` dict pattern does — and where it breaks.

**The dict pattern works when the chain input is a plain string.** The `RunnableParallel` receives the string, fans it out to every branch in the dict simultaneously, and collects the results into a new dict. `RunnablePassthrough` in the `"question"` branch just forwards the string unchanged. The output is `{"context": "..retrieved..", "question": "How do I fix auth failures?"}` — exactly what the prompt needs.

```
Input:  "How do I fix auth failures?"   (a plain string)

RunnableParallel:
  "context" branch → retriever | format_docs  → "TICK-001: ..."
  "question" branch → RunnablePassthrough()   → "How do I fix auth failures?"

Output: {"context": "TICK-001: ...", "question": "How do I fix auth failures?"}
                ↓ passed to prompt ✓
```

**The dict pattern breaks when the chain input is already a dict.** For a conversational chain you call `.invoke({"question": "...", "chat_history": [...]})`. Now the `RunnableParallel` receives that dict — not a string. Three things go wrong simultaneously:

1. The `"context"` branch passes the whole dict to the retriever. The retriever expects a string. It fails.
2. The `"question"` branch uses `RunnablePassthrough()` which forwards the whole dict as the value of `"question"`. Wrong type.
3. The `RunnableParallel` **replaces** the input with a brand-new dict containing only `"context"` and `"question"`. The `"chat_history"` key is completely discarded — the prompt's `MessagesPlaceholder` will raise a `KeyError`.

```
Input:  {"question": "How was it resolved?", "chat_history": [...]}  (a dict)

RunnableParallel:
  "context" branch → retriever receives the whole dict → FAILS (expects str)
  "question" branch → RunnablePassthrough() → {"question": "...", "chat_history": [...]}

Output: {"context": ERROR, "question": <whole dict>}
        chat_history is GONE
```

**`RunnablePassthrough.assign()` solves all three problems at once.** Instead of *replacing* the input with a new dict, `.assign()` starts with the original input dict and *adds* new keys to it. Every key that was already in the input — `question`, `chat_history` — is passed through to the output unchanged. Only the keys you name in `.assign()` are computed and added on top.

```
Input:  {"question": "How was it resolved?", "chat_history": [...]}

RunnablePassthrough.assign(context=...):
  Pass through "question"      → "How was it resolved?"      ✓ unchanged
  Pass through "chat_history"  → [HumanMessage, AIMessage]   ✓ unchanged
  Compute     "context"        → run sub-pipeline → "TICK-001: ..."

Output: {
  "question":     "How was it resolved?",    ← passed through
  "chat_history": [...],                     ← passed through
  "context":      "TICK-001: ..."            ← newly added
}
           ↓ passed to conv_prompt ✓ (all three keys present)
```

The sub-pipeline that computes `context` is `itemgetter("question") | retriever | format_docs`. This is necessary because `.assign()` passes the whole input dict into the sub-pipeline. `itemgetter("question")` is the first step — it extracts just the question string from the dict so the retriever receives a plain string as it expects.

```python
# ── What breaks ───────────────────────────────────────────────────────
broken_conv = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | conv_prompt
    # retriever gets the whole dict → error
    # chat_history is dropped → MessagesPlaceholder KeyError
)

# ── What works ────────────────────────────────────────────────────────
working_conv = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
        #        ↑ extracts str from dict  ↑ now receives str  ↑ formats List[Document]
    )
    | conv_prompt   # receives {"question": ..., "chat_history": ..., "context": ...} ✓
    | llm
    | StrOutputParser()
)

working_conv.invoke({
    "question": "How was it resolved?",
    "chat_history": history
})
```

**Rule of thumb:** use the `RunnableParallel` dict pattern when your chain input is a single string and you need to fan it out into multiple named values. Use `RunnablePassthrough.assign()` when your chain input is already a multi-key dict and you need to add one more computed key without losing the others.

##### `itemgetter` as a Runnable

`itemgetter` is from Python's standard library `operator` module — it has nothing to do with LangChain. But LCEL automatically coerces any callable into a `RunnableLambda` when it appears in a chain, so `itemgetter("question")` composes cleanly with `|` without any boilerplate. Writing `itemgetter("question") | retriever | format_docs` is the idiomatic style for "extract a field, then run a sub-pipeline on it." It is functionally identical to `(lambda x: x["question"]) | retriever | format_docs` but significantly more readable.

```python
from operator import itemgetter

# These are equivalent — itemgetter is preferred style
context_pipe_idiomatic = itemgetter("question") | retriever | format_docs
context_pipe_lambda    = (lambda x: x["question"]) | retriever | format_docs

# Full conversational chain putting everything together:
conv_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
        # itemgetter("question")  → str          (extracts from input dict)
        # | retriever             → List[Document] (semantic search)
        # | format_docs           → str           (joins into context string)
    )
    | conv_prompt    # str, List[Document] → ChatPromptValue
    | llm            # ChatPromptValue     → AIMessage
    | StrOutputParser()  # AIMessage       → str
)

answer = conv_chain.invoke({
    "question": "How was it resolved?",
    "chat_history": history
})
```

##### Data flow through a full chain

It helps to trace exactly what type each `|` link receives and produces:

```
Input:   "How do I fix authentication failures?"   ← plain string

  retriever        List[Document]   ← 3 most relevant tickets
  format_docs      str              ← tickets joined into one context block
  ↕ (RunnableParallel merges context + original question)
  prompt           ChatPromptValue  ← filled template, ready to send
  llm              AIMessage        ← model response object
  StrOutputParser  str              ← .content extracted — final answer

Output:  "Based on TICK-001, authentication failures after password reset
          can be resolved by clearing active sessions..."
```

Each link's output type is exactly what the next link expects. If you ever see a type error in an LCEL chain, it is almost always because one step returned something unexpected — tracing the types like this is the fastest way to debug it.

---

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
