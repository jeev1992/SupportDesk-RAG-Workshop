# -*- coding: utf-8 -*-
"""
Module 4 Solutions: RAG Pipeline
================================

Solutions for all exercises in exercises.md
"""

import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ============================================================================
# Setup: Load data and create vector store
# ============================================================================
print("Loading data...")
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)

documents = []
for ticket in tickets:
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}

Problem Description:
{ticket['description']}

Resolution:
{ticket['resolution']}
    """.strip()
    
    doc = Document(
        page_content=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category'],
            'priority': ticket['priority']
        }
    )
    documents.append(doc)

print(f"✓ Loaded {len(documents)} documents")

# Create vector store
print("Building vector store...")
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="solutions_test",
    collection_metadata={"hnsw:space": "cosine"}
)
print("✓ Vector store ready")

# Create retriever and LLM
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, timeout=120, max_retries=3)

def format_docs(docs):
    """
    Convert retrieved Document objects into one prompt-ready context string.

    Why this is useful:
    - Keeps all retrieved evidence in one place for the LLM.
    - Preserves chunk boundaries with separators for readability.
    - Reused across many exercises to keep chain assembly concise.
    """
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


# ============================================================================
# Exercise 1: Modify the Prompt Template (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 1: Modify the Prompt Template")
print("=" * 80)

# Version A: Concise
template_a = """Answer the question using only the ticket context below. Cite ticket IDs.

Context: {context}

Question: {question}

Answer:"""

# Version B: Step-by-step
template_b = """You are a support assistant. Answer using ONLY the context below.

Context: {context}

Question: {question}

Think step by step:
1. What tickets are relevant?
2. What information do they contain?
3. How does this answer the question?

Answer:"""

# Version C: Bullet points
template_c = """Answer using only the context. Format as bullet points with ticket citations.

Context: {context}

Question: {question}

Answer (bullet points with sources):"""

templates = [("Concise", template_a), ("Step-by-step", template_b), ("Bullet points", template_c)]
test_query = "How do I fix authentication issues?"

for name, template in templates:
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print(f"\n{name} template:")
    response = chain.invoke(test_query)
    print(f"  {response[:200]}...")


# ============================================================================
# Exercise 2: Adjust Retrieval Parameters (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 2: Adjust Retrieval Parameters")
print("=" * 80)

test_query = "Payment processing failures"

# Test different k values
for k in [1, 3, 5, 10]:
    retriever_k = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever_k.invoke(test_query)
    print(f"\nk={k}: Retrieved {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.metadata['ticket_id']}: {doc.metadata['title'][:40]}...")

# ── MMR (Maximal Marginal Relevance) ──────────────────────────────────
#
# Problem with plain similarity search:
#   If you search "database issues", you might get 3 tickets that all
#   describe the *exact same* connection timeout bug — redundant results
#   that waste your context window.
#
# MMR solves this by balancing:
#   - RELEVANCE  (similarity to query)
#   - DIVERSITY  (dissimilarity to already-selected results)
#
# The MMR formula (Carbonell & Goldstein, 1998):
#   MMR = argmax [ λ · Sim(doc, query) − (1−λ) · max Sim(doc, already_selected) ]
#         λ controls the trade-off (Chroma defaults to 0.5)
#
# Two-stage algorithm:
#   Stage 1 — Cast a wide net:
#       Fetch the top `fetch_k` (here 10) most similar docs via cosine similarity.
#   Stage 2 — Select diverse subset:
#       From those 10 candidates, iteratively pick `k` (here 3) docs using MMR.
#       Each pick maximizes relevance while penalizing similarity to prior picks.
#
# Example — searching "database issues" against the top-10 similarity candidates:
#
#   Rank | Ticket   | Topic                          | Similarity
#   ─────┼──────────┼────────────────────────────────┼───────────
#     1  | TICK-005 | DB connection timeout           | 0.92
#     2  | TICK-012 | DB connection timeout (dup)     | 0.90
#     3  | TICK-008 | DB connection timeout (variant) | 0.88
#     4  | TICK-015 | DB deadlock issue               | 0.85
#     5  | TICK-003 | DB migration failure            | 0.82
#    ... | ...      | ...                             | ...
#
#   Plain similarity k=3 returns:
#     → TICK-005, TICK-012, TICK-008  — three nearly identical timeout tickets!
#
#   MMR k=3, fetch_k=10 returns:
#     Pick 1: TICK-005 (highest relevance)
#     Pick 2: TICK-015 (relevant AND different from TICK-005 — deadlock vs timeout)
#     Pick 3: TICK-003 (relevant AND different from both — migration topic)
#     → Diverse coverage of different database problems.
#
# Rule of thumb: set fetch_k = 3×k to 5×k for good diversity without too much noise.
#
# When to use MMR vs similarity:
#   - Similarity: precise query, want single best answer ("TICK-001 resolution")
#   - MMR: broad query, want coverage across subtopics ("database issues")
print("\n\nMMR search (diverse results):")
retriever_mmr = vector_store.as_retriever(
    search_type="mmr",              # Use MMR instead of plain cosine similarity
    search_kwargs={
        "k": 3,                     # Final number of documents to return
        "fetch_k": 10,              # Candidate pool size (fetched via similarity first)
        # "lambda_mult": 0.5,       # Trade-off: 1.0 = pure relevance, 0.0 = pure diversity
        #                           # Default 0.5 balances both equally.
    }
)
# fetch_k > k is required — MMR needs a larger candidate pool to select diverse items from.
docs_mmr = retriever_mmr.invoke(test_query)
for doc in docs_mmr:
    print(f"  - {doc.metadata['ticket_id']}: {doc.metadata['title'][:40]}...")


# ============================================================================
# Exercise 3: Implement Citation Formatting (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 3: Implement Citation Formatting")
print("=" * 80)

citation_prompt = """Answer the question using the context. Include inline citations [TICK-XXX] after each fact.

Example format:
"Database connection timeouts occur when the pool is undersized [TICK-002]. Increase max_connections and monitor usage [TICK-002]."

Context:
{context}

Question: {question}

Answer with inline citations:"""

prompt = ChatPromptTemplate.from_template(citation_prompt)
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What causes authentication failures and how do I fix them?"
print(f"\nQuery: {query}")
response = chain.invoke(query)
print(f"\nAnswer with citations:\n{response}")


# ============================================================================
# Exercise 4: Build a Fallback System (Easy)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 4: Build a Fallback System")
print("=" * 80)

def smart_rag(query, vector_store, llm, min_score_threshold=0.7):
    """
    RAG with confidence-based fallbacks.

    Strategy:
    - High confidence (small distance): answer normally with full context.
    - Medium confidence: return a cautious clarification-style response.
    - Low confidence: refuse with a safe fallback.

    This pattern prevents overconfident hallucinations on weak retrieval.
    """
    # Retrieve candidate evidence + numeric distances from the vector store.
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    if not docs_with_scores:
        return "No relevant tickets found.", "no_results"
    
    # Lower distance = better match (cosine distance: 0=identical, 1=orthogonal).
    # We use the top distance as a lightweight proxy for answer reliability.
    best_distance = docs_with_scores[0][1]
    
    print(f"  Best match distance: {best_distance:.4f}")
    
    if best_distance < 0.5:  # Very relevant
        # High-confidence path: construct full grounded prompt and answer directly.
        docs = [doc for doc, score in docs_with_scores]
        context = format_docs(docs)
        
        prompt = f"""Answer using only this context. Cite ticket IDs.

Context: {context}

Question: {query}

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content, "high_confidence"
    
    elif best_distance < 1.0:  # Somewhat relevant
        # Medium-confidence path: avoid overclaiming, ask for confirmation/context.
        ticket_id = docs_with_scores[0][0].metadata['ticket_id']
        return f"Found possibly relevant ticket ({ticket_id}), but confidence is moderate. Would you like me to show details?", "medium_confidence"
    
    else:  # Not relevant
        # Low-confidence path: safe refusal to avoid unsupported generation.
        return "I don't have relevant ticket history for this question.", "low_confidence"

# Test with different queries
test_queries = [
    ("authentication problems", "high confidence expected"),
    ("system performance", "medium confidence expected"),
    ("how to bake cookies", "low confidence expected")
]

for query, note in test_queries:
    print(f"\nQuery: '{query}' ({note})")
    answer, confidence = smart_rag(query, vector_store, llm)
    print(f"  Confidence: {confidence}")
    print(f"  Answer: {answer[:150]}...")


# ============================================================================
# Exercise 5: Compare Chain Types (Medium)
# ============================================================================
#
# When you have multiple retrieved documents, how do you combine them
# to generate a single answer?  This exercise compares three strategies:
#
# ── Strategy Overview ─────────────────────────────────────────────────
#
#   Strategy    | LLM Calls | Speed   | Quality | Best For
#   ────────────┼───────────┼─────────┼─────────┼──────────────────────────────
#   Stuff       | 1         | Fastest | Good    | Small context (< token limit)
#   Map-Reduce  | N + 1     | Slow    | Good    | Many/large docs that won't
#               |           |         |         | fit in one prompt
#   Refine      | N         | Slowest | Best    | When quality > latency
#
# ── STUFF ─────────────────────────────────────────────────────────────
#   Query → Retrieve 3 docs → Concatenate all into ONE prompt → LLM → Answer
#
#   ┌──────────────────────────────────────────┐
#   │  Prompt:                                 │
#   │  Context: [Doc1] --- [Doc2] --- [Doc3]   │  ← all docs concatenated
#   │  Question: How do I fix DB timeouts?     │
#   └──────────────────────────┬───────────────┘
#                              ▼
#                         LLM (1 call)
#                              ▼
#                           Answer
#
#   Pros: Fast (1 LLM call), simple to implement.
#   Cons: Breaks if total context exceeds the model's token limit.
#
# ── MAP_REDUCE ────────────────────────────────────────────────────────
#   Query → Retrieve 3 docs → Process EACH doc separately → Combine → Answer
#
#   Two phases:
#     1. Map   — Each doc gets its own LLM call ("extract key info").
#     2. Reduce — A final LLM call combines the summaries into one answer.
#
#                       ┌─── Doc1 → LLM → Summary1
#                       │
#   Query → Retrieve ───┼─── Doc2 → LLM → Summary2    (Map: 3 LLM calls)
#                       │
#                       └─── Doc3 → LLM → Summary3
#                                   │
#                                   ▼
#                       Combine summaries → LLM → Final Answer  (Reduce: 1 call)
#
#   Pros: Handles unlimited docs (each fits in one prompt); map calls can
#         run in parallel.
#   Cons: Slower (N+1 LLM calls), more expensive, summaries may lose detail.
#
# ── REFINE ─────────────────────────────────────────────────────────────
#   Query → Retrieve 3 docs → Process sequentially, refining each time
#
#   Doc1 → LLM → Draft answer
#   Doc2 + Draft → LLM → Refined answer
#   Doc3 + Refined → LLM → Final answer
#
#   Pros: Highest quality — each step builds on the previous answer.
#   Cons: Slowest (sequential, can't parallelize), N serial LLM calls.
#
# ── For this support ticket system (small docs, k=3) ──────────────────
#   STUFF is the right default — all 3 tickets easily fit within the
#   context window, so there's no reason to pay the latency/cost of
#   multiple LLM calls.
#
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXERCISE 5: Compare Chain Types")
print("=" * 80)

import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Compare different retrieval strategies using LCEL
strategies = {
    "stuff": "Concatenate all documents into context (fast, simple)",
    "map_reduce": "Process each doc separately, then combine (parallel)",
    "refine": "Iteratively refine answer with each doc (highest quality)"
}

test_query = "How do I fix database timeouts?"

# ── STUFF strategy (default) ──────────────────────────────────────────
# All retrieved docs are "stuffed" into a single {context} variable
# and sent to the LLM in ONE call.
print("\nSTUFF strategy:")
start = time.time()
stuff_prompt = ChatPromptTemplate.from_template(
    "Answer using the context:\n\nContext: {context}\n\nQuestion: {question}"
)
stuff_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | stuff_prompt
    | llm
    | StrOutputParser()
)
result = stuff_chain.invoke(test_query)
print(f"  Time: {time.time() - start:.2f}s")
print(f"  Answer: {result[:150]}...")

# ── MAP_REDUCE strategy ───────────────────────────────────────────────
# Phase 1 (Map):  Each doc → individual LLM call → summary
# Phase 2 (Reduce): All summaries → one LLM call → final answer
print("\nMAP_REDUCE strategy:")
start = time.time()
docs = retriever.invoke(test_query)

# Map phase: extract key info from each document independently
individual_answers = []
for doc in docs:
    single_prompt = ChatPromptTemplate.from_template(
        "Extract key info about this issue:\n{doc}\n\nKey points:"
    )
    chain = single_prompt | llm | StrOutputParser()
    individual_answers.append(chain.invoke({"doc": doc.page_content}))

# Reduce phase: combine all per-doc summaries into a single coherent answer
combine_prompt = ChatPromptTemplate.from_template(
    "Combine these points to answer: {question}\n\nPoints:\n{summaries}"
)
combine_chain = combine_prompt | llm | StrOutputParser()
result = combine_chain.invoke({"question": test_query, "summaries": "\n".join(individual_answers)})
print(f"  Time: {time.time() - start:.2f}s")
print(f"  Answer: {result[:150]}...")

# ── REFINE strategy ───────────────────────────────────────────────────
# Process docs sequentially. Start with the first doc to produce a draft,
# then feed each subsequent doc + the current draft to the LLM to refine.
#
#   Doc1 → LLM → Draft answer
#   Doc2 + Draft → LLM → Refined answer
#   Doc3 + Refined → LLM → Final answer
#
# Each step can incorporate new info AND correct earlier mistakes.
print("\nREFINE strategy:")
start = time.time()
docs = retriever.invoke(test_query)

# Step 1: Generate initial draft from the first document
initial_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only this context.\n\n"
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)
initial_chain = initial_prompt | llm | StrOutputParser()
current_answer = initial_chain.invoke({
    "context": docs[0].page_content,
    "question": test_query
})

# Steps 2..N: Refine the draft with each remaining document
refine_prompt = ChatPromptTemplate.from_template(
    "Here is an existing answer to the question:\n\n"
    "Existing answer: {existing_answer}\n\n"
    "Now consider this additional context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Refine the existing answer using the new context. "
    "If the new context isn't useful, return the existing answer unchanged.\n\n"
    "Refined answer:"
)
refine_chain = refine_prompt | llm | StrOutputParser()

for doc in docs[1:]:  # Iterate over remaining docs (skip the first)
    current_answer = refine_chain.invoke({
        "existing_answer": current_answer,
        "context": doc.page_content,
        "question": test_query
    })

print(f"  Time: {time.time() - start:.2f}s")
print(f"  Answer: {current_answer[:150]}...")

print("\n→ 'stuff': Fastest (1 LLM call), concatenates all docs into one prompt")
print("→ 'map_reduce': Parallel processing (N+1 calls), good for many docs")
print("→ 'refine': Iterative (N sequential calls), highest quality but slowest")


# ============================================================================
# Exercise 6: Add Metadata Filtering (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 6: Add Metadata Filtering")
print("=" * 80)

query = "system problem"

# Without filter
print("\nWithout filter:")
docs = vector_store.similarity_search(query, k=3)
for doc in docs:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")

# With category filter
print("\nWith category='Authentication' filter:")
docs_filtered = vector_store.similarity_search(
    query, 
    k=3,
    filter={"category": "Authentication"}
)
for doc in docs_filtered:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")

print("\nWith category='Database' filter:")
docs_filtered = vector_store.similarity_search(
    query, 
    k=3,
    filter={"category": "Database"}
)
for doc in docs_filtered:
    print(f"  - {doc.metadata['ticket_id']} ({doc.metadata['category']})")


# ============================================================================
# Exercise 7: Add Streaming Responses (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Add Streaming Responses")
print("=" * 80)

from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Create streaming LLM
streaming_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    timeout=120,
    max_retries=3
)

# Build streaming chain
prompt = ChatPromptTemplate.from_template("""Answer using the context. Provide a detailed, thorough response.
Include step-by-step troubleshooting instructions, root causes, and preventive measures.
Cite ticket IDs for every fact.

Context: {context}

Question: {question}

Detailed Answer:""")

streaming_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | streaming_llm
    | StrOutputParser()
)

print("\nStreaming response:")
query = "What causes database connection issues and how do I troubleshoot and prevent them?"
result = streaming_chain.invoke(query)
print("\n")  # Newline after streaming


# ============================================================================
# Exercise 8: Multi-Turn Conversation (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Multi-Turn Conversation")
print("=" * 80)

from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store chat history
chat_history = []

# ── Step 1: Query Reformulation ───────────────────────────────────────
# Problem: The retriever only sees the raw question string.
#   Turn 1: "What causes authentication failures?" → retriever works fine
#   Turn 2: "How do I fix it?"                     → retriever gets "it" — no context!
#
# Solution: Use the LLM to rewrite follow-up questions into standalone queries
# BEFORE they reach the retriever.
#   "How do I fix it?" + chat_history → "How do I fix authentication failures?"
#
# This is called "query condensing" or "history-aware retrieval".
condense_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and a follow-up question, rephrase the "
     "follow-up as a standalone question that includes all necessary "
     "context from the history. If the question is already standalone, "
     "return it unchanged."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# Chain that rewrites the question: dict → standalone question string
condense_chain = condense_prompt | llm | StrOutputParser()

# Create conversational prompt with history placeholder
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are SupportDesk AI. Answer using ONLY the ticket context below.
Always cite ticket IDs. If the answer isn't in the context, say "I don't have that information."

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# ── Step 2: Build the full chain ──────────────────────────────────────
# Flow:
#   Input: {"question": str, "chat_history": List[Message]}
#     │
#     ├─→ condense_chain  → standalone question (str)
#     │                        │
#     │                        ├─→ retriever → format_docs → context (str)
#     │                        │
#     ├─→ (passes through chat_history and question unchanged)
#     │
#     ▼
#   conv_prompt  → receives {context, chat_history, question}
#     │
#     ▼
#   LLM → StrOutputParser → answer (str)
conv_chain = (
    # First, rewrite the question into a standalone form
    RunnablePassthrough.assign(
        standalone=condense_chain
    )
    # Then, use the standalone question to retrieve relevant docs
    | RunnablePassthrough.assign(
        context=itemgetter("standalone") | retriever | format_docs
    )
    | conv_prompt
    | llm
    | StrOutputParser()
)

def ask_with_history(question, history):
    """
    Execute one conversational turn and persist memory.

    If chat_history is empty (first turn), skip the condense step and
    send the question directly to the retriever — no LLM rewrite needed.
    For follow-up turns, condense the question first so the retriever
    gets a standalone query instead of vague references like "it" or "that".
    """
    if not history:
        # First turn: no history to condense, retrieve directly with original question
        standalone = question
    else:
        # Follow-up turn: rewrite the question using history context
        # e.g. "How do I fix it?" → "How do I fix authentication failures?"
        standalone = condense_chain.invoke({"question": question, "chat_history": history})

    # Retrieve docs using the standalone query and generate the answer
    context = format_docs(retriever.invoke(standalone))
    answer = (conv_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "chat_history": history,
        "question": question,  # Show the original question in the prompt, not the rewritten one
    })

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    return answer

# Simulate a 3-turn conversation
# Use specific queries that match ticket content for reliable retrieval
conversation = [
    "Why are users unable to log in after a password reset?",
    "How do I fix it?",          # "it" → login issue (remembered from turn 1)
    "What about database issues?"  # new topic — no cross-contamination from history
]

print("\nSimulated conversation:")
for user_msg in conversation:
    print(f"\nUser: {user_msg}")
    result = ask_with_history(user_msg, chat_history)
    print(f"Assistant: {result[:200]}...")


# ============================================================================
# Bonus: Hallucination Detection (Challenge)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Hallucination Detection")
print("=" * 80)

def detect_hallucination(query, answer, source_documents, llm):
    """
    Use LLM-as-judge to check if an answer is grounded in retrieved sources.

    Practical purpose:
    - Adds a post-generation QA gate.
    - Helps flag unsupported claims before returning responses in production.
    """
    source_text = "\n\n".join([doc.page_content for doc in source_documents])
    
    detection_prompt = f"""You are a fact-checker. Determine if the answer is fully grounded in the source documents.

SOURCE DOCUMENTS:
{source_text}

ANSWER TO CHECK:
{answer}

Is every claim in the answer supported by the source documents?
Respond with:
- "GROUNDED" if all claims are supported
- "HALLUCINATION" if any claims are not in sources
- Brief explanation

Response:"""

    response = llm.invoke(detection_prompt)
    return response.content

# Test hallucination detection using the LCEL chain already set up above
test_query = "How do I fix authentication issues?"

# Get answer and source docs using LCEL (no deprecated RetrievalQA)
source_docs = retriever.invoke(test_query)
answer_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. Cite ticket IDs.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)
answer_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | answer_prompt | llm | StrOutputParser()
)
answer = answer_chain.invoke(test_query)

print(f"\nQuery: {test_query}")
print(f"Answer: {answer[:200]}...")
print(f"\nHallucination check:")
check_result = detect_hallucination(test_query, answer, source_docs, llm)
print(check_result)


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1.  Prompt template design significantly affects answer quality
2.  k parameter trades off coverage vs relevance
3.  MMR provides diverse results when needed
4.  Fallbacks handle low-confidence situations gracefully
5.  Different chain types suit different use cases:
    - stuff: Fast, good for small context
    - map_reduce: Parallelizable, handles large docs
    - refine: Highest quality, slow
6.  Streaming improves UX for long answers
7.  Memory enables multi-turn conversations
8.  LLM-as-judge can detect hallucinations

Next: Move on to Module 5 - Evaluation!
""")
