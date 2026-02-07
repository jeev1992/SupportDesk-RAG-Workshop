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
from langchain_community.vectorstores import Chroma
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
    collection_name="solutions_test"
)
print("✓ Vector store ready")

# Create retriever and LLM
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, timeout=120, max_retries=3)

def format_docs(docs):
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
    for doc in docs[:3]:
        print(f"  - {doc.metadata['ticket_id']}: {doc.metadata['title'][:40]}...")

# Test MMR (Maximal Marginal Relevance) for diversity
print("\n\nMMR search (diverse results):")
retriever_mmr = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)
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
    RAG with intelligent fallbacks based on retrieval confidence
    """
    # Get docs with scores
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    if not docs_with_scores:
        return "No relevant tickets found.", "no_results"
    
    # Lower distance = better match (Chroma uses L2 distance)
    best_distance = docs_with_scores[0][1]
    
    print(f"  Best match distance: {best_distance:.4f}")
    
    if best_distance < 0.5:  # Very relevant
        # Proceed with RAG
        docs = [doc for doc, score in docs_with_scores]
        context = format_docs(docs)
        
        prompt = f"""Answer using only this context. Cite ticket IDs.

Context: {context}

Question: {query}

Answer:"""
        
        response = llm.invoke(prompt)
        return response.content, "high_confidence"
    
    elif best_distance < 1.0:  # Somewhat relevant
        ticket_id = docs_with_scores[0][0].metadata['ticket_id']
        return f"Found possibly relevant ticket ({ticket_id}), but confidence is moderate. Would you like me to show details?", "medium_confidence"
    
    else:  # Not relevant
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

# STUFF strategy (default) - all docs in one prompt
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

# MAP_REDUCE simulation - process docs individually then summarize
print("\nMAP_REDUCE strategy:")
start = time.time()
docs = retriever.invoke(test_query)
individual_answers = []
for doc in docs:
    single_prompt = ChatPromptTemplate.from_template(
        "Extract key info about this issue:\n{doc}\n\nKey points:"
    )
    chain = single_prompt | llm | StrOutputParser()
    individual_answers.append(chain.invoke({"doc": doc.page_content}))
    
combine_prompt = ChatPromptTemplate.from_template(
    "Combine these points to answer: {question}\n\nPoints:\n{summaries}"
)
combine_chain = combine_prompt | llm | StrOutputParser()
result = combine_chain.invoke({"question": test_query, "summaries": "\n".join(individual_answers)})
print(f"  Time: {time.time() - start:.2f}s")
print(f"  Answer: {result[:150]}...")

print("\n→ 'stuff': Fastest, concatenates all docs into one prompt")
print("→ 'map_reduce': Parallel processing, good for many docs")
print("→ 'refine': Iterative, highest quality but slowest (not shown - similar to map_reduce)")


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

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

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
prompt = ChatPromptTemplate.from_template("""Answer using the context. Be concise.

Context: {context}

Question: {question}

Answer:""")

streaming_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | streaming_llm
    | StrOutputParser()
)

print("\nStreaming response:")
query = "What causes database connection issues?"
result = streaming_chain.invoke(query)
print("\n")  # Newline after streaming


# ============================================================================
# Exercise 8: Multi-Turn Conversation (Medium)
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Multi-Turn Conversation")
print("=" * 80)

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Store chat history
chat_history = []

# Create conversational prompt with history placeholder
conv_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful support assistant. Answer using the context provided. Context: {context}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Build conversational chain
conv_chain = conv_prompt | llm | StrOutputParser()

def ask_with_history(question):
    """Ask a question with conversation history"""
    context = format_docs(retriever.invoke(question))
    response = conv_chain.invoke({
        "context": context,
        "history": chat_history,
        "question": question
    })
    # Update history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response))
    return response

# Simulate conversation
conversation = [
    "What causes authentication failures?",
    "How do I fix it?",  # Should remember we're talking about auth
    "What about database issues?"  # New topic
]

print("\nSimulated conversation:")
for user_msg in conversation:
    print(f"\nUser: {user_msg}")
    result = ask_with_history(user_msg)
    print(f"Assistant: {result[:200]}...")


# ============================================================================
# Bonus: Hallucination Detection (Challenge)
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Hallucination Detection")
print("=" * 80)

def detect_hallucination(query, answer, source_documents, llm):
    """
    Use LLM-as-judge to check if answer is grounded in sources
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

# Test hallucination detection
test_query = "How do I fix authentication issues?"
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
result = qa_chain.invoke({"query": test_query})

print(f"\nQuery: {test_query}")
print(f"Answer: {result['result'][:200]}...")
print(f"\nHallucination check:")
check_result = detect_hallucination(
    test_query, 
    result['result'], 
    result['source_documents'],
    llm
)
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
1. Prompt template design significantly affects answer quality
2. k parameter trades off coverage vs relevance
3. MMR provides diverse results when needed
4. Fallbacks handle low-confidence situations gracefully
5. Different chain types suit different use cases:
   - stuff: Fast, good for small context
   - map_reduce: Parallelizable, handles large docs
   - refine: Highest quality, slow
6. Streaming improves UX for long answers
7. Memory enables multi-turn conversations
8. LLM-as-judge can detect hallucinations

Next: Move on to Module 5 - Evaluation!
""")
