# -*- coding: utf-8 -*-
"""
Building the Complete RAG Pipeline Demo
================================================

This demo teaches:
1. Complete RAG architecture: retrieve → inject → generate
2. LangChain components (retrievers, prompts, chains)
3. Anti-hallucination strategies
4. Building a production-ready Q&A system

LEARNING RESOURCES:
- RAG Paper (Lewis et al.): https://arxiv.org/abs/2005.11401
- LangChain Documentation: https://python.langchain.com/docs/get_started/introduction
- LCEL Guide: https://python.langchain.com/docs/expression_language/
- Prompt Engineering: https://platform.openai.com/docs/guides/prompt-engineering
- Chroma Vector DB: https://docs.trychroma.com/
"""

import json
import os
from operator import itemgetter  # Used for idiomatic LCEL key extraction (LangChain pattern)
# LangChain is a framework for building LLM applications
# Reference: https://python.langchain.com/docs/get_started/introduction
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # OpenAI integrations
from langchain_chroma import Chroma  # Vector database for similarity search
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Smart text chunking
from langchain_core.documents import Document  # Document abstraction
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # Prompt templates
from langchain_core.messages import HumanMessage, AIMessage  # Chat history message types
from langchain_core.output_parsers import StrOutputParser  # Parse LLM output
from langchain_core.runnables import RunnablePassthrough  # Pass data through pipeline

# Load environment variables (API keys, model names)
from dotenv import load_dotenv
load_dotenv()

print("="*80)
print("MODULE 4: RAG PIPELINE")
print("="*80)
print("""
STRUCTURE:

  PART 1: Data Ingestion & Vector Store
  PART 2: Retriever Setup
  PART 3: Prompt Engineering (Anti-Hallucination)
  PART 4: Language Model
  PART 5: LCEL Chain Assembly
  PART 6: Testing the RAG System
  PART 7: Validation & Fallback
  PART 8: Conversation with History (Multi-Turn RAG)
  PART 9: Interactive Demo
""")

# ============================================================================
# PART 1: Data Ingestion & Vector Store Setup
# ============================================================================
print("\n" + "="*80)
print("PART 1: Data Ingestion Pipeline")
print("="*80)

# Load tickets
with open('../../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

# Convert to LangChain Document objects
# Documents are the core abstraction in LangChain - they combine content with metadata
# Reference: https://python.langchain.com/docs/modules/data_connection/document_loaders/
documents = []
for ticket in tickets:
    # Create rich document with all context
    # TIP: Structure your content logically - LLMs understand formatted text better
    content = f"""
Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Date: {ticket['created_date']} to {ticket['resolved_date']}

Problem Description:
{ticket['description']}

Resolution:
{ticket['resolution']}
    """.strip()
    
    # Create Document with metadata
    # Metadata is crucial for filtering, citation, and source tracking
    # Best practice: Include all information you might want to filter or display later
    doc = Document(
        page_content=content,  # The actual text content
        metadata={  # Structured data about the document
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'source': f"Ticket {ticket['ticket_id']}"
        }
    )
    documents.append(doc)

print(f"✓ Created {len(documents)} documents with metadata")

# Initialize OpenAI embeddings
# Embeddings convert text into vectors for semantic search
# Reference: https://platform.openai.com/docs/guides/embeddings
print("\nInitializing OpenAI embedding model...")
embeddings = OpenAIEmbeddings(
    model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
)
print("✓ OpenAI embedding model ready")

# Build vector store using Chroma
# Chroma is an open-source vector database optimized for AI applications
# It stores embeddings and enables fast similarity search
# Reference: https://docs.trychroma.com/
print("\nBuilding Chroma vector store...")
# Always rebuild from scratch so repeated runs don't accumulate duplicate documents.
import shutil
if os.path.exists("./rag_vectorstore"):
    shutil.rmtree("./rag_vectorstore")
vector_store = Chroma.from_documents(
    documents=documents,  # Our support ticket documents
    embedding=embeddings,  # Embedding function to use
    collection_name="supportdesk_rag",  # Name for this collection
    persist_directory="./rag_vectorstore",  # Where to save the database
    collection_metadata={"hnsw:space": "cosine"}  # Use cosine distance for similarity search
)
print("✓ Vector store created and persisted")

# ============================================================================
# PART 2: Create Retriever
# ============================================================================
print("\n" + "="*80)
print("PART 2: Setting Up Retriever")
print("="*80)

# Create a retriever from the vector store
# Retrievers are the interface for querying the vector store
# Reference: https://python.langchain.com/docs/modules/data_connection/retrievers/
retriever = vector_store.as_retriever(
    search_type="similarity",  # Use cosine similarity for ranking
    search_kwargs={"k": 3}  # Retrieve top-3 most similar documents
    # Other options:
    # - "mmr" (Maximal Marginal Relevance): Balances relevance with diversity
    # - "similarity_score_threshold": Only return docs above a score threshold
)

print("✓ Retriever configured:")
print(f"  - Search type: similarity")
print(f"  - Top-K results: 3")
print("\nTIP: k=3-5 is usually optimal. Too few → missing context, too many → noise")

# Test retriever
test_query = "Users can't log in after changing passwords"
print(f"\nTest query: '{test_query}'")
retrieved_docs = retriever.invoke(test_query)

print(f"\nRetrieved {len(retrieved_docs)} documents:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n#{i} - {doc.metadata['ticket_id']}: {doc.metadata['title']}")
    print(f"  Category: {doc.metadata['category']}")

# ============================================================================
# PART 3: Create Prompt Template with Anti-Hallucination Rules
# ============================================================================
print("\n" + "="*80)
print("PART 3: Prompt Engineering for RAG")
print("="*80)

# Define strict grounding prompt
# Prompt engineering is CRUCIAL for RAG - it tells the LLM how to use the context
# Reference: https://platform.openai.com/docs/guides/prompt-engineering
# Key principles:
# 1. Be explicit about using ONLY the provided context
# 2. Define what to do when information is missing
# 3. Request citations for transparency and verification
# 4. Set the role/persona for appropriate tone
prompt_template = """You are SupportDesk AI, a technical support assistant that helps engineers troubleshoot issues using historical support ticket data.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the answer is not in the context, say "I don't have enough information in the ticket history to answer that question."
3. DO NOT make up information or use external knowledge
4. Always cite the ticket ID when referencing information
5. If multiple tickets are relevant, mention all of them

Context from support tickets:
{context}

Question: {question}

Helpful Answer (with ticket citations):"""

# Convert string template to ChatPromptTemplate
# This creates a reusable template with variable placeholders
# Reference: https://python.langchain.com/docs/modules/model_io/prompts/
PROMPT = ChatPromptTemplate.from_template(prompt_template)

print("✓ Prompt template created with anti-hallucination rules:")
print("\n" + "-"*80)
print(prompt_template)
print("-"*80)

# ============================================================================
# PART 4: Initialize LLM
# ============================================================================
print("\n" + "="*80)
print("PART 4: Initializing Language Model")
print("="*80)

# Check if OpenAI key is available
if os.getenv("OPENAI_API_KEY"):
    print("✓ OpenAI API key found")
    # Initialize ChatOpenAI for generation
    # Reference: https://python.langchain.com/docs/integrations/chat/openai
    llm = ChatOpenAI(
        model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
        temperature=0,  # Temperature controls randomness (0 = deterministic, 2 = very creative)
        # For RAG, use temperature=0 to ensure consistent, factual responses
        # Reference: https://platform.openai.com/docs/guides/text-generation/how-should-i-set-the-temperature-parameter
        timeout=120,  # Increase timeout for slower connections
        max_retries=3,  # Retry on transient failures
    )
    print(f"✓ Using {os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')}")
else:
    print("⚠ OpenAI API key not found!")
    print("  Please set OPENAI_API_KEY environment variable")
    print("  Or use Ollama: ollama pull llama2")
    print("\nFor this demo, we'll show the prompt without generating answers.")
    llm = None

# ============================================================================
# PART 5: Build RAG Chain using LCEL (LangChain Expression Language)
# ============================================================================
print("\n" + "="*80)
print("PART 5: Assembling RAG Chain")
print("="*80)

# Helper function to format retrieved documents
# This concatenates all retrieved document contents into a single context string
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

if llm:
    # Build RAG chain using LCEL (LangChain Expression Language)
    # LCEL allows you to chain components using the | operator (like Unix pipes)
    # Reference: https://python.langchain.com/docs/expression_language/
    #
    # This chain does:
    # 1. Takes a question (string input)
    # 2. Retriever gets relevant docs, format_docs combines them
    # 3. PROMPT fills in {context} and {question} variables
    # 4. LLM generates answer based on filled prompt
    # 5. StrOutputParser extracts the string response
    #
    # The dict {"context": ..., "question": ...} creates the input for the prompt
    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    print("✓ RAG chain assembled:")
    print("  Retriever → Context Injection → LLM → Answer")
    print("\nThis is the complete RAG pipeline! Query in → Answer out")
else:
    qa_chain = None
    print("⚠ LLM not available, showing architecture only")

# ============================================================================
# PART 6: Test the RAG System
# ============================================================================
print("\n" + "="*80)
print("PART 6: Testing the RAG System")
print("="*80)

test_queries = [
    "How do I fix authentication failures after password reset?",
    "What causes database connection timeouts?",
    "Why are emails not being delivered?",
    "How do I make the perfect pizza?"  # Should refuse to answer!
]

for query in test_queries:
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("="*80)
    
    # Show retrieved context
    docs = retriever.invoke(query)
    print(f"\nRetrieved {len(docs)} relevant tickets:")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.metadata['ticket_id']}: {doc.metadata['title']}")
    
    if qa_chain:
        # Generate answer
        print("\nGenerating answer...")
        result = qa_chain.invoke(query)
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print("-"*80)
        print(result)
        
        print("\n" + "-"*80)
        print("SOURCE DOCUMENTS:")
        print("-"*80)
        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata['source']}")
    else:
        print("\n(LLM not configured - would generate answer here)")

# ============================================================================
# PART 7: Validation & Fallback
# ============================================================================
print("\n" + "="*80)
print("PART 7: Enhanced RAG with Answer Validation")
print("="*80)

def rag_with_validation(query, retriever, llm, min_similarity_score=0.5):
    """
    RAG pipeline with additional validation and fallback
    """
    # Retrieve documents with scores
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)

    print(f"\nQuery: {query}")
    print(f"\nRetrieval scores (cosine distance: 0=identical, 1=orthogonal, 2=opposite):")
    for doc, score in docs_with_scores:
        print(f"  - {doc.metadata['ticket_id']}: {score:.4f}")

    # Check if best match is good enough
    best_score = docs_with_scores[0][1]

    # Cosine distance: lower = more similar. 0.5 means cosine_similarity < 0.5 — too weak to answer.
    if best_score > min_similarity_score:
        print(f"\n⚠ Best match distance ({best_score:.4f}) exceeds threshold ({min_similarity_score}) — too dissimilar to answer confidently")
        return "I don't have enough relevant information in the ticket history to answer that question confidently."
    
    # If good matches, proceed with RAG
    docs = [doc for doc, score in docs_with_scores]
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""{prompt_template.replace('{context}', context).replace('{question}', query)}"""
    
    if llm:
        response = llm.predict(prompt)
        return response
    else:
        return "(LLM not configured)"

print("\nTesting validation logic:")
print("\n1. Relevant query (should answer):")
rag_with_validation(
    "How to fix database connection timeouts?",
    retriever,
    llm,
    min_similarity_score=0.5
)

print("\n2. Irrelevant query (should refuse):")
rag_with_validation(
    "What is the capital of France?",
    retriever,
    llm,
    min_similarity_score=0.5
)

# ============================================================================
# PART 8: Conversation with History (Multi-Turn RAG)
# ============================================================================
print("\n" + "="*80)
print("PART 8: Conversation with History (Multi-Turn RAG)")
print("="*80)
print("""
Problem with single-turn RAG:
  Turn 1: "How do I fix authentication failures?"  → good answer
  Turn 2: "How long did it take to resolve?"       → loses context! "it" = ???

Solution: Pass chat_history to the prompt so the LLM remembers previous turns.
  MessagesPlaceholder injects prior HumanMessage / AIMessage objects into the prompt.
""")

if llm:
    # Conversation-aware prompt with chat history slot
    # MessagesPlaceholder expands the history list into the prompt at call time
    conv_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are SupportDesk AI. Answer using ONLY the ticket context below.
Always cite ticket IDs. If the answer isn't in the context, say "I don't have that information."

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    # Chain: retrieve context for the CURRENT question, inject history, generate
    # itemgetter is the idiomatic LangChain LCEL way to extract dict keys —
    # it creates a Runnable that pulls a named field from the input dict.
    # Reference: https://python.langchain.com/docs/expression_language/cookbook/memory
    conv_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | conv_prompt
        | llm
        | StrOutputParser()
    )

    def ask_with_history(question, history):
        """Ask a question and automatically append the turn to history."""
        answer = conv_chain.invoke({"question": question, "chat_history": history})
        history.append(HumanMessage(content=question))
        history.append(AIMessage(content=answer))
        return answer

    # ── Demonstrate a 3-turn conversation ──────────────────────────────────
    print("Multi-turn conversation demo:\n")
    history = []

    q1 = "How do I fix authentication failures after a password reset?"
    print(f"Turn 1 — User: {q1}")
    a1 = ask_with_history(q1, history)
    print(f"         Assistant: {a1[:300]}{'...' if len(a1) > 300 else ''}")

    q2 = "How long did it typically take to resolve that kind of issue?"
    print(f"\nTurn 2 — User: {q2}")
    a2 = ask_with_history(q2, history)
    print(f"         Assistant: {a2[:300]}{'...' if len(a2) > 300 else ''}")

    q3 = "Were there any database-related tickets with similar resolution times?"
    print(f"\nTurn 3 — User: {q3}")
    a3 = ask_with_history(q3, history)
    print(f"         Assistant: {a3[:300]}{'...' if len(a3) > 300 else ''}")

    print(f"\n✓ History contains {len(history)} messages ({len(history)//2} complete turns)")
    print("TIP: In production, cap history to avoid token bloat:")
    print("       history = history[-6:]  # keep last 3 turns")

else:
    print("(LLM not configured — would run multi-turn conversation here)")
    print("\nKey pattern:\n")
    print("  history = []")
    print("  answer = chain.invoke({'question': q, 'chat_history': history})")
    print("  history.append(HumanMessage(content=q))")
    print("  history.append(AIMessage(content=answer))")
    print()
    print("MessagesPlaceholder injects the history list directly into the prompt,")
    print("so the LLM sees every prior turn when generating the next answer.")

# ============================================================================
# PART 9: Interactive Demo
# ============================================================================
print("\n" + "="*80)
print("PART 9: Interactive SupportDesk Assistant")
print("="*80)

if qa_chain:
    print("\nSupportDesk RAG Assistant Ready!")
    print("Ask questions about support ticket history.")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_query = input("You: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        print("\nAssistant: ", end="")
        answer = qa_chain.invoke(user_query)
        print(answer)
        
        docs = retriever.invoke(user_query)
        print(f"\n📎 Sources: {', '.join([doc.metadata['ticket_id'] for doc in docs])}")
        print()
else:
    print("\n⚠ Interactive mode requires OpenAI API key")
    print("Set OPENAI_API_KEY to try the interactive assistant!")

print("\n" + "="*80)
print("DEMO COMPLETE!")
print("="*80)
print("\nKey Takeaways:")
print("1. RAG pipeline: Retrieve → Inject Context → Generate")
print("2. Strict prompt engineering prevents hallucinations")
print("3. Always return source documents for verification")
print("4. Implement fallbacks for low-confidence matches")
print("5. Temperature=0 for deterministic, grounded answers")
print("\nNext: Evaluation & Metrics")
