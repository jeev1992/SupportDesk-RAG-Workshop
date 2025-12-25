"""
Hour 3: Building the Complete RAG Pipeline Demo
================================================

This demo teaches:
1. Complete RAG architecture: retrieve → inject → generate
2. LangChain components (retrievers, prompts, chains)
3. Anti-hallucination strategies
4. Building a production-ready Q&A system
"""

import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Try to load OpenAI key
from dotenv import load_dotenv
load_dotenv()

print("="*80)
print("HOUR 3: BUILDING THE RAG PIPELINE")
print("="*80)

# ============================================================================
# PART 1: Data Ingestion & Vector Store Setup
# ============================================================================
print("\n" + "="*80)
print("PART 1: Data Ingestion Pipeline")
print("="*80)

# Load tickets
with open('../data/synthetic_tickets.json', 'r') as f:
    tickets = json.load(f)
print(f"✓ Loaded {len(tickets)} support tickets")

# Convert to LangChain documents
documents = []
for ticket in tickets:
    # Create rich document with all context
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
    
    doc = Document(
        page_content=content,
        metadata={
            'ticket_id': ticket['ticket_id'],
            'title': ticket['title'],
            'category': ticket['category'],
            'priority': ticket['priority'],
            'source': f"Ticket {ticket['ticket_id']}"
        }
    )
    documents.append(doc)

print(f"✓ Created {len(documents)} documents with metadata")

# Initialize embeddings
print("\nInitializing embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)
print("✓ Embedding model ready")

# Build vector store
print("\nBuilding Chroma vector store...")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="supportdesk_rag",
    persist_directory="./rag_vectorstore"
)
print("✓ Vector store created and persisted")

# ============================================================================
# PART 2: Create Retriever
# ============================================================================
print("\n" + "="*80)
print("PART 2: Setting Up Retriever")
print("="*80)

# Basic retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve top-3 documents
)

print("✓ Retriever configured:")
print(f"  - Search type: similarity")
print(f"  - Top-K results: 3")

# Test retriever
test_query = "Users can't log in after changing passwords"
print(f"\nTest query: '{test_query}'")
retrieved_docs = retriever.get_relevant_documents(test_query)

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

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

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
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,  # Deterministic output, less creativity
    )
    print("✓ Using GPT-3.5-turbo")
else:
    print("⚠ OpenAI API key not found!")
    print("  Please set OPENAI_API_KEY environment variable")
    print("  Or use Ollama: ollama pull llama2")
    print("\nFor this demo, we'll show the prompt without generating answers.")
    llm = None

# ============================================================================
# PART 5: Build RAG Chain
# ============================================================================
print("\n" + "="*80)
print("PART 5: Assembling RAG Chain")
print("="*80)

if llm:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved docs into prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("✓ RAG chain assembled:")
    print("  Retriever → Context Injection → LLM → Answer")
else:
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
    docs = retriever.get_relevant_documents(query)
    print(f"\nRetrieved {len(docs)} relevant tickets:")
    for i, doc in enumerate(docs, 1):
        print(f"\n  [{i}] {doc.metadata['ticket_id']}: {doc.metadata['title']}")
    
    if llm:
        # Generate answer
        print("\nGenerating answer...")
        result = qa_chain({"query": query})
        
        print("\n" + "-"*80)
        print("ANSWER:")
        print("-"*80)
        print(result['result'])
        
        print("\n" + "-"*80)
        print("SOURCE DOCUMENTS:")
        print("-"*80)
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"{i}. {doc.metadata['source']}")
    else:
        print("\n(LLM not configured - would generate answer here)")

# ============================================================================
# PART 7: Advanced - Custom Chain with Validation
# ============================================================================
print("\n" + "="*80)
print("PART 7: Enhanced RAG with Answer Validation")
print("="*80)

def rag_with_validation(query, retriever, llm, min_similarity_score=0.7):
    """
    RAG pipeline with additional validation and fallback
    """
    # Retrieve documents with scores
    docs_with_scores = vector_store.similarity_search_with_score(query, k=3)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieval scores:")
    for doc, score in docs_with_scores:
        print(f"  - {doc.metadata['ticket_id']}: {score:.4f}")
    
    # Check if best match is good enough
    best_score = docs_with_scores[0][1]
    
    if best_score > min_similarity_score:
        print(f"\n⚠ Best match score ({best_score:.4f}) below threshold ({min_similarity_score})")
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
    min_similarity_score=0.7
)

print("\n2. Irrelevant query (should refuse):")
rag_with_validation(
    "What is the capital of France?",
    retriever,
    llm,
    min_similarity_score=0.7
)

# ============================================================================
# PART 8: Interactive Demo
# ============================================================================
print("\n" + "="*80)
print("PART 8: Interactive SupportDesk Assistant")
print("="*80)

if llm:
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
        result = qa_chain({"query": user_query})
        print(result['result'])
        
        print(f"\n📎 Sources: {', '.join([doc.metadata['ticket_id'] for doc in result['source_documents']])}")
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
print("\nNext: Hour 4 - Evaluation & Metrics")
