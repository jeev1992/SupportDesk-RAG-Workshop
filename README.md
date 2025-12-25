# SupportDesk-RAG: A Support Ticket Retrieval & Troubleshooting Assistant

## 4-Hour Training Workshop for Software Development Engineers

### Workshop Overview
This hands-on workshop teaches you to build a production-ready Retrieval-Augmented Generation (RAG) system using support ticket history. By the end, you'll have a working assistant that answers incident queries using ONLY retrieved ticket context, preventing hallucinations.

### Learning Objectives
- ✅ Understand embeddings and similarity search
- ✅ Master chunking strategies for optimal retrieval
- ✅ Build and query vector stores (FAISS, Chroma)
- ✅ Implement a LangChain-based RAG pipeline end-to-end
- ✅ Evaluate retrieval quality (precision/recall/F1)
- ✅ Evaluate generated answers (ROUGE/BLEU)
- ✅ Prevent hallucinations using prompt guards

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up OpenAI API Key (Optional)
```bash
# Windows
$env:OPENAI_API_KEY="sk-your-key-here"

# Linux/Mac
export OPENAI_API_KEY="sk-your-key-here"
```

### 3. Test Installation
```bash
cd hour_1_embeddings
python demo.py
```

---

## Workshop Schedule

### Hour 1: Embeddings & Similarity Search (60 min)
**Concepts (20 min)**
- What are embeddings? Vector representations of text
- Why embeddings matter for semantic search
- Distance metrics: cosine similarity, dot product, Euclidean distance
- Choosing the right embedding model

**Hands-on Demo (30 min)**
- Generate embeddings for sample support tickets
- Compute similarity scores between queries and tickets
- Rank and retrieve most relevant matches
- Visualize embeddings in 2D space (PCA/t-SNE)

**Exercise (10 min)**
- Find the top-5 most similar tickets for given queries
- Experiment with different embedding models

📂 **Materials**: `hour_1_embeddings/`

---

### Hour 2: Chunking & Vector Stores (60 min)
**Concepts (20 min)**
- Why chunking matters: context window limits and precision
- Fixed-size chunking vs semantic chunking vs windowed chunping
- Vector stores: FAISS, Chroma, Pinecone comparison
- Indexing strategies and trade-offs

**Hands-on Demo (30 min)**
- Implement 3 chunking strategies on support tickets
- Build a FAISS index from scratch
- Query by vector similarity
- Compare with Chroma's high-level abstraction

**Exercise (10 min)**
- Index tickets with different chunk sizes
- Measure retrieval quality for each strategy

📂 **Materials**: `hour_2_chunking/`

---

### Hour 3: Build the RAG Pipeline (60 min)
**Concepts (15 min)**
- RAG architecture: retrieve → inject → generate
- LangChain components: retrievers, prompt templates, chains
- Anti-hallucination strategies
- Context injection patterns

**Hands-on Demo (35 min)**
- Ingest synthetic support tickets
- Build the full pipeline: chunk → embed → index → retrieve
- Create prompt templates with strict grounding rules
- Generate answers with retrieved context
- Implement "No relevant tickets found" fallback

**Exercise (10 min)**
- Modify prompts to test hallucination resistance
- Add citation formatting to responses

📂 **Materials**: `hour_3_rag_pipeline/`

---

### Hour 4: Evaluation & Final Demo (60 min)
**Concepts (15 min)**
- Retrieval metrics: precision@k, recall@k, F1
- Generation metrics: ROUGE-L, BLEU
- Manual evaluation for hallucinations
- A/B testing RAG configurations

**Hands-on Demo (30 min)**
- Create evaluation dataset (queries + labeled relevant tickets)
- Compute retrieval metrics
- Evaluate generated answers against reference answers
- Manual hallucination check workflow

**Final Demo (15 min)**
- Interactive SupportDesk-RAG assistant
- Query real scenarios and see retrieved context

📂 **Materials**: `hour_4_evaluation/`

---

## 📁 Repository Structure

```
SupportDesk-RAG-Workshop/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── QUICKSTART.md               # Detailed setup guide
├── POST_WORKSHOP_GUIDE.md      # Next steps after workshop
├── data/
│   └── synthetic_tickets.json  # Sample support tickets
├── hour_1_embeddings/
│   ├── demo.py                 # Live demo code
│   └── exercises.md            # Practice exercises
├── hour_2_chunking/
│   ├── demo.py
│   └── exercises.md
├── hour_3_rag_pipeline/
│   ├── demo.py
│   └── exercises.md
└── hour_4_evaluation/
    ├── demo.py
    ├── evaluation_queries.json
    └── exercises.md
```

---

## 🎯 Prerequisites

- Python 3.8+
- Basic understanding of Python programming
- Familiarity with APIs (optional)
- OpenAI API key (optional - can use local models)

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [Chroma Documentation](https://docs.trychroma.com/)

---

## 🤝 Contributing

Found a bug or have suggestions? Feel free to open an issue or submit a pull request!

---

## 📄 License

This workshop material is provided for educational purposes.
