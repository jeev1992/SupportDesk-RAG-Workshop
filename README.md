# SupportDesk-RAG: A Support Ticket Retrieval & Troubleshooting Assistant

## Hands-On RAG Workshop with OpenAI

### Workshop Overview
This comprehensive workshop teaches you to build a production-ready Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and language models. By the end, you'll have a working assistant that answers incident queries using retrieved ticket context, with strong safeguards against hallucinations.

### Learning Objectives
- ✅ Generate and work with OpenAI embeddings
- ✅ Master chunking strategies for optimal retrieval  
- ✅ Compare 5 different indexing strategies (LlamaIndex)
- ✅ Implement a complete RAG pipeline with LangChain
- ✅ Evaluate with two-layer metrics (retrieval + generation)
- ✅ Deploy anti-hallucination safeguards
- ✅ Build agentic RAG systems with multi-step reasoning

---

## 🚀 Quick Start

### 1. Install Python (one-time)
- Install Python **3.10+** (recommended: **3.11**) from https://www.python.org/downloads/
- During install on Windows, enable **"Add Python to PATH"**

### 2. Clone or open this repo
```bash
# If you already have the repo, skip this step
git clone <your-repo-url>
cd SupportDesk-RAG-Workshop
```

### 3. Create a virtual environment (recommended)

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
```

**macOS/Linux:**
```bash
python3 -m venv .venv
```

### 4. Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```bat
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 5. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure OpenAI API

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

Then edit `.env` and set:
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### 7. Run a smoke test
```bash
cd modules/1_embeddings
python demo.py
```

If Module 1 runs, your environment is ready.

---

## Workshop Modules

### Module 1: Embeddings (`modules/1_embeddings/`)
**Learn:**
- Generate embeddings using OpenAI API
- Compute semantic similarity scores
- Visualize similarity relationships with heatmaps

**Run:**
```bash
cd modules/1_embeddings
python demo.py
```

---

### Module 2: Chunking (`modules/2_chunking/`)
**Learn:**
- Fixed-size vs recursive vs semantic chunking
- Structure-aware splitting (Markdown/HTML)
- Build vector stores with Chroma

**Run:**
```bash
cd modules/2_chunking
python demo.py
```

---

### Module 3: Indexing Strategies (`modules/3_indexing/`)
**Learn:**
- Vector Index - Semantic similarity search (most common)
- Summary Index - High-level document summaries
- Tree Index - Hierarchical retrieval patterns
- Keyword Table Index - Traditional keyword matching
- Hybrid Retrieval - Combining multiple strategies

**Technologies:** LlamaIndex for clean indexing abstractions

**Run:**
```bash
cd modules/3_indexing
python demo.py
```

---

### Module 4: RAG Pipeline (`modules/4_rag_pipeline/`)
**Learn:**
- Complete RAG architecture
- LangChain integration
- Prompt engineering for grounded responses
- Anti-hallucination strategies

**Run:**
```bash
cd modules/4_rag_pipeline
python demo.py
```

---

### Module 5: Evaluation (`modules/5_evaluation/`)
**Learn:**
- Two-layer evaluation approach (Retrieval + Generation)
- Retrieval metrics (Precision@K, Recall@K, F1)
- Generation metrics (Groundedness, Completeness)
- LLM-as-judge for generation evaluation
- Creating comprehensive evaluation reports

**Technologies:** FAISS, LLM-as-Judge evaluation

**Run:**
```bash
cd modules/5_evaluation
python demo.py
```

---

### Module 6: Agentic RAG (`modules/6_agentic_rag/`)
**Learn:**
- Creating custom tools for LangChain agents
- Building agents with OpenAI function calling
- Implementing conversation memory
- Multi-step reasoning with tool selection
- Comparing agentic vs direct RAG approaches

**Technologies:** LangChain Agents, OpenAI Function Calling

**Run:**
```bash
cd modules/6_agentic_rag
python demo.py
```

---

## 📁 Repository Structure

```
SupportDesk-RAG-Workshop/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                # Environment template
├── POST_CLASS_GUIDE.md         # Post-workshop learning guide
├── data/
│   └── synthetic_tickets.json  # Sample support tickets
└── modules/
    ├── 1_embeddings/
    │   ├── demo.py             # Working demo code
    │   ├── notes.md            # Instructor notes
    │   └── exercises.md        # Practice exercises
    ├── 2_chunking/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 3_indexing/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 4_rag_pipeline/
    │   ├── demo.py
    │   ├── notes.md
    │   └── exercises.md
    ├── 5_evaluation/
    │   ├── demo.py
    │   ├── notes.md
    │   ├── exercises.md
    │   ├── solutions.py
    │   └── evaluation_queries.json
    └── 6_agentic_rag/
        ├── demo.py
        ├── notes.md
        ├── exercises.md
        ├── solutions.py
        ├── tools.py
        ├── test_setup.py
        └── README.md
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (defaults shown)
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Model Options

**Embeddings:**
- `text-embedding-3-small` (1536 dims, recommended)
- `text-embedding-3-large` (3072 dims, highest quality)

**Chat:**
- `gpt-4o-mini` (recommended for cost/performance)
- `gpt-4o` (most capable)

---

## 💰 Cost Estimate

Running all modules: **< $0.10**
- Embeddings: ~$0.01 (20 tickets + queries)
- Chat completions: ~$0.05 (RAG pipeline demos)

See [OpenAI Pricing](https://openai.com/pricing) for current rates.

---

## 🎯 Prerequisites

- Python 3.10+ (recommended: 3.11)
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Basic understanding of Python
- Familiarity with APIs (helpful but not required)

---

## 🛠️ Troubleshooting

### Virtual Environment Activation Fails (Windows PowerShell)
If you see an execution policy error:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### `python` Command Not Found (Windows)
Use:
```powershell
py --version
py -3.11 -m venv .venv
```

### OpenAI API Errors
- Verify API key in `.env` file
- Check credits: https://platform.openai.com/usage
- Rate limits: Wait 60s if you get 429 errors

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

If issues persist, confirm your venv is active and reinstall cleanly:
```bash
pip uninstall -y -r requirements.txt
pip install -r requirements.txt
```

### Path Issues
- Always run demos from their module directory
- Ensure `data/synthetic_tickets.json` exists

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI API Reference](https://platform.openai.com/docs)
- [Chroma Documentation](https://docs.trychroma.com/)

---

## 🤝 Contributing

Found a bug or have suggestions? Open an issue or submit a pull request!

---

## 📄 License

MIT License - Feel free to use for learning and teaching!
