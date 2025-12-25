# SupportDesk-RAG Workshop: Quick Start Guide

## 🚀 Setup Instructions (15 minutes)

### 1. Install Python Dependencies

```bash
cd SupportDesk-RAG-Workshop

# Install all requirements
pip install -r requirements.txt
```

**If you encounter issues**, install packages individually:
```bash
pip install langchain langchain-openai langchain-community
pip install faiss-cpu chromadb sentence-transformers
pip install rouge-score nltk openai numpy pandas matplotlib scikit-learn
pip install python-dotenv tqdm
```

### 2. Set Up OpenAI API Key (Optional but Recommended)

**Option A: Environment Variable (Windows)**
```powershell
$env:OPENAI_API_KEY="sk-your-key-here"
```

**Option B: Environment Variable (Linux/Mac)**
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

**Option C: Create `.env` file**
```bash
# Create .env file in project root
echo OPENAI_API_KEY=sk-your-key-here > .env
```

**Option D: Use Local LLMs (Free)**
If you don't have OpenAI API key, you can use local models with Ollama:
1. Install [Ollama](https://ollama.ai/)
2. Run: `ollama pull llama2`
3. Modify demo scripts to use Ollama instead of OpenAI

### 3. Test Installation

```bash
# Test Hour 1 (embeddings)
cd hour_1_embeddings
python demo.py
```

If you see embeddings generated successfully, you're ready to go! ✅

---

## 📅 Workshop Timeline (4 Hours)

### Hour 1: Embeddings & Similarity Search (9:00 - 10:00 AM)
- **9:00-9:20**: Concepts (What are embeddings? Why similarity search?)
- **9:20-9:50**: Demo (Run `hour_1_embeddings/demo.py`)
- **9:50-10:00**: Exercises & Q&A

**Output**: Understanding of semantic search and vector representations

---

### Hour 2: Chunking & Vector Stores (10:15 - 11:15 AM)
*15-minute break*

- **10:15-10:35**: Concepts (Chunking strategies, FAISS vs Chroma)
- **10:35-11:05**: Demo (Run `hour_2_chunking/demo.py`)
- **11:05-11:15**: Exercises & Q&A

**Output**: Ability to build and query vector stores at scale

---

### Hour 3: Build the RAG Pipeline (11:30 AM - 12:30 PM)
*15-minute break*

- **11:30-11:45**: Concepts (RAG architecture, anti-hallucination)
- **11:45-12:20**: Demo (Run `hour_3_rag_pipeline/demo.py`)
- **12:20-12:30**: Exercises & Q&A

**Output**: Working end-to-end RAG system

---

### 🍕 Lunch Break (12:30 - 1:30 PM)

---

### Hour 4: Evaluation & Final Demo (1:30 - 2:30 PM)
- **1:30-1:45**: Concepts (Evaluation metrics, hallucination detection)
- **1:45-2:15**: Demo (Run `hour_4_evaluation/demo.py`)
- **2:15-2:25**: Interactive final demo
- **2:25-2:30**: Wrap-up & next steps

**Output**: Evaluated RAG system ready for production

---

## 🎯 Learning Path

### Progressive Complexity
Each hour builds on the previous:
```
Hour 1: Embeddings → Hour 2: Storage → Hour 3: Generation → Hour 4: Validation
```

### Hands-On Philosophy
- 70% coding, 30% theory
- Every concept demonstrated with working code
- Exercises reinforce learning immediately

---

## 🛠️ Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'sentence_transformers'`
**Solution**: 
```bash
pip install sentence-transformers
```

### Issue: `ImportError: cannot import name 'ChatOpenAI'`
**Solution**: 
```bash
pip install --upgrade langchain langchain-openai
```

### Issue: OpenAI API rate limit errors
**Solution**: 
1. Use a different API key
2. Switch to local models (Ollama)
3. Add delays between API calls

### Issue: FAISS installation fails on Windows
**Solution**: 
```bash
pip install faiss-cpu --no-cache-dir
```

---

## 💡 Tips for Success

1. **Read the exercises.md** in each folder before starting
2. **Run demos first** to see expected output
3. **Modify code** - break things and fix them to learn
4. **Ask questions** - there are no silly questions
5. **Take notes** - document your learnings

---

## 📞 Support

- Raise your hand during the workshop
- Check `exercises.md` for hints
- Review demo code for reference implementations

---

**Ready to build your first RAG system? Let's go! 🚀**
