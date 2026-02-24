# Talking Script — Week 3: RAG-Powered Knowledge Agents 2
## RAG Pipelines, Evaluation & Agentic RAG

> **How to use this script:** Each section corresponds to a slide. Text in *[brackets]* are stage directions. Bullet points are optional elaborations you can pick from based on time and audience engagement.

---

## SLIDE 1 — Title Slide

Good [morning / afternoon / evening] everyone, and welcome to Week 3 of RAG-Powered Knowledge Agents. Today we're going to level up significantly from where we left off. Last week we built the foundation — embeddings, chunking, and indexing. This week we're putting it all together into a **full RAG pipeline**, learning how to **evaluate** whether our system is actually working, and then taking the exciting leap into **Agentic RAG** — where we give the AI the ability to reason, plan, and use tools to answer complex questions.

By the end of today, you'll understand not just how to build a RAG system, but how to prove it's working and how to make it smarter. Let's get started.

---

## SLIDE 2 — Instructor Introduction

*[Show if new students are in the class, otherwise keep brief]*

Quick intro for anyone joining us for the first time. I'm Jeevendra Singh. I've been working as a Software Engineer since 2013, with companies like Ericsson, SAP, Q2, and currently at Microsoft where I've spent the last 4+ years. My focus has shifted toward Agentic AI — building intelligent API platforms, copilots, and RAG-powered systems. I've trained over 2,000 learners on Agentic AI and related topics, so this is a space I'm genuinely passionate about. Enough about me — let's hear about you.

---

## SLIDE 3 — Welcome & Icebreaker

Before we dive in, pop into the chat and tell us three things: **your name**, **what you do**, and **one thing you're hoping to get out of today**. We've got a great mix of data scientists, engineers, and product folks, and the questions you ask will shape how we explore this material together.

*[Give 60-90 seconds for people to type in the chat]*

Fantastic — I can already see some great answers in there. Let's make sure everyone gets the most out of the next few hours.

---

## SLIDE 4 — Structure of Class

Just a quick overview of how this course is structured. We have a **Sunday live class** that runs about 4 hours — that's what we're in right now. You also have **assignments** — both MCQ and coding exercises — that reinforce what we cover today. Don't skip them; the coding assignments are where the real learning happens. You'll go from understanding concepts here to actually building them in your own environment.

---

## SLIDE 5 — Optimize Your Experience

A few things that will make today much more valuable for you. First — **please interact**. Ask questions, unmute, drop things in chat. This is a live class, not a lecture recording. Second — **attempt the assignments**. I can't stress this enough. The Thursday review session is there specifically to give you feedback. Third — **use your resources**. You have TAs, Discord, coaching sessions, and support tickets. Your success here is directly proportional to how actively you engage.

---

## SLIDE 6 — Don't Worry

I know some of you might be thinking — "this RAG and Agentic AI stuff sounds complex, what if I don't get it?" Here's my promise: we're here every step of the way. Nobody gets left behind. If something isn't clicking, say so — because I guarantee at least five other people in this room are thinking the same thing.

---

## SLIDE 7 — IK Support Features

Let me quickly walk through what support looks like beyond this class. You have:

- **Uplevel** — your central hub for all videos, MCQs, and assignments
- **Post-class videos** — optional but great for revisiting anything you missed
- **Technical coaching on Wednesdays** — bring your questions, bring your code
- **Discord** — an active community where peers and TAs answer questions daily
- **Support tickets on UpLevel** — for any technical or administrative concerns

Use all of these. They exist because learning complex topics takes repetition and support.

---

## SLIDE 8 — Success Hacks

Here's the formula that I've seen work for the 2,000+ learners I've trained. Watch the pre-class videos before you show up. Attend every session. Participate — even if you think your question is basic. Attempt every assignment, even if you don't finish it. And above everything else: **be consistent**. Missing one week here or skipping one assignment there compounds fast. The people who get the most out of this program are the ones who show up consistently, even when they're busy.

And — be patient with yourself. This stuff is genuinely hard. Give yourself permission to not understand everything immediately.

---

## SLIDE 9 — API Key Setup

Okay, practical note — check your email. You should have received an **OpenAI API key** from Interview Kickstart. If you don't see it in your inbox, check your spam folder. If it's still not there, email operations@interviewkickstart.com.

A few important rules about this key:
- It's for **course work only** — not personal projects
- Credits are limited per learner — don't burn through them on experiments outside class
- **Do not share it** with anyone

Keep it secure. Treat it like a password.

---

## SLIDE 10 — Workshop Repository

Everything we're going to work with today lives in this GitHub repository. If you haven't already, now is a great time to clone it:

```
git clone https://github.com/jeev1992/SupportDesk-RAG-Workshop.git
```

The repo has **6 complete modules** — from embeddings all the way to Agentic RAG. Each one has working demo code, practice exercises, and solutions. Today we're working primarily with Modules 4, 5, and 6.

Once cloned:
1. Run `pip install -r requirements.txt`
2. Set your OpenAI API key as an environment variable
3. You're ready to run any module demo

If you hit setup issues, drop them in the chat and our TAs will help you out.

---

## SLIDE 11 — Week 2 Recap Title

Let's do a quick recap of what we covered in Week 2 before we build on it today.

---

## SLIDE 12 — Week 2 Architecture Diagram

In Week 2 we covered three modules that form the foundation of every RAG system:

**Module 1 — Embeddings & Similarity.** We learned how text gets converted into vectors — specifically 1536-dimensional vectors using OpenAI's embedding model. We learned how cosine similarity works, scoring from -1 to 1. And the key insight: semantic search finds meaning, not just keywords. Searching for "reset" can find "forgot password" because semantically, they're close in vector space.

**Module 2 — Chunking & Vector Stores.** We learned that how you chunk your documents matters enormously. Recursive chunking is your default, semantic chunking is for quality-sensitive use cases. We stored vectors in ChromaDB with metadata, and we explored two search modes: standard similarity search and MMR — Maximum Marginal Relevance — which balances relevance with diversity to avoid redundant results.

**Module 3 — Indexing Strategies.** We compared four approaches: vector indexes for semantic search, keyword indexes for exact matching, tree-based indexes for hierarchical data, and — the recommendation for production — **hybrid retrieval** that combines vector and keyword search for the best of both worlds.

The bottom line from Week 2: **always measure retrieval quality**. Building is the easy part; knowing whether it's working is where most teams fall short.

Today we fix that.

---

## SLIDE 13 — Today's Agenda

Here's what we're covering today in three main sections:

1. **RAG Pipeline** — We'll build a complete, production-ready RAG pipeline end-to-end, including conversation history
2. **RAG Evaluation** — We'll introduce a rigorous two-layer evaluation framework with six metrics you can use to prove your system is working
3. **Agentic RAG with LangChain** — We'll move beyond fixed pipelines into systems where an AI agent reasons, selects tools, and handles complex multi-step queries

Each section builds on the last. Let's go.

---

---

# SECTION 1: THE RAG FOUNDATION

---

## SLIDE 14 — The RAG Foundation (Section Title)

We're starting with the "why" before we get into the "how." Understanding what problem RAG solves makes all the architectural decisions we'll make afterward make sense.

---

## SLIDE 15 — The Hallucination Problem

Here's the core problem with large language models: they are extremely good at sounding correct even when they're wrong. This is called **hallucination** — the model generates plausible-sounding but completely fabricated information.

Look at this spectrum on the slide. At the top — the green zone — we have **Grounded answers backed by real documents**. That's RAG. The answer comes directly from retrieved source material: "TICK-001: Clear sessions to fix login." Minimal risk.

As we move down the spectrum, answers become less and less grounded. **Partially grounded** — some facts are real, some are inferred. Then we hit **plausible but invented** — the model sounds confident but there's no source. And at the bottom: **full hallucination** — the model invents a ticket ID that doesn't exist and tells the user it resolved the problem in March 2024.

In a support desk context, that last scenario isn't just annoying — it's potentially harmful. A user follows fabricated advice and wastes hours chasing a ghost.

**RAG keeps you in the green zone** by grounding every answer in real documents you control.

---

## SLIDE 16 — What RAG Solves

Let me give you five concrete reasons why RAG has become the default architecture for enterprise AI applications:

1. **Factual Accuracy.** Answers are grounded in your actual documents. The support bot cites TICK-001 directly. No hallucinations.

2. **Always Up-to-Date.** When your knowledge base changes, you re-index. The model doesn't need to be retrained. A new KB article can be available to all queries within minutes.

3. **Traceable Sources.** Every answer comes with citations. Users can click through to verify. This is critical for trust in regulated industries.

4. **Cost-Effective.** Embedding 10,000 documents costs about 50 cents. Fine-tuning a large model on the same data? Easily $1,000 or more. RAG gives you similar benefits at a fraction of the cost.

5. **Domain-Specific.** RAG works with your private data — HR policies, medical records, legal contracts, customer support history. The LLM has never seen this data in training, but RAG puts it right in front of the model at query time.

The punchline: **RAG sits in the sweet spot** — more accurate than pure prompting, far cheaper than fine-tuning, and it works with data the LLM has never seen.

---

## SLIDE 17 — The Two Phases of RAG

Let me walk you through the complete RAG architecture. There are two distinct phases that run at different times for completely different reasons.

**The Offline Phase — runs once, when your data is ready.**

Think of this like building the library index. You do it once, and then everyone can query it efficiently.

- Step 1: **Load Documents.** Raw data — JSON tickets, PDFs, Confluence pages, database exports — gets loaded into structured LangChain Document objects with metadata attached. That metadata — ticket ID, category, priority, date — is critical. Include everything you might want to filter on later.

- Step 2: **Chunking.** Documents are split into smaller pieces using RecursiveCharacterTextSplitter. We use 500-token chunks with 50-token overlap. The overlap is important — it prevents losing context at chunk boundaries. A 2,000-character ticket becomes approximately 4 overlapping chunks.

- Step 3: **Embedding.** Each chunk is converted into a 1,536-dimensional vector using OpenAI's `text-embedding-3-small` model. Here's the key insight: semantically similar text produces nearby vectors. "Password reset failure" and "can't log in after changing credentials" end up close together in vector space despite using completely different words.

- Step 4: **Vector Store.** Vectors plus original text plus metadata get stored in a specialized database. For prototyping, use FAISS. For local production, Chroma. For scale, Pinecone or Weaviate.

**The Online Phase — runs for every user query.**

The user is waiting. Target latency: under 3 seconds total.

- Step 1: User types a natural language question. No special syntax needed.
- Step 2: Same embedding model converts the query to a vector. *Critical: must be the same model used during indexing. A mismatch causes silent failures.*
- Step 3: Cosine similarity compares the query vector against all stored vectors. Returns the top-K most relevant documents.
- Step 4: Retrieved documents are injected into a prompt template — this is the "Augmented" in RAG.
- Step 5: LLM generates an answer at temperature=0 for consistent, factual outputs.
- Step 6: Answer is returned with source citations and a confidence score.

The Vector Store is the bridge between these two phases. Everything feeds through it.

---

## SLIDE 18 — Recap: Offline Phase

*[Point to each stage column as you describe it]*

Let's do a proper walkthrough of the offline phase — this is a recap from Week 2, but I want every detail to land because everything in today's pipeline builds on top of it.

**Stage 1 — Documents.**
Raw data files are loaded and converted into structured LangChain Document objects. Each object carries the text content AND metadata. Look at the metadata example on the slide: `ticket_id: TICK-001`, `category: Authentication`, `priority: Critical`, `created_date: 2024-01-15`, `source: tickets.json`. This metadata travels with every chunk all the way into the vector store. Include everything you might ever want to filter on or display later — you can't add it retroactively without re-indexing.

**Stage 2 — Chunking.**
Documents are split into smaller, focused pieces that embedding models can process effectively. Our settings: `RecursiveCharacterTextSplitter`, chunk size of 500 tokens, overlap of 50 tokens. It splits on natural breaks — paragraphs first, then sentences, then words. The overlap is the part people often skip over, but it's critical. Look at the diagram: Chunk 1 ends with `[...problem desc===]`, Chunk 2 starts with `[===resolution...]`. That `^^^` shared zone — that 50-character overlap — is what prevents losing context at boundaries. A 2,000-character ticket becomes 4 chunks with overlapping edges.

**Stage 3 — Embedding.**
Each text chunk gets converted into a dense numeric vector by `text-embedding-3-small`. Input: a text string. Output: a 1,536-dimensional array like `[0.23, 0.11, -0.05, 0.42, 0.07, ...]`. Here's the key insight on this slide: "password reset failure" and "can't log in after changing credentials" are **CLOSE in vector space despite using completely different words.** That's the magic — and that's what makes semantic search fundamentally better than keyword matching.

**Stage 4 — Vector Store.**
All vectors get stored in a database optimized for high-dimensional similarity search. Your options: FAISS for local in-memory prototyping, Chroma for local persistent storage in small production, Pinecone for cloud-managed scale, Weaviate for cloud hybrid enterprise use. What actually gets stored for each chunk: the numeric vector, the original text, all metadata fields, and the document source reference. Our demo uses Chroma with `persist_directory` so the index survives between sessions.

Bottom line on this slide: **this phase runs once.** Update it when your documents change. The cost to embed 30 support tickets? Fractions of a cent.

---

## SLIDE 19 — Online Phase Deep Dive

Now the online phase — this runs for **every single query**, and the user is waiting. Let's walk through all six stages with the actual latency numbers from the slide.

**Stage 1 — User Query.** (Latency: instant)
User types a natural language question — in our example: *"How do I fix authentication failures after password reset?"* No special syntax needed. Plain English works best. Key point on this slide: **the system understands intent, not just keywords.** That's what separates semantic RAG from a keyword search engine.

**Stage 2 — Embed Query.** (Latency: ~50ms)
The same embedding model used during indexing converts the query into a vector. Input: "How do I fix auth..." Output: `[0.45, 0.22, 0.18, -0.31, 0.09, 0.55, ...]` — 1,536 dimensions.

Here's the **Critical Rule** highlighted on this slide: you must use the **exact same model** as the offline phase. Mismatched models produce incompatible vectors — the cosine similarity scores become meaningless and search will fail silently. You'll get results back, but they won't correspond to anything real. This is the most common silent failure I see in RAG implementations.

**Stage 3 — Similarity Search.** (Latency: ~20ms)
Query vector is compared against all stored vectors using cosine similarity. The results come back ranked: TICK-001 at 0.92, TICK-011 at 0.87, TICK-014 at 0.83 — these are above the relevance threshold, so we keep them. TICK-020 at 0.45, TICK-007 at 0.38 — below threshold, discarded as noise. We return the top K=3. Higher score = more relevant, lower scores = irrelevant content that would just confuse the LLM.

**Stage 4 — Augment Prompt.** (Latency: ~5ms)
Retrieved documents are injected into a structured prompt template. The prompt structure is:
- Context: TICK-001 full text, TICK-011 full text, TICK-014 full text
- Question: How do I fix...
- Rules: Answer ONLY from context. Cite ticket IDs.

This is the **Augmented** in Retrieval-Augmented Generation. Without this step, the LLM has no context to ground its answers in — it's back to hallucinating.

**Stage 5 — LLM Generate.** (Latency: 1–3 seconds — your bottleneck)
Model: GPT-4o-mini. Temperature: 0 — deterministic output. Max retries: 3. Timeout: 120 seconds.

The key principle on this slide: **the LLM is constrained to ONLY use provided context. It synthesizes and formats — but it does not add outside knowledge or speculate.** Temperature=0 enforces this. If you set temperature to 0.7, you're opening the door to creative — and potentially wrong — additions.

**Stage 6 — Respond + Sources.** (Total end-to-end: 2–4s)
Final answer is formatted with source citations and delivered. The example output: *"Based on TICK-001 and TICK-011, clear active sessions and verify SAML config uses SHA-256. Sources: TICK-001, TICK-011, TICK-014. Confidence: High."* Post-processing: format the answer cleanly, attach source document IDs, include a confidence score, highlight key action items, enable click-to-verify.

The complete flow in one line — as shown at the bottom of the slide:
**Question → embed → Query Vector → search → Top-K Docs → inject → Rich Prompt → generate → Raw Answer → format → Cited Response**

---

## SLIDE 20 — Complete Pipeline Summary

Here's the beautiful thing about LangChain's LCEL — LangChain Expression Language. The entire pipeline we just walked through? It can be expressed in 4 lines:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
answer = chain.invoke("How do I fix authentication failures?")
```

That's the whole RAG pipeline. Document loading, chunking, embedding, vector storage — that's setup. The runtime chain is this elegant pipe expression.

This is the power of LangChain's component model. Each piece is a standardized interface. You can swap out the LLM, the vector store, the embedding model, and the rest of the chain doesn't change.

---

## SLIDE 21 — Complete Pipeline Data Flow

Let me show you the data flow visually. There are two parallel pipelines:

**Indexing pipeline (offline):**
Raw Documents → chunk → Text Chunks → embed → Vectors → store → Vector DB

**Query pipeline (online):**
User Query → embed → Query Vector → search (via Vector DB) → Top-K Docs → template → Augmented Prompt → generate → Grounded Answer

The Vector DB is the single bridge connecting both pipelines. This is why choosing the right vector store matters — it has to be fast enough for real-time queries and reliable enough for production.

---

## SLIDE 22 — Building the Baseline (Section Title)

Now that we understand the architecture, let's talk about actually building it — the components you need and how they connect.

---

## SLIDE 23 — The LangChain Component Stack

Think of LangChain as a box of Lego bricks. Each component has a standardized interface, and LCEL is how you snap them together.

Four categories of components:

**Data Layer:** `Document` objects wrap your text with metadata. `RecursiveCharacterTextSplitter` handles chunking. These are your inputs.

**Embedding + Storage:** `OpenAIEmbeddings` converts text to vectors. Chroma (or FAISS/Pinecone) stores and searches them. These make your data searchable.

**Prompt + LLM:** `ChatPromptTemplate` structures your prompt with placeholders for context and question. `ChatOpenAI` with temperature=0 generates grounded answers.

**Orchestration:** LCEL's pipe operator (`|`) chains everything together. `RunnablePassthrough` lets you pass the original query through while other branches retrieve context.

The key mental model: **LangChain gives you Lego blocks. LCEL is how you snap them together.**

---

## SLIDE 24 — Technology Stack Behind RAG

Just to give you the bigger picture — RAG doesn't exist in isolation. It sits at the intersection of four technology domains:

- **Vector Databases** — engineered specifically for high-dimensional similarity search
- **Transformer Models** — both embedding models (text → vector) and language models (vector-grounded text → answer)
- **Semantic & Keyword Search** — the retrieval mechanisms themselves
- **Classic ML Techniques** — NLP preprocessing, classification for routing, clustering for organization

When people say "naive RAG," they mean a basic pipeline that uses these four pillars without query optimization, iterative refinement, or agentic reasoning. That's what we're building today as a baseline — and in Module 6, we'll see what happens when we add intelligence on top of it.

---

## SLIDE 25 — Conversation with History

Here's a capability that makes the difference between a useful chatbot and a frustrating one: **conversation memory**.

Without memory, watch what happens:
- Turn 1: "What's ticket TICK-001?" → Agent answers correctly
- Turn 2: "How was it resolved?" → Agent replies: "Resolved? What ticket are you referring to?"

That's infuriating. Every turn is treated as a brand new conversation.

With memory:
- Turn 1: Same question, same answer
- Turn 2: "How was it resolved?" → Agent remembers TICK-001 from Turn 1, answers directly: "It was resolved by clearing active sessions and updating the SAML configuration."

The implementation is straightforward. You maintain a `chat_history` list. Each turn appends a `HumanMessage` and an `AIMessage`. Your prompt template has a `MessagesPlaceholder` that injects this history before the current question.

Notice what the prompt looks like at Turn 3:
1. System message with instructions
2. Full conversation history injected
3. Fresh retrieved context for the new question
4. The current question

The LLM sees all of this context at once and can reason across the entire conversation.

One thing to be aware of: history grows linearly. By Turn 10 you've got 20 messages. This has token cost and context window implications — something we'll come back to when we talk about memory strategies in Module 6.

---

## SLIDE 26 — Module 4 Demo

*[Transition to live demo]*

Alright — this is where we move from slides to code. Let's run Module 4 together. This demo walks through the complete RAG pipeline — from loading the support tickets, chunking them, building the vector store, and running multi-turn conversations against it.

Open your terminal, navigate to the Module 4 directory, and let's do this together.

*[Run the demo — encourage students to follow along on their own machines]*

---

---

# Q&A AND BREAK

---

## SLIDE 27 — Q&A

Before we move to evaluation — any questions on the RAG pipeline? On the offline vs. online phases? On how LangChain components connect? On the conversation history implementation?

*[Address questions. Remind people: no question is too basic. Others are thinking the same thing.]*

---

## SLIDE 28 — 10-Minute Break

We're going to take a 10-minute break right now. The timer is on the screen. Please be back when it hits zero — we've got a lot of ground to cover on evaluation and Agentic RAG.

Stretch, grab a drink, check your environment setup. If you had any issues with Module 4, TAs are available in Discord right now.

---

---

# SECTION 2: RAG EVALUATION

---

## SLIDE 29 — Today's Agenda (Evaluation Focus)

Welcome back. We've completed the pipeline section. Now we're moving into what I consider one of the **most underrated topics in AI engineering**: evaluation.

Most teams skip this. They build the pipeline, it "seems to work," they ship it, and then users start complaining. Evaluation is how you avoid that outcome.

---

## SLIDE 30 — Why Evaluation Matters

Let me be direct about this.

**Without evaluation, you are guessing.**

When you change your chunk size from 500 to 300 tokens, did that help or hurt? When you switch from similarity search to MMR, did retrieval quality improve? When you update your prompt template, did the answers get better or did you just introduce a regression somewhere?

Without measurement, you cannot answer any of these questions. You're flying blind.

**With evaluation:**
- You make data-driven decisions
- You catch regressions before users do
- You can systematically optimize
- You can demonstrate real business value to stakeholders

The core principle: **what gets measured gets improved.**

This isn't optional for production systems. It's the difference between a demo and a product.

---

## SLIDE 31 — The Two-Layer Framework

Here's the evaluation framework we're going to use. Six metrics, organized into two layers.

**Layer 1: Retrieval Metrics** — Did we pull the right documents?

- **Precision@K:** Of the K documents we retrieved, how many were actually relevant? Formula: `Relevant in Top-K / K`. If we retrieved 5 docs and 3 are relevant, Precision@5 = 0.60. High precision = less noise.

- **Recall@K:** Of all the relevant documents that exist, how many did we find? Formula: `Relevant in Top-K / Total Relevant`. If 4 relevant docs exist and we found 3, Recall@5 = 0.75. High recall = good coverage.

- **F1@K:** The harmonic mean of precision and recall. `2 × (P × R) / (P + R)`. Gives you a single balanced metric. With P=0.60 and R=0.75, F1 = 0.67.

**Layer 2: Generation Metrics** — Did we generate a good answer from those documents?

- **Groundedness:** Is every claim in the answer supported by retrieved documents? Target: above 0.85. Below that is hallucination risk. This is the most critical production metric.

- **Completeness:** Does the answer cover all key points from the relevant documents? Target: above 0.75. A grounded but incomplete answer leaves users missing critical information.

- **Answer Relevance:** Does the answer actually address what the user asked? Target: above 0.80. You can have a grounded and complete answer that's still off-topic.

**Critical principle:** Both layers must pass. Great retrieval + bad generation = bad answer. Bad retrieval + great generation = hallucination. You need both.

---

## SLIDE 32 — Ground Truth Setup

Here's the thing about evaluation metrics: they're meaningless without ground truth. You need a labeled dataset — a set of queries where you know which documents are relevant.

Here's how to build one:

1. **Collect real queries** — from production logs or stakeholder input. Not made-up ones.
2. **Manually tag relevant documents** — for each query, identify which ticket IDs are relevant.
3. **Use multiple annotators** — 2-3 reviewers per query. Measure agreement with Cohen's Kappa — this tells you how much annotators agree beyond random chance. A value of 1.0 means perfect agreement, 0 means chance-level agreement, and negative values mean systematic disagreement. Above 0.6 is acceptable; above 0.8 is excellent.
4. **Start small** — 15-50 queries covering your main categories (Auth, DB, Payment, Performance).
5. **Scale up** — expand to 50-100 queries for production-grade evaluation.

A sample ground truth entry looks like:
```
query_id: Q1
question: "How do I fix auth failures?"
relevant_ids: [TICK-001, TICK-011, TICK-014]
category: Authentication
```

This labeled dataset is the foundation of everything that follows. The quality of your evaluation is only as good as the quality of your ground truth.

---

## SLIDE 33 — Precision@K, Recall@K, F1@K

Let me make these formulas concrete with an example.

Say our ground truth says 4 tickets are relevant for the query "How do I fix auth failures?" Our retriever returns 5 tickets, 3 of which are in the ground truth set.

- **Precision@5** = 3 relevant / 5 retrieved = **0.60** — 60% of what we returned was useful
- **Recall@5** = 3 found / 4 total relevant = **0.75** — we found 75% of what existed
- **F1@5** = 2 × (0.60 × 0.75) / (0.60 + 0.75) = **0.67**

The F1 score tells you the balanced picture. A high F1 means you're neither missing important documents nor flooding the context with irrelevant ones.

Key tradeoff to remember: **you cannot maximize both precision and recall simultaneously.** Increasing K tends to improve recall but can hurt precision. F1 is your balancing tool.

---

## SLIDE 34 — Interpreting K Tradeoffs

This is a question I get a lot: "What K should I use?"

Here's the decision tree:

**K = 1-3:** High precision, lower recall. Good when you need every retrieved document to be spot-on. Regulatory workflows, compliance contexts.

**K = 3-5:** The sweet spot for most RAG systems. Start here. Measure both P@K and R@K. Adjust based on what you find.

**K = 7+:** Higher recall, lower precision. You might capture more relevant documents, but you're also flooding the LLM's context window with noise. Beyond K=7 rarely helps.

**Use case guidance:**
- **Support/troubleshooting:** Favor recall. You don't want to miss the one ticket that has the fix. False negatives are worse than noise.
- **Compliance/regulatory:** Favor precision. Every retrieved document must be defensibly relevant.

Start with K=3. Measure. Adjust. Don't guess.

---

## SLIDE 35 — Interpreting K Tradeoffs (Visual)

*[Point to the flowchart as you walk through each box]*

This slide takes the same logic and makes it visual — a decision tree that maps K values to outcomes and use cases. Let me walk through it left to right.

**The K Tradeoff spectrum (center column):**

- **Low K (e.g. K=1–3):** Higher Precision, less noise. Lower Recall — you may miss relevant documents. Fewer docs retrieved, but most are relevant. This is the right zone when every document must defensibly earn its place in the prompt.

- **Sweet Spot (K=3–5):** Best balance for most RAG systems. Start here. Measure P@K and R@K together, then adjust left or right based on what the numbers tell you. This is where the green box sits for a reason.

- **High K (e.g. K=7+):** Higher Recall — captures more documents. Lower Precision — more noise enters the context window. Rarely helps beyond K=7, and it can overload the LLM with irrelevant content that actively degrades the answer quality.

**Practical Guidance (top-right dark box):**
- Start with K=3
- Measure both P@K and R@K
- Adjust based on use case
- K=3 to K=5 is optimal for most RAG
- Beyond K=7 rarely helps

**How to Choose K (right panel) — the use-case decision:**

**Favor Precision:** Strict compliance or regulatory workflows. Every retrieved document must be relevant. Optimize for Precision@1 and MRR — Mean Reciprocal Rank — which rewards getting the most relevant document ranked first.

**Favor Recall:** Troubleshooting or support use cases. Don't miss the one document that contains the fix. In a support desk, false negatives — missing relevant tickets — are far more costly than retrieving a little extra noise.

The key message: **there is no universally correct K.** It depends on what your users need and what the cost of mistakes looks like in your domain. That's why we measure both P@K and R@K rather than optimizing for just one.

---

## SLIDE 36 — Generation Evaluation Metrics

Even when retrieval is perfect — you found exactly the right documents — generation can still fail.

The LLM can:
- Add information not present in the retrieved context (hallucination)
- Answer only part of the question (incompleteness)
- Generate an answer that's technically grounded but doesn't address what the user actually asked (irrelevance)

That's why we evaluate generation independently.

**Groundedness (Faithfulness):** The most critical metric. Every claim in the answer must be traceable to the retrieved context. We measure this by having a judge LLM enumerate each claim and check whether the context supports it. Score below 0.85 = hallucination risk. Fix this before you deploy anything.

**Completeness:** A system might be perfectly grounded but only address one of three relevant aspects in the documents. Target above 0.75. Evaluated by comparing against a reference answer that covers all expected points.

**Answer Relevance:** Is the answer on-topic? An answer can be grounded, complete, and still miss what the user actually needed. Target above 0.80. Evaluated via semantic alignment between question and answer.

Bottom line: **right documents can still produce wrong answers.** Always evaluate generation.

---

## SLIDE 37 — Generation Metrics Framework (Visual)

*[Use this slide as a visual reference while discussing the three metrics]*

The generation quality layer sits on top of the retrieval layer. Both must pass for the system to be production-ready.

Quick visual anchors:
- Groundedness → *Is every claim supported?* → Target 0.85+
- Completeness → *Did we cover all key points?* → Target 0.75+
- Answer Relevance → *Is the answer on-topic?* → Target 0.80+

---

## SLIDE 38 — LLM-as-Judge

So how do we actually measure groundedness and completeness at scale? We can't have humans review every answer — that doesn't scale.

Enter **LLM-as-Judge**.

The idea: use a powerful LLM to score answer quality against a rubric. You send it:
- The original question
- The retrieved context
- The generated answer
- A scoring rubric with clear anchors

The judge LLM returns a score from 0.0 to 1.0 and a rationale.

**Best practices:**
- Use rubric-based prompts with clear scoring anchors: 1.0 = fully supported, 0.5 = partially supported, 0.0 = contradicts or unsupported
- Ask for chain-of-thought: "List all claims → check each against context → compute score." This makes the scoring auditable.
- Keep temperature at 0 for consistency across runs
- Use a stronger model than the one being evaluated when possible — you want the judge to be smarter than the defendant

**Honest caveats to acknowledge:** LLM judges can have model bias. They have API cost per evaluation call. They're not always accurate on edge cases. They're not a perfect replacement for human review — they're a force multiplier.

The right approach: LLM-as-Judge for scale and rapid iteration, with spot-checks by humans on 10-15% of scored examples.

---

## SLIDE 39 — LLM-as-Judge Workflow (Visual)

*[Walk through the diagram left to right — INPUT → JUDGE LLM → OUTPUT, then Advantages vs Caveats, then Recommendation]*

Input → Judge LLM → Output. Let's unpack each box.

**INPUT:** Question + Retrieved Context + Generated Answer + Scoring Rubric. All four go in together. The judge sees the full picture — not just the answer in isolation.

**JUDGE LLM (center, dark blue):** Four steps internally:
1. List all claims in the generated answer
2. Check each claim against the retrieved context
3. Score per rubric
4. Provide a rationale

Temperature = 0 for consistency. Every evaluation run produces the same score for the same input.

**OUTPUT:** A score from 0.0 to 1.0, plus a short rationale per metric. The scoring anchors: 1.0 = fully supported, 0.5 = partial, 0.0 = unsupported or contradicts context.

**Advantages vs Caveats:**

The green advantages box: captures semantic meaning, understands nuance, scales to large datasets, cost-effective vs. human review, and chain-of-thought reasoning makes it auditable.

The red caveats box: model bias is possible, there's API cost per evaluation call, it's not always accurate on edge cases, and there's parsing variability across runs.

**The Recommendation (bottom-right dark box):**
Use LLM-as-Judge for scale and rapid iteration. Spot-check 10–15% of scores with human reviewers. Measure agreement between human and LLM scores — when it's high, you can trust the automation. Use a stronger model than the one being evaluated.

Now that we've defined all six metrics — three retrieval, three generation — plus our evaluation methodology, let's look at how you actually use these numbers to manage a production RAG system. That's the Targets and Alert Thresholds coming up next.

---

## SLIDE 40 — Targets & Alert Thresholds

Here are the production benchmarks I recommend for RAG systems:

| Metric | Target (Green) | Warning (Yellow) | Critical (Red) |
|---|---|---|---|
| Precision@3 | > 0.80 | < 0.70 | < 0.50 |
| Recall@3 | > 0.70 | < 0.60 | < 0.40 |
| F1@3 | > 0.75 | < 0.65 | < 0.45 |
| Groundedness | > 0.85 | < 0.75 | < 0.60 |
| Completeness | > 0.75 | < 0.65 | < 0.50 |

**What to do at each level:**
- **Target met:** Keep monitoring. Log metrics for trend analysis.
- **Warning band:** Investigate root cause. Plan an improvement sprint.
- **Critical band:** Block deployment immediately. Escalate. The system is producing unreliable outputs.

These thresholds aren't arbitrary — they're calibrated for a support desk context where incorrect guidance has real consequences. Adjust them for your specific use case and risk tolerance.

---

## SLIDE 41 — Failure Pattern Playbook

When your metrics are below target, this flowchart tells you where to look.

**Step 1: Read your metrics.** Run both retrieval evaluation (P@K, R@K, F1) and generation evaluation (Groundedness, Completeness). Match results to one of four failure patterns.

**Pattern A — High Precision + Low Recall:**
Your retriever is too narrow. It finds highly relevant docs but misses others. Fixes: increase K, use smaller chunks, add query expansion, include synonyms.

**Pattern B — Low Precision + High Recall:**
Your retriever is too noisy. It finds everything but drowns the LLM in irrelevant content. Fixes: use MMR, add metadata filters, improve embeddings, switch to hybrid search.

**Pattern C — Strong Retrieval + Weak Generation:**
You're finding the right docs but the LLM isn't using them well. Fixes: improve your prompt template, add few-shot examples, increase model strength, set temperature to 0.

**Pattern D — Low Groundedness:**
Hallucination risk. The LLM is inventing claims. Fixes: use grounding prompts that explicitly forbid adding information not in context, require citations, add a verification step.

**Step 2: Apply the fix, re-evaluate, and confirm the targeted metric improved without causing regressions in other metrics.**

---

## SLIDE 42 — A/B Testing & Evaluation Pipeline

The failure playbook tells you what to fix. A/B testing tells you whether your fix worked.

The process:
1. Define your baseline configuration (what you have today)
2. Build one or more candidate configurations (your proposed fixes)
3. Run every evaluation query through all configurations
4. Compute all six metrics for each
5. Compare side-by-side

**The configuration that lifts the weakest metric without regressing others wins.**

One critical addition: always track **latency and cost** alongside quality metrics. A 5% accuracy gain that costs 3x more latency is probably not worth it in a real-time support context. Every improvement decision is a tradeoff.

---

## SLIDE 43 — A/B Testing Results (Visual)

*[Walk through the example results on the slide]*

Here's a real example of what A/B test results look like:

- **Baseline:** P@K 0.750, F1 0.667, Groundedness 0.750
- **Hybrid Retriever (Candidate A):** +11% F1 lift → F1 0.741, Groundedness 0.780
- **Optimized Prompt (Candidate B):** +7% additional lift → F1 0.793, Groundedness 0.850

The hybrid retriever improved retrieval metrics. The optimized prompt pushed groundedness above target. Combined, they moved the system from warning band to full green.

**Release gate logic:**
- **All metrics at target → PASS.** Deploy.
- **Any metric in warning band → REVIEW.** Flag for human review. Plan improvement sprint.
- **Any metric in critical band → BLOCK.** Do not deploy. Escalate.

And remember: **evaluation is not a one-time event.** Re-run as your corpus grows, as user queries evolve, and as models update. This is an ongoing quality gate, not a checkbox.

---

## SLIDE 44 — Module 5 Demo

*[Transition to live demo]*

Let's make this concrete with Module 5. This demo implements the complete evaluation framework — ground truth setup, all six metric calculations, LLM-as-Judge, the alert thresholds — everything we just talked about in working code.

Navigate to the Module 5 directory and follow along.

*[Run the demo]*

---

---

# Q&A AND BREAK

---

## SLIDE 45 — Q&A

Questions on evaluation? On any of the six metrics? On how to build ground truth? On LLM-as-Judge?

*[Address questions — take at least 5-10 minutes here, evaluation concepts take time to settle]*

---

## SLIDE 46 — 5-Minute Break

Five-minute break. Back when the timer hits zero — we've got the most exciting section coming up: Agentic RAG.

---

---

# SECTION 3: AGENTIC RAG WITH LANGCHAIN

---

## SLIDE 47 — Today's Agenda (Agentic RAG Focus)

Last section of the day, and in my opinion the most exciting. We've built a pipeline, we've learned to evaluate it. Now we're going to make it intelligent.

---

## SLIDE 48 — Direct RAG vs. Agentic RAG

Let's start with a clear comparison.

**Direct RAG (what we built in Module 4):**
`Query → Embed → Retrieve → Generate`

Fixed pipeline. Every query follows the same path regardless of what it's asking. Simple, fast, predictable. Great for single-step questions like "What is ticket TICK-001?" or "How do I fix authentication errors?"

**Agentic RAG (Module 6):**
`Query → Agent Thinks → Pick Tool → Observe → Act or Answer`

Reasoning loop. Different path for different queries. The agent decides which tool to use, uses it, observes the result, and decides what to do next. Higher latency, higher cost, harder to debug — but capable of handling complexity that Direct RAG fundamentally cannot.

**The synthesis:** You don't have to choose one. Build both. Use Direct RAG for simple, predictable queries where speed and cost matter. Route complex, multi-step queries to the Agentic layer. This **hybrid approach** gives you the best of both worlds.

---

## SLIDE 49 — Agent Architecture

Here's what the agent's brain looks like.

On the left: a **Tool Registry** — a set of functions the agent can call:
1. `SearchSimilarTickets` — semantic vector search
2. `GetTicketByID` — direct lookup by ticket ID
3. `SearchByCategory` — filter by category label
4. `GetTicketStatistics` — aggregate counts and distributions

In the center: a user query — "Find critical database issues and compare their solutions."

On the right: the **Agent Reasoning Loop**:
1. **Observe:** Read the query or the previous tool result
2. **Think:** Which tool do I need? Do I have enough information to answer?
3. **Act:** Call the selected tool with appropriate inputs
4. Back to Observe with the tool's result
5. When the agent determines it has enough information → **Final Response** with sources and citations

The agent is not hard-coded to follow a sequence. It decides at each step what to do next.

---

## SLIDE 50 — The ReAct Pattern

The agent loop we just described has a formal name: **ReAct** — Reasoning + Acting.

The theoretical ReAct pattern:
1. Input query
2. LLM produces a **Thought** — reasoning about what to do
3. LLM outputs an **Action** — which tool to call
4. External environment executes the action
5. **Observation** — result comes back to LLM
6. Decision: "Done?" If yes → Final Answer. If no → back to step 3

In practice, **LangGraph** implements this loop. The key components: a start node, a model node (Thought + Action), a tools node (execution), and an end node (Final Answer). LangGraph manages the state between iterations.

The pattern: **Reason → Act → Observe → Repeat.**

This is what makes agents powerful — they can handle queries that require multiple steps, multiple sources, and adaptive decision-making.

---

## SLIDE 51 — Tool Design Principles

Here's where most agent implementations fall apart: **tool design**. The quality of your tools is the single biggest factor in agent performance.

**Principle 1 — Single-purpose, clearly scoped tools.**
Each tool does one thing well. "search_tool" is bad. "SearchSimilarTickets" is good. The name alone tells the agent when to use it. Vague names lead to wrong choices.

**Principle 2 — Precise descriptions.**
The description is the only guidance the agent has for selecting tools. It must specify: what the tool does, when to use it, the expected input format, and what the output looks like.

Think of the description as a docstring written for a colleague who has never seen your codebase. If they couldn't pick the right tool from the description alone, rewrite it.

**Principle 3 — Keep the toolset focused: 3 to 7 tools.**
Too few tools: the agent lacks capability. Too many: selection accuracy drops sharply because the LLM starts confusing similar tools or picking suboptimal ones.

If you find yourself needing more than 7 tools: group related functions into a single tool with sub-commands, or route to specialized sub-agents.

---

## SLIDE 52 — Tool Design Principles (Visual Reference)

*[Use this as a reference during Q&A and demo]*

The three-part anatomy of every tool:
1. `name` → identifier
2. `func` → the function it calls
3. `description` → when and how the agent uses it

**Good example:**
- Name: `SearchSimilarTickets`
- Description: "Search for similar tickets using semantic similarity. Perfect for finding related issues. Use for 'how-to' or 'what causes' queries."

**Bad example:**
- Name: `search_tool`
- Description: "Searches stuff."

The bad description gives the agent no information. It will use this tool randomly or not at all.

**DOs:**
- Clear, specific descriptions with use-case guidance
- Max iterations set to 3-5 to prevent infinite loops
- Error handling in every tool — return a helpful string, never raise an unhandled exception
- Limit tool count to 3-7

**DON'Ts:**
- Generic, vague tool names
- Tools that can crash silently
- Using agents when Direct RAG would do
- Ignoring context window limits — long conversations eat tokens fast

---

## SLIDE 53 — Example Toolset in Support Desk

Here's the concrete toolset for our support desk agent:

1. **SearchSimilarTickets** — Semantic vector search. Input: problem description. Output: top-K similar tickets. Use this for troubleshooting queries.

2. **GetTicketByID** — Direct lookup. Input: exact ticket ID like TICK-005. Output: full ticket details. Use this when the user references a specific ticket.

3. **SearchByCategory** — Filtered retrieval. Input: category label (Authentication, Database, Payment, Performance). Output: all tickets in that category. Use this for scoped exploration.

4. **GetTicketStatistics** — Aggregate analytics. Input: optional category filter. Output: counts, priority distributions, resolution rates. Use this for overview and reporting queries.

Four tools. Clean. Purposeful. Each one has a distinct use case that the agent can identify from the description.

---

## SLIDE 54 — Tool Routing (Visual)

*[Walk through the routing diagram]*

The agent's routing intelligence emerges from the tool descriptions combined with the query semantics:

- "How do I fix login errors?" → **SearchSimilarTickets** — semantic troubleshooting match
- "What is TICK-005?" → **GetTicketByID** — direct lookup, no search needed
- "Show all payment issues" → **SearchByCategory** with "Payment"
- "Find critical DB issues and compare their solutions" → **SearchByCategory** then **GetTicketByID** per result — multi-step chaining

That last one is where it gets interesting. No single tool call can answer that question. The agent has to chain tools. That's the real power of agentic RAG.

---

## SLIDE 55 — Tool Selection Examples

Let me reinforce the routing with a clear example for each tool:

1. "How do I fix login issues?" — Describing a problem, not referencing a ticket. → **SearchSimilarTickets**

2. "Show me ticket TICK-005" — Explicit ticket ID. No search needed. → **GetTicketByID**

3. "What payment issues have we seen?" — Topic exploration, not a specific ticket. → **SearchByCategory("Payment")**

4. "Give me a database overview" — Aggregate information wanted, not individual tickets. → **GetTicketStatistics("Database")**

**The key insight:** It's the tool descriptions that make these routing decisions possible. Ambiguous descriptions → wrong routing → wrong answers. Every minute you spend improving tool descriptions pays dividends in agent accuracy.

---

## SLIDE 56 — Multi-Step Reasoning

Here's the query that demonstrates the real value of agentic RAG: **"Find critical database issues and compare their solutions."**

Direct RAG cannot answer this. It's one retrieval, one generation. This query requires:
1. Finding all database tickets
2. Filtering for critical priority
3. Fetching full details for each critical ticket
4. Synthesizing a comparison across multiple tickets

Watch how the agent decomposes it:
1. `SearchByCategory("Database")` → returns all database-related tickets
2. Agent reasons: *filter for critical priority* — no tool needed, this is a reasoning step
3. `GetTicketByID("TICK-002")` → full details for first critical ticket
4. `GetTicketByID("TICK-018")` → full details for second critical ticket
5. Agent synthesizes a comparison → delivers final answer

**Decomposition + Synthesis.** That's the core pattern. Breaking a complex question into sub-tasks, executing them, and assembling the results into a coherent response.

This is something no single retrieval pass can do.

---

## SLIDE 57 — Multi-Step Reasoning (Visual Diagram)

*[Walk through the three iterations]*

Let's trace the agent's reasoning across three iterations:

**Iteration 1:** Think: "I need database tickets first. SearchByCategory." → Act: SearchByCategory("Database") → Observe: Got TICK-002, TICK-012, TICK-019. Need details. → Not done.

**Iteration 2:** Think: "I need full details for each to compare." → Act: GetTicketByID("TICK-002"), GetTicketByID("TICK-007") → Observe: Got resolution details. Need more. → Not done.

**Iteration 3:** Think: "I have enough data to compare solutions." → Act: Synthesize comparison → Decide: Have full coverage. → Done → Final Response.

Three tool calls. Three iterations. One complex question answered correctly.

---

## SLIDE 58 — Best Practices

Five practices that will determine whether your agent works in production:

1. **System prompt clarity.** Be explicit. Tell the agent: use tools before guessing, cite sources, admit when you don't know, stop when you have enough information. A vague system prompt produces vague behavior.

2. **Tool descriptions are critical.** I'll say it again: spend more time on tool descriptions than on anything else. A poorly described tool creates confident wrong behavior. That's worse than no tool.

3. **Error handling in every tool.** Tools should never raise unhandled exceptions. Return a clear, actionable error string. "No ticket found with ID TICK-999" gives the agent something to work with. A Python traceback does not.

4. **Set max iterations.** Cap the agent loop at 3-5 iterations. If it hasn't answered by then, it's stuck. Return a graceful fallback: "I wasn't able to fully answer. Here's what I found..."

5. **Monitor costs.** Agentic RAG is more expensive than Direct RAG — every reasoning step and tool call consumes tokens. Track per-query token usage. Consider using a cheaper model for agent reasoning (e.g., GPT-4o-mini) and a stronger model only for final synthesis.

---

## SLIDE 59 — Live Poll

*[Launch Zoom poll]*

Quick temperature check before we wrap up. Two minutes. We're launching a Zoom poll — please participate.

*[Pause for poll responses]*

---

## SLIDE 60 — Module 6 Demo

*[Transition to live demo]*

Last demo of the day. Module 6 — the full Agentic RAG implementation. This brings together everything: the tool registry, the ReAct reasoning loop, conversational memory, and best practices.

Navigate to Module 6 and follow along.

*[Run the demo — show multi-step reasoning in action with the support desk query]*

---

## SLIDE 61 — Thank You

That's a wrap on Week 3.

Today we covered a lot of ground:
- **Module 4:** Complete RAG pipeline with conversation history
- **Module 5:** Two-layer evaluation framework with six metrics and LLM-as-Judge
- **Module 6:** Agentic RAG with the ReAct pattern, tool design, memory strategies, and debugging

**Your assignments this week:** Complete the MCQ and the coding assignment in UpLevel. The coding assignment will have you building and evaluating a RAG pipeline. Don't skip the evaluation part — it's the most valuable skill you'll practice.

**Resources:** Discord is active, TAs are available, coaching sessions are on Wednesday. Use everything.

Thank you all for a great session. See you next week.

---

*End of Talking Script*

---

## Appendix: Timing Guide

| Section | Slides | Suggested Time |
|---|---|---|
| Intro & Admin | 1–10 | 15 min |
| Week 2 Recap | 11–12 | 5 min |
| RAG Foundation | 13–16 | 20 min |
| RAG Pipeline Architecture | 17–21 | 25 min |
| Building the Baseline | 22–26 | 20 min |
| **Q&A + 10-min Break** | 27–28 | 20 min |
| Why Evaluation | 29–30 | 10 min |
| Evaluation Framework (Metrics) | 31–38 | 35 min |
| Targets & Failure Playbook | 39–43 | 20 min |
| **Module 5 Demo** | 44 | 15 min |
| **Q&A + 5-min Break** | 45–46 | 15 min |
| Direct RAG vs Agentic RAG | 47–50 | 20 min |
| Tool Design | 51–55 | 20 min |
| Multi-Step & Memory | 56–59 | 20 min |
| Best Practices & Pitfalls | 60–62 | 15 min |
| **Poll + Module 6 Demo** | 63–64 | 20 min |
| Wrap-up | 65 | 5 min |
| **TOTAL** | | **~4 hours** |
