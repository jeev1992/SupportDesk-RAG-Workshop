# Module 6: Agentic RAG with LangChain

## Overview

This module demonstrates how to build intelligent agents that use RAG as one of many tools, enabling flexible, multi-step reasoning and decision-making.

## What You'll Learn

- ✅ Creating custom tools for LangChain agents
- ✅ Building agents with OpenAI function calling
- ✅ Implementing conversation memory
- ✅ Multi-step reasoning with tool selection
- ✅ Comparing agentic vs direct RAG approaches

## Files

- **`demo.py`** - Complete demonstration with 7 different scenarios
- **`tools.py`** - Custom tool implementations for support ticket operations
- **`notes.md`** - Comprehensive learning notes and concepts
- **`exercises.md`** - Hands-on exercises from beginner to advanced
- **`README.md`** - This file

## Quick Start

### 1. Use Python 3.12 and activate the workshop virtual environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
python --version   # should show 3.12.x
```

**macOS/Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version   # should show 3.12.x
```

### 2. Ensure Dependencies are Installed

```bash
# From the workshop root directory
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Make sure your `.env` file in the project root contains:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### 4. Run the Demo

```bash
cd modules/6_agentic_rag
python demo.py
```

## What the Demo Shows

### Part 1: Setup Agent with Tools
- Initializes LLM and creates 4 custom tools
- Shows tool descriptions and capabilities

### Part 2: Simple Query - RAG Tool Selection
- Agent automatically chooses `SearchSimilarTickets` tool
- Demonstrates semantic search capability

### Part 3: Specific Ticket Lookup
- Agent uses `GetTicketByID` for direct lookups
- Shows intelligent tool selection

### Part 4: Category-Based Search
- Agent filters tickets by category
- Uses `SearchByCategory` tool

### Part 5: Database Statistics
- Agent provides overview and insights
- Uses `GetTicketStatistics` tool

### Part 6: Multi-Step Reasoning
- Agent combines multiple tools
- Demonstrates complex query handling

### Part 7: Conversational Agent with Memory
- Multi-turn conversation with context
- Follow-up questions without re-explanation

## Architecture

```
┌─────────────────────────────────────────┐
│           User Query                     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│      LangChain Agent (LLM Brain)        │
│  • Analyzes query                       │
│  • Selects appropriate tool             │
│  • Reasons about next steps             │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│           Available Tools                │
├─────────────────────────────────────────┤
│ 1. SearchSimilarTickets (RAG)          │
│ 2. GetTicketByID (Direct Lookup)       │
│ 3. SearchByCategory (Filter)           │
│ 4. GetTicketStatistics (Analytics)     │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│        Tool Execution                    │
│  • Vector search (Chroma)               │
│  • Data retrieval                       │
│  • Calculations                         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│      Agent Synthesizes Response         │
│  • Combines tool outputs                │
│  • Generates natural language answer    │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│           User Response                  │
└─────────────────────────────────────────┘
```

## Key Concepts

### Agent vs Direct RAG

| Feature | Direct RAG (Module 4) | Agentic RAG (Module 6) |
|---------|----------------------|------------------------|
| **Flow** | Query → Retrieve → Generate | Query → Think → Tool → Generate |
| **Flexibility** | Fixed pipeline | Dynamic tool selection |
| **Multi-step** | No | Yes |
| **Cost** | Lower | Higher (more LLM calls) |
| **Use Case** | Simple retrieval | Complex reasoning |

### The Four Tools

1. **SearchSimilarTickets** - Semantic RAG search for similar issues
2. **GetTicketByID** - Direct lookup by ticket identifier
3. **SearchByCategory** - Filter tickets by type (Auth, Database, etc.)
4. **GetTicketStatistics** - Analytics and overview of ticket database

## Example Interactions

### Simple Troubleshooting
```
User: "How do I fix authentication problems after password reset?"
Agent: [Uses SearchSimilarTickets]
      → Finds TICK-001
      → Explains the solution
```

### Multi-Step Query
```
User: "Find database-related critical issues and tell me how they were resolved"
Agent: [Uses SearchByCategory("Database")]
      → [Filters for Critical priority]
      → [Uses GetTicketByID for each]
      → [Synthesizes resolutions]
```

### Conversational Follow-up
```
User: "What issues have we had with iOS?"
Agent: [Searches and finds TICK-004]

User: "What was the ticket ID?"
Agent: [References conversation history]
      → "That was TICK-004"

User: "How was it resolved?"
Agent: [Retrieves TICK-004 details again]
      → Explains the resolution
```

## When to Use Agentic RAG

### ✅ Good Use Cases:
- Complex, multi-step queries
- Conversational interfaces
- Combining retrieval with other operations (CRUD, calculations)
- When users need flexibility in querying
- Tasks requiring reasoning and planning

### ❌ Not Ideal For:
- Simple, single-step retrieval
- Latency-critical applications
- Highly predictable query patterns
- Cost-sensitive scenarios
- When you need 100% deterministic behavior

## Exercises

See [`exercises.md`](exercises.md) for 10+ hands-on exercises ranging from beginner to advanced:

- ⭐ Basic: Run demo and observe behavior
- ⭐⭐ Intermediate: Add new tools, improve descriptions
- ⭐⭐⭐ Advanced: Multi-step queries, ticket creation
- ⭐⭐⭐⭐ Expert: Streaming, evaluation, hybrid routing
- ⭐⭐⭐⭐⭐ Challenge: Multi-agent systems

## Troubleshooting

### Agent Loops or Repeats Same Tool
- Check tool descriptions - make them more specific
- Increase `max_iterations` if legitimate multi-step
- Add more context in system prompt

### Agent Chooses Wrong Tool
- Improve tool descriptions
- Add examples in descriptions
- Refine system prompt with clearer guidelines

### Memory Issues / Token Limits
- Use `RunnableWithMessageHistory` with windowed or summary-style history
- Limit conversation history
- Consider truncating old messages

### Performance Issues
- Use faster embedding models
- Cache vector store results
- Use cheaper LLM for agent reasoning (e.g., gpt-4o-mini)

## Next Steps

1. Complete the exercises in [`exercises.md`](exercises.md)
2. Read the detailed concepts in [`notes.md`](notes.md)
3. Experiment with different queries
4. Compare with Module 4's direct RAG
5. Try adding your own custom tools
6. Explore LangGraph for more complex workflows

## Additional Resources

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [ReAct: Reasoning and Acting Paper](https://arxiv.org/abs/2210.03629)
- [Tool Use Best Practices](https://python.langchain.com/docs/modules/agents/tools/)

---

**Happy Learning!** 🚀

Questions? Check the notes or try the exercises!
