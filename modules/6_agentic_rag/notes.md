# Module 6: Agentic RAG with LangChain

## Overview
This module teaches how to build intelligent agents that can use RAG as one of many tools, making decisions about when and how to retrieve information.

---

## What is Agentic RAG?

### Traditional RAG (Module 4)
```
User Query → Embed → Retrieve → Generate Response
```
- Fixed pipeline
- Always retrieves, even if not needed
- Single retrieval strategy

### Agentic RAG (Module 6)
```
User Query → Agent → [Tool Selection] → Generate Response
                ↓
         Multiple Tools:
         • RAG Retrieval
         • Direct Lookup
         • Calculations
         • API Calls
```
- Flexible decision-making
- Multi-step reasoning
- Combines multiple information sources

---

## Key Concepts

### 1. LangChain Agents

**Agent Components:**
- **LLM**: The "brain" that makes decisions
- **Tools**: Functions the agent can call
- **Prompt**: Instructions for the agent
- **Executor**: Runs the agent loop

**Agent Loop:**
```python
while not done:
    1. Observe current state
    2. Think: which tool to use?
    3. Act: call the tool
    4. Observe: see the result
    5. Repeat or finish
```

### 2. Tool Design

**Tool Structure:**
```python
Tool(
    name="ToolName",
    func=function_to_call,
    description="Clear description of what this tool does"
)
```

**Good Tool Descriptions:**
- Clearly state the tool's purpose
- Specify when to use it
- Describe expected input format
- Mention what output to expect

**Example:**
```python
Tool(
    name="SearchSimilarTickets",
    func=search_function,
    description="""Search for similar support tickets.
    Input: Problem description or question.
    Use this for 'how to fix' queries."""
)
```

### 3. Memory with RunnableWithMessageHistory

Use `RunnableWithMessageHistory` as the modern memory orchestration layer.
You can still implement three practical strategies with it.

**Strategy A — Full history (buffer-like):**
- Best for: short conversations (< 10 turns)
- Token cost: grows linearly with conversation length

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}

def get_history(session_id: str):
   return store.setdefault(session_id, InMemoryChatMessageHistory())

prompt = ChatPromptTemplate.from_messages([
   ("system", "You are SupportDesk AI."),
   MessagesPlaceholder("history"),
   ("human", "{question}")
])

chain_with_history = RunnableWithMessageHistory(
   prompt | llm,
   get_history,
   input_messages_key="question",
   history_messages_key="history"
)
```

**Strategy B — Windowed history (window-like):**
- Best for: most production use cases
- Token cost: fixed (bounded by window size)

```python
WINDOW_TURNS = 3

def apply_window(session_id: str):
   history = store[session_id]
   history.messages = history.messages[-(WINDOW_TURNS * 2):]
```

**Strategy C — Summary-style history (summary-like):**
- Best for: long conversations where early context still matters
- Token cost: medium (extra compression step)

```python
from langchain_core.messages import SystemMessage

def compress_older_turns(session_id: str, keep_recent_turns: int = 2):
   history = store[session_id]
   recent = keep_recent_turns * 2
   if len(history.messages) <= recent + 1:
      return

   older = history.messages[:-recent]
   summary = " | ".join(f"{m.type}: {m.content[:80]}" for m in older)
   history.messages = [SystemMessage(content=f"Summary: {summary[:600]}")] + history.messages[-recent:]
```

**Comparison:**

| Strategy | Token Cost | Context Loss | Best For |
|----------|-----------|--------------|----------|
| Full history | High (grows) | None | Short chats < 10 turns |
| Windowed history | Low (fixed) | Drops old messages | Most production cases |
| Summary-style history | Medium (compression step) | Low (summarised) | Long sessions |

---

## Comparison: Direct RAG vs Agentic RAG

| Aspect | Direct RAG | Agentic RAG |
|--------|-----------|-------------|
| **Architecture** | Fixed pipeline | Flexible tools |
| **Decision Making** | No decisions | Agent chooses tools |
| **Multi-step** | No | Yes |
| **Latency** | Lower | Higher |
| **Cost** | Lower (fewer LLM calls) | Higher (reasoning calls) |
| **Complexity** | Simpler | More complex |
| **Use Case** | Single retrieval queries | Complex, multi-step tasks |

---

## Agent Patterns

### Pattern 1: RAG as Primary Tool
```python
# Agent mainly uses RAG, but can do other things
tools = [
    rag_tool,
    fallback_tool,
    clarification_tool
]
```

### Pattern 2: Multi-Source Retrieval
```python
# Agent combines multiple retrieval sources
tools = [
    vector_search_tool,
    keyword_search_tool,
    database_lookup_tool,
    api_call_tool
]
```

### Pattern 3: Reasoning + Retrieval
```python
# Agent interleaves reasoning with retrieval
Query: "Find critical database issues and compare solutions"
Agent:
  1. SearchByCategory("Database")
  2. Filter for critical
  3. GetTicketByID for each
  4. Compare and synthesize
```

---

## Best Practices

### ✅ DO:

1. **Write Clear Tool Descriptions**
   - Help the agent choose correctly
   - Include example inputs
   - Specify when to use

2. **Limit Tool Count**
   - Too many tools confuse the agent
   - 3-7 tools is optimal
   - Combine similar functions

3. **Handle Errors Gracefully**
   - Return helpful error messages
   - Agent can retry with corrections
   - Log failures for debugging

4. **Set Max Iterations**
   - Prevent infinite loops
   - 3-5 iterations usually enough
   - Fail gracefully if exceeded

5. **Monitor Token Usage**
   - Agents use more tokens (reasoning)
   - Consider cost implications
   - Use cheaper models for agent reasoning

### ❌ DON'T:

1. **Don't Make Tools Too Generic**
   - Bad: "search_tool" (search what?)
   - Good: "search_similar_tickets", "search_by_category"

2. **Don't Skip Error Handling**
   - Tools should never crash
   - Always return a string
   - Include helpful error messages

3. **Don't Forget Memory Limits**
   - Long conversations → many tokens
   - Summarize or truncate history
   - Consider context window

4. **Don't Use Agents for Simple Tasks**
   - Overkill for single retrieval
   - Higher cost and latency
   - Use direct RAG instead

---

## Advanced Topics

### Multi-Agent Systems
- Multiple specialized agents
- Each agent has specific expertise
- Agents communicate and delegate
- Examples: CrewAI, AutoGen

### Streaming Responses
- Show agent thinking in real-time
- Better UX for long operations
- Use `agent_executor.stream()`

### Custom Agent Types
- `create_react_agent()`: ReAct pattern
- `create_openai_functions_agent()`: Function calling
- `create_structured_chat_agent()`: Structured tool inputs

### Tool Validation
- Pydantic models for tool inputs
- Type checking
- Automatic error messages

---

## Common Pitfalls

### Pitfall 1: Agent Loops
**Problem:** Agent keeps using same tool repeatedly
**Solution:** Better tool descriptions, add iteration limits

### Pitfall 2: Wrong Tool Selection
**Problem:** Agent chooses incorrect tool
**Solution:** Improve descriptions, add examples in prompt

### Pitfall 3: Incomplete Reasoning
**Problem:** Agent stops too early
**Solution:** Better system prompt, encourage thoroughness

### Pitfall 4: Token Explosion
**Problem:** Conversation history grows too large
**Solution:** Use summary memory, prune old messages

---

## Debugging Tips

1. **Enable Verbose Mode**
   ```python
   agent_executor = AgentExecutor(agent, tools, verbose=True)
   ```

2. **Check Tool Descriptions**
   - Are they clear and specific?
   - Do they have examples?

3. **Monitor Agent Scratchpad**
   - See agent's reasoning
   - Understand tool selection

4. **Test Tools Individually**
   - Call tools directly first
   - Ensure they work correctly

---

## When to Choose Which Approach

### Use Direct RAG (Module 4) when:
- ✓ Simple retrieval queries
- ✓ Predictable patterns
- ✓ Low latency needed
- ✓ Cost-sensitive
- ✓ Production stability critical

### Use Agentic RAG (Module 6) when:
- ✓ Complex, multi-step queries
- ✓ Need flexibility
- ✓ Conversational interface
- ✓ Multiple information sources
- ✓ Task decomposition needed

---

## Resources

- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Tool Use Best Practices](https://python.langchain.com/docs/modules/agents/tools/)
