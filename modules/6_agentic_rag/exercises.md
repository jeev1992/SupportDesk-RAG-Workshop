# Module 6 Exercises: Agentic RAG

Complete these exercises after studying `demo.py`. Solutions are in `solutions.py`.

> ✅ **Exercise style for this workshop:** finish each task with **small edits** to existing files (typically 3–15 lines). Avoid creating new large scripts.

---

## Easy Exercises (Start Here!)

### Exercise 1: Run the Demo and Observe

**Task**: Run the demo and understand agent behavior.

```bash
python demo.py
```

**Observe and answer**:
1. Which tool did the agent select for "How do I fix authentication problems?"
2. Which tool did the agent select for "Show me ticket TICK-005"?
3. How many reasoning steps did each query take?

**Key insight**: The agent decides which tool to use based on the query!

---

### Exercise 2: Test Different Queries

**Task**: Run queries and predict which tool will be used.

**Add these queries to demo.py (after PART 2) and run:**
```python
# Test query - prediction: which tool?
test_query = "Show me all database-related tickets"
response = run_agent(test_query)
print(response)
```

**Test these queries** (predict the tool before running):

| Query | Predicted Tool | Actual Tool |
|-------|---------------|-------------|
| "What issues have we seen with payments?" | ? | ? |
| "Get ticket TICK-003" | ? | ? |
| "How many tickets are in each category?" | ? | ? |
| "How to resolve mobile app crashes" | ? | ? |

---

### Exercise 3: Improve Tool Descriptions

**Task**: Make tool selection more accurate by improving descriptions.

**In tools.py, find the `get_tools()` method and improve descriptions:**

```python
Tool(
    name="SearchSimilarTickets",
    # BEFORE: Generic description
    # description="Search for similar tickets"
    
    # AFTER: More specific
    description="""Use this for troubleshooting questions like "how to fix", 
    "why is X happening", or "similar issues to Y". 
    Searches semantically - finds related tickets even if words don't match exactly."""
)
```

**Test these queries after improving descriptions**:
- "What database problems have we seen?" (Should use SearchByCategory)
- "Users can't log in" (Should use SearchSimilarTickets)
- "Get TICK-010 details" (Should use GetTicketByID)

---

### Exercise 4: Add Priority Support (Small Patch)

**Task**: Add priority handling with minimal changes (no new class/file).

**In `tools.py`, pick one existing tool and add a small priority-aware branch.**

Example patch:
```python
if "critical" in query.lower():
    docs = vector_store.similarity_search(query, k=3, filter={"priority": "Critical"})
    return format_results(docs)
```

**Test with**: "Show me all critical priority tickets"

---

## Medium Exercises

### Exercise 5: Multi-Step Queries

**Task**: Write queries that require the agent to use multiple tools.

**Try these multi-step queries:**
```python
query = "How many payment tickets do we have and what was the resolution for the most recent one?"
response = run_agent(query)
```

**Queries to test**:
1. "Find all high priority tickets and show details of the first one"
2. "Compare the resolution for TICK-001 and TICK-005"
3. "What Authentication issues do we have? Give me details on each one."

**Questions**:
- Does the agent decompose the query correctly?
- In what order does it use tools?
- Does it synthesize information from multiple tools?

---

### Exercise 6: Custom Agent Prompt

**Task**: Modify the system prompt to change agent behavior.

**In demo.py, find the system message in `run_agent()` and modify it:**

```python
SystemMessage(content="""You are an expert support desk assistant.

ALWAYS follow these rules:
1. State which tool you're using before searching
2. If the query is ambiguous, ask a clarifying question FIRST
3. When providing solutions, rate your confidence (High/Medium/Low)
4. Suggest related tickets the user might want to explore
""")
```

**Test with**: "Issues with users logging in"

**Expected changes**:
- Agent should mention tool usage
- Should suggest related topics
- Should indicate confidence level

---

### Exercise 7: Interactive Conversation Loop

**Task**: Use a tiny in-file loop for quick testing (no new file).

**Add this at the bottom of `demo.py` temporarily:**
```python
while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    response = run_agent(user_input)
    print(f"\nAssistant: {response}")
```

Keep the loop under ~12 lines.

**Test conversation**:
1. "What authentication issues have we seen?"
2. "Tell me more about TICK-001"
3. "What was the resolution?"

---

### Exercise 8: Error Handling

**Task**: Make tools more robust with error handling.

**Update tools to handle edge cases:**
```python
def get_ticket_by_id(self, ticket_id: str) -> str:
    """Retrieve a specific ticket by its ID."""
    # Handle edge cases
    if not ticket_id or not ticket_id.strip():
        return "Error: Please provide a ticket ID (e.g., TICK-001)"
    
    ticket_id = ticket_id.upper().strip()
    
    # Check format
    if not ticket_id.startswith("TICK-"):
        return f"Error: Invalid format. Ticket IDs look like TICK-001, TICK-002, etc."
    
    # ... rest of the function
```

**Test with**:
- "Get ticket XYZ123" (invalid format)
- "Get ticket TICK-999" (not found)
- "Get ticket" (empty input)

---

## Bonus Challenges

### Bonus: Conversation with Memory

**Task**: Reuse existing memory code in `demo.py` and change only one setting.

```python
# In Part 7, change the window size from k=2 to k=1 and compare behavior.
apply_window(session_id, k_turns=1)
```

**Test this conversation flow**:
1. User: "What issues have we had with databases?"
2. User: "What was the ticket ID?"  ← Should remember context
3. User: "How was it resolved?"  ← Should still remember

---

### Bonus: Agent Evaluation

**Task**: Add a tiny manual evaluation pass (no framework build).

```python
test_cases = [
    {
        "query": "How do I fix login issues?",
        "expected_tool": "SearchSimilarTickets",
        "should_contain": ["authentication", "TICK"]
    },
    {
        "query": "Show ticket TICK-001",
        "expected_tool": "GetTicketByID",
        "should_contain": ["TICK-001"]
    },
    {
        "query": "How many tickets are there?",
        "expected_tool": "GetTicketStatistics",
        "should_contain": ["total", "category"]
    }
]

for test in test_cases:
    response = run_agent(test["query"])
    print(test["query"])
    print(response[:150])
```

Goal: observe behavior quickly without writing a full evaluator.

---

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **Agent** | LLM that decides which tools to use based on the query |
| **Tool** | Function the agent can call with specific inputs |
| **Tool Selection** | Agent reads tool descriptions to choose the right one |
| **Multi-step** | Agent can use multiple tools to answer complex queries |
| **Memory** | Maintaining conversation history for follow-up questions |

---

## When to Use Agentic RAG

| Use Case | Direct RAG | Agentic RAG |
|----------|-----------|-------------|
| Simple Q&A | ✅ | ❌ |
| Low latency needed | ✅ | ❌ |
| Complex multi-step queries | ❌ | ✅ |
| Multiple data sources | ❌ | ✅ |
| Interactive conversation | ❌ | ✅ |
| Predictable behavior | ✅ | ❌ |
| Cost-sensitive | ✅ | ❌ |

---

## 🎉 Congratulations!

You've completed the Agentic RAG module! You now know how to:

1. Build agents with custom tools
2. Optimize tool descriptions for accurate selection
3. Handle multi-step reasoning
4. Implement conversational memory
5. Add error handling for robustness

---

**Need help?** Check `solutions.py` or ask the instructor!
