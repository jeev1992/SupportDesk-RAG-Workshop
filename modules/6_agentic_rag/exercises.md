# Module 6 Exercises: Agentic RAG

Complete these exercises after studying `demo.py`. Solutions are in `solutions.py`.

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

### Exercise 4: Add a Priority Search Tool

**Task**: Add a new tool that filters tickets by priority.

**In tools.py, add this method to the SupportTicketTools class:**
```python
def search_by_priority(self, priority: str) -> str:
    """Find all tickets with a specific priority level."""
    priority = priority.strip().capitalize()
    matching = [t for t in self.tickets if t['priority'].lower() == priority.lower()]
    
    if not matching:
        available = list(set(t['priority'] for t in self.tickets))
        return f"No tickets with priority '{priority}'. Available: {', '.join(available)}"
    
    output = f"Found {len(matching)} tickets with {priority} priority:\n\n"
    for ticket in matching:
        output += f"• [{ticket['ticket_id']}] {ticket['title']} ({ticket['category']})\n"
    
    return output
```

**Then add it to get_tools():**
```python
Tool(
    name="SearchByPriority",
    func=self.search_by_priority,
    description="""Find all tickets with a specific priority level.
    Input should be: Critical, High, Medium, or Low.
    Use when user asks about urgent/important issues or priority levels."""
)
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

**Task**: Build an interactive chat with the agent.

**Create a new file `chat.py`:**
```python
from demo import run_agent

print("=" * 60)
print("Support Assistant (type 'quit' to exit)")
print("=" * 60)

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    
    if not user_input:
        continue
    
    print("\nThinking...")
    response = run_agent(user_input)
    print(f"\nAssistant: {response}")
```

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

**Task**: Build a conversational agent that remembers context.

**Create `conversational_chat.py`:**
```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from tools import SupportTicketTools
import os
from dotenv import load_dotenv

load_dotenv()

# Setup
llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'), temperature=0)
tool_manager = SupportTicketTools()
tools = tool_manager.get_tools()

# ... bind tools (copy from demo.py)

# Maintain conversation history across turns
conversation_history = []

def chat(user_message):
    """Process a message with full conversation memory."""
    global conversation_history
    
    # Add user message to history
    conversation_history.append(HumanMessage(content=user_message))
    
    # Create full message list with system prompt
    messages = [
        SystemMessage(content="You are a support assistant with memory of our conversation.")
    ] + conversation_history
    
    # Run agent loop with tools...
    # Add assistant response to history
    # Return response

# Interactive loop
print("Conversational Support Agent")
while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit']:
        break
    
    response = chat(user_input)
    print(f"\nAssistant: {response}")
```

**Test this conversation flow**:
1. User: "What issues have we had with databases?"
2. User: "What was the ticket ID?"  ← Should remember context
3. User: "How was it resolved?"  ← Should still remember

---

### Bonus: Agent Evaluation

**Task**: Measure agent performance on a test set.

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

def evaluate_agent(test_cases):
    results = []
    for test in test_cases:
        # Run agent, track which tool was called
        # Check if expected_tool was used
        # Check if response contains expected keywords
        pass
    
    return results
```

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
