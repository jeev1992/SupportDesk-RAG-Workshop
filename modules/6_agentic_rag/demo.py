# -*- coding: utf-8 -*-
"""
Agentic RAG Demo - LangChain Agent with Tools
==============================================

This demo teaches:
1. Creating custom tools for RAG retrieval
2. Building an agent with tool calling
3. Handling conversational context
4. Multi-step reasoning and tool selection
5. Comparing agent-based vs direct RAG
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

from tools import SupportTicketTools

# Load environment variables
load_dotenv()

print("="*80)
print("AGENTIC RAG: LangChain Agent with RAG Tools")
print("="*80)
print("\nThis demo shows how to build an intelligent agent that:")
print("✓ Uses RAG retrieval as a tool (not the only approach)")
print("✓ Decides when to use which tool based on the query")
print("✓ Maintains conversation context across turns")
print("✓ Performs multi-step reasoning")

# ============================================================================
# PART 1: Setup Agent with Tools
# ============================================================================
print("\n" + "="*80)
print("PART 1: Setting Up the Agent")
print("="*80)

print("\nInitializing LLM...")
llm = ChatOpenAI(
    model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
    temperature=0,
    api_key=os.getenv('OPENAI_API_KEY')
)
print("✓ LLM initialized")

print("\nCreating agent tools...")
tool_manager = SupportTicketTools()
tools = tool_manager.get_tools()

# Convert tools to OpenAI function format
tool_definitions = []
for tool in tools:
    tool_definitions.append({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The input to the tool"
                    }
                },
                "required": ["input"]
            }
        }
    })

# Bind tools to LLM
llm_with_tools = llm.bind(tools=tool_definitions)

print(f"✓ Created {len(tools)} tools:")
for tool in tools:
    print(f"  • {tool.name}: {tool.description.split('.')[0]}")

def run_agent(query: str, max_iterations: int = 5) -> str:
    """
    Run a ReAct-style tool-calling loop until the model returns a final answer.

    Loop behavior:
    1) Model sees conversation + tool schema.
    2) Model either answers directly OR emits one/more tool calls.
    3) We execute each tool call, append ToolMessage results.
    4) Repeat until no tool calls remain or iteration cap is reached.
    """
    messages = [
        SystemMessage(content="""You are an expert support desk assistant that helps troubleshoot technical issues.

You have access to a database of previous support tickets with their resolutions. 
Use your tools to find relevant information and provide helpful, accurate answers.

Guidelines:
- ALWAYS search for similar tickets when asked about troubleshooting or "how to fix" questions
- Be specific and reference ticket IDs when providing solutions
- If multiple similar issues exist, mention the most relevant ones
- Admit when you don't have enough information
- Be concise but thorough in your responses
- When appropriate, use multiple tools to gather complete information

Remember: Your primary value is retrieving and applying solutions from past tickets!"""),
        HumanMessage(content=query)
    ]
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # If the model produced no tool calls, treat content as final answer.
        if not response.tool_calls:
            # No more tool calls, return the response
            return response.content
        
        # Execute each requested tool exactly as the model specified.
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            
            print(f"\n🔧 Calling tool: {tool_name}")
            print(f"   Input: {tool_input}")
            
            # Resolve tool by name from the registered tool list.
            # This explicit lookup keeps control in application code (safer than eval).
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            
            if tool_output is None:
                tool_output = f"Error: Tool {tool_name} not found"
            
            print(f"   Output: {tool_output[:200]}...")
            
            # Feed tool output back to model in the expected ToolMessage format.
            # The `tool_call_id` links this output to the originating request.
            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            ))
    
    return "Maximum iterations reached. Could not complete the task."

print("\n✓ Agent ready!")

# ============================================================================
# PART 2: Simple Query - RAG Tool Selection
# ============================================================================
print("\n" + "="*80)
print("PART 2: Simple Query - Agent Selects RAG Tool")
print("="*80)

query1 = "How do I fix authentication problems after password reset?"
print(f"\nQuery: '{query1}'")
print("\nAgent will automatically:")
print("1. Recognize this is a troubleshooting question")
print("2. Choose the SearchSimilarTickets tool")
print("3. Retrieve relevant tickets")
print("4. Synthesize an answer\n")
print("-" * 80)

response1 = run_agent(query1)
print("\n" + "-" * 80)
print("FINAL ANSWER:")
print(response1)

# ============================================================================
# PART 3: Specific Lookup - Different Tool
# ============================================================================
print("\n" + "="*80)
print("PART 3: Specific Ticket Lookup")
print("="*80)

query2 = "Show me details of ticket TICK-005"
print(f"\nQuery: '{query2}'")
print("\nAgent will:")
print("1. Recognize this asks for a specific ticket")
print("2. Choose the GetTicketByID tool")
print("3. Return the exact ticket\n")
print("-" * 80)

response2 = run_agent(query2)
print("\n" + "-" * 80)
print("FINAL ANSWER:")
print(response2)

# ============================================================================
# PART 4: Category Filtering
# ============================================================================
print("\n" + "="*80)
print("PART 4: Category-Based Search")
print("="*80)

query3 = "What payment-related issues have we seen?"
print(f"\nQuery: '{query3}'")
print("\nAgent will:")
print("1. Identify this asks about a category of issues")
print("2. Choose the SearchByCategory tool")
print("3. Show all payment tickets\n")
print("-" * 80)

response3 = run_agent(query3)
print("\n" + "-" * 80)
print("FINAL ANSWER:")
print(response3)

# ============================================================================
# PART 5: Statistics Query
# ============================================================================
print("\n" + "="*80)
print("PART 5: Database Statistics")
print("="*80)

query4 = "Give me an overview of the ticket database"
print(f"\nQuery: '{query4}'")
print("\nAgent will:")
print("1. Recognize this asks for statistics")
print("2. Choose the GetTicketStatistics tool")
print("3. Provide summary insights\n")
print("-" * 80)

response4 = run_agent(query4)
print("\n" + "-" * 80)
print("FINAL ANSWER:")
print(response4)

# ============================================================================
# PART 6: Multi-Step Reasoning
# ============================================================================
print("\n" + "="*80)
print("PART 6: Multi-Step Reasoning")
print("="*80)

query5 = "Find database-related critical issues and tell me how they were resolved"
print(f"\nQuery: '{query5}'")
print("\nAgent will need to:")
print("1. First get category statistics or search by category")
print("2. Then look up specific ticket details")
print("3. Synthesize the resolution information\n")
print("-" * 80)

response5 = run_agent(query5)
print("\n" + "-" * 80)
print("FINAL ANSWER:")
print(response5)

# ============================================================================
# PART 7: Conversational Agent with Memory
# ============================================================================
print("\n" + "="*80)
print("PART 7: Conversational Agent with Memory")
print("="*80)

print("\nSimulating a multi-turn conversation...")

def run_conversational_agent(conversation_history, query: str, max_iterations: int = 5) -> tuple:
    """
    Run the agent while preserving prior conversation turns.

    `conversation_history` should already contain Human/AI/Tool messages from
    previous turns so follow-up questions can resolve references like "that".
    """
    messages = [SystemMessage(content="""You are an expert support desk assistant that helps troubleshoot technical issues.
Use your tools to find relevant information and maintain context across our conversation.""")]
    
    # Replay prior turns before appending the new user query.
    messages.extend(conversation_history)
    messages.append(HumanMessage(content=query))
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return messages, response.content
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            
            print(f"\n🔧 Calling tool: {tool_name}")
            
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            
            if tool_output is None:
                tool_output = f"Error: Tool {tool_name} not found"
            
            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            ))
    
    return messages, "Maximum iterations reached."

# Start conversation
conversation = []

print("\n--- Conversation Turn 1 ---")
conv_query1 = "What issues have we had with iOS?"
print(f"User: {conv_query1}\n")
conversation, conv_response1 = run_conversational_agent(conversation, conv_query1)
print(f"\nAssistant: {conv_response1}")

print("\n--- Conversation Turn 2 (Follow-up) ---")
conv_query2 = "What was the ticket ID for that?"
print(f"User: {conv_query2}")
print("(Notice: The agent remembers the previous context!)\n")
conversation, conv_response2 = run_conversational_agent(conversation, conv_query2)
print(f"\nAssistant: {conv_response2}")

print("\n--- Conversation Turn 3 (Another Follow-up) ---")
conv_query3 = "How was it resolved?"
print(f"User: {conv_query3}\n")
conversation, conv_response3 = run_conversational_agent(conversation, conv_query3)
print(f"\nAssistant: {conv_response3}")

# ============================================================================
# PART 8: Memory Type Comparison
# ============================================================================
print("\n" + "="*80)
print("PART 8: Memory Type Comparison")
print("="*80)
print("""
Three history strategies with RunnableWithMessageHistory:
  Full history    → keeps all turns verbatim                  (grows linearly)
  Windowed        → keeps only last N turns                   (fixed cost)
  Summary-style   → compresses older turns into a summary     (medium cost)
""")

history_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support assistant. Answer concisely based on conversation context."),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

history_chain = history_prompt | llm

def _apply_window(history: InMemoryChatMessageHistory, window_turns: int) -> None:
    """Keep only the most recent N human+assistant turns (2 messages per turn)."""
    max_messages = window_turns * 2
    if len(history.messages) > max_messages:
        history.messages = history.messages[-max_messages:]


def _apply_summary_style(history: InMemoryChatMessageHistory, keep_recent_turns: int = 2) -> None:
    """
    Compress older messages into one summary-like SystemMessage.

    This is a lightweight summary strategy for teaching purposes. In production,
    replace with an LLM-generated summary for higher fidelity.
    """
    max_recent_messages = keep_recent_turns * 2
    if len(history.messages) <= max_recent_messages + 1:
        return

    older = history.messages[:-max_recent_messages]
    recent = history.messages[-max_recent_messages:]

    # Build compact textual snippets from older turns to preserve broad context.
    snippets = []
    for message in older:
        role = "User" if message.type == "human" else "Assistant"
        snippets.append(f"{role}: {message.content[:80]}")

    summary_text = "Conversation summary of earlier turns: " + " | ".join(snippets)
    history.messages = [SystemMessage(content=summary_text[:600])] + recent


sample_turns = [
    "What authentication issues have we seen?",
    "How was TICK-001 resolved?",
    "Any database-related critical issues?",
]

# --- Strategy 1: Full history ---
print("-"*60)
print("1. Full history via RunnableWithMessageHistory")
full_store = {}

def get_full_history(session_id: str) -> InMemoryChatMessageHistory:
    """Session store for full-history strategy (buffer-style memory)."""
    return full_store.setdefault(session_id, InMemoryChatMessageHistory())

full_chain = RunnableWithMessageHistory(
    history_chain,
    get_full_history,
    input_messages_key="question",
    history_messages_key="history",
)

for turn in sample_turns:
    full_chain.invoke(
        {"question": turn},
        config={"configurable": {"session_id": "full-demo"}}
    )

print(f"  Stored messages: {len(full_store['full-demo'].messages)}")
print("  → All turns retained. Best for short chats (< 10 turns).")

# --- Strategy 2: Windowed history ---
print("\n2. Windowed history via RunnableWithMessageHistory (k=2 turns)")
window_store = {}

def get_window_history(session_id: str) -> InMemoryChatMessageHistory:
    """Session store for windowed strategy (fixed memory cost)."""
    return window_store.setdefault(session_id, InMemoryChatMessageHistory())

window_chain = RunnableWithMessageHistory(
    history_chain,
    get_window_history,
    input_messages_key="question",
    history_messages_key="history",
)

for turn in sample_turns:
    window_chain.invoke(
        {"question": turn},
        config={"configurable": {"session_id": "window-demo"}}
    )
    _apply_window(window_store["window-demo"], window_turns=2)

print(f"  Stored messages: {len(window_store['window-demo'].messages)} (window=2 means last 4 messages)")
print("  → Older turns dropped. Best for most production use cases.")

# --- Strategy 3: Summary-style history ---
print("\n3. Summary-style history via RunnableWithMessageHistory")
summary_store = {}

def get_summary_history(session_id: str) -> InMemoryChatMessageHistory:
    """Session store for summary-style strategy (compressed long history)."""
    return summary_store.setdefault(session_id, InMemoryChatMessageHistory())

summary_chain = RunnableWithMessageHistory(
    history_chain,
    get_summary_history,
    input_messages_key="question",
    history_messages_key="history",
)

for turn in sample_turns:
    summary_chain.invoke(
        {"question": turn},
        config={"configurable": {"session_id": "summary-demo"}}
    )
    _apply_summary_style(summary_store["summary-demo"], keep_recent_turns=2)

summary_messages = summary_store["summary-demo"].messages
print(f"  Stored messages: {len(summary_messages)}")
if summary_messages and summary_messages[0].type == "system":
    print("  → Older turns compressed into a summary message. Best for long sessions.")
else:
    print("  → Recent-only state retained; add more turns to trigger summarization.")

print("\n✓ Memory type comparison complete")
print("  TIP: Start with windowed history (k=3 turns) in production — fixed cost, good context.")

# ============================================================================
# PART 9: Hybrid Routing
# ============================================================================
print("\n" + "="*80)
print("PART 9: Hybrid Routing — Direct RAG vs Agentic RAG")
print("="*80)
print("""
A lightweight heuristic router inspects each query and dispatches it to the
right handler — avoiding the full agent loop cost for simple lookups.

  Simple query  →  Direct RAG  (fast, cheap, predictable)
  Complex query →  Agentic RAG (flexible, multi-step)
""")

import re as _re

def route_query(query: str) -> str:
    """
    Route a query to either direct retrieval or full agentic reasoning.

    Heuristic routing keeps costs low:
    - Direct path for simple lookups/overview requests.
    - Agentic path for multi-step reasoning and troubleshooting.
    """
    q = query.lower().strip()

    # Direct: specific ticket ID or statistics
    direct_patterns = [
        r'\bTICK-\d+\b',
        r'\b(statistics|overview|summary|count|how many)\b',
    ]
    for pattern in direct_patterns:
        if _re.search(pattern, q, _re.IGNORECASE):
            return 'direct'

    # Agentic: troubleshooting, comparison, multi-step
    agentic_patterns = [
        r'\b(how (do|to|can)|why (is|does|did)|fix|resolve|compare)\b',
        r'\b(find .* and|critical|all .* tickets|details? (of|on) each)\b',
    ]
    for pattern in agentic_patterns:
        if _re.search(pattern, q, _re.IGNORECASE):
            return 'agentic'

    return 'direct'  # default to cheaper path

def direct_rag(query: str) -> str:
    """Lightweight retrieval-only path for cheap, fast answers."""
    results = tool_manager.search_similar_tickets(query)
    return f"[Direct RAG] {results[:300]}..."

routing_test_cases = [
    ("Show me TICK-005",                              "direct"),
    ("Give me an overview of all tickets",            "direct"),
    ("How do I fix login failures?",                  "agentic"),
    ("Find critical database issues and compare",     "agentic"),
    ("What payment-related issues have we seen?",     "agentic"),
]

print(f"{'Query':<48} {'Expected':<10} {'Got':<10} {'Match'}")
print("-" * 80)
for query, expected in routing_test_cases:
    got = route_query(query)
    match = "✓" if got == expected else "✗"
    print(f"{query:<48} {expected:<10} {got:<10} {match}")

print("\nDispatching two example queries through the hybrid router:\n")

for query, _ in routing_test_cases[:2]:
    route = route_query(query)
    print(f"Query:  '{query}'")
    print(f"Route:  {route.upper()}")
    if route == 'direct':
        answer = direct_rag(query)
    else:
        answer = run_agent(query)
    print(f"Answer: {answer[:200]}...\n")

print("✓ Hybrid routing demo complete")
print("  TIP: Replace the regex heuristic with a small classifier for production.")

# ============================================================================
# Summary and Key Learnings
# ============================================================================
print("\n" + "="*80)
print("KEY LEARNINGS: Agentic RAG")
print("="*80)

print("""
✅ Agent Architecture Benefits:
   • Tools give the agent structured capabilities
   • Agent decides WHEN and WHICH tool to use
   • More flexible than hardcoded RAG pipelines
   • Can combine multiple tools for complex queries

✅ Tool Design Best Practices:
   • Clear, specific tool descriptions help agent selection
   • Each tool should have a single, well-defined purpose
   • Return formatted strings for easy agent consumption
   • Include error handling and helpful messages

✅ Memory Management:
   • Conversation history maintained by passing messages
   • Enables follow-up questions without re-explaining
   • Be mindful of token limits with long conversations
   • Consider summarization for longer chats

✅ When to Use Agentic RAG:
   • Multi-step queries requiring reasoning
   • Need to combine retrieval with other operations
   • Interactive/conversational applications
   • When users need flexible query patterns

✅ When Direct RAG is Better:
   • Simple, single-step retrieval needs
   • Lower latency requirements
   • More predictable/controllable behavior
   • Cost-sensitive applications (agents use more tokens)

🎯 Next Steps:
   1. Try different queries in the exercises
   2. Add custom tools (e.g., ticket creation)
   3. Experiment with different agent prompts
   4. Compare performance vs Module 4's direct RAG
   5. Add evaluation metrics for agent responses
""")

print("\n" + "="*80)
print("Demo completed! Check exercises.md for hands-on practice.")
print("="*80)
