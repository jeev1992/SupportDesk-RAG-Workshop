# -*- coding: utf-8 -*-
"""
Module 6 Solutions: Agentic RAG
===============================

Solutions for all exercises in exercises.md
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# ============================================================================
# Setup: Load tools and agent
# ============================================================================
print("Setting up agent...")


class SupportTicketToolsEnhanced:
    """Enhanced tools with better error handling and small-patch extensions."""
    
    def __init__(self, tickets_path: str = '../../data/synthetic_tickets.json'):
        # Load canonical ticket dataset once at startup for all tools.
        with open(tickets_path, 'r', encoding='utf-8') as f:
            self.tickets = json.load(f)
        
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        self._setup_vectorstore()
    
    def _setup_vectorstore(self):
        """Index all tickets into a semantic vector store for similarity retrieval."""
        documents = []
        for ticket in self.tickets:
            content = f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}"""
            
            doc = Document(
                page_content=content,
                metadata={
                    'ticket_id': ticket['ticket_id'],
                    'category': ticket['category'],
                    'priority': ticket['priority']
                }
            )
            documents.append(doc)
        
        # Dedicated collection name avoids collisions with other module demos.
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name="solutions_agent"
        )
    
    def search_similar_tickets(self, query: str) -> str:
        """Search for similar tickets using semantic similarity."""
        if not query or not query.strip():
            return "Error: Please provide a search query."

        # Small-patch Exercise 4 behavior:
        # If the query explicitly asks for critical issues, apply a priority filter
        # using the existing search tool instead of adding a brand-new tool.
        if "critical" in query.lower():
            results = self.vectorstore.similarity_search(
                query,
                k=3,
                filter={"priority": "Critical"}
            )
        else:
            # Semantic retrieval is best for "how to fix" and fuzzy symptom queries.
            results = self.vectorstore.similarity_search(query, k=3)
        
        if not results:
            return "No similar tickets found."
        
        output = "Found similar tickets:\n\n"
        for i, doc in enumerate(results, 1):
            output += f"--- Ticket {i} ---\n{doc.page_content}\n\n"
        
        return output
    
    def get_ticket_by_id(self, ticket_id: str) -> str:
        """Retrieve a specific ticket by ID with error handling."""
        # Error handling
        if not ticket_id or not ticket_id.strip():
            return "Error: Please provide a ticket ID (e.g., TICK-001)"
        
        ticket_id = ticket_id.upper().strip()
        
        # Validate expected identifier shape before scanning dataset.
        if not ticket_id.startswith("TICK-"):
            return f"Error: Invalid format. Ticket IDs look like TICK-001, TICK-002, etc. You provided: '{ticket_id}'"
        
        for ticket in self.tickets:
            if ticket['ticket_id'] == ticket_id:
                return f"""Ticket ID: {ticket['ticket_id']}
Title: {ticket['title']}
Description: {ticket['description']}
Resolution: {ticket['resolution']}
Category: {ticket['category']}
Priority: {ticket['priority']}
Created: {ticket['created_date']}
Resolved: {ticket['resolved_date']}"""
        
        return f"Ticket '{ticket_id}' not found. Check the ID and try again."
    
    def search_by_category(self, category: str) -> str:
        """Find all tickets in a specific category."""
        if not category or not category.strip():
            available = list(set(t['category'] for t in self.tickets))
            return f"Error: Please provide a category. Available: {', '.join(available)}"
        
        category = category.strip()
        # Simple exact category matching (case-insensitive).
        matching = [t for t in self.tickets if t['category'].lower() == category.lower()]
        
        if not matching:
            available = list(set(t['category'] for t in self.tickets))
            return f"No tickets in category '{category}'. Available categories: {', '.join(available)}"
        
        output = f"Found {len(matching)} tickets in '{category}':\n\n"
        for ticket in matching:
            output += f"• [{ticket['ticket_id']}] {ticket['title']} (Priority: {ticket['priority']})\n"
        
        return output
    
    # Optional reference implementation (not required by small-edit exercise style)
    def search_by_priority(self, priority: str) -> str:
        """Find all tickets with a specific priority level."""
        if not priority or not priority.strip():
            available = list(set(t['priority'] for t in self.tickets))
            return f"Error: Please provide a priority. Available: {', '.join(available)}"
        
        priority = priority.strip()
        # Priority filtering supports urgency-focused triage style queries.
        matching = [t for t in self.tickets if t['priority'].lower() == priority.lower()]
        
        if not matching:
            available = list(set(t['priority'] for t in self.tickets))
            return f"No tickets with priority '{priority}'. Available: {', '.join(available)}"
        
        output = f"Found {len(matching)} tickets with {priority} priority:\n\n"
        for ticket in matching:
            output += f"• [{ticket['ticket_id']}] {ticket['title']} ({ticket['category']})\n"
        
        return output
    
    def get_ticket_statistics(self, input: str = "") -> str:
        """Get statistics about the ticket database."""
        total = len(self.tickets)
        
        # Aggregate counts manually for transparency in workshop code.
        categories = {}
        priorities = {}
        for ticket in self.tickets:
            categories[ticket['category']] = categories.get(ticket['category'], 0) + 1
            priorities[ticket['priority']] = priorities.get(ticket['priority'], 0) + 1
        
        output = f"Ticket Database Statistics:\n"
        output += f"Total Tickets: {total}\n\n"
        
        output += "By Category:\n"
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {cat}: {count}\n"
        
        output += "\nBy Priority:\n"
        for pri, count in sorted(priorities.items(), key=lambda x: x[1], reverse=True):
            output += f"  • {pri}: {count}\n"
        
        return output
    
    def get_tools(self):
        """Get all tools with improved descriptions."""
        return [
            Tool(
                name="SearchSimilarTickets",
                func=self.search_similar_tickets,
                description="""Search for similar support tickets using semantic similarity.
Use this for troubleshooting questions like "how to fix X", "why is X happening", 
"users experiencing X", or finding similar past issues.
Input: A clear description of the problem or question.
This is the PRIMARY tool for "how to fix" questions."""
            ),
            Tool(
                name="GetTicketByID",
                func=self.get_ticket_by_id,
                description="""Retrieve details of a specific ticket by its exact ID.
Use when user mentions a specific ticket number like TICK-001, TICK-005, etc.
Input: A ticket ID (e.g., 'TICK-001').
Do NOT use this for searching - only for getting known ticket details."""
            ),
            Tool(
                name="SearchByCategory",
                func=self.search_by_category,
                description="""Find ALL tickets in a specific category.
Use when user asks "what X issues have we seen" or "show me all X tickets".
Categories include: Authentication, Database, Payment, Performance, Email, Mobile.
Input: Category name (e.g., 'Database', 'Authentication')."""
            ),
            Tool(
                name="GetTicketStatistics",
                func=self.get_ticket_statistics,
                description="""Get statistics and overview of the ticket database.
Use when user asks "how many tickets", "give me an overview", or "statistics".
No input needed - provides total count, category breakdown, and priority breakdown."""
            )
        ]


# Initialize enhanced tools
print("Creating enhanced tools...")
tool_manager = SupportTicketToolsEnhanced()
tools = tool_manager.get_tools()
print(f"✓ Created {len(tools)} tools")

# Initialize LLM with tools
llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'), temperature=0)

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
                    "input": {"type": "string", "description": "The input to the tool"}
                },
                "required": ["input"]
            }
        }
    })

llm_with_tools = llm.bind(tools=tool_definitions)


# ============================================================================
# Exercise 6: Custom Agent Prompt
# ============================================================================
CUSTOM_SYSTEM_PROMPT = """You are an expert support desk assistant with access to a ticket database.

ALWAYS follow these rules:
1. State which tool you're about to use before calling it
2. If the query is ambiguous, ask a clarifying question FIRST
3. When providing solutions, rate your confidence (High/Medium/Low)
4. After answering, suggest one related topic the user might want to explore
5. Reference ticket IDs when providing information

Your tools:
- SearchSimilarTickets: For troubleshooting and "how to fix" questions
- GetTicketByID: For looking up specific tickets (TICK-001, etc.)
- SearchByCategory: For finding all tickets in a category
- GetTicketStatistics: For database overview and counts
"""


def run_agent(query: str, max_iterations: int = 5, custom_prompt: str = None) -> dict:
    """
    Run the agent with tracking for evaluation.
    Returns dict with response, tools_used, and iterations.
    """
    # Allow callers to swap prompts for experimentation while keeping a safe default.
    system_prompt = custom_prompt or CUSTOM_SYSTEM_PROMPT
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    # Track tool trajectory for later evaluation/debugging.
    tools_used = []
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Final answer path: model returned plain content with no tool requests.
        if not response.tool_calls:
            return {
                'response': response.content,
                'tools_used': tools_used,
                'iterations': i + 1
            }
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            
            print(f"🔧 Tool: {tool_name} | Input: {tool_input[:50]}...")
            tools_used.append(tool_name)
            
            # Resolve and execute requested tool safely via name lookup.
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            
            if tool_output is None:
                tool_output = f"Error: Tool {tool_name} not found"
            
            # Feed tool output back into model context for the next reasoning step.
            messages.append(ToolMessage(
                content=tool_output,
                tool_call_id=tool_call["id"]
            ))
    
    return {
        'response': "Maximum iterations reached.",
        'tools_used': tools_used,
        'iterations': max_iterations
    }


# ============================================================================
# Exercise 1 & 2: Test Different Queries
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 1 & 2: Test Different Queries")
print("=" * 80)

test_queries = [
    ("How do I fix authentication problems?", "SearchSimilarTickets"),
    ("Show me ticket TICK-005", "GetTicketByID"),
    ("What payment issues have we seen?", "SearchByCategory"),
    ("Give me an overview of tickets", "GetTicketStatistics"),
    ("How many critical tickets are there?", "GetTicketStatistics"),  # Exercise 4 small patch
]

for query, expected_tool in test_queries:
    print(f"\n--- Query: '{query}' ---")
    print(f"Expected tool: {expected_tool}")
    
    result = run_agent(query)
    
    print(f"Actual tools used: {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")
    correct = expected_tool in result['tools_used']
    print(f"Correct: {'✓' if correct else '✗'}")
    print(f"Response: {result['response'][:150]}...")


# ============================================================================
# Exercise 5: Multi-Step Queries
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 5: Multi-Step Queries")
print("=" * 80)

multi_step_queries = [
    "How many payment tickets do we have and what was the most recent resolution?",
    "Find all high priority tickets and give me details on the first one",
    "Compare the resolution for TICK-001 and TICK-005",
]

for query in multi_step_queries:
    print(f"\n--- Multi-step Query ---")
    print(f"Query: '{query}'")
    
    result = run_agent(query)
    
    print(f"Tools used (in order): {result['tools_used']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Response: {result['response'][:200]}...")


# ============================================================================
# Exercise 7: Interactive Conversation Loop
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 7: Interactive Chat (Simulated)")
print("=" * 80)

# Simulated conversation
simulated_conversation = [
    "What authentication issues have we seen?",
    "Tell me about TICK-001",
    "What was the priority?"
]

print("\nSimulating interactive conversation:")
for i, query in enumerate(simulated_conversation, 1):
    print(f"\n--- Turn {i} ---")
    print(f"User: {query}")
    result = run_agent(query)
    print(f"Assistant: {result['response'][:150]}...")


# ============================================================================
# Exercise 8: Error Handling
# ============================================================================
print("\n" + "=" * 80)
print("EXERCISE 8: Error Handling")
print("=" * 80)

error_test_cases = [
    "Get ticket XYZ123",           # Invalid format
    "Get ticket TICK-999",          # Not found
    "Show me category FooBar",      # Invalid category
    "Show me critical incidents",   # Priority-aware branch in existing tool
]

for query in error_test_cases:
    print(f"\n--- Error case: '{query}' ---")
    result = run_agent(query)
    print(f"Response: {result['response'][:150]}...")


# ============================================================================
# Bonus: Conversation with Memory
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Conversation with Memory")
print("=" * 80)

conversation_history = []

def chat_with_memory(user_message: str) -> str:
    """
    Minimal memory-enabled chat loop using explicit message history.

    This demonstrates the core concept (carry prior messages forward) without
    introducing additional abstractions.
    """
    global conversation_history
    
    messages = [
        SystemMessage(content="""You are a support assistant. 
Remember our conversation and use context from previous messages.
When the user asks follow-up questions, refer to what we discussed.""")
    ] + conversation_history + [HumanMessage(content=user_message)]
    
    for i in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            conversation_history.append(HumanMessage(content=user_message))
            conversation_history.append(AIMessage(content=response.content))
            return response.content
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"].get("input", "")
            
            print(f"  🔧 {tool_name}")
            
            tool_output = None
            for tool in tools:
                if tool.name == tool_name:
                    tool_output = tool.func(tool_input)
                    break
            
            messages.append(ToolMessage(
                content=tool_output or "Tool not found",
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached"

# Test memory
memory_conversation = [
    "What database issues have we had?",
    "What was the ticket ID for that?",  # Should remember context
    "How was it resolved?"                # Should still remember
]

print("\nTesting conversation memory:")
for query in memory_conversation:
    print(f"\nUser: {query}")
    response = chat_with_memory(query)
    print(f"Assistant: {response[:150]}...")


# ============================================================================
# Bonus: Agent Evaluation
# ============================================================================
print("\n" + "=" * 80)
print("BONUS: Agent Evaluation")
print("=" * 80)

evaluation_cases = [
    {"query": "How do I fix login issues?", "expected_tool": "SearchSimilarTickets"},
    {"query": "Show ticket TICK-001", "expected_tool": "GetTicketByID"},
    {"query": "How many tickets?", "expected_tool": "GetTicketStatistics"},
    {"query": "All payment tickets", "expected_tool": "SearchByCategory"},
    {"query": "How many critical tickets are there?", "expected_tool": "GetTicketStatistics"},
]

correct = 0
total = len(evaluation_cases)

print("\nRunning evaluation...")
for test in evaluation_cases:
    result = run_agent(test['query'])
    passed = test['expected_tool'] in result['tools_used']
    if passed:
        correct += 1
    print(f"  {'✓' if passed else '✗'} '{test['query'][:30]}...' → {result['tools_used']}")

accuracy = correct / total * 100
print(f"\nTool Selection Accuracy: {correct}/{total} ({accuracy:.0f}%)")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("ALL SOLUTIONS COMPLETE!")
print("=" * 80)
print("""
Key Takeaways:
──────────────
1. Tool Descriptions Matter:
   - Clear, specific descriptions improve tool selection
   - Include keywords that match likely user queries
   - State when NOT to use a tool

2. Error Handling:
   - Validate inputs before processing
   - Return helpful error messages
   - Suggest corrections when possible

3. Multi-Step Reasoning:
   - Agents can chain multiple tools
   - Order of tool calls matters
   - Synthesis happens after all tools run

4. Conversation Memory:
   - Pass message history to maintain context
   - Enables follow-up questions
   - Be mindful of token limits

5. Evaluation:
   - Track tool selection accuracy
   - Measure iterations and latency
   - Use test sets for regression testing

When to Use Agentic RAG:
────────────────────────
✓ Complex multi-step queries
✓ Multiple data sources/actions
✓ Interactive conversations
✓ Flexible query patterns

When to Use Direct RAG:
───────────────────────
✓ Simple single-step retrieval
✓ Low latency requirements
✓ Predictable behavior needed
✓ Cost-sensitive applications
""")
