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
print("✓ Shows decision trace for tool selection")

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
- Before each tool call, provide a short public rationale in content using this exact prefix:
    "Decision: <one sentence explaining why this tool is needed>"

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

        # Print model-provided public rationale (not hidden chain-of-thought).
        decision_trace = (response.content or "").strip()
        if decision_trace:
            print(f"\n🧭 {decision_trace}")
        
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
Use your tools to find relevant information and maintain context across our conversation.
Before each tool call, provide a short public rationale in content using this exact prefix:
"Decision: <one sentence explaining why this tool is needed>".""")]
    
    # Replay prior turns before appending the new user query.
    messages.extend(conversation_history)
    messages.append(HumanMessage(content=query))
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return messages, response.content

        decision_trace = (response.content or "").strip()
        if decision_trace:
            print(f"\n🧭 {decision_trace}")
        
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
