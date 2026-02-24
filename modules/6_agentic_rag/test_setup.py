# -*- coding: utf-8 -*-
"""
Quick Test Script - Verify Module Setup
========================================
"""

import sys
import os

print("Testing Module 6 Setup...")
print("="*60)

# Test 1: Check if we can import required packages
# -----------------------------------------------------------------------------
# Why this test matters:
# - Import failures are the most common early setup issue.
# - Testing each key package separately makes troubleshooting faster.
# -----------------------------------------------------------------------------
print("\n1. Testing imports...")
try:
    from langchain_openai import ChatOpenAI
    print("   ✓ langchain_openai")
except ImportError as e:
    print(f"   ✗ langchain_openai: {e}")

try:
    from langchain.agents import initialize_agent, AgentExecutor, AgentType
    print("   ✓ langchain.agents")
except ImportError as e:
    print(f"   ✗ langchain.agents: {e}")

try:
    from langchain_community.vectorstores import Chroma
    print("   ✓ langchain_community.vectorstores")
except ImportError as e:
    print(f"   ✗ langchain_community.vectorstores: {e}")

try:
    from langchain_core.tools import Tool
    print("   ✓ langchain_core.tools")
except ImportError as e:
    print(f"   ✗ langchain_core.tools: {e}")

# Test 2: Check if data file exists
# -----------------------------------------------------------------------------
# The demos rely on synthetic ticket data. If the path is wrong, many later
# failures can look like logic bugs, so we validate filesystem access early.
# -----------------------------------------------------------------------------
print("\n2. Testing data file access...")
data_path = '../../data/synthetic_tickets.json'
if os.path.exists(data_path):
    print(f"   ✓ Found {data_path}")
    import json
    with open(data_path, 'r') as f:
        tickets = json.load(f)
    print(f"   ✓ Loaded {len(tickets)} tickets")
else:
    print(f"   ✗ Cannot find {data_path}")

# Test 3: Check environment variables
# -----------------------------------------------------------------------------
# OpenAI API key is required for embeddings/chat calls. We only print a prefix
# for safety so users can confirm presence without exposing full secrets.
# -----------------------------------------------------------------------------
print("\n3. Testing environment variables...")
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    print(f"   ✓ OPENAI_API_KEY is set ({api_key[:8]}...)")
else:
    print("   ✗ OPENAI_API_KEY not found")

# Test 4: Try importing our tools
# -----------------------------------------------------------------------------
# This validates local module wiring and confirms the tool factory can build
# all expected tools (which indirectly validates vectorstore initialization).
# -----------------------------------------------------------------------------
print("\n4. Testing custom tools module...")
try:
    from tools import SupportTicketTools
    print("   ✓ Successfully imported SupportTicketTools")
    
    # Try initializing
    tool_manager = SupportTicketTools()
    tools = tool_manager.get_tools()
    print(f"   ✓ Created {len(tools)} tools:")
    for tool in tools:
        print(f"      - {tool.name}")
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "="*60)
print("Setup test complete!")
print("\nIf all checks passed, run: python demo.py")
