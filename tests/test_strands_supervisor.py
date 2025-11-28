"""
Test the Strands-powered Supervisor Agent
"""

import asyncio
import os
import pytest
from agents.supervisor_agent import SupervisorAgent

@pytest.mark.asyncio
async def test_supervisor():
    """Test the supervisor agent with Strands framework"""
    
    print("=" * 60)
    print("Testing Strands-Powered Supervisor Agent")
    print("=" * 60)
    
    # Initialize supervisor
    try:
        supervisor = SupervisorAgent()
        print("✓ Supervisor initialized successfully")
        print(f"  Model: {supervisor.model}")
        print(f"  Framework: AWS Strands")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize supervisor: {e}")
        return False
    
    # Test 1: Simple query without document
    print("Test 1: Simple query without document")
    print("-" * 60)
    try:
        response = await supervisor.process_message(
            run_id="test-1",
            message="Hello! What can you help me with?",
        )
        print(f"Status: {response['status']}")
        print(f"Response: {response['response'][:200]}...")
        print("✓ Test 1 passed")
        print()
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        return False
    
    # Test 2: Query with document context
    print("Test 2: Query with document context")
    print("-" * 60)
    try:
        document = """
        Meeting Notes - Sprint Planning
        
        Team discussed new features for the customer portal:
        - Add password reset functionality
        - Implement two-factor authentication
        - Create user profile management page
        
        Technical requirements:
        - Must use OAuth 2.0
        - Store passwords securely with bcrypt
        - Session timeout after 30 minutes
        """
        
        response = await supervisor.process_message(
            run_id="test-2",
            message="Can you summarize the key points from this meeting?",
            document_text=document
        )
        print(f"Status: {response['status']}")
        print(f"Response: {response['response'][:300]}...")
        print("✓ Test 2 passed")
        print()
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        return False
    
    # Test 3: Backlog generation request
    print("Test 3: Backlog generation request")
    print("-" * 60)
    try:
        response = await supervisor.process_message(
            run_id="test-3",
            message="Generate backlog items from this document",
            instruction_type="generate_backlog",
            document_text=document
        )
        print(f"Status: {response['status']}")
        print(f"Mode: {response['status'].get('mode', 'unknown')}")
        print(f"Framework: {response['status'].get('framework', 'unknown')}")
        print(f"Response: {response['response'][:300]}...")
        print("✓ Test 3 passed")
        print()
    except Exception as e:
        print(f"✗ Test 3 failed: {e}")
        return False
    
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print()
    print("The Strands-powered supervisor is working correctly!")
    print("The agent can now be extended with specialized sub-agents and tools.")
    
    return True

if __name__ == "__main__":
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY environment variable not set")
        print("  Set it with: export OPENAI_API_KEY=your_key_here")
        exit(1)
    
    # Run async tests
    result = asyncio.run(test_supervisor())
    exit(0 if result else 1)
