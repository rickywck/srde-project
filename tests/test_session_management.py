"""
Test script to verify Strands Session Management implementation
Demonstrates that conversation history persists across multiple requests
"""

import asyncio
import sys
from pathlib import Path
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supervisor import SupervisorAgent


@pytest.mark.asyncio
async def test_conversation_history():
    """
    Test that conversation history is maintained across multiple interactions
    """
    print("=" * 70)
    print("Testing Strands Session Management - Conversation History")
    print("=" * 70)
    
    # Initialize supervisor
    supervisor = SupervisorAgent()
    
    # Test with a specific run_id
    test_run_id = "test-session-001"
    
    print(f"\n1️⃣  First message (run_id: {test_run_id})")
    print("-" * 70)
    
    # First interaction - introduce user
    response1 = await supervisor.process_message(
        run_id=test_run_id,
        message="Hello! My name is Alice and I work on the telecom project."
    )
    
    print(f"User: Hello! My name is Alice and I work on the telecom project.")
    print(f"Agent: {response1['response'][:200]}...")
    print(f"Status: {response1['status']}")
    
    print(f"\n2️⃣  Second message (same run_id: {test_run_id})")
    print("-" * 70)
    
    # Second interaction - test if agent remembers the name
    response2 = await supervisor.process_message(
        run_id=test_run_id,
        message="What is my name and which project do I work on?"
    )
    
    print(f"User: What is my name and which project do I work on?")
    print(f"Agent: {response2['response']}")
    print(f"Status: {response2['status']}")
    
    # Verify conversation history
    print(f"\n3️⃣  Verify conversation history")
    print("-" * 70)
    
    history = supervisor.get_conversation_history(test_run_id)
    print(f"Total messages in history: {len(history)}")
    
    # Check if session management is working
    if response2['status'].get('session_managed'):
        print("✅ Session management is ENABLED")
        print(f"✅ Conversation length: {response2['status'].get('conversation_length')}")
    else:
        print("❌ Session management is NOT enabled")
    
    # Test with different run_id - should be independent conversation
    print(f"\n4️⃣  New conversation with different run_id")
    print("-" * 70)
    
    different_run_id = "test-session-002"
    response3 = await supervisor.process_message(
        run_id=different_run_id,
        message="What is my name?"
    )
    
    print(f"User (different run_id): What is my name?")
    print(f"Agent: {response3['response']}")
    print(f"Status: {response3['status']}")
    
    # Verify session isolation
    print(f"\n5️⃣  Verify session isolation")
    print("-" * 70)
    
    history1 = supervisor.get_conversation_history(test_run_id)
    history2 = supervisor.get_conversation_history(different_run_id)
    
    print(f"Session 1 ({test_run_id}): {len(history1)} messages")
    print(f"Session 2 ({different_run_id}): {len(history2)} messages")
    
    if len(history1) > len(history2):
        print("✅ Sessions are properly isolated")
    else:
        print("⚠️  Sessions might not be properly isolated")
    
    # Check session files on disk
    print(f"\n6️⃣  Check persisted session files")
    print("-" * 70)
    
    sessions_dir = Path("sessions")
    if sessions_dir.exists():
        session_dirs = list(sessions_dir.glob("session_*"))
        print(f"Found {len(session_dirs)} session(s) on disk:")
        for session_dir in session_dirs:
            print(f"  - {session_dir.name}")
            agent_dirs = list(session_dir.glob("agents/agent_*"))
            if agent_dirs:
                for agent_dir in agent_dirs:
                    messages_dir = agent_dir / "messages"
                    if messages_dir.exists():
                        message_files = list(messages_dir.glob("message_*.json"))
                        print(f"    - {len(message_files)} messages persisted")
    else:
        print("⚠️  Sessions directory not found")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_conversation_history())
