# Strands Session Management Implementation

## Overview

This document explains how conversation history is maintained across chatbot interactions using AWS Strands Session Management.

## The Problem

Previously, the supervisor agent created a **new Agent instance for every request**, which meant:
- Each request started with empty conversation history
- Previous context was lost between interactions
- Every conversation appeared as an independent request/response

While Strands automatically maintains conversation history **within a single Agent instance** via `agent.messages`, creating a new agent for each request discarded that history.

## The Solution

We now implement **Strands Session Management** to persist conversation history across requests:

### Key Changes

1. **FileSessionManager**: Persists agent state and messages to disk
2. **Agent Caching**: Reuses existing agent instances per `run_id` instead of creating new ones
3. **Session Directory**: Stores session data in `sessions/` folder

### Implementation Details

```python
from strands.session.file_session_manager import FileSessionManager

# Create session manager for each run_id
session_manager = FileSessionManager(
    session_id=run_id,
    storage_dir="sessions"
)

# Create agent with session management
agent = Agent(
    model=self.model,
    system_prompt=self.system_prompt,
    session_manager=session_manager,  # Enable persistence
    tools=[...]
)

# Cache agent for reuse
self.agents_cache[run_id] = agent
```

## How It Works

### First Request (run_id = "abc123")
1. No agent exists in cache
2. Create new `FileSessionManager(session_id="abc123")`
3. Create new `Agent` with session manager
4. Agent processes message and updates `agent.messages`
5. Session manager automatically persists conversation to `sessions/session_abc123/`
6. Agent cached in `self.agents_cache["abc123"]`

### Subsequent Requests (same run_id = "abc123")
1. Agent exists in cache, reuse it
2. Agent already has previous conversation history in `agent.messages`
3. New message added to existing conversation
4. Session manager automatically persists updated history
5. Agent maintains full context across all interactions

### Different Session (run_id = "xyz789")
1. New run_id, no agent in cache
2. Create separate session manager and agent
3. Independent conversation history maintained

## Session Storage Structure

Sessions are stored in the filesystem:

```
sessions/
└── session_abc123/
    ├── session.json              # Session metadata
    └── agents/
        └── agent_<agent_id>/
            ├── agent.json         # Agent state
            └── messages/
                ├── message_0.json # First message
                ├── message_1.json # Second message
                └── message_2.json # Third message
```

## API Changes

### Status Response
The chat response now includes session information:

```json
{
  "response": "...",
  "status": {
    "run_id": "abc123",
    "session_managed": true,
    "conversation_length": 5,
    ...
  }
}
```

### New Methods

#### Get Conversation History
```python
supervisor.get_conversation_history(run_id)
# Returns: List of message dictionaries from agent.messages
```

#### Clear Session Cache
```python
supervisor.clear_session(run_id)
# Removes agent from memory cache (persisted data remains on disk)
```

## Benefits

✅ **Automatic Persistence**: Strands handles all saving/loading automatically  
✅ **Context Continuity**: Agent remembers previous interactions  
✅ **Multi-Turn Conversations**: Natural follow-up questions work correctly  
✅ **Tool Context**: Tools invoked in previous turns are remembered  
✅ **Per-Session Isolation**: Each `run_id` has independent conversation history  

## Best Practices

### 1. Unique Session IDs
- Use unique `run_id` for each user/conversation
- Current implementation uses UUID per document upload
- Frontend maintains `run_id` across chat interactions

### 2. Session Cleanup
Consider implementing cleanup for old sessions:
```python
# Clean up sessions older than 30 days
# Clean up memory cache for inactive sessions
supervisor.clear_session(run_id)
```

### 3. Production Considerations
For production deployments:

**Option 1: File-based (Current)**
- Simple and works for single-server deployments
- Suitable for development and small-scale production

**Option 2: S3-based (Recommended for Production)**
```python
from strands.session.s3_session_manager import S3SessionManager

session_manager = S3SessionManager(
    session_id=run_id,
    bucket="my-agent-sessions",
    prefix="production/"
)
```
- Centralized storage for multi-server deployments
- Durable and scalable
- Works with load balancers and serverless

**Option 3: Custom Backend**
Implement `SessionRepository` for databases (PostgreSQL, Redis, etc.)

## Testing Session Management

### Test Conversation History
```bash
# Start conversation
curl -X POST "http://localhost:8000/chat/test-123" \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is John"}'

# Follow-up (should remember name)
curl -X POST "http://localhost:8000/chat/test-123" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my name?"}'
# Expected: "Your name is John"
```

### Verify Session Files
```bash
# Check session directory
ls -la sessions/session_test-123/agents/

# View message history
cat sessions/session_test-123/agents/agent_*/messages/message_*.json
```

## Troubleshooting

### Issue: Conversation history not persisting
**Cause**: New agent created each request  
**Solution**: ✅ Fixed - agents are now cached per `run_id`

### Issue: Memory usage grows over time
**Cause**: Agent cache never cleared  
**Solution**: Implement periodic cache cleanup:
```python
# Clear inactive sessions (>1 hour idle)
for run_id in list(supervisor.agents_cache.keys()):
    if is_inactive(run_id):
        supervisor.clear_session(run_id)
```

### Issue: Session data not found after server restart
**Cause**: Using memory cache only  
**Solution**: ✅ Fixed - `FileSessionManager` persists to disk and restores on agent creation

## References

- [Strands State Management](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/state/)
- [Strands Session Management](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/agents/session-management/)
- [FileSessionManager API](https://strandsagents.com/latest/documentation/docs/api-reference/session/)
