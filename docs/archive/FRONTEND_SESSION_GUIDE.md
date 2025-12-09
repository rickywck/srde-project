# Frontend Integration Guide - Session Management

## Overview

With session management enabled, the chatbot now maintains conversation history across interactions. The frontend needs to properly manage `run_id` to ensure continuity.

## Key Concepts

### run_id (Session Identifier)
- Each `run_id` represents an independent conversation session
- Same `run_id` = continued conversation with history
- Different `run_id` = new conversation without previous context

## API Usage

### 1. Starting a New Conversation

**Option A: Upload Document (Recommended)**
```javascript
// Upload document - backend generates run_id
const formData = new FormData();
formData.append('file', documentFile);

const response = await fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
});

const { run_id } = await response.json();
// Store run_id for this conversation
sessionStorage.setItem('currentRunId', run_id);
```

**Option B: Chat Without Document**
```javascript
// Generate a new run_id on frontend
const run_id = crypto.randomUUID();
sessionStorage.setItem('currentRunId', run_id);
```

### 2. Sending Chat Messages

```javascript
// Retrieve the current run_id
const run_id = sessionStorage.getItem('currentRunId');

const response = await fetch(`http://localhost:8000/chat/${run_id}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: userMessage,
    instruction_type: null,  // optional
    document_text: null      // optional
  })
});

const data = await response.json();
console.log('Response:', data.response);
console.log('Session info:', data.status.session_managed, 
            data.status.conversation_length);
```

### 3. Getting Conversation History

```javascript
const run_id = sessionStorage.getItem('currentRunId');

const response = await fetch(`http://localhost:8000/chat-history/${run_id}`);
const { history } = await response.json();

// Display conversation history
history.forEach(msg => {
  console.log(`${msg.role}: ${msg.message}`);
});
```

### 4. Starting a New Conversation

```javascript
// Generate new run_id for fresh start
const newRunId = crypto.randomUUID();
sessionStorage.setItem('currentRunId', newRunId);

// Clear chat UI
clearChatMessages();
```

## Response Structure

### Chat Response with Session Info

```json
{
  "run_id": "abc-123-def",
  "response": "I understand you're working on the telecom project...",
  "status": {
    "run_id": "abc-123-def",
    "model": "gpt-4o",
    "has_document": false,
    "mode": "strands_orchestration",
    "framework": "aws_strands",
    "session_managed": true,           // ← Session management enabled
    "conversation_length": 5,          // ← Number of messages in history
    "agents_available": [...],
    "tools_invoked": [...]
  },
  "timestamp": "2025-11-25T10:30:00.000Z"
}
```

## Frontend Implementation Example

### React Hook Example

```javascript
import { useState, useEffect } from 'react';

function useChatSession() {
  const [runId, setRunId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [conversationLength, setConversationLength] = useState(0);

  useEffect(() => {
    // Initialize or restore run_id from session storage
    let currentRunId = sessionStorage.getItem('currentRunId');
    if (!currentRunId) {
      currentRunId = crypto.randomUUID();
      sessionStorage.setItem('currentRunId', currentRunId);
    }
    setRunId(currentRunId);
    
    // Load existing conversation history
    loadHistory(currentRunId);
  }, []);

  const loadHistory = async (runId) => {
    try {
      const response = await fetch(`/chat-history/${runId}`);
      const { history } = await response.json();
      setMessages(history);
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const sendMessage = async (message) => {
    if (!runId) return;

    // Add user message to UI immediately
    const userMsg = { role: 'user', message, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);

    try {
      const response = await fetch(`/chat/${runId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });

      const data = await response.json();
      
      // Add assistant response to UI
      const assistantMsg = { 
        role: 'assistant', 
        message: data.response, 
        timestamp: data.timestamp 
      };
      setMessages(prev => [...prev, assistantMsg]);
      
      // Update conversation length indicator
      setConversationLength(data.status.conversation_length);

    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const startNewConversation = () => {
    const newRunId = crypto.randomUUID();
    sessionStorage.setItem('currentRunId', newRunId);
    setRunId(newRunId);
    setMessages([]);
    setConversationLength(0);
  };

  return { 
    runId, 
    messages, 
    conversationLength, 
    sendMessage, 
    startNewConversation 
  };
}

// Usage in component
function ChatComponent() {
  const { runId, messages, conversationLength, sendMessage, startNewConversation } 
    = useChatSession();

  return (
    <div>
      <div className="chat-header">
        <span>Session: {runId}</span>
        <span>Messages: {conversationLength}</span>
        <button onClick={startNewConversation}>New Chat</button>
      </div>
      
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.role}>
            {msg.message}
          </div>
        ))}
      </div>
      
      <input 
        onSubmit={(e) => sendMessage(e.target.value)} 
        placeholder="Type a message..."
      />
    </div>
  );
}
```

### Vanilla JavaScript Example

```javascript
class ChatSession {
  constructor() {
    this.runId = this.initializeRunId();
    this.loadHistory();
  }

  initializeRunId() {
    let runId = sessionStorage.getItem('currentRunId');
    if (!runId) {
      runId = crypto.randomUUID();
      sessionStorage.setItem('currentRunId', runId);
    }
    return runId;
  }

  async loadHistory() {
    const response = await fetch(`/chat-history/${this.runId}`);
    const { history } = await response.json();
    this.displayHistory(history);
  }

  async sendMessage(message) {
    this.displayMessage('user', message);

    const response = await fetch(`/chat/${this.runId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });

    const data = await response.json();
    this.displayMessage('assistant', data.response);
    
    // Update UI with session info
    document.getElementById('conversation-length').textContent = 
      data.status.conversation_length;
  }

  displayMessage(role, message) {
    const chatDiv = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    msgDiv.textContent = message;
    chatDiv.appendChild(msgDiv);
  }

  displayHistory(history) {
    history.forEach(msg => {
      this.displayMessage(msg.role, msg.message);
    });
  }

  newConversation() {
    this.runId = crypto.randomUUID();
    sessionStorage.setItem('currentRunId', this.runId);
    document.getElementById('chat-messages').innerHTML = '';
  }
}

// Initialize
const chatSession = new ChatSession();

document.getElementById('send-btn').addEventListener('click', () => {
  const input = document.getElementById('message-input');
  chatSession.sendMessage(input.value);
  input.value = '';
});

document.getElementById('new-chat-btn').addEventListener('click', () => {
  chatSession.newConversation();
});
```

## Best Practices

### 1. Session Persistence
```javascript
// Store run_id in sessionStorage for tab persistence
sessionStorage.setItem('currentRunId', run_id);

// Or localStorage for cross-tab/browser restart persistence
localStorage.setItem('currentRunId', run_id);
```

### 2. User Feedback
Show session information to users:
```javascript
// Display conversation length
<span>Messages in conversation: {conversationLength}</span>

// Show session indicator
<div className="session-indicator">
  {sessionManaged ? '🟢 Connected' : '🔴 Disconnected'}
</div>
```

### 3. Error Handling
```javascript
try {
  const response = await sendMessage(message);
  if (!response.status.session_managed) {
    console.warn('Session management not active');
  }
} catch (error) {
  // Handle network errors
  showError('Failed to send message. Please try again.');
}
```

### 4. Document Upload Integration
```javascript
// When uploading a document, start a new conversation
async function uploadDocument(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch('/upload', {
    method: 'POST',
    body: formData
  });
  
  const { run_id } = await response.json();
  
  // Start new conversation with this run_id
  sessionStorage.setItem('currentRunId', run_id);
  
  // Reset UI
  clearMessages();
  
  // Optionally send a greeting message
  await sendMessage('Hello, I just uploaded a document. Can you help me analyze it?');
}
```

## Testing Session Management

### Browser Console Test
```javascript
// Test 1: First message
fetch('/chat/test-123', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'My name is Alice' })
}).then(r => r.json()).then(console.log);

// Test 2: Follow-up (should remember name)
fetch('/chat/test-123', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'What is my name?' })
}).then(r => r.json()).then(d => console.log(d.response));
// Expected: "Your name is Alice"

// Test 3: Get history
fetch('/chat-history/test-123')
  .then(r => r.json())
  .then(d => console.log(d.history));
```

## Troubleshooting

### Problem: Agent doesn't remember previous messages
**Check:**
1. Using same `run_id` for all messages in conversation?
2. `session_managed: true` in response status?
3. `conversation_length` increasing with each message?

### Problem: Conversation resets unexpectedly
**Check:**
1. `run_id` being overwritten by mistake?
2. Browser session storage cleared?
3. Different tabs using different `run_id`s?

### Problem: Multiple conversations mixed together
**Check:**
1. Using unique `run_id` for each conversation?
2. Not accidentally reusing `run_id` from previous session?

## Migration Notes

If you have existing frontend code:

1. **Before (No Session Management)**:
   - Each request was independent
   - No conversation history
   - Agent had no memory

2. **After (With Session Management)**:
   - Must maintain `run_id` across requests
   - Conversation history automatically maintained
   - Agent remembers context from previous messages

**Code Changes Required:**
- Store and reuse `run_id` for conversation continuity
- Generate new `run_id` when starting fresh conversation
- No backend changes needed - session management is automatic
