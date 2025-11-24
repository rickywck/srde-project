// Backlog Synthesizer Chat Interface JavaScript

let currentRunId = null;
let isBusy = false;

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const runIdDisplay = document.getElementById('runIdDisplay');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const tokenInfo = document.getElementById('tokenInfo');
const runsList = document.getElementById('runsList');

// Quick action buttons
const generateBtn = document.getElementById('generateBtn');
const showBacklogBtn = document.getElementById('showBacklogBtn');
const showTaggingBtn = document.getElementById('showTaggingBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadRuns();
});

function setupEventListeners() {
    // File upload
    uploadBox.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });
    
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
    
    // Chat input
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Quick actions
    generateBtn.addEventListener('click', () => {
        sendQuickMessage("Generate backlog items from this document.");
    });
    
    showBacklogBtn.addEventListener('click', () => {
        loadBacklog();
    });
    
    showTaggingBtn.addEventListener('click', () => {
        loadTagging();
    });
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFileUpload(file);
    }
}

async function handleFileUpload(file) {
    setBusy(true, 'Uploading document...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Upload failed');
        }
        
        const data = await response.json();
        currentRunId = data.run_id;
        
        // Update UI
        runIdDisplay.textContent = `Run: ${currentRunId.substring(0, 8)}...`;
        messageInput.disabled = false;
        sendBtn.disabled = false;
        generateBtn.disabled = false;
        showBacklogBtn.disabled = false;
        showTaggingBtn.disabled = false;
        
        // Add system message
        addMessage('system', `✅ Document "${file.name}" uploaded successfully!`);
        
        // Load chat history if exists
        await loadChatHistory();
        
        // Refresh runs list
        await loadRuns();
        
    } catch (error) {
        console.error('Upload error:', error);
        addMessage('system', `❌ Upload failed: ${error.message}`);
    } finally {
        setBusy(false);
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isBusy || !currentRunId) return;
    
    // Clear input
    messageInput.value = '';
    
    // Add user message
    addMessage('user', message);
    
    setBusy(true, 'Thinking...');
    
    try {
        const response = await fetch(`/chat/${currentRunId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });
        
        if (!response.ok) {
            throw new Error('Chat request failed');
        }
        
        const data = await response.json();
        
        // Add assistant response
        addMessage('assistant', data.response);
        
        // Update token info if available
        if (data.status.tokens_used) {
            tokenInfo.textContent = `Tokens: ${data.status.tokens_used}`;
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        addMessage('system', `❌ Error: ${error.message}`);
    } finally {
        setBusy(false);
    }
}

function sendQuickMessage(message) {
    messageInput.value = message;
    sendMessage();
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function loadChatHistory() {
    if (!currentRunId) return;
    
    try {
        const response = await fetch(`/chat-history/${currentRunId}`);
        if (!response.ok) return;
        
        const data = await response.json();
        
        // Clear existing messages except welcome
        chatMessages.innerHTML = '';
        
        // Add messages from history
        for (const entry of data.history) {
            if (entry.role !== 'system' || !entry.message.includes('Document uploaded')) {
                addMessage(entry.role, entry.message);
            }
        }
        
    } catch (error) {
        console.error('Failed to load chat history:', error);
    }
}

async function loadBacklog() {
    if (!currentRunId) return;
    
    setBusy(true, 'Loading backlog...');
    
    try {
        const response = await fetch(`/backlog/${currentRunId}`);
        if (!response.ok) {
            throw new Error('Failed to load backlog');
        }
        
        const data = await response.json();
        
        if (data.items.length === 0) {
            addMessage('system', '📋 No backlog items generated yet. Ask me to generate them!');
        } else {
            const summary = `📋 Found ${data.count} backlog items:\n\n` +
                data.items.map((item, i) => 
                    `${i + 1}. [${item.type}] ${item.title}`
                ).join('\n');
            addMessage('assistant', summary);
        }
        
    } catch (error) {
        console.error('Backlog load error:', error);
        addMessage('system', `❌ Error loading backlog: ${error.message}`);
    } finally {
        setBusy(false);
    }
}

async function loadTagging() {
    if (!currentRunId) return;
    
    setBusy(true, 'Loading tagging results...');
    
    try {
        const response = await fetch(`/tagging/${currentRunId}`);
        if (!response.ok) {
            throw new Error('Failed to load tagging');
        }
        
        const data = await response.json();
        
        if (data.items.length === 0) {
            addMessage('system', '🏷️ No tagging results yet. Generate backlog items first!');
        } else {
            const tagCounts = data.items.reduce((acc, item) => {
                acc[item.decision_tag] = (acc[item.decision_tag] || 0) + 1;
                return acc;
            }, {});
            
            const summary = `🏷️ Tagging Results (${data.count} stories):\n\n` +
                Object.entries(tagCounts).map(([tag, count]) => 
                    `${tag}: ${count}`
                ).join('\n');
            addMessage('assistant', summary);
        }
        
    } catch (error) {
        console.error('Tagging load error:', error);
        addMessage('system', `❌ Error loading tagging: ${error.message}`);
    } finally {
        setBusy(false);
    }
}

async function loadRuns() {
    try {
        const response = await fetch('/runs');
        if (!response.ok) return;
        
        const data = await response.json();
        
        runsList.innerHTML = '';
        
        if (data.runs.length === 0) {
            runsList.innerHTML = '<p style="font-size: 12px; color: #999;">No runs yet</p>';
            return;
        }
        
        for (const run of data.runs) {
            const runItem = document.createElement('div');
            runItem.className = 'run-item';
            if (run.run_id === currentRunId) {
                runItem.classList.add('active');
            }
            
            const date = new Date(run.created);
            runItem.innerHTML = `
                <div style="font-weight: 500; margin-bottom: 4px;">
                    ${run.run_id.substring(0, 8)}...
                </div>
                <div style="color: #999; font-size: 11px;">
                    ${date.toLocaleString()}
                </div>
            `;
            
            runItem.addEventListener('click', () => loadRun(run.run_id));
            runsList.appendChild(runItem);
        }
        
    } catch (error) {
        console.error('Failed to load runs:', error);
    }
}

async function loadRun(runId) {
    currentRunId = runId;
    runIdDisplay.textContent = `Run: ${runId.substring(0, 8)}...`;
    
    messageInput.disabled = false;
    sendBtn.disabled = false;
    generateBtn.disabled = false;
    showBacklogBtn.disabled = false;
    showTaggingBtn.disabled = false;
    
    // Clear messages and load history
    chatMessages.innerHTML = '';
    await loadChatHistory();
    
    // Update runs list highlighting
    await loadRuns();
}

function setBusy(busy, message = 'Ready') {
    isBusy = busy;
    statusText.textContent = message;
    
    if (busy) {
        statusDot.classList.add('busy');
        sendBtn.disabled = true;
        messageInput.disabled = true;
    } else {
        statusDot.classList.remove('busy');
        if (currentRunId) {
            sendBtn.disabled = false;
            messageInput.disabled = false;
        }
    }
}
