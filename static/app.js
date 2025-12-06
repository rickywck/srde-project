// Backlog Synthesizer Chat Interface JavaScript

let currentRunId = null;
let isBusy = false;
let chatAttachedDocument = null; // For chat-specific document upload
let chatAttachedFileName = null;
let dashboardLoading = false;
let lastDashboardRunId = null;
let selectedDashboardAgentId = 'supervisor';
let lastDashboardData = null;

// DOM elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const runIdDisplay = document.getElementById('runIdDisplay');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const loadingSpinner = document.getElementById('loadingSpinner');
const tokenInfo = document.getElementById('tokenInfo');
const runsList = document.getElementById('runsList');
const modelSelect = document.getElementById('modelSelect');
const resetSessionBtn = document.getElementById('resetSessionBtn');
const dashboardBtn = document.getElementById('dashboardBtn');
const dashboardPanel = document.getElementById('dashboardPanel');
const dashboardOverlay = document.getElementById('dashboardOverlay');
const dashboardSubtitle = document.getElementById('dashboardSubtitle');
const dashboardContent = document.getElementById('dashboardContent');
const dashboardCloseBtn = document.getElementById('dashboardCloseBtn');
const dashboardRefreshBtn = document.getElementById('dashboardRefreshBtn');

// Chat document upload elements
const attachBtn = document.getElementById('attachBtn');
const chatFileInput = document.getElementById('chatFileInput');
const chatFileIndicator = document.getElementById('chatFileIndicator');
const chatFileName = document.getElementById('chatFileName');
const removeChatFile = document.getElementById('removeChatFile');

// Quick action buttons
const generateBtn = document.getElementById('generateBtn');
const showBacklogBtn = document.getElementById('showBacklogBtn');
const showTaggingBtn = document.getElementById('showTaggingBtn');
const evaluateBtn = document.getElementById('evaluateBtn');
const adoPreviewBtn = document.getElementById('adoPreviewBtn');
const adoExportBtn = document.getElementById('adoExportBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    loadRuns();
    // Enable chat interface on load (no document upload required)
    enableChatInterface();
    initModelPicker();
});

function enableChatInterface() {
    if (messageInput) messageInput.disabled = false;
    if (sendBtn) sendBtn.disabled = false;
    if (attachBtn) attachBtn.disabled = false;
}

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
        generateBacklogWorkflow();
    });
    
    showBacklogBtn.addEventListener('click', () => {
        loadBacklog();
    });
    
    showTaggingBtn.addEventListener('click', () => {
        loadTagging();
    });

    evaluateBtn.addEventListener('click', () => {
        evaluateQuality();
    });

    // ADO export actions
    if (adoPreviewBtn) {
        adoPreviewBtn.onclick = async () => {
            await adoExport(true);
        };
    }
    if (adoExportBtn) {
        adoExportBtn.onclick = async () => {
            const proceed = confirm('This will attempt to export to Azure DevOps. Continue?');
            if (proceed) {
                await adoExport(false);
            }
        };
    }

    // Chat document attachment
    if (attachBtn && chatFileInput) {
        attachBtn.onclick = function(e) {
            e.preventDefault();
            e.stopPropagation();
            chatFileInput.click();
            return false;
        };
        chatFileInput.addEventListener('change', handleChatFileSelect);
    }
    
    if (removeChatFile) {
        removeChatFile.onclick = function(e) {
            e.preventDefault();
            removeChatAttachment();
            return false;
        };
    }
    
    // Drag and drop for chat input area
    const chatInputArea = document.querySelector('.chat-input-area');
    chatInputArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        chatInputArea.style.background = '#e8f5e9';
    });
    
    chatInputArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        chatInputArea.style.background = 'white';
    });
    
    chatInputArea.addEventListener('drop', (e) => {
        e.preventDefault();
        chatInputArea.style.background = 'white';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            // Simulate file input change
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(files[0]);
            chatFileInput.files = dataTransfer.files;
            handleChatFileSelect({ target: chatFileInput });
        }
    });

    // Dashboard controls
    if (dashboardBtn) {
        dashboardBtn.addEventListener('click', () => {
            if (isDashboardOpen()) {
                closeDashboard();
            } else {
                openDashboard();
            }
        });
    }
    if (dashboardCloseBtn) {
        dashboardCloseBtn.addEventListener('click', () => closeDashboard());
    }
    if (dashboardOverlay) {
        dashboardOverlay.addEventListener('click', () => closeDashboard());
    }
    if (dashboardRefreshBtn) {
        dashboardRefreshBtn.addEventListener('click', () => {
            loadDashboardStats(true);
        });
    }
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && isDashboardOpen()) {
            closeDashboard();
        }
    });

    // Reset session
    if (resetSessionBtn) {
        resetSessionBtn.addEventListener('click', () => {
            resetSession();
        });
    }
}

// Initialize model picker with current OpenAI chat models (excluding special-purpose)
function initModelPicker() {
    if (!modelSelect) return;
    // Default model aligned with server config (`config.poc.yaml`).
    // Will attempt to fetch server-provided default via `/config` endpoint.
    const FALLBACK_OPENAI_MODEL = 'gpt-5-mini';
    const MODEL_OPTIONS = [
        'gpt-4o',
        'gpt-4o-mini',
        'gpt-4.1',
        'gpt-4.1-mini',
        'gpt-5.1',
        'gpt-5',
        'gpt-5-mini',
        'gpt-5-nano'
    ];
    // Fetch configured default from server, fall back to stored or hardcoded default
    (async () => {
        let defaultModel = FALLBACK_OPENAI_MODEL;
        try {
            const resp = await fetch('/config');
            if (resp.ok) {
                const cfg = await resp.json();
                if (cfg && cfg.openai_model) {
                    defaultModel = cfg.openai_model;
                }
            }
        } catch (e) {
            console.warn('Could not fetch /config, using fallback model', e);
        }

        const saved = localStorage.getItem('openai_model') || defaultModel;
        modelSelect.innerHTML = '';
        MODEL_OPTIONS.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m === saved) opt.selected = true;
            modelSelect.appendChild(opt);
        });
    })();
    modelSelect.addEventListener('change', () => {
        localStorage.setItem('openai_model', modelSelect.value);
        addMessage('system', `🧠 Model set to ${modelSelect.value}`);
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
        resetDashboardAgentState();
        
        // Update UI
        runIdDisplay.textContent = `Run: ${currentRunId.substring(0, 8)}...`;
        setDashboardSubtitleForCurrentRun();
        messageInput.disabled = false;
        sendBtn.disabled = false;
        attachBtn.disabled = false;
        generateBtn.disabled = false;
        showBacklogBtn.disabled = false;
        showTaggingBtn.disabled = false;
        evaluateBtn.disabled = false;
        if (adoPreviewBtn) adoPreviewBtn.disabled = false;
        if (adoExportBtn) adoExportBtn.disabled = false;
        
        // Add system message
        addMessage('system', `✅ Document "${file.name}" uploaded successfully!`);
        
        // Load chat history if exists
        await loadChatHistory();
        
        // Refresh runs list
        await loadRuns();
        refreshDashboardIfOpen({ forceRefresh: true });
        
    } catch (error) {
        console.error('Upload error:', error);
        addMessage('system', `❌ Upload failed: ${error.message}`);
    } finally {
        setBusy(false);
        refreshDashboardIfOpen({ forceRefresh: true });
    }
}

async function adoExport(dryRun = true) {
    if (!currentRunId) {
        addMessage('system', '❌ No run available. Please upload a document or start a chat.');
        return;
    }
    setBusy(true, dryRun ? 'Building ADO export preview...' : 'Exporting to ADO...');
    try {
        const url = `/ado-export/${currentRunId}?dry_run=${dryRun}&filter_tags=new,gap`;
        const res = await fetch(url, { method: 'POST' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Request failed (${res.status})`);
        }
        const data = await res.json();
        const counts = data?.summary?.counts || {};
        
        if (dryRun) {
            // Preview mode - show what will be created
            const adoConfig = data?.summary?.ado_config || {};
            let previewMsg = `🧾 <strong>ADO Export Preview</strong>\n\n`;
            previewMsg += `<strong>What will be created:</strong>\n`;
            previewMsg += `  📦 Epics: ${counts.epics || 0}\n`;
            previewMsg += `  🎯 Features: ${counts.features || 0}\n`;
            previewMsg += `  📝 User Stories: ${counts.stories || 0}\n\n`;
            previewMsg += `<strong>Target:</strong> ${adoConfig.organization}/${adoConfig.project}\n`;
            previewMsg += `<strong>Filter:</strong> ${(data?.summary?.filter_tags || []).join(', ')}\n`;
            previewMsg += `<strong>Status:</strong> ${adoConfig.pat_present ? '✅ Ready to export' : '❌ PAT not configured'}`;
            addMessage('system', previewMsg);
        } else {
            // Actual export - show detailed results
            const createdItems = data?.created_items || [];
            const exportCounts = data?.counts || {};
            const errors = data?.errors || [];
            
            let resultMsg = '';
            
            if (data.status === 'ok' || data.status === 'partial') {
                resultMsg += `🚀 <strong>ADO Export Complete!</strong>\n\n`;
                
                if (createdItems.length > 0) {
                    resultMsg += `<strong>Created ${createdItems.length} work item${createdItems.length !== 1 ? 's' : ''}:</strong>\n\n`;
                    
                    // Group by type
                    const epics = createdItems.filter(item => item.type === 'Epic');
                    const features = createdItems.filter(item => item.type === 'Feature');
                    const stories = createdItems.filter(item => item.type === 'User Story');
                    
                    if (epics.length > 0) {
                        resultMsg += `📦 <strong>Epics (${epics.length}):</strong>\n`;
                        epics.forEach(item => {
                            resultMsg += `  • <a href="${item.url}" target="_blank" style="color:#667eea;text-decoration:none;">#${item.ado_id}</a> ${item.title}\n`;
                        });
                        resultMsg += `\n`;
                    }
                    
                    if (features.length > 0) {
                        resultMsg += `🎯 <strong>Features (${features.length}):</strong>\n`;
                        features.forEach(item => {
                            resultMsg += `  • <a href="${item.url}" target="_blank" style="color:#667eea;text-decoration:none;">#${item.ado_id}</a> ${item.title}`;
                            if (item.parent_ado_id) {
                                resultMsg += ` <span style="color:#999;">(parent: #${item.parent_ado_id})</span>`;
                            }
                            resultMsg += `\n`;
                        });
                        resultMsg += `\n`;
                    }
                    
                    if (stories.length > 0) {
                        resultMsg += `📝 <strong>User Stories (${stories.length}):</strong>\n`;
                        stories.forEach(item => {
                            resultMsg += `  • <a href="${item.url}" target="_blank" style="color:#667eea;text-decoration:none;">#${item.ado_id}</a> ${item.title}`;
                            if (item.parent_ado_id) {
                                resultMsg += ` <span style="color:#999;">(parent: #${item.parent_ado_id})</span>`;
                            }
                            resultMsg += `\n`;
                        });
                    }
                } else {
                    resultMsg += `⚠️ No items were created.\n\n`;
                }
                
                if (errors.length > 0) {
                    resultMsg += `\n<strong style="color:#d32f2f;">⚠️ Errors encountered (${errors.length}):</strong>\n`;
                    errors.forEach(err => {
                        resultMsg += `  • ${err}\n`;
                    });
                }
                
                addMessage('system', resultMsg);
            } else {
                resultMsg += `❌ <strong>ADO Export Failed</strong>\n\n`;
                if (errors.length > 0) {
                    errors.forEach(err => {
                        resultMsg += `• ${err}\n`;
                    });
                } else {
                    resultMsg += `Status: ${data.status}\n`;
                    resultMsg += `Mode: ${data.mode || 'unknown'}`;
                }
                addMessage('system', resultMsg);
            }
        }
    } catch (error) {
        console.error('ADO export error:', error);
        addMessage('system', `❌ ADO export error: ${error.message}`);
    } finally {
        setBusy(false);
        refreshDashboardIfOpen({ forceRefresh: true });
    }
}

// Render ADO export result or preview returned by server
function renderAdoExportFromData(data) {
    const mode = data?.mode;
    const counts = data?.summary?.counts || {};
    if (mode === 'dry_run') {
        const adoConfig = data?.summary?.ado_config || {};
        let previewMsg = `🧾 <strong>ADO Export Preview</strong>\n\n`;
        previewMsg += `<strong>What will be created:</strong>\n`;
        previewMsg += `  📦 Epics: ${counts.epics || 0}\n`;
        previewMsg += `  🎯 Features: ${counts.features || 0}\n`;
        previewMsg += `  📝 User Stories: ${counts.stories || 0}\n\n`;
        previewMsg += `<strong>Target:</strong> ${adoConfig.organization}/${adoConfig.project}\n`;
        previewMsg += `<strong>Filter:</strong> ${(data?.summary?.filter_tags || []).join(', ')}\n`;
        previewMsg += `<strong>Status:</strong> ${adoConfig.pat_present ? '✅ Ready to export' : '❌ PAT not configured'}`;
        addMessage('system', previewMsg);
        return;
    }

    // Otherwise render write/actual export result
    const createdItems = data?.created_items || [];
    const errors = data?.errors || [];
    let resultMsg = '';
    if (data.status === 'ok' || data.status === 'partial') {
        resultMsg += `🚀 <strong>ADO Export Complete!</strong>\n\n`;
        if (createdItems.length > 0) {
            resultMsg += `<strong>Created ${createdItems.length} work item${createdItems.length !== 1 ? 's' : ''}:</strong>\n\n`;
            const epics = createdItems.filter(i => i.type === 'Epic');
            const features = createdItems.filter(i => i.type === 'Feature');
            const stories = createdItems.filter(i => i.type === 'User Story');
            if (epics.length) {
                resultMsg += `📦 <strong>Epics (${epics.length}):</strong>\n`;
                epics.forEach(item => { resultMsg += `  • <a href="${item.url}" target="_blank" style="color:#667eea;text-decoration:none;">#${item.ado_id}</a> ${item.title}\n`; });
                resultMsg += `\n`;
            }
            if (features.length) {
                resultMsg += `🎯 <strong>Features (${features.length}):</strong>\n`;
                features.forEach(item => {
                    resultMsg += `  • <a href="${item.url}" target="_blank" style="color:#667eea;text-decoration:none;">#${item.ado_id}</a> ${item.title}`;
                    if (item.parent_ado_id) resultMsg += ` <span style="color:#999;">(parent: #${item.parent_ado_id})</span>`;
                    resultMsg += `\n`;
                });
                resultMsg += `\n`;
            }
            if (stories.length) {
                resultMsg += `📝 <strong>User Stories (${stories.length}):</strong>\n`;
                stories.forEach(item => {
                    resultMsg += `  • <a href="${item.url}" target="_blank" style="color:#667eea;text-decoration:none;">#${item.ado_id}</a> ${item.title}`;
                    if (item.parent_ado_id) resultMsg += ` <span style="color:#999;">(parent: #${item.parent_ado_id})</span>`;
                    resultMsg += `\n`;
                });
            }
        } else {
            resultMsg += `⚠️ No items were created.\n\n`;
        }
        if (errors.length) {
            resultMsg += `\n<strong style=\"color:#d32f2f;\">⚠️ Errors encountered (${errors.length}):</strong>\n`;
            errors.forEach(err => { resultMsg += `  • ${err}\n`; });
        }
        addMessage('system', resultMsg);
    } else {
        resultMsg += `❌ <strong>ADO Export Failed</strong>\n\n`;
        if (errors.length) {
            errors.forEach(err => { resultMsg += `• ${err}\n`; });
        } else {
            resultMsg += `Status: ${data.status}\n`;
            resultMsg += `Mode: ${data.mode || 'unknown'}`;
        }
        addMessage('system', resultMsg);
    }
}

// Build standardized evaluation panel (table + optional metadata)
function buildEvaluationPanel(evaluation, meta = {}) {
    const ev = evaluation || {};
    let html = '<div style="font-weight:600;margin-bottom:4px;">📊 Quality Evaluation</div>';
    html += '<table class="evaluation-table" style="border-collapse:collapse;width:100%;font-size:12px;table-layout:fixed;">';
    html += '<colgroup><col style="width:30%"><col style="width:20%"><col style="width:50%"></colgroup>';
    html += '<thead><tr>' +
        '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Metric</th>' +
        '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Score</th>' +
        '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Reasoning</th>' +
        '</tr></thead><tbody>';
    const rows = [
        ['Completeness', ev.completeness?.score, ev.completeness?.reasoning],
        ['Relevance', ev.relevance?.score, ev.relevance?.reasoning],
        ['Quality', ev.quality?.score, ev.quality?.reasoning]
    ];
    for (const r of rows) {
        if (r[1] !== undefined) {
            html += `<tr><td style="border:1px solid #ddd;padding:4px;">${r[0]}</td><td style="border:1px solid #ddd;padding:4px;">${r[1]}</td><td style=\"border:1px solid #ddd;padding:4px;\">${r[2] || ''}</td></tr>`;
        }
    }
    if (ev.overall_score !== undefined) {
        html += `<tr><td style=\"border:1px solid #ddd;padding:4px;font-weight:600;\">Overall</td><td style=\"border:1px solid #ddd;padding:4px;font-weight:600;\">${ev.overall_score}</td><td style=\"border:1px solid #ddd;padding:4px;\"></td></tr>`;
    }
    if (ev.summary) {
        html += `<tr><td style=\"border:1px solid #ddd;padding:4px;\">Summary</td><td style=\"border:1px solid #ddd;padding:4px;\">-</td><td style=\"border:1px solid #ddd;padding:4px;\">${ev.summary}</td></tr>`;
    }
    html += '</tbody></table>';
    // Footer metadata
    const parts = [];
    if (meta.items_evaluated != null) parts.push(`Items evaluated: ${meta.items_evaluated}`);
    if (meta.segment_length != null) parts.push(`Document length: ${Number(meta.segment_length).toLocaleString()} chars`);
    if (meta.model_used) parts.push(`Model: ${meta.model_used}`);
    if (meta.timestamp) parts.push(`Timestamp: ${new Date(meta.timestamp).toLocaleString()}`);
    if (parts.length) {
        html += `<div style=\"margin-top:6px;color:#999;font-size:12px;\">${parts.join(' | ')}</div>`;
    }
    return html;
}

// Render evaluation result (used by chatbot mode)
function renderEvaluationFromData(data) {
    const evaluation = data?.evaluation || {};
    const meta = {
        items_evaluated: data?.items_evaluated,
        segment_length: data?.segment_length,
        model_used: data?.model_used,
        timestamp: data?.timestamp
    };
    const html = buildEvaluationPanel(evaluation, meta);
    addMessage('assistant', html);
}

async function handleChatFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        setBusy(true, 'Reading document...');
        
        try {
            let text = '';
            
            // Check file type and read accordingly
            if (file.name.endsWith('.txt') || file.name.endsWith('.md')) {
                text = await file.text();
            } else if (file.name.endsWith('.docx') || file.name.endsWith('.pdf')) {
                // For binary formats, we'll send to backend for processing
                const formData = new FormData();
                formData.append('file', file);
                
                const response = await fetch('/extract-text', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Failed to extract text from document');
                }
                
                const data = await response.json();
                text = data.text;
            } else {
                // Try to read as text for other formats
                text = await file.text();
            }
            
            chatAttachedDocument = text;
            chatAttachedFileName = file.name;
            
            // Update UI to show file is attached
            chatFileName.textContent = `📎 ${file.name}`;
            chatFileIndicator.classList.add('active');
            
            addMessage('system', `📎 File "${file.name}" attached to next message`);
        } catch (error) {
            console.error('Error reading chat file:', error);
            addMessage('system', `❌ Failed to read file: ${error.message}`);
        } finally {
            setBusy(false);
        }
    }
}

function removeChatAttachment() {
    chatAttachedDocument = null;
    chatAttachedFileName = null;
    chatFileInput.value = '';
    chatFileIndicator.classList.remove('active');
    chatFileName.textContent = '📎 No file attached';
    addMessage('system', '📎 Attachment removed');
}

async function createEmptyRun() {
    // Create a new run without uploading a document
    const runId = generateRunId();
    currentRunId = runId;
    resetDashboardAgentState();
    
    // Update UI
    runIdDisplay.textContent = `Run: ${currentRunId.substring(0, 8)}...`;
    setDashboardSubtitleForCurrentRun();
    refreshDashboardIfOpen({ forceRefresh: true });
    
    // Note: Quick action buttons remain disabled until document is uploaded
    return runId;
}

function generateRunId() {
    // Generate a UUID v4 compatible string
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function resetSession() {
    // Create a fresh run id and clear UI state
    currentRunId = generateRunId();
    resetDashboardAgentState();
    try { sessionStorage.setItem('currentRunId', currentRunId); } catch (_) {}
    runIdDisplay.textContent = `Run: ${currentRunId.substring(0, 8)}...`;

    // Clear chat UI and show welcome
    chatMessages.innerHTML = '';
    addMessage('system', '🔄 Started a new session. You can upload a file or chat directly.');

    // Clear attachment state
    if (chatAttachedDocument) {
        removeChatAttachment();
    }

    // Reset indicators
    tokenInfo.textContent = '';
    statusText.textContent = 'Ready';

    // Enable chat; disable quick actions until document exists
    enableChatInterface();
    generateBtn.disabled = true;
    showBacklogBtn.disabled = true;
    showTaggingBtn.disabled = true;
    evaluateBtn.disabled = true;
    if (adoPreviewBtn) adoPreviewBtn.disabled = true;
    if (adoExportBtn) adoExportBtn.disabled = true;

    // Refresh runs list highlighting (new run not persisted until used)
    loadRuns();
    setDashboardSubtitleForCurrentRun();
    refreshDashboardIfOpen({ forceRefresh: true });
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isBusy) return;
    
    // Auto-create run if one doesn't exist
    if (!currentRunId) {
        await createEmptyRun();
    }
    
    // Clear input
    messageInput.value = '';
    
    // Add user message
    addMessage('user', message);
    
    setBusy(true, 'Thinking...');
    
    try {
        // Prepare request body with optional document
        const requestBody = {
            message: message
        };
        // Include model override if selected
        if (modelSelect && modelSelect.value) {
            requestBody.model_override = modelSelect.value;
        } else {
            const savedModel = localStorage.getItem('openai_model');
            if (savedModel) requestBody.model_override = savedModel;
        }
        
        // Include chat-attached document if present
        if (chatAttachedDocument) {
            requestBody.document_text = chatAttachedDocument;
            addMessage('system', `📄 Sending message with attached document: ${chatAttachedFileName}`);
        }
        
        const response = await fetch(`/chat/${currentRunId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error('Chat request failed');
        }
        
        const data = await response.json();

        // If supervisor signals backlog was generated, fetch and display it
        if (data && data.response_type === 'backlog_generated') {
            if (data.response) {
                addMessage('assistant', data.response);
            }
            await loadBacklog();
        } else if (data && data.response_type === 'tagging') {
            if (data.response) {
                addMessage('assistant', data.response);
            }
            await loadTagging();
        } else if (data && data.response_type === 'ado_export') {
            if (data.response) {
                addMessage('assistant', data.response);
            }
            // Fetch last export result and render it
            try {
                const res = await fetch(`/ado-export/last/${currentRunId}`);
                if (res.ok) {
                    const exportData = await res.json();
                    renderAdoExportFromData(exportData);
                } else {
                    const err = await res.json().catch(() => ({}));
                    addMessage('system', `❌ Failed to load ADO export result: ${err.detail || res.status}`);
                }
            } catch (e) {
                addMessage('system', `❌ Failed to load ADO export result: ${e.message}`);
            }
        } else if (data && data.response_type === 'evaluation') {
            if (data.response) {
                addMessage('assistant', data.response);
            }
            // Fetch evaluation result and render it
            try {
                const res = await fetch(`/evaluation/${currentRunId}`);
                if (res.ok) {
                    const evalData = await res.json();
                    renderEvaluationFromData(evalData);
                } else {
                    const err = await res.json().catch(() => ({}));
                    addMessage('system', `❌ Failed to load evaluation result: ${err.detail || res.status}`);
                }
            } catch (e) {
                addMessage('system', `❌ Failed to load evaluation result: ${e.message}`);
            }
        } else {
            // Default assistant text response
            addMessage('assistant', data.response);
        }
        
        // Clear chat attachment after successful send
        if (chatAttachedDocument) {
            removeChatAttachment();
        }
        
        // Update token info if available
        if (data.status.tokens_used) {
            tokenInfo.textContent = `Tokens: ${data.status.tokens_used}`;
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        addMessage('system', `❌ Error: ${error.message}`);
    } finally {
        setBusy(false);
        refreshDashboardIfOpen({ forceRefresh: true });
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

    // Metadata (role + timestamp)
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    const roleLabel = (role === 'user') ? 'You' : (role === 'assistant') ? 'Assistant' : 'System';
    metaDiv.textContent = `${roleLabel} • ${new Date().toLocaleString()}`;

    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'message-body';

    // Render HTML for system messages or if content looks like HTML; otherwise use textContent
    const looksLikeHtml = typeof content === 'string' && /</.test(content);
    if (role === 'system' || looksLikeHtml) {
        bodyDiv.innerHTML = content;
    } else {
        bodyDiv.textContent = content;
    }

    contentDiv.appendChild(metaDiv);
    contentDiv.appendChild(bodyDiv);
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
            // Build table view
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            let html = `<div style="font-weight:600;margin-bottom:4px;">📋 Generated Backlog (${data.count} items)</div>`;
            // Use colgroup with larger AC column (last) and balanced title/description widths
            html += '<table style="border-collapse:collapse;width:100%;font-size:12px;table-layout:fixed;">';
            html += '<colgroup><col style="width:5%"><col style="width:8%"><col style="width:20%"><col style="width:35%"><col style="width:7%"><col style="width:25%"></colgroup>';
            html += '<thead><tr>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">#</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Type</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Title</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Description</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Parent</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">ACs</th>' +
                '</tr></thead><tbody>';
            for (let i = 0; i < data.items.length; i++) {
                const item = data.items[i];
                const acs = (item.acceptance_criteria || []).join('; ');
                const fullDesc = item.description || '';
                const desc = fullDesc.slice(0, 240) + (fullDesc.length > 240 ? '…' : '');
                html += '<tr>' +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;">${i + 1}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;">${item.type || ''}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;font-weight:500;">${item.title || ''}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;" title="${fullDesc.replace(/"/g,'&quot;')}">${desc}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;">${item.parent_reference || '-'}</td>` +
                    `<td class="ac-cell" style="border:1px solid #ddd;padding:4px;vertical-align:top;">${acs || '-'}</td>` +
                    '</tr>';
            }
            html += '</tbody></table>';
            // Render as assistant message (addMessage will include metadata)
            addMessage('assistant', html);
        }
        
    } catch (error) {
        console.error('Backlog load error:', error);
        addMessage('system', `❌ Error loading backlog: ${error.message}`);
    } finally {
        setBusy(false);
        refreshDashboardIfOpen({ forceRefresh: true });
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
            // Fetch backlog to enrich story titles
            let backlogMap = {};
            try {
                const backlogResp = await fetch(`/backlog/${currentRunId}`);
                if (backlogResp.ok) {
                    const backlogData = await backlogResp.json();
                    for (const itm of backlogData.items) {
                        if (itm.internal_id) backlogMap[itm.internal_id] = itm;
                    }
                }
            } catch (_) { /* ignore enrichment errors */ }
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            let html = `<div style="font-weight:600;margin-bottom:4px;">🏷️ Tagging Results (${data.count} stories)</div>`;
            html += '<table style="border-collapse:collapse;width:100%;font-size:12px;table-layout:fixed;">';
            html += '<colgroup><col style="width:5%"><col style="width:35%"><col style="width:12%"><col style="width:35%"><col style="width:6%"><col style="width:7%"></colgroup>';
            html += '<thead><tr>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">#</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Story Title</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Decision</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Reason</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Similar Count</th>' +
                '<th style="border:1px solid #ccc;padding:4px;background:#f5f5f5;text-align:left;">Early Exit</th>' +
                '</tr></thead><tbody>';
            for (let i = 0; i < data.items.length; i++) {
                const rec = data.items[i];
                let title = '';
                if (rec.story_internal_id && backlogMap[rec.story_internal_id] && backlogMap[rec.story_internal_id].title) {
                    title = backlogMap[rec.story_internal_id].title;
                } else if (rec.story_title) {
                    title = rec.story_title;
                } else if (rec.story_internal_id) {
                    title = rec.story_internal_id;
                }
                const fullReason = rec.reason || '';
                const reason = fullReason.slice(0,180) + (fullReason.length > 180 ? '…' : '');
                html += '<tr>' +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;">${i + 1}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;font-weight:500;" title="${title.replace(/"/g,'&quot;')}">${title}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;">${rec.decision_tag}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;" title="${fullReason.replace(/"/g,'&quot;')}">${reason}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;">${rec.similar_count}</td>` +
                    `<td style="border:1px solid #ddd;padding:4px;vertical-align:top;">${rec.early_exit ? 'Yes' : 'No'}</td>` +
                    '</tr>';
            }
            html += '</tbody></table>';
            // Render as assistant message (addMessage will include metadata)
            addMessage('assistant', html);
        }
        
    } catch (error) {
        console.error('Tagging load error:', error);
        addMessage('system', `❌ Error loading tagging: ${error.message}`);
    } finally {
        setBusy(false);
        refreshDashboardIfOpen({ forceRefresh: true });
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
    resetDashboardAgentState();
    runIdDisplay.textContent = `Run: ${runId.substring(0, 8)}...`;
    setDashboardSubtitleForCurrentRun();
    
    messageInput.disabled = false;
    sendBtn.disabled = false;
    generateBtn.disabled = false;
    showBacklogBtn.disabled = false;
    showTaggingBtn.disabled = false;
    evaluateBtn.disabled = false;
    
    // Clear messages and load history
    chatMessages.innerHTML = '';
    await loadChatHistory();
    
    // Update runs list highlighting
    await loadRuns();
    refreshDashboardIfOpen({ forceRefresh: true });
}

async function generateBacklogWorkflow() {
    if (!currentRunId) {
        addMessage('system', '❌ Please upload a document first.');
        return;
    }
    
    setBusy(true, 'Running backlog synthesis...');
    addMessage('system', '🚀 Running full workflow (segment → retrieve → generate → tag)');
    try {
        const resp = await fetch(`/generate-backlog/${currentRunId}`, { method: 'POST' });
        if (!resp.ok) throw new Error('Workflow request failed');
        const data = await resp.json();
        addMessage('assistant', data.response);
        
        // Display workflow steps status
        if (data.workflow_steps) {
            const steps = data.workflow_steps;
            let statusMsg = '📊 Workflow Status:\n';
            
            if (steps.segmentation) {
                const icon = steps.segmentation.status === 'success' ? '✅' : '⚠️';
                statusMsg += `${icon} Segmentation: ${steps.segmentation.segments_count} segments created\n`;
            }
            
            if (steps.retrieval) {
                const icon = steps.retrieval.status === 'success' ? '✅' : '⚠️';
                statusMsg += `${icon} Retrieval: ${steps.retrieval.message}\n`;
            }
            
            if (steps.generation) {
                const icon = steps.generation.status === 'success' ? '✅' : '⚠️';
                statusMsg += `${icon} Generation: ${steps.generation.message}\n`;
            }
            
            if (steps.tagging) {
                const icon = steps.tagging.status === 'success' ? '✅' : '⚠️';
                statusMsg += `${icon} Tagging: ${steps.tagging.message}\n`;
                if (steps.tagging.tag_distribution) {
                    statusMsg += '  Tag distribution:\n';
                    Object.entries(steps.tagging.tag_distribution).forEach(([tag, count]) => {
                        statusMsg += `    - ${tag}: ${count}\n`;
                    });
                }
            }
            
            addMessage('system', statusMsg);

            // If tagging succeeded as part of the workflow, automatically display tagging results
            try {
                if (steps.tagging && steps.tagging.status === 'success') {
                    await loadTagging();
                }
            } catch (_) { /* ignore */ }
        }
        
        // Display counts summary
        const counts = data.counts || {};
        let summary = '📊 Summary\n';
        summary += `Segments: ${counts.segments || 0}\n`;
        summary += `Items: ${counts.backlog_items || 0} (Stories: ${counts.stories || 0})\n`;
        if (counts.tags) {
            summary += 'Tags:\n' + Object.entries(counts.tags).map(([k,v]) => `- ${k}: ${v}`).join('\n');
        }
        addMessage('system', summary);
    } catch (error) {
        console.error('Workflow error:', error);
        addMessage('system', `❌ Workflow failed: ${error.message}`);
    } finally {
        setBusy(false);
        refreshDashboardIfOpen({ forceRefresh: true });
    }
}

// Poll function removed (direct synchronous workflow now)

function setBusy(busy, message = 'Ready') {
    isBusy = busy;
    statusText.textContent = message;
    
    if (busy) {
        statusDot.classList.add('busy');
        if (loadingSpinner) loadingSpinner.style.display = 'inline-block';
        // Update send button to show small spinner and processing label
        if (sendBtn) sendBtn.innerHTML = `<span class="loading" style="width:14px;height:14px;border-width:2px;margin-right:8px;"></span>Processing...`;
        sendBtn.disabled = true;
        messageInput.disabled = true;
    } else {
        statusDot.classList.remove('busy');
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        if (sendBtn) sendBtn.textContent = 'Send';
        // Always re-enable chat inputs (no longer requires Quick Actions upload)
        sendBtn.disabled = false;
        messageInput.disabled = false;
    }
}

async function evaluateQuality() {
    if (!currentRunId) {
        addMessage('system', '❌ Please upload a document first.');
        return;
    }
    setBusy(true, 'Evaluating quality...');
    addMessage('system', '🧪 Running evaluation (LLM judge)');
    try {
        const resp = await fetch(`/evaluate/${currentRunId}`, { method: 'POST' });
        if (!resp.ok) throw new Error('Evaluation request failed');
        const data = await resp.json();
        const ev = data.evaluation || {};
        const meta = {
            items_evaluated: data?.items_evaluated,
            segment_length: data?.segment_length,
            model_used: data?.status?.model || data?.model_used,
            timestamp: data?.timestamp
        };
        const html = buildEvaluationPanel(ev, meta);
        addMessage('assistant', html);
    } catch (error) {
        console.error('Evaluation error:', error);
        addMessage('system', `❌ Evaluation failed: ${error.message}`);
    } finally {
        setBusy(false);
        refreshDashboardIfOpen({ forceRefresh: true });
    }
}

function isDashboardOpen() {
    return dashboardPanel ? dashboardPanel.classList.contains('active') : false;
}

function openDashboard() {
    if (!dashboardPanel) return;
    dashboardPanel.classList.add('active');
    if (dashboardOverlay) dashboardOverlay.classList.add('active');
    if (dashboardBtn) dashboardBtn.classList.add('active');
    if (!currentRunId) {
        setDashboardSubtitle('No session selected');
        renderDashboardEmpty('Start a run to view stats.');
        return;
    }
    setDashboardSubtitle(`Run ${currentRunId.substring(0,8)} • loading…`);
    loadDashboardStats();
}

function closeDashboard() {
    if (!dashboardPanel) return;
    dashboardPanel.classList.remove('active');
    if (dashboardOverlay) dashboardOverlay.classList.remove('active');
    if (dashboardBtn) dashboardBtn.classList.remove('active');
}

function setDashboardSubtitle(text) {
    if (dashboardSubtitle) {
        dashboardSubtitle.textContent = text;
    }
}

function resetDashboardAgentState() {
    selectedDashboardAgentId = 'supervisor';
    lastDashboardData = null;
    lastDashboardRunId = null;
}

function setDashboardSubtitleForCurrentRun(statusText) {
    if (!dashboardSubtitle) return;
    if (!currentRunId) {
        dashboardSubtitle.textContent = statusText || 'No session selected';
        return;
    }
    const shortId = currentRunId.substring(0, 8);
    dashboardSubtitle.textContent = statusText ? `Run ${shortId} • ${statusText}` : `Run ${shortId} • live session`;
}

function renderDashboardEmpty(message) {
    if (!dashboardContent) return;
    lastDashboardData = null;
    dashboardContent.innerHTML = `<div class="dashboard-empty">${message}</div>`;
}

function renderDashboardLoading() {
    if (!dashboardContent) return;
    dashboardContent.innerHTML = `
        <div class="dashboard-empty">
            <span class="loading" style="width:18px;height:18px;border-width:2px;margin-right:8px;"></span>
            Loading insights…
        </div>
    `;
}

async function loadDashboardStats(force = false) {
    if (!dashboardPanel || !currentRunId) {
        renderDashboardEmpty('Select a run to view stats.');
        return;
    }
    if (dashboardLoading && !force) return;
    if (!force && lastDashboardRunId === currentRunId) return;

    dashboardLoading = true;
    if (dashboardRefreshBtn) dashboardRefreshBtn.disabled = true;
    renderDashboardLoading();

    try {
        const resp = await fetch(`/session-dashboard/${currentRunId}?t=${Date.now()}`);
        if (!resp.ok) throw new Error('Failed to load dashboard');
        const data = await resp.json();
        lastDashboardRunId = currentRunId;
        renderDashboard(data);
    } catch (error) {
        console.error('Dashboard load error:', error);
        renderDashboardEmpty('Unable to load stats right now.');
    } finally {
        dashboardLoading = false;
        if (dashboardRefreshBtn) dashboardRefreshBtn.disabled = false;
    }
}

function renderDashboard(data) {
    if (!dashboardContent || !data) return;
    lastDashboardData = data;

    const agents = normalizeDashboardAgents(data);
    if (!agents.length) {
        renderDashboardEmpty('No stats available for this session yet.');
        return;
    }

    const agentMap = {};
    agents.forEach(agent => {
        agentMap[agent.id] = agent;
    });

    if (!agentMap[selectedDashboardAgentId]) {
        selectedDashboardAgentId = agents[0].id;
    }

    const selectedAgent = agentMap[selectedDashboardAgentId] || agents[0];
    const conversation = selectedAgent.summary || {};
    const artifacts = data.artifacts || {};
    const sessionState = data.session_state || {};
    const toolUsage = conversation.tool_usage || [];
    const backlogItems = artifacts.backlog_items || 0;
    const taggingItems = artifacts.tagging_items || 0;
    const adoInfo = artifacts.ado_export || {};
    const evalCount = artifacts.evaluation_runs || 0;

    const modelLabel = data.model_id || localStorage.getItem('openai_model') || 'default model';
    const shortId = (data.run_id || currentRunId || '').substring(0,8);
    const subtitle = selectedAgent.type === 'supervisor'
        ? `Run ${shortId} • ${modelLabel}`
        : `Run ${shortId} • ${modelLabel} • ${selectedAgent.label}`;
    setDashboardSubtitle(subtitle);

    const selectorHtml = buildAgentSelectorHtml(agents, selectedDashboardAgentId);
    const messagesCard = `
        <div class="dashboard-card">
            <h4>Total Messages</h4>
            <strong>${formatInteger(conversation.total_messages)}</strong>
            <div class="role-breakdown">${buildRoleBreakdown(conversation.by_role)}</div>
        </div>
    `;
    const tokenCard = `
        <div class="dashboard-card">
            <h4>Token Estimate</h4>
            <strong>${formatInteger(conversation.token_estimate)}</strong>
            <div class="role-breakdown"><span>Approximate tokens exchanged</span></div>
        </div>
    `;
    const thirdCard = selectedAgent.type === 'supervisor'
        ? `
            <div class="dashboard-card">
                <h4>Artifacts</h4>
                <strong>${formatInteger(backlogItems)}</strong>
                <div class="role-breakdown">
                    <div>Stories tagged: <strong>${formatInteger(taggingItems)}</strong></div>
                    <div>Evaluations: <strong>${formatInteger(evalCount)}</strong></div>
                </div>
            </div>
        `
        : buildAgentActivityCard(conversation.by_role);

    let supplementalHtml = '';
    if (selectedAgent.type === 'supervisor') {
        supplementalHtml += `
            <div class="dashboard-card">
                <h4>ADO Export State</h4>
                <div class="artifact-list">${buildAdoSummary(adoInfo)}</div>
            </div>
        `;
        supplementalHtml += `
            <div class="dashboard-card">
                <h4>Session Health</h4>
                <div class="artifact-list">
                    <div>Conversation manager: <strong>${sessionState.conversation_manager || 'SlidingWindow'}</strong></div>
                    <div>Trimmed turns: <strong>${formatInteger(sessionState.removed_message_count)}</strong></div>
                    <div>Started: <strong>${formatTimestamp(sessionState.session_created_at)}</strong></div>
                    <div>Updated: <strong>${formatTimestamp(sessionState.session_updated_at)}</strong></div>
                </div>
            </div>
        `;
    } else {
        supplementalHtml += buildAgentContextCard(selectedAgent);
    }

    dashboardContent.innerHTML = `
        ${selectorHtml}
        <div class="dashboard-grid">
            ${messagesCard}
            ${tokenCard}
            ${thirdCard}
        </div>
        <div class="dashboard-card">
            <h4>Last Exchange</h4>
            <div class="role-breakdown">
                <div>${formatRole(conversation.last_message_role)} • ${formatTimestamp(conversation.last_message_at)}</div>
                <div>${conversation.last_message_preview || 'No messages yet.'}</div>
            </div>
        </div>
        <div class="dashboard-card">
            <h4>Tool Usage</h4>
            ${buildToolList(toolUsage)}
        </div>
        ${supplementalHtml}
    `;

    dashboardContent.querySelectorAll('.dashboard-agent-chip').forEach(chip => {
        chip.addEventListener('click', (event) => {
            const id = event.currentTarget && event.currentTarget.dataset ? event.currentTarget.dataset.agentId : null;
            if (id) {
                selectDashboardAgent(id);
            }
        });
    });
}

function buildRoleBreakdown(byRole = {}) {
    const entries = Object.entries(byRole || {});
    if (!entries.length) return '<span>No turns yet.</span>';
    return entries.map(([role, count]) => `<div>${formatRole(role)} <strong>${formatInteger(count)}</strong></div>`).join('');
}

function buildToolList(toolUsage) {
    if (!toolUsage || toolUsage.length === 0) {
        return '<div class="role-breakdown">No structured tool calls yet.</div>';
    }
    const chips = toolUsage.slice(0, 6).map(tool => `<li>${tool.name} · ${formatInteger(tool.count)}</li>`).join('');
    return `<ul class="tool-list">${chips}</ul>`;
}

function buildAdoSummary(info = {}) {
    if (!info.has_export) {
        return '<div>No exports recorded.</div>';
    }
    const lines = [];
    if (info.status) lines.push(`Status: <strong>${info.status}</strong>`);
    if (info.mode) lines.push(`Mode: <strong>${info.mode}</strong>`);
    if (info.updated_at) lines.push(`Updated: <strong>${formatTimestamp(info.updated_at)}</strong>`);
    if (info.counts && Object.keys(info.counts).length) {
        const summary = Object.entries(info.counts).map(([k,v]) => `${k}: ${formatInteger(v)}`).join(' • ');
        lines.push(`Items: <strong>${summary}</strong>`);
    }
    return lines.map(line => `<div>${line}</div>`).join('');
}

function normalizeDashboardAgents(data) {
    if (data && Array.isArray(data.agents) && data.agents.length) {
        return data.agents.map(agent => ({
            id: agent.id,
            label: agent.label || agent.id,
            type: agent.type || 'tool',
            tool_name: agent.tool_name || agent.id,
            available: Boolean(agent.available),
            source: agent.source,
            summary: agent.summary || {}
        }));
    }
    const fallbackConversation = (data && data.conversation) || {};
    const fallbackSource = data && data.sources && data.sources.conversation ? data.sources.conversation : [];
    return [
        {
            id: 'supervisor',
            label: 'Supervisor',
            type: 'supervisor',
            tool_name: null,
            available: Boolean(fallbackConversation.total_messages),
            source: fallbackSource,
            summary: fallbackConversation
        }
    ];
}

function buildAgentSelectorHtml(agents, selectedId) {
    if (!agents || !agents.length) {
        return '';
    }
    const chips = agents.map(agent => {
        const secondary = `${agent.type === 'supervisor' ? 'Supervisor' : 'Tool'} • ${agent.available ? 'active' : 'idle'}`;
        const activeClass = agent.id === selectedId ? 'active' : '';
        return `
            <button type="button" class="dashboard-agent-chip ${activeClass}" data-agent-id="${agent.id}">
                ${agent.label}
                <span>${secondary}</span>
            </button>
        `;
    }).join('');
    return `<div class="dashboard-agent-selector">${chips}</div>`;
}

function buildAgentActivityCard(byRole = {}) {
    const userTurns = Number(byRole.user || 0);
    const assistantTurns = Number(byRole.assistant || 0);
    const systemTurns = Number(byRole.system || 0);
    return `
        <div class="dashboard-card">
            <h4>Agent Activity</h4>
            <strong>${formatInteger(assistantTurns)}</strong>
            <div class="role-breakdown">
                <div>Inputs: <strong>${formatInteger(userTurns)}</strong></div>
                <div>Outputs: <strong>${formatInteger(assistantTurns)}</strong></div>
                <div>System: <strong>${formatInteger(systemTurns)}</strong></div>
            </div>
        </div>
    `;
}

function buildAgentContextCard(agent) {
    if (!agent) {
        return '<div class="dashboard-card"><h4>Tool Context</h4><div class="artifact-list">No agent stats available.</div></div>';
    }
    const status = agent.available ? 'Active' : 'Idle';
    const sourceLabel = formatAgentSource(agent.source);
    const toolName = agent.tool_name || agent.label || 'Tool';
    return `
        <div class="dashboard-card">
            <h4>Tool Context</h4>
            <div class="artifact-list">
                <div>Tool name: <strong>${toolName}</strong></div>
                <div>Status: <strong>${status}</strong></div>
                <div>Data source: <strong>${sourceLabel}</strong></div>
            </div>
        </div>
    `;
}

function formatAgentSource(source) {
    if (!source) return 'unknown';
    if (Array.isArray(source)) {
        if (!source.length) return 'unknown';
        return source.join(', ');
    }
    if (typeof source === 'string') return source;
    return 'unknown';
}

function formatRole(role) {
    if (!role) return '—';
    switch (role) {
        case 'user': return '🙋 User';
        case 'assistant': return '🤖 Assistant';
        case 'system': return '🛠️ System';
        default: return role;
    }
}

function formatTimestamp(value) {
    if (!value) return 'n/a';
    try {
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return value;
        return date.toLocaleString();
    } catch (_) {
        return value;
    }
}

function formatInteger(value) {
    const num = Number(value || 0);
    if (Number.isNaN(num)) return '0';
    return num.toLocaleString();
}

function refreshDashboardIfOpen(options = {}) {
    if (!isDashboardOpen()) return;
    if (!currentRunId) {
        setDashboardSubtitle('No session selected');
        renderDashboardEmpty('Start a run to view stats.');
        return;
    }
    const force = Boolean(options.forceRefresh);
    loadDashboardStats(force);
}

function selectDashboardAgent(agentId) {
    if (!agentId || agentId === selectedDashboardAgentId) return;
    selectedDashboardAgentId = agentId;
    if (lastDashboardData) {
        renderDashboard(lastDashboardData);
    } else {
        refreshDashboardIfOpen({ forceRefresh: true });
    }
}
