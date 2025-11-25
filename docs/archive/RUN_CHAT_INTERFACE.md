# Running the Chat Interface

## Prerequisites

1. Python 3.9 or higher
2. OpenAI API key

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   
   Create a `.env` file in the project root (or export directly):
   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Verify configuration:**
   
   Ensure `config.poc.yaml` has correct settings. The default config should work for the POC.

## Running the Application

1. **Start the FastAPI server:**
   ```bash
   python app.py
   ```
   
   Or with uvicorn directly:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open the chat interface:**
   
   Navigate to http://localhost:8000 in your browser.

## Using the Interface

### Upload a Document

1. Drag and drop a file (`.txt`, `.md`, `.docx`) onto the upload area, or click to browse
2. The document will be uploaded and assigned a unique run ID
3. The chat interface will activate

### Chat with the System

Type messages in the chat input to interact with the system. Try:

- **"What's in this document?"** - Get a summary of the uploaded content
- **"Analyze this document"** - Request document analysis
- **"Generate backlog items"** - Request backlog generation (functionality coming in next iteration)
- **"Show me the segments"** - Ask about document segments (coming soon)

### Quick Actions

Use the sidebar buttons for common tasks:
- **📊 Analyze Document** - Quick analysis request
- **✨ Generate Backlog** - Request backlog generation
- **📋 Show Backlog Items** - View generated items (when available)
- **🏷️ Show Tagging Results** - View tagging results (when available)

### Recent Runs

The sidebar shows your recent document processing runs. Click any run to switch to it and view its chat history.

## Current Implementation Status

✅ **Implemented (POC Phase 1):**
- File upload (meeting notes/transcripts)
- Chat interface with conversation history
- Passthrough supervisor agent (sends to LLM)
- Run management (multiple sessions)
- Document context awareness

⏳ **Coming in Next Iterations:**
- Document segmentation agent
- Backlog generation agent
- Tagging agent
- Retrieval from Pinecone (ADO items & architecture)
- ADO write functionality

## API Endpoints

If you want to use the API directly:

### POST /upload
Upload a document to create a new run.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@meeting_notes.txt"
```

### POST /chat/{run_id}
Send a chat message.

```bash
curl -X POST http://localhost:8000/chat/{run_id} \
  -H "Content-Type: application/json" \
  -d '{"message": "What is in this document?"}'
```

### GET /chat-history/{run_id}
Get conversation history.

```bash
curl http://localhost:8000/chat-history/{run_id}
```

### GET /runs
List all runs.

```bash
curl http://localhost:8000/runs
```

## Directory Structure

```
runs/
  {run_id}/
    raw.txt                    # Uploaded document
    chat_history.jsonl         # Conversation history
    segments.jsonl             # Document segments (future)
    generated_backlog.jsonl    # Generated items (future)
    tagging.jsonl              # Tagging results (future)
```

## Troubleshooting

**Port already in use:**
```bash
# Use a different port
uvicorn app:app --port 8001
```

**OpenAI API errors:**
- Verify your API key is set correctly
- Check your OpenAI account has credits
- Ensure you have access to the gpt-4o model

**File upload issues:**
- Check file size (max ~10MB recommended)
- Ensure file format is supported (.txt, .md, .docx)
- Check browser console for errors

## Next Steps

The current implementation provides a working chat interface with document upload. Future iterations will add:

1. **Segmentation Agent** - Split documents into coherent chunks
2. **Retrieval Tool** - Search Pinecone for relevant context
3. **Generation Agent** - Create backlog items
4. **Tagging Agent** - Classify stories relative to existing backlog
5. **ADO Writer** - Persist items to Azure DevOps

Each of these will be integrated into the supervisor agent orchestration flow.
