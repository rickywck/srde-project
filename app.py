"""
FastAPI application for Backlog Synthesizer POC
Provides chat interface and API endpoints for document processing
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import uuid
from datetime import datetime
from pathlib import Path

from supervisor import SupervisorAgent

app = FastAPI(title="Backlog Synthesizer POC")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize supervisor agent
supervisor = SupervisorAgent()

# Data models
class ChatMessage(BaseModel):
    message: str
    instruction_type: Optional[str] = None

class ChatResponse(BaseModel):
    run_id: str
    response: str
    status: Dict[str, Any]
    timestamp: str

class UploadResponse(BaseModel):
    run_id: str
    filename: str
    message: str

# Ensure runs directory exists
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

def get_run_dir(run_id: str) -> Path:
    """Get the directory for a specific run"""
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(exist_ok=True)
    return run_dir

def save_chat_history(run_id: str, role: str, message: str):
    """Save chat message to history"""
    run_dir = get_run_dir(run_id)
    history_file = run_dir / "chat_history.jsonl"
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "role": role,
        "message": message
    }
    
    with open(history_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

@app.get("/")
async def root():
    """Serve the chat interface"""
    return FileResponse("static/index.html")

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (meeting notes/transcript) to start a new run
    """
    try:
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        run_dir = get_run_dir(run_id)
        
        # Save uploaded file
        content = await file.read()
        raw_file = run_dir / "raw.txt"
        
        with open(raw_file, "wb") as f:
            f.write(content)
        
        # Initialize chat history
        save_chat_history(run_id, "system", f"Document uploaded: {file.filename}")
        
        return UploadResponse(
            run_id=run_id,
            filename=file.filename,
            message=f"Document uploaded successfully. Use run_id: {run_id} to interact."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/chat/{run_id}", response_model=ChatResponse)
async def chat(run_id: str, message: ChatMessage):
    """
    Send a chat message and get response from supervisor agent
    """
    try:
        run_dir = get_run_dir(run_id)
        
        # Check if run exists
        if not run_dir.exists():
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        
        # Save user message to history
        save_chat_history(run_id, "user", message.message)
        
        # Load document if it exists
        raw_file = run_dir / "raw.txt"
        document_text = None
        if raw_file.exists():
            with open(raw_file, "r") as f:
                document_text = f.read()
        
        # Get response from supervisor agent (passthrough to LLM)
        response = await supervisor.process_message(
            run_id=run_id,
            message=message.message,
            instruction_type=message.instruction_type,
            document_text=document_text
        )
        
        # Save assistant response to history
        save_chat_history(run_id, "assistant", response["response"])
        
        return ChatResponse(
            run_id=run_id,
            response=response["response"],
            status=response.get("status", {}),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/chat-history/{run_id}")
async def get_chat_history(run_id: str):
    """
    Get conversation history for a run
    """
    try:
        run_dir = get_run_dir(run_id)
        history_file = run_dir / "chat_history.jsonl"
        
        if not history_file.exists():
            return {"run_id": run_id, "history": []}
        
        history = []
        with open(history_file, "r") as f:
            for line in f:
                if line.strip():
                    history.append(json.loads(line))
        
        return {"run_id": run_id, "history": history}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@app.get("/backlog/{run_id}")
async def get_backlog(run_id: str):
    """
    Get generated backlog items for a run
    """
    try:
        run_dir = get_run_dir(run_id)
        backlog_file = run_dir / "generated_backlog.jsonl"
        
        if not backlog_file.exists():
            return {"run_id": run_id, "items": [], "message": "No backlog generated yet"}
        
        items = []
        with open(backlog_file, "r") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        return {"run_id": run_id, "items": items, "count": len(items)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get backlog: {str(e)}")

@app.get("/tagging/{run_id}")
async def get_tagging(run_id: str):
    """
    Get tagging results for a run
    """
    try:
        run_dir = get_run_dir(run_id)
        tagging_file = run_dir / "tagging.jsonl"
        
        if not tagging_file.exists():
            return {"run_id": run_id, "items": [], "message": "No tagging results yet"}
        
        items = []
        with open(tagging_file, "r") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        
        return {"run_id": run_id, "items": items, "count": len(items)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tagging: {str(e)}")

@app.get("/runs")
async def list_runs():
    """
    List all available runs
    """
    try:
        runs = []
        for run_dir in RUNS_DIR.iterdir():
            if run_dir.is_dir():
                # Get basic info about the run
                raw_file = run_dir / "raw.txt"
                history_file = run_dir / "chat_history.jsonl"
                
                run_info = {
                    "run_id": run_dir.name,
                    "has_document": raw_file.exists(),
                    "has_history": history_file.exists(),
                    "created": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat()
                }
                runs.append(run_info)
        
        # Sort by creation time (most recent first)
        runs.sort(key=lambda x: x["created"], reverse=True)
        
        return {"runs": runs, "count": len(runs)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(e)}")

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
