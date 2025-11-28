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
import yaml
from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agents.supervisor_agent import SupervisorAgent
from workflows import BacklogSynthesisWorkflow, StrandsBacklogWorkflow
from tools.ado_writer_tool import create_ado_writer_tool
from tools.file_extractor import FileExtractor

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
    document_text: Optional[str] = None  # Optional document for chat-specific upload
    model_override: Optional[str] = None  # Optional: override OpenAI chat model

class ChatResponse(BaseModel):
    run_id: str
    response: str
    status: Dict[str, Any]
    timestamp: str
    # Optional UI hint to render structured content (e.g., backlog)
    response_type: Optional[str] = None

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
    Supports .txt, .md, .docx, .pdf formats
    """
    try:
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        run_dir = get_run_dir(run_id)
        
        # Read uploaded file
        content = await file.read()
        
        # Extract text from file using FileExtractor
        try:
            text = FileExtractor.extract_text(content, file.filename)
        except Exception as extract_error:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to extract text from file: {str(extract_error)}"
            )
        
        # Save extracted text to raw.txt
        raw_file = run_dir / "raw.txt"
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        # Initialize chat history
        save_chat_history(run_id, "system", f"Document uploaded: {file.filename}")
        
        return UploadResponse(
            run_id=run_id,
            filename=file.filename,
            message=f"Document uploaded successfully. Use run_id: {run_id} to interact."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from uploaded document (for chat attachment)
    Supports .txt, .md, .docx, .pdf formats
    """
    try:
        content = await file.read()
        
        # Use FileExtractor to extract text
        text = FileExtractor.extract_text(content, file.filename)
        
        return {"text": text, "filename": file.filename}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")

@app.post("/ado-export/{run_id}")
async def ado_export(run_id: str, dry_run: bool = True, filter_tags: Optional[str] = None):
    """
    Export generated backlog to Azure DevOps via ADO Writer tool.
    - dry_run: when true (default), produces a plan without creating items
    - filter_tags: comma-separated list of tags to include (default new,gap)
    """
    try:
        run_dir = get_run_dir(run_id)
        backlog_file = run_dir / "generated_backlog.jsonl"
        if not backlog_file.exists():
            raise HTTPException(status_code=404, detail="No generated backlog found for this run")

        tags = [t.strip() for t in (filter_tags.split(",") if filter_tags else ["new", "gap"]) if t.strip()]
        writer = create_ado_writer_tool(run_id)
        payload = json.dumps({
            "run_id": run_id,
            "filter_tags": tags,
            "dry_run": dry_run
        })
        result_json = writer(payload)
        try:
            result = json.loads(result_json)
        except Exception:
            # If tool returned non-JSON, wrap it
            result = {"status": "ok", "raw": result_json}
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ADO export failed: {str(e)}")

@app.get("/ado-export/last/{run_id}")
async def ado_export_last(run_id: str):
    """
    Return the most recent ADO export result (preview or actual) for a run.
    """
    try:
        run_dir = get_run_dir(run_id)
        last_path = run_dir / "ado_export_last.json"
        if not last_path.exists():
            raise HTTPException(status_code=404, detail="No ADO export result found for this run")
        try:
            data = json.loads(last_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to parse ADO result: {str(e)}")
        return JSONResponse(content=data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load ADO result: {str(e)}")

@app.post("/chat/{run_id}", response_model=ChatResponse)
async def chat(run_id: str, message: ChatMessage):
    """
    Send a chat message and get response from supervisor agent.
    
    The supervisor agent orchestrates all specialized tools:
    - segment_document: Document segmentation with intent detection
    - generate_backlog: Backlog item generation from segments (no retrieval)
    - tag_story: Story classification (new/gap/conflict)
    - generate_backlog_with_retrieval: Combined retrieval + backlog generation (retrieval results not returned)
    - evaluate_backlog_quality: LLM-as-judge quality evaluation
    
    The supervisor uses Strands Agent to decide which tools to invoke based on user message.
    
    Document handling:
    - Chat-specific document: Use document_text provided in the message body
    - Quick Actions document: NOT used by chat interface (stored in raw.txt)
    """
    try:
        # Auto-create run directory if it doesn't exist (for chat-only sessions)
        run_dir = get_run_dir(run_id)
        
        # Save user message to history
        save_chat_history(run_id, "user", message.message)
        
        # Use chat-provided document (if any), NOT the Quick Actions document
        document_text = message.document_text
        
        # Get response from supervisor agent (passthrough to LLM)
        response = await supervisor.process_message(
            run_id=run_id,
            message=message.message,
            instruction_type=message.instruction_type,
            document_text=document_text,
            model_override=message.model_override
        )
        
        # Save assistant response to history
        save_chat_history(run_id, "assistant", response["response"])
        
        return ChatResponse(
            run_id=run_id,
            response=response["response"],
            status=response.get("status", {}),
            timestamp=datetime.utcnow().isoformat(),
            response_type=response.get("response_type")
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

@app.post("/generate-backlog/{run_id}")
async def generate_backlog(run_id: str, use_strands_workflow: bool = False):
    """
    Run full backlog synthesis workflow: segment → retrieve → generate → tag.
    
    Args:
        run_id: Unique run identifier
        use_strands_workflow: If True, use Strands built-in workflow tool for orchestration.
                             If False (default), use custom sequential workflow.
    
    The workflow executes 4 stages with explicit dependency management:
    1. Segmentation (document → segments with intent detection)
    2. Retrieval (segments → RAG context from Pinecone)
    3. Generation (segments + context → backlog items)
    4. Tagging (stories → gap/conflict/new classification)
    """
    try:
        run_dir = get_run_dir(run_id)
        raw_file = run_dir / "raw.txt"
        if not raw_file.exists():
            raise HTTPException(status_code=404, detail=f"No document found for run {run_id}")

        document_text = raw_file.read_text()
        
        # Choose workflow implementation
        if use_strands_workflow:
            # Use Strands built-in workflow tool (automatic dependency resolution, parallel execution)
            try:
                workflow = StrandsBacklogWorkflow(run_id, run_dir)
            except ImportError as e:
                raise HTTPException(
                    status_code=501,
                    detail=f"Strands workflow not available: {str(e)}"
                )
        else:
            # Use custom sequential workflow (explicit stage management)
            workflow = BacklogSynthesisWorkflow(run_id, run_dir)
        
        # Execute workflow pipeline
        result = await workflow.execute(document_text)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        save_chat_history(run_id, "system", f"❌ Workflow failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

@app.post("/evaluate/{run_id}")
async def evaluate_backlog(run_id: str):
    """
    Run evaluation agent on the generated backlog for a run.
    Uses externalized BacklogSynthesisWorkflow for evaluation stage.
    """
    try:
        run_dir = get_run_dir(run_id)
        backlog_file = run_dir / "generated_backlog.jsonl"

        if not backlog_file.exists():
            raise HTTPException(status_code=404, detail="No generated backlog found for this run")

        # Initialize workflow orchestrator
        workflow = BacklogSynthesisWorkflow(run_id, run_dir)
        
        # Execute evaluation stage
        eval_result = await workflow.evaluate()
        
        # Extract evaluation details
        evaluation = eval_result.get("evaluation", {})
        
        return {
            "run_id": run_id,
            "status": eval_result.get("status"),
            "items_evaluated": eval_result.get("items_evaluated"),
            "evaluation": evaluation,
            "raw": eval_result,
            "timestamp": eval_result.get("timestamp")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        save_chat_history(run_id, "system", f"❌ Evaluation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

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
