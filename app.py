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

from supervisor import SupervisorAgent
from agents.segmentation_agent import create_segmentation_agent
from tools.retrieval_tool import create_retrieval_tool
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.tagging_agent import create_tagging_agent
from agents.evaluation_agent import create_evaluation_agent

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

@app.post("/generate-backlog/{run_id}")
async def generate_backlog(run_id: str):
    """Run full backlog synthesis workflow: segment → retrieve → generate → tag."""
    try:
        run_dir = get_run_dir(run_id)
        raw_file = run_dir / "raw.txt"
        if not raw_file.exists():
            raise HTTPException(status_code=404, detail=f"No document found for run {run_id}")

        # Load config (thresholds, models)
        config = {}
        if Path("config.poc.yaml").exists():
            with open("config.poc.yaml", "r") as f:
                config = yaml.safe_load(f) or {}
        min_similarity = config.get("retrieval", {}).get("min_similarity_threshold", 0.7)
        embedding_model = config.get("openai", {}).get("embedding_model", "text-embedding-3-small")

        document_text = raw_file.read_text()
        save_chat_history(run_id, "system", "🚀 Starting full backlog synthesis workflow")

        # Initialize tool instances
        segmentation_tool = create_segmentation_agent(run_id)
        retrieval_tool = create_retrieval_tool(run_id)
        generation_tool = create_backlog_generation_agent(run_id)
        tagging_tool = create_tagging_agent(run_id)

        # Step 1: Segmentation
        save_chat_history(run_id, "system", "Step 1: Segmenting document")
        seg_result = json.loads(segmentation_tool(document_text))
        if seg_result.get("status") != "success" and seg_result.get("status") != "success_mock":
            raise HTTPException(status_code=500, detail=seg_result.get("error", "Segmentation failed"))
        segments = seg_result.get("segments", [])
        save_chat_history(run_id, "system", f"✓ Segmented into {len(segments)} segments")

        # Prepare containers
        segment_contexts = []
        generation_summaries = []

        # Step 2 & 3: Retrieval + Generation per segment
        for segment in segments:
            seg_id = segment["segment_id"]
            payload = {
                "segment_id": seg_id,
                "segment_text": segment["raw_text"],
                "intent_labels": segment.get("intent_labels", []),
                "dominant_intent": segment.get("dominant_intent", "")
            }
            retrieval_result = json.loads(retrieval_tool(json.dumps(payload)))
            segment_contexts.append(retrieval_result)

            gen_payload = {
                "segment_id": seg_id,
                "segment_text": segment["raw_text"],
                "intent_labels": segment.get("intent_labels", []),
                "dominant_intent": segment.get("dominant_intent", ""),
                "retrieved_context": {
                    "ado_items": retrieval_result.get("ado_items", []),
                    "architecture_constraints": retrieval_result.get("architecture_constraints", [])
                }
            }
            gen_result = json.loads(generation_tool(json.dumps(gen_payload)))
            generation_summaries.append(gen_result)

        save_chat_history(run_id, "system", "✓ Retrieval & generation completed for all segments")

        # Collect generated stories for tagging
        backlog_file = run_dir / "generated_backlog.jsonl"
        generated_items: List[Dict[str, Any]] = []
        if backlog_file.exists():
            with open(backlog_file, "r") as bf:
                for line in bf:
                    if line.strip():
                        generated_items.append(json.loads(line))
        stories = [i for i in generated_items if i.get("type", "").lower() == "story"]

        # Initialize clients for story-level retrieval
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
        index_name = config.get("pinecone", {}).get("index_name", "rde-lab")
        index = pc.Index(index_name) if pc else None

        tagging_records = []
        tagging_file = run_dir / "tagging.jsonl"

        def build_story_text(story: Dict[str, Any]) -> str:
            ac = story.get("acceptance_criteria", []) or []
            return story.get("title", "") + "\n" + story.get("description", "") + "\n" + "\n".join(ac)

        for story in stories:
            story_text = build_story_text(story)
            similar_existing_stories = []
            if openai_client and index:
                try:
                    emb_resp = openai_client.embeddings.create(model=embedding_model, input=story_text[:3000])
                    vec = emb_resp.data[0].embedding
                    query_res = index.query(vector=vec, top_k=10, namespace="ado_items", include_metadata=True)
                    for match in query_res.get("matches", []):
                        score = match.get("score", 0)
                        if score >= min_similarity:
                            md = match.get("metadata", {})
                            # Only keep if existing item is a story/user story
                            item_type = (md.get("type") or md.get("work_item_type") or "").lower()
                            if "story" in item_type:
                                similar_existing_stories.append({
                                    "work_item_id": md.get("work_item_id") or match.get("id"),
                                    "title": md.get("title", ""),
                                    "description": md.get("description", "")[:500],
                                    "similarity": score
                                })
                except Exception as e:  # Fallback silent
                    similar_existing_stories = []

            # Early exit if none above threshold
            if not similar_existing_stories:
                tagging_output = {
                    "status": "ok",
                    "run_id": run_id,
                    "decision_tag": "new",
                    "related_ids": [],
                    "reason": "No similar existing stories found (all below threshold)",
                    "early_exit": True,
                    "similarity_threshold": min_similarity,
                    "similar_count": 0,
                    "model_used": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
                    "story_internal_id": story.get("internal_id")
                }
            else:
                tag_payload = {
                    "story": {
                        "title": story.get("title"),
                        "description": story.get("description"),
                        "acceptance_criteria": story.get("acceptance_criteria", [])
                    },
                    "similar_existing_stories": similar_existing_stories,
                    "similarity_threshold": min_similarity
                }
                tag_result = json.loads(tagging_tool(json.dumps(tag_payload)))
                tag_result["story_internal_id"] = story.get("internal_id")
                tagging_output = tag_result

            tagging_records.append(tagging_output)
            with open(tagging_file, "a") as tf:
                tf.write(json.dumps(tagging_output) + "\n")

        save_chat_history(run_id, "system", f"✓ Tagged {len(stories)} stories")

        # Summaries
        tag_counts = {}
        for rec in tagging_records:
            tag = rec.get("decision_tag", "unknown")
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        response_lines = [
            "🎯 Backlog Synthesis Complete",
            "",
            "=" * 60,
            "SEGMENTATION ✅",
            f"Segments: {len(segments)}",
            "",
            "RETRIEVAL ✅",
            f"Context retrieved for {len(segment_contexts)} segments",
            "",
            "GENERATION ✅",
            f"Total backlog items: {len(generated_items)} (Stories: {len(stories)})",
            "",
            "TAGGING ✅",
            "Tag distribution:" + "\n" + "\n".join([f"- {k}: {v}" for k, v in tag_counts.items()]),
            "",
            "FIRST 5 ITEMS:",
        ]
        for itm in generated_items[:5]:
            response_lines.append(f"[{itm.get('type')}] {itm.get('title')}")

        response_lines.extend([
            "",
            "ARTIFACTS:",
            f"- segments.jsonl",
            f"- generated_backlog.jsonl",
            f"- tagging.jsonl",
            "",
            "NEXT: Review items, optionally write to ADO."
        ])

        response_text = "\n".join(response_lines)
        save_chat_history(run_id, "assistant", response_text)

        return {
            "run_id": run_id,
            "status": "success",
            "message": "Workflow completed",
            "response": response_text,
            "counts": {
                "segments": len(segments),
                "backlog_items": len(generated_items),
                "stories": len(stories),
                "tags": tag_counts
            },
            "files": {
                "segments": str(run_dir / "segments.jsonl"),
                "backlog": str(backlog_file),
                "tagging": str(tagging_file)
            },
            "workflow_steps": {
                "segmentation": {
                    "status": "success",
                    "segments_count": len(segments),
                    "segments_file": str(run_dir / "segments.jsonl")
                },
                "retrieval": {
                    "status": "success",
                    "message": f"Context retrieved for {len(segment_contexts)} segments"
                },
                "generation": {
                    "status": "success",
                    "message": f"Generated {len(generated_items)} items ({len(stories)} stories)"
                },
                "tagging": {
                    "status": "success",
                    "message": f"Tagged {len(stories)} stories",
                    "tag_distribution": tag_counts
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        save_chat_history(run_id, "system", f"❌ Workflow failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

@app.post("/evaluate/{run_id}")
async def evaluate_backlog(run_id: str):
    """Run evaluation agent on the generated backlog for a run."""
    try:
        run_dir = get_run_dir(run_id)
        backlog_file = run_dir / "generated_backlog.jsonl"
        raw_file = run_dir / "raw.txt"
        segments_file = run_dir / "segments.jsonl"

        if not backlog_file.exists():
            raise HTTPException(status_code=404, detail="No generated backlog found for this run")

        # Load generated backlog items
        generated_items: List[Dict[str, Any]] = []
        with open(backlog_file, "r") as bf:
            for line in bf:
                if line.strip():
                    try:
                        generated_items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not generated_items:
            raise HTTPException(status_code=400, detail="Backlog file is empty or invalid")

        # Derive a representative segment text: use first segment if available, else raw doc head
        segment_text = ""
        if segments_file.exists():
            with open(segments_file, "r") as sf:
                first_line = sf.readline()
                if first_line.strip():
                    try:
                        seg_obj = json.loads(first_line)
                        segment_text = seg_obj.get("raw_text", "")
                    except json.JSONDecodeError:
                        segment_text = ""
        if not segment_text and raw_file.exists():
            segment_text = raw_file.read_text()[:4000]

        # Minimal retrieved context (not persisted yet) - could be enhanced later
        retrieved_context = {"ado_items": [], "architecture_constraints": []}

        # Prepare evaluation agent
        evaluation_tool = create_evaluation_agent(run_id)
        payload = {
            "segment_text": segment_text,
            "retrieved_context": retrieved_context,
            "generated_backlog": generated_items,
            "evaluation_mode": "live"
        }
        eval_result = json.loads(evaluation_tool(json.dumps(payload)))

        if eval_result.get("status") not in ("success", "success_mock"):
            raise HTTPException(status_code=500, detail=eval_result.get("error", "Evaluation failed"))

        # Save summary line to chat history
        evaluation = eval_result.get("evaluation", {})
        summary_lines = [
            "🧪 Evaluation Results",
            f"Completeness: {evaluation.get('completeness', {}).get('score')} - {evaluation.get('completeness', {}).get('reasoning','')[:120]}",
            f"Relevance: {evaluation.get('relevance', {}).get('score')} - {evaluation.get('relevance', {}).get('reasoning','')[:120]}",
            f"Quality: {evaluation.get('quality', {}).get('score')} - {evaluation.get('quality', {}).get('reasoning','')[:120]}",
            f"Overall: {evaluation.get('overall_score')}",
        ]
        suggestions = evaluation.get("suggestions", [])
        if suggestions:
            summary_lines.append("Suggestions:")
            for s in suggestions[:5]:
                summary_lines.append(f"- {s}")
        save_chat_history(run_id, "assistant", "\n".join(summary_lines))

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
