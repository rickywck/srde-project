"""
Backlog Synthesis Workflow - Externalized orchestration logic
Implements the full pipeline: segment → retrieve → generate → tag → evaluate
Uses Strands Workflow for structured multi-agent coordination
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from pinecone import Pinecone

from agents.segmentation_agent import create_segmentation_agent
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.tagging_agent import create_tagging_agent
from agents.evaluation_agent import create_evaluation_agent
from tools.retrieval_tool import create_retrieval_tool
from services.similar_story_retriever import SimilarStoryRetriever


class BacklogSynthesisWorkflow:
    """
    Orchestrates the backlog synthesis pipeline with clear dependency management.
    
    Workflow stages:
    1. Segmentation: Document → Segments with intent detection
    2. Retrieval: Segments → Retrieved context (ADO items + arch constraints)
    3. Generation: Segments + Context → Backlog items (Epics/Features/Stories)
    4. Tagging: Stories → Classifications (gap/conflict/new)
    5. Evaluation: Backlog → Quality assessment
    """
    
    def __init__(self, run_id: str, run_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self.run_dir = run_dir
        self.config = config or self._load_config()
        
        # Configuration
        self.min_similarity = self.config.get("retrieval", {}).get("tagging", {}).get("min_similarity_threshold", 0.5)
        self.embedding_model = self.config.get("openai", {}).get("embedding_model", "text-embedding-3-small")
        # Ensure embedding dims align with Pinecone index dimension (default 512)
        self.embedding_dimensions = int(self.config.get("openai", {}).get("embedding_dimensions", 512))
        self.index_name = self.config.get("pinecone", {}).get("index_name", "rde-lab")
        # Optional Pinecone namespace (project scoped)
        self.pinecone_namespace = (self.config.get("pinecone", {}).get("project") or "").strip()
        
        # Initialize clients (lazy)
        self._openai_client = None
        self._pinecone_client = None
        self._index = None
        self._similar_retriever: Optional[SimilarStoryRetriever] = None
        
        # Workflow state
        self.results = {
            "segmentation": None,
            "retrieval": [],
            "generation": [],
            "tagging": [],
            "evaluation": None
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path("config.poc.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    @property
    def openai_client(self) -> Optional[OpenAI]:
        """Lazy initialization of OpenAI client"""
        if self._openai_client is None and os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai_client
    
    @property
    def pinecone_client(self) -> Optional[Pinecone]:
        """Lazy initialization of Pinecone client"""
        if self._pinecone_client is None and os.getenv("PINECONE_API_KEY"):
            self._pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        return self._pinecone_client
    
    @property
    def index(self):
        """Lazy initialization of Pinecone index"""
        if self._index is None and self.pinecone_client:
            self._index = self.pinecone_client.Index(self.index_name)
        return self._index

    @property
    def similar_retriever(self) -> SimilarStoryRetriever:
        """Lazy initialization of shared similar story retriever"""
        if self._similar_retriever is None:
            self._similar_retriever = SimilarStoryRetriever(
                config=self.config,
                min_similarity=self.min_similarity,
                log_fn=lambda msg: self.log_progress(msg, save_to_history=False),
            )
        return self._similar_retriever
    
    def log_progress(self, message: str, save_to_history: bool = True):
        """Log workflow progress"""
        print(f"[{self.run_id}] {message}")
        if save_to_history:
            self._save_chat_history("system", message)
    
    def _save_chat_history(self, role: str, message: str):
        """Save message to chat history"""
        history_file = self.run_dir / "chat_history.jsonl"
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "role": role,
            "message": message
        }
        with open(history_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    async def execute(self, document_text: str) -> Dict[str, Any]:
        """
        Execute the complete workflow pipeline.
        
        Returns comprehensive results including all intermediate artifacts.
        """
        self.log_progress("🚀 Starting backlog synthesis workflow")
        
        # Clear previous run artifacts to ensure fresh state
        for filename in ["generated_backlog.jsonl", "tagging.jsonl", "segments.jsonl", "evaluation.jsonl"]:
            fpath = self.run_dir / filename
            if fpath.exists():
                try:
                    fpath.unlink()
                    self.log_progress(f"Cleared previous {filename}", save_to_history=False)
                except Exception as e:
                    self.log_progress(f"⚠ Failed to clear {filename}: {e}", save_to_history=False)
        
        try:
            # Stage 1: Segmentation
            segments = await self._stage_segmentation(document_text)
            
            # Stage 2 & 3: Retrieval + Generation (per segment)
            await self._stage_retrieval_and_generation(segments)
            
            # Stage 4: Tagging (per story)
            await self._stage_tagging()
            
            # Stage 5: Evaluation (optional, can be called separately)
            # evaluation = await self._stage_evaluation()
            
            # Compile final results
            return self._compile_results()
            
        except Exception as e:
            self.log_progress(f"❌ Workflow failed: {str(e)}")
            raise
    
    async def _stage_segmentation(self, document_text: str) -> List[Dict[str, Any]]:
        """Stage 1: Document Segmentation"""
        self.log_progress("Stage 1: Segmenting document")
        
        segmentation_tool = create_segmentation_agent(self.run_id)
        seg_result = json.loads(segmentation_tool(document_text))
        
        if seg_result.get("status") not in ("success", "success_mock"):
            raise ValueError(seg_result.get("error", "Segmentation failed"))
        
        segments = seg_result.get("segments", [])
        self.results["segmentation"] = seg_result
        self.log_progress(f"✓ Segmented into {len(segments)} segments")
        
        return segments
    
    async def _stage_retrieval_and_generation(self, segments: List[Dict[str, Any]]):
        """Stages 2 & 3: Retrieval + Generation per segment"""
        self.log_progress(f"Stage 2 & 3: Retrieval + Generation for {len(segments)} segments")
        
        retrieval_tool = create_retrieval_tool(self.run_id)
        generation_tool = create_backlog_generation_agent(self.run_id)
        
        for segment in segments:
            seg_id = segment["segment_id"]
            
            # Retrieval for this segment
            retrieval_payload = {
                "segment_id": seg_id,
                "segment_text": segment["raw_text"],
                "intent_labels": segment.get("intent_labels", []),
                "dominant_intent": segment.get("dominant_intent", "")
            }
            retrieval_result = json.loads(retrieval_tool(json.dumps(retrieval_payload)))
            self.results["retrieval"].append(retrieval_result)
            
            # Generation using retrieved context
            generation_payload = {
                "segment_id": seg_id,
                "segment_text": segment["raw_text"],
                "intent_labels": segment.get("intent_labels", []),
                "dominant_intent": segment.get("dominant_intent", ""),
                "retrieved_context": {
                    "ado_items": retrieval_result.get("ado_items", []),
                    "architecture_constraints": retrieval_result.get("architecture_constraints", [])
                }
            }
            generation_result = json.loads(generation_tool(json.dumps(generation_payload)))
            self.results["generation"].append(generation_result)
        
        self.log_progress("✓ Retrieval & generation completed for all segments")
    
    async def _stage_tagging(self):
        """Stage 4: Tag generated stories"""
        # Load generated backlog items
        # Always read from the canonical runs/<run_id> location where agents write
        backlog_file = Path(f"runs/{self.run_id}") / "generated_backlog.jsonl"
        if not backlog_file.exists():
            self.log_progress("⚠ No generated backlog found, skipping tagging")
            return
        
        generated_items = []
        with open(backlog_file, "r") as bf:
            for line in bf:
                if line.strip():
                    generated_items.append(json.loads(line))
        
        def _is_user_story(item: Dict[str, Any]) -> bool:
            t = (item.get("type") or item.get("work_item_type") or "").lower()
            return t in {"user story", "story", "user_story"}

        stories = [i for i in generated_items if _is_user_story(i)]
        
        if not stories:
            self.log_progress("⚠ No stories found, skipping tagging")
            return
        
        self.log_progress(f"Stage 4: Tagging {len(stories)} stories")

        tagging_tool = create_tagging_agent(self.run_id)

        # Batch tagging via agent so it manages file initialization once and appends internally
        stories_payload = []
        for s in stories:
            stories_payload.append({
                "title": s.get("title"),
                "description": s.get("description"),
                "acceptance_criteria": s.get("acceptance_criteria", []),
                "internal_id": s.get("internal_id")
            })

        batch_payload = {
            "stories": stories_payload,
            # Leave similar empty to let the agent perform internal retrieval per story
            "similar_existing_stories": [],
            "similarity_threshold": self.min_similarity
        }

        tagging_result = json.loads(tagging_tool(json.dumps(batch_payload)))

        # tagging_result in multi mode returns aggregated results
        tag_counts = {}
        if isinstance(tagging_result, dict) and tagging_result.get("status") == "ok":
            if tagging_result.get("mode") == "multi":
                results = tagging_result.get("results", [])
                self.results["tagging"].extend(results)
                for rec in results:
                    tag = (rec.get("decision_tag") or "unknown").lower()
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            elif tagging_result.get("mode") == "batch":
                # batch mode may only return counts; persist what we can
                self.results["tagging"].append(tagging_result)
                tag_counts = tagging_result.get("tag_counts", {}) or {}
        else:
            # Fallback: no structured result; keep empty but don't fail workflow
            self.log_progress("⚠ Tagging returned no structured results", save_to_history=False)

        self.log_progress(f"✓ Tagged {len(stories)} stories: {tag_counts}")
    
    def _find_similar_stories(self, story: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar existing stories using the shared retriever class"""
        try:
            return self.similar_retriever.find_similar_stories(story, min_similarity=self.min_similarity)
        except Exception as e:
            self.log_progress(f"⚠ Similar story search failed: {str(e)}", save_to_history=False)
            return []
    
    async def evaluate(self) -> Dict[str, Any]:
        """Stage 5: Evaluate generated backlog quality"""
        self.log_progress("Stage 5: Evaluating backlog quality")
        
        # Load generated backlog
        backlog_file = Path(f"runs/{self.run_id}") / "generated_backlog.jsonl"
        if not backlog_file.exists():
            raise FileNotFoundError("No generated backlog found for evaluation")
        
        generated_items = []
        with open(backlog_file, "r") as bf:
            for line in bf:
                if line.strip():
                    generated_items.append(json.loads(line))
        
        if not generated_items:
            raise ValueError("Backlog file is empty")
        
        # Get segment text for context
        segment_text = self._get_representative_segment_text()
        
        # Prepare evaluation payload
        evaluation_tool = create_evaluation_agent(self.run_id)
        payload = {
            "segment_text": segment_text,
            "retrieved_context": {"ado_items": [], "architecture_constraints": []},
            "generated_backlog": generated_items,
            "evaluation_mode": "live"
        }
        
        eval_result = json.loads(evaluation_tool(json.dumps(payload)))
        
        if eval_result.get("status") not in ("success", "success_mock"):
            raise ValueError(eval_result.get("error", "Evaluation failed"))
        
        self.results["evaluation"] = eval_result
        
        # Log summary
        evaluation = eval_result.get("evaluation", {})
        summary = [
            "🧪 Evaluation Results",
            f"Completeness: {evaluation.get('completeness', {}).get('score')}",
            f"Relevance: {evaluation.get('relevance', {}).get('score')}",
            f"Quality: {evaluation.get('quality', {}).get('score')}",
            f"Overall: {evaluation.get('overall_score')}"
        ]
        self.log_progress("\n".join(summary))
        
        return eval_result
    
    def _get_representative_segment_text(self) -> str:
        """Get representative segment text for evaluation context"""
        segments_file = self.run_dir / "segments.jsonl"
        raw_file = self.run_dir / "raw.txt"
        
        # Try to get first segment
        if segments_file.exists():
            with open(segments_file, "r") as sf:
                first_line = sf.readline()
                if first_line.strip():
                    try:
                        seg_obj = json.loads(first_line)
                        return seg_obj.get("raw_text", "")
                    except json.JSONDecodeError:
                        pass
        
        # Fallback to raw document head
        if raw_file.exists():
            return raw_file.read_text()[:4000]
        
        return ""
    
    def _compile_results(self) -> Dict[str, Any]:
        """Compile comprehensive workflow results"""
        # Load generated items count
        backlog_file = Path(f"runs/{self.run_id}") / "generated_backlog.jsonl"
        generated_items = []
        if backlog_file.exists():
            with open(backlog_file, "r") as bf:
                for line in bf:
                    if line.strip():
                        generated_items.append(json.loads(line))
        
        def _is_user_story(item: Dict[str, Any]) -> bool:
            t = (item.get("type") or item.get("work_item_type") or "").lower()
            return t in {"user story", "story", "user_story"}
        
        stories = [i for i in generated_items if _is_user_story(i)]
        
        # Tag distribution
        tag_counts = {}
        for rec in self.results["tagging"]:
            tag = rec.get("decision_tag", "unknown")
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Build response summary
        segments = self.results["segmentation"].get("segments", [])
        response_lines = [
            "🎯 Backlog Synthesis Complete",
            "",
            "=" * 60,
            "SEGMENTATION ✅",
            f"Segments: {len(segments)}",
            "",
            "RETRIEVAL ✅",
            f"Context retrieved for {len(self.results['retrieval'])} segments",
            "",
            "GENERATION ✅",
            f"Total backlog items: {len(generated_items)} (Stories: {len(stories)})",
            "",
            "TAGGING ✅",
            "Tag distribution:",
            *[f"- {k}: {v}" for k, v in tag_counts.items()],
            "",
            "FIRST 5 ITEMS:",
            *[f"[{itm.get('type')}] {itm.get('title')}" for itm in generated_items[:5]],
            "",
            "ARTIFACTS:",
            "- segments.jsonl",
            "- generated_backlog.jsonl",
            "- tagging.jsonl",
            "",
            "NEXT: Review items, optionally evaluate or write to ADO."
        ]
        
        response_text = "\n".join(response_lines)
        self._save_chat_history("assistant", response_text)
        
        return {
            "run_id": self.run_id,
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
                "segments": str(self.run_dir / "segments.jsonl"),
                "backlog": str(backlog_file),
                "tagging": str(self.run_dir / "tagging.jsonl")
            },
            "workflow_steps": {
                "segmentation": {
                    "status": "success",
                    "segments_count": len(segments),
                    "segments_file": str(self.run_dir / "segments.jsonl")
                },
                "retrieval": {
                    "status": "success",
                    "message": f"Context retrieved for {len(self.results['retrieval'])} segments"
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
