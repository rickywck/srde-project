"""
Backlog Synthesis Workflow - Externalized orchestration logic
Implements the full pipeline: segment → retrieve → generate → tag → evaluate
(Orchestrator-only: provider clients are handled inside agents/tools.)
"""

import os
import json
import yaml
from typing import Dict, Any, List, Optional
import asyncio
import functools
from pathlib import Path
from datetime import datetime
# No direct OpenAI/Pinecone usage in the orchestrator

from agents.segmentation_agent import create_segmentation_agent
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.tagging_agent import create_tagging_agent
from agents.evaluation_agent import create_evaluation_agent

from strands.multiagent import GraphBuilder

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
        # Orchestrator configuration (provider-agnostic)
        
        # Workflow state
        self.results = {
            "segmentation": None,
            "retrieval": [],
            "generation": [],
            "tagging": [],
            "evaluation": None
        }
        self.document_text = None  # Store full document text for evaluation
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = Path("config.poc.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}
    
    # No provider client properties; agents/tools handle their own clients internally.
    
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
        try:
            from agents.utils.strands_hooks import StrandsHookClient
            hook = StrandsHookClient(self.run_id)
            await hook.emit(event="started", agent="workflow", message="Workflow started")
        except Exception:
            pass
        self.document_text = document_text  # Store full document text for evaluation
        
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
            if 'hook' in locals() and hook:
                try:
                    await hook.emit(event="progress", agent="segmentation", message=f"Created {len(segments)} segments")
                except Exception:
                    pass

            # Stage 2 & 3: Retrieval + Generation (per segment)
            gen_items = await self._stage_retrieval_and_generation(segments)
            if hook:
                try:
                    await hook.emit(event="progress", agent="generation", message=f"Generated {len(gen_items)} items")
                except Exception:
                    pass

            # Stage 4: Tagging (per story)
            tag_stats = await self._stage_tagging()
            if hook:
                try:
                    await hook.emit(event="progress", agent="tagging", message=f"Tagged {tag_stats.get('processed',0)} stories")
                except Exception:
                    pass
            
            # Stage 5: Evaluation (optional, can be called separately)
            # evaluation = await self._stage_evaluation()
            
            # Compile final results
            res = self._compile_results()
            if hook:
                try:
                    await hook.emit(event="finished", agent="workflow", message="Workflow completed")
                except Exception:
                    pass
            return res
            
        except Exception as e:
            self.log_progress(f"❌ Workflow failed: {str(e)}")
            raise
    
    async def _stage_segmentation(self, document_text: str) -> List[Dict[str, Any]]:
        """Stage 1: Document Segmentation"""
        self.log_progress("Stage 1: Segmenting document")
        
        segmentation_tool = create_segmentation_agent(self.run_id)
        seg_json = await asyncio.to_thread(segmentation_tool, document_text)
        seg_result = json.loads(seg_json)
        
        if seg_result.get("status") not in ("success", "success_mock"):
            raise ValueError(seg_result.get("error", "Segmentation failed"))
        
        segments = seg_result.get("segments", [])
        self.results["segmentation"] = seg_result
        self.log_progress(f"✓ Segmented into {len(segments)} segments")
        
        return segments
    
    async def _stage_retrieval_and_generation(self, segments: List[Dict[str, Any]]):
        """Stages 2 & 3: Retrieval + Generation per segment (combined tool)"""
        self.log_progress(f"Stage 2 & 3: Retrieval + Generation for {len(segments)} segments")

        from tools.retrieval_backlog_tool import create_retrieval_backlog_tool
        combined_tool = create_retrieval_backlog_tool(self.run_id)

        processed = 0
        for segment in segments:
            seg_id = segment["segment_id"]
            fn = functools.partial(
                combined_tool,
                segment_id=seg_id,
                segment_text=segment.get("raw_text", ""),
                intent_labels=segment.get("intent_labels", []),
                dominant_intent=segment.get("dominant_intent", ""),
                user_instructions="",
            )
            gen_json = await asyncio.to_thread(fn)
            try:
                generation_result = json.loads(gen_json)
            except Exception as e:
                generation_result = {"status": "error", "error": f"Parse generation result failed: {e}", "segment_id": seg_id}
            self.results["generation"].append(generation_result)
            processed += 1
            # Emit coarse progress every 5 segments
            try:
                from agents.utils.strands_hooks import StrandsHookClient
                hook = StrandsHookClient(self.run_id)
                if processed % 5 == 0 or processed == len(segments):
                    await hook.emit(event="progress", agent="generation", message=f"Processed {processed}/{len(segments)} segments")
            except Exception:
                pass

        # Consolidate all generated items and rewrite the backlog file atomically
        try:
            all_items: List[Dict[str, Any]] = []
            for gen in self.results["generation"]:
                try:
                    if isinstance(gen, dict) and (gen.get("status") in ("success", "success_mock")):
                        all_items.extend(gen.get("backlog_items", []) or [])
                except Exception:
                    continue
            out_dir = Path(f"runs/{self.run_id}")
            out_dir.mkdir(parents=True, exist_ok=True)
            backlog_path = out_dir / "generated_backlog.jsonl"
            with open(backlog_path, "w") as f:
                for it in all_items:
                    f.write(json.dumps(it) + "\n")
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            self.log_progress(f"✓ Retrieval & generation completed for all segments (combined) — wrote {len(all_items)} items")
            return all_items
        except Exception as e:
            self.log_progress(f"⚠ Failed to consolidate backlog items: {e}")
            return []
    
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
        
        self.log_progress(f"Stage 4: Tagging {len(stories)} stories (per-story)")

        tagging_tool = create_tagging_agent(self.run_id)

        # Per-story tagging loop
        tag_counts: Dict[str, int] = {}
        processed = 0
        errors = 0
        for s in stories:
            payload = {
                "title": s.get("title") or "",
                "description": s.get("description") or "",
                "acceptance_criteria": s.get("acceptance_criteria", []) or [],
                "internal_id": s.get("internal_id"),
            }
            try:
                out = await asyncio.to_thread(tagging_tool, payload)
                res = json.loads(out)
                if isinstance(res, dict) and res.get("status") == "ok":
                    self.results["tagging"].append(res)
                    tag = (res.get("decision_tag") or "unknown").lower()
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    processed += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                self.log_progress(f"⚠ Tagging failed for story {s.get('internal_id')}: {e}", save_to_history=False)

        self.log_progress(f"✓ Tagged {processed} stories (errors: {errors}): {tag_counts}")
    
    def _find_similar_stories(self, story: Dict[str, Any]) -> List[Dict[str, Any]]:
        """No-op: similarity retrieval is performed by agents/tools."""
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
        segment_text = self.document_text or self._get_representative_segment_text()
        
        # Prepare evaluation payload
        evaluation_tool = create_evaluation_agent(self.run_id)
        payload = {
            "segment_text": segment_text,
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
