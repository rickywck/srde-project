"""
Combined Retrieval + Backlog Generation Tool

This tool encapsulates both context retrieval (from Pinecone) and backlog generation
(LLM-based) to minimize conversation payload. It returns only the backlog generation
result, not the raw retrieval context, to reduce history size.

It can take a segment (id) from runs/<run_id>/segments.jsonl or accept raw segment
text and intents directly.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from strands import tool

# Module logger
logger = logging.getLogger(__name__)

from tools.retrieval_tool import create_retrieval_tool
from agents.backlog_generation_agent import create_backlog_generation_agent


def _load_segment_from_file(run_id: str, segment_id: int, segments_file: Optional[str] = None) -> Dict[str, Any]:
    """Load a segment record from runs/<run_id>/segments.jsonl by id."""
    file_path = Path(segments_file) if segments_file else Path(f"runs/{run_id}/segments.jsonl")
    if not file_path.exists():
        raise FileNotFoundError(f"Segments file not found: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if int(rec.get("segment_id", -1)) == int(segment_id):
                return rec
    raise ValueError(f"Segment with id {segment_id} not found in {file_path}")


def create_retrieval_backlog_tool(run_id: str):
    """
    Factory to create the combined retrieval + backlog generation tool for a run.

    Returns a Strands tool function that:
    1) Accepts either raw segment data or a segment_id to read from segments.jsonl
    2) Performs retrieval using the existing retrieval_tool
    3) Performs backlog generation using the existing backlog_generation_agent
    4) Returns ONLY the backlog generation summary (no raw retrieval context in response)
    """

    retrieval_fn = create_retrieval_tool(run_id)
    backlog_fn = create_backlog_generation_agent(run_id)

    @tool
    def generate_backlog_with_retrieval(
        segment_id: Optional[int] = None,
        segment_text: Optional[str] = None,
        intent_labels: Optional[List[str]] = None,
        dominant_intent: Optional[str] = None,
        segments_file_path: Optional[str] = None,
    ) -> str:
        """
        Generate backlog items WITH context retrieval, returning only generation results.

        This tool supports TWO input modes:
        1. From uploaded document: Load a pre-segmented chunk via segment_id from segments.jsonl
        2. From chat input: Provide raw requirement text directly via segment_text (no document upload needed)

        Parameters:
        - segment_id: integer id to load from runs/<run_id>/segments.jsonl (for uploaded documents)
        - segment_text: raw requirement text (use this for direct chat input OR segment text)
        - intent_labels: list of intent labels (optional, can be inferred)
        - dominant_intent: dominant intent label (optional, can be inferred)
        - segments_file_path: optional override path to segments.jsonl

        Output: JSON string with backlog generation summary (retrieval context is NOT included in output)
        
        Use cases:
        - User uploaded document: Call with segment_id after segmentation
        - User typed requirements in chat: Call with segment_text directly (no segmentation needed)
        """
        logger.debug("generate_backlog_with_retrieval called with: segment_id=%r, segment_text=%s..., intent_labels=%r, dominant_intent=%r, segments_file_path=%r",
                     segment_id, segment_text[:100] if segment_text else None, intent_labels, dominant_intent, segments_file_path)
        try:
            segments_file = segments_file_path

            # If no raw text provided, load from segments.jsonl using segment_id
            if (not segment_text) and segment_id is not None:
                rec = _load_segment_from_file(run_id, int(segment_id), segments_file)
                segment_text = rec.get("raw_text", "")
                # allow provided intents to override file if supplied
                intent_labels = intent_labels or rec.get("intent_labels", [])
                dominant_intent = dominant_intent or rec.get("dominant_intent")

            # Validate required inputs
            if not segment_text:
                raise ValueError("Missing segment_text. Provide segment_text directly or a valid segment_id present in segments.jsonl.")
            intent_labels = intent_labels or []
            dominant_intent = dominant_intent or (intent_labels[0] if intent_labels else "")
            seg_id = int(segment_id or 0)

            print(f"Combined Tool: Starting retrieval + generation for segment {seg_id} (run_id: {run_id})")

            # 1) Retrieval (returns JSON string)
            retrieval_json = retrieval_fn(
                query_data=None,
                segment_text=segment_text,
                intent_labels=intent_labels,
                dominant_intent=dominant_intent,
                segment_id=seg_id,
            )
            try:
                retrieved = json.loads(retrieval_json)
            except Exception:
                # fail open: use minimal empty context
                retrieved = {"ado_items": [], "architecture_constraints": []}

            # Optional: store a lightweight retrieval summary to disk (not returned)
            try:
                out_dir = Path(f"runs/{run_id}")
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / f"retrieval_summary_seg_{seg_id}.json", "w") as f:
                    summary = {
                        "segment_id": seg_id,
                        "ado_items_count": len(retrieved.get("ado_items", []) or []),
                        "architecture_items_count": len(retrieved.get("architecture_constraints", []) or []),
                    }
                    json.dump(summary, f)
            except Exception:
                pass

            # 2) Backlog generation (returns JSON string) — do NOT include retrieval payload in response
            gen_result_json = backlog_fn(
                segment_id=seg_id,
                segment_text=segment_text,
                intent_labels=intent_labels,
                dominant_intent=dominant_intent,
                retrieved_context={
                    "ado_items": retrieved.get("ado_items", []) or [],
                    "architecture_constraints": retrieved.get("architecture_constraints", []) or [],
                },
            )

            # Return generation result directly
            return gen_result_json

        except Exception as e:
            err = {
                "status": "error",
                "error": f"Combined retrieval+generation failed: {str(e)}",
                "run_id": run_id,
            }
            return json.dumps(err, indent=2)

    return generate_backlog_with_retrieval
