"""
Segmentation Agent - Specialized agent for document segmentation with intent detection
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
from strands import Agent, tool
from strands.types.exceptions import StructuredOutputException
from .prompt_loader import get_prompt_loader
from .model_factory import ModelFactory

# Module logger
logger = logging.getLogger(__name__)


class SegmentOut(BaseModel):
    segment_id: int
    segment_order: int
    raw_text: str
    intent_labels: List[str]
    dominant_intent: str

    class Config:
        extra = "allow"


class SegmentationResponseIn(BaseModel):
    segments: List[SegmentOut]

    class Config:
        extra = "allow"


def create_segmentation_agent(run_id: str):
    """
    Create a segmentation agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    prompt_loader.load_prompt("segmentation_agent")
    system_prompt = prompt_loader.get_system_prompt("segmentation_agent")
    prompt_loader.get_user_prompt_template("segmentation_agent")
    prompt_params = prompt_loader.get_parameters("segmentation_agent") or {}

    # Create a Strands OpenAIModel using factory helper (agent does not access config or API key directly)
    try:
        model = ModelFactory.create_openai_model_for_agent(agent_params=prompt_params)
        model_name = getattr(model, "model_id", None) or ModelFactory.get_default_model_id()
        logger.debug("Initialized Strands OpenAIModel for segmentation: %s", model_name)
    except Exception as e:
        logger.exception("Failed to create Strands OpenAIModel for segmentation: %s", e)
        model = None
        model_name = ModelFactory.get_default_model_id()

    # Initialize Strands Agent
    agent = None
    if model is not None:
        try:
            agent = Agent(model=model, system_prompt=system_prompt)
        except Exception as e:
            logger.exception("Failed to initialize Strands Agent (segmentation): %s", e)
            agent = None
    
    @tool
    def segment_document(document_text: str) -> str:
        """
        Segments a document into coherent chunks with intent detection.
        
        Args:
            document_text: The full text of the document to segment
            
        Returns:
            JSON string containing segmentation results with segment_id, raw_text, intent_labels, and dominant_intent
        """
        
        # Build segmentation prompt from template
        segmentation_prompt = prompt_loader.format_user_prompt(
            "segmentation_agent",
            document_text=document_text
        )
        
        try:
            logger.info("Segmentation Agent: Processing document (run_id: %s)", run_id)

            # Mock mode for offline / no network testing
            if os.getenv("SEGMENTATION_AGENT_MOCK") == "1":
                logger.info("Segmentation Agent: Using MOCK mode (SEGMENTATION_AGENT_MOCK=1)")
                # Simple segmentation by double newlines
                raw_segments = [s.strip() for s in document_text.split("\n\n") if s.strip()]

                def _generate_intents(text: str):
                    lower = text.lower()
                    intents = []
                    patterns = [
                        ("multi-factor authentication", ["multi_factor_authentication_security_upgrade", "sms_email_totp_channel_support", "auth_service_schema_migration_requirement", "secure_account_access_for_users"]),
                        ("performance", ["dashboard_search_api_latency_optimization", "database_query_index_additions", "p95_response_time_reduction_goal"]),
                        ("offline mode", ["offline_sync_local_cache_strategy", "offline_document_access_user_value"]),
                        ("api documentation", ["api_docs_code_comment_generation_adoption", "developer_integration_experience_improvement", "technical_debt_documentation_modernization"]),
                        ("open questions", ["timeline_for_auth_enhancement_delivery", "budget_for_offline_mode_initiative", "ownership_api_doc_tooling_setup"]),
                    ]
                    for trigger, labels in patterns:
                        if trigger in lower:
                            intents.extend(labels)
                    # Fallback heuristic: take top keywords
                    if not intents:
                        import re
                        words = re.findall(r"[a-zA-Z_]{4,}", lower)
                        stop = {"this", "that", "with", "from", "have", "will", "need", "users", "been", "were", "also", "even", "such", "into", "then", "them", "they", "high", "mode", "docs", "api", "page"}
                        filtered = [w for w in words if w not in stop]
                        unique = []
                        for w in filtered:
                            if w not in unique:
                                unique.append(w)
                        base = unique[:4] if unique else ["general", "segment"]
                        intents.append("_".join(base) + "_topic_focus")
                    # Trim to 3–6 intents
                    if len(intents) > 6:
                        intents = intents[:6]
                    dominant = intents[0]
                    return intents, dominant

                segments = []
                for i, seg_text in enumerate(raw_segments, 1):
                    intents, dominant = _generate_intents(seg_text)
                    segments.append({
                        "segment_id": i,
                        "segment_order": i,
                        "raw_text": seg_text,
                        "intent_labels": intents,
                        "dominant_intent": dominant
                    })
                result = {"status": "success_mock", "segments": segments}
            else:
                if agent is None:
                    raise ValueError("Segmentation agent not initialized and mock mode not enabled. Set SEGMENTATION_AGENT_MOCK=1 to use mock.")
                # Use Strands with Structured Output
                try:
                    agent_result = agent(
                        segmentation_prompt,
                        structured_output_model=SegmentationResponseIn,
                    )
                    validated: SegmentationResponseIn = agent_result.structured_output  # type: ignore[assignment]
                    # Convert to plain dict for downstream compatibility
                    result: Dict[str, Any] = {"segments": []}
                    for seg in validated.segments:
                        try:
                            seg_dict = seg.model_dump() if hasattr(seg, "model_dump") else seg.dict()
                        except Exception:
                            seg_dict = dict(seg)
                        result["segments"].append(seg_dict)
                except (StructuredOutputException, ValidationError) as e:
                    raise ValueError(f"Structured output failed: {e}")
            
            # Validate structure
            if "segments" not in result:
                raise ValueError("Response missing 'segments' key")
            
            segments = result["segments"]
            
            # Ensure output directory exists
            output_dir = Path(f"runs/{run_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write segments to JSONL file
            segments_file = output_dir / "segments.jsonl"
            with open(segments_file, "w") as f:
                for segment in segments:
                    f.write(json.dumps(segment) + "\n")
            
            logger.info("Segmentation Agent: Created %s segments", len(segments))
            
            # Prepare summary for display
            summary = {
                "status": "success",
                "run_id": run_id,
                "total_segments": len(segments),
                "segments_file": str(segments_file),
                "segments": segments
            }
            
            return json.dumps(summary, indent=2)
            
        except json.JSONDecodeError as e:
            error_msg = {
                "status": "error",
                "error": f"Failed to parse response as JSON: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)

        except Exception as e:
            error_msg = {
                "status": "error",
                "error": f"Segmentation failed: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
    
    return segment_document


# Note: System prompt now loaded from prompts/segmentation_agent.yaml
# This ensures consistency and easier prompt management
