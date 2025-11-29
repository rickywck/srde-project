"""
Segmentation Agent - Specialized agent for document segmentation with intent detection
"""

import os
import json
import yaml
import logging
from pathlib import Path
from strands import Agent, tool
from .prompt_loader import get_prompt_loader
from .model_factory import ModelFactory

# Module logger
logger = logging.getLogger(__name__)


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
    prompt_config = prompt_loader.load_prompt("segmentation_agent")
    system_prompt = prompt_loader.get_system_prompt("segmentation_agent")
    user_template = prompt_loader.get_user_prompt_template("segmentation_agent")
    prompt_params = prompt_loader.get_parameters("segmentation_agent") or {}

    # Load app config via ModelFactory so model selection and params match backlog agent
    config_path = "config.poc.yaml"
    try:
        _cfg = ModelFactory._load_config(config_path)
        logger.debug("Loaded config for segmentation agent: %s", {k: v for k, v in (_cfg or {}).items()})
    except Exception as e:
        logger.exception("Error loading config via ModelFactory: %s", e)
        _cfg = {}

    # Determine effective max tokens (priority: agent prompt params -> app config -> model default)
    agent_max_tokens = prompt_params.get("max_completion_tokens") or prompt_params.get("max_tokens")
    app_max_tokens = _cfg.get("openai", {}).get("max_tokens")
    if agent_max_tokens is not None:
        eff_max_tokens = int(agent_max_tokens)
    elif app_max_tokens is not None:
        eff_max_tokens = int(app_max_tokens)
    else:
        eff_max_tokens = None

    model_params = {}
    if eff_max_tokens is not None:
        model_params["max_completion_tokens"] = eff_max_tokens

    try:
        model_descriptor = ModelFactory.create_openai_model(config_path=config_path, model_params=model_params)
        model_name = getattr(model_descriptor, "model_id", None) or ModelFactory.get_default_model_id(config_path)
        model_params_from_factory = getattr(model_descriptor, "params", None) or {}
        logger.debug("Model descriptor from factory: model_name=%s params=%s", model_name, model_params_from_factory)
    except Exception as e:
        logger.exception("ModelFactory.create_openai_model failed, falling back to env/config defaults: %s", e)
        model_name = ModelFactory.get_default_model_id(config_path)
        model_params_from_factory = {}

    # Import OpenAI lazily so tests without package can still import this module
    OpenAI = None
    try:
        from openai import OpenAI as _OpenAI
        OpenAI = _OpenAI
        logger.debug("`openai` package available; OpenAI client can be created")
    except ModuleNotFoundError:
        logger.warning("`openai` package not installed; running in MOCK mode unless an alternative client is provided")

    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key) if (api_key and OpenAI is not None) else None

    # Merge prompt params with any params from the model factory (factory wins)
    params = dict(model_params_from_factory)
    for k, v in (prompt_params or {}).items():
        if k not in params:
            params[k] = v
    
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
                if not openai_client:
                    raise ValueError("OpenAI client not initialized and mock mode not enabled. Set OPENAI_API_KEY or SEGMENTATION_AGENT_MOCK=1.")
                # Call LLM for segmentation
                # Normalize token parameter to Responses-style key; omit temperature to avoid 400s on newer models
                max_comp = (
                    params.get("max_completion_tokens")
                    or params.get("max_output_tokens")
                    or params.get("max_tokens")
                    or 4000
                )
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": segmentation_prompt}],
                    temperature=params.get("temperature", 0.7),
                    max_tokens=eff_max_tokens,
                    **({"response_format": {"type": params.get("response_format", "json_object")}} if params.get("response_format") else {})
                )
                # Parse response
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
            
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
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
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
