"""
Backlog Generation Agent - Specialized agent for generating backlog items from segments
"""

import os
import json
import yaml
import logging
import re
from typing import Dict, Any, List, Union
from pathlib import Path
from pydantic import BaseModel, ValidationError
from strands import tool, Agent
from strands.types.exceptions import StructuredOutputException
from .prompt_loader import get_prompt_loader
from tools.token_utils import estimate_tokens
from .model_factory import ModelFactory

# Module logger
logger = logging.getLogger(__name__)


# Pydantic models for strict validation of LLM output
class BacklogItemIn(BaseModel):
    type: str
    title: str
    description: str | None = None
    acceptance_criteria: List[str] | None = None
    parent_reference: str | None = None
    rationale: str | None = None

    class Config:
        extra = "allow"


class BacklogResponseIn(BaseModel):
    backlog_items: List[BacklogItemIn]

    class Config:
        extra = "allow"


def _pydantic_validate_response(payload: Dict[str, Any]) -> BacklogResponseIn:
    """Validate payload using Pydantic (v1/v2 compatible)."""
    try:
        return BacklogResponseIn.model_validate(payload)  # type: ignore[attr-defined]
    except AttributeError:
        return BacklogResponseIn.parse_obj(payload)


def extract_json(text: str) -> str | None:
    """Deprecated: no longer needed with Strands Structured Output. Retained for compatibility."""
    if not text:
        return None
    return text.strip()


def create_backlog_generation_agent(run_id: str):
    """
    Create a backlog generation agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    # Get OpenAI configuration via ModelFactory (centralized)
    api_key = os.getenv("OPENAI_API_KEY")
    config_path = "config.poc.yaml"
    # Use ModelFactory to load configuration and determine model id / params
    try:
        _cfg = ModelFactory._load_config(config_path)
        logger.debug("Loaded config for backlog agent: %s", {k: v for k, v in (_cfg or {}).items()})
    except Exception as e:
        logger.exception("Error loading config via ModelFactory: %s", e)
        _cfg = {}
    # Generation model (factory maps overrides and env vars)
    # We'll create a lightweight model descriptor from the factory so we can
    # consistently obtain the model id and mapped params below.

    # Generation limits from prompt parameters (loaded later, but defaults needed here if we want to use them before prompt loader?)
    # Actually, we should load prompt parameters earlier or move this logic down.
    # Let's move the prompt loading up.
    
    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("backlog_generation_agent")
    # Merge prompt parameters with any params provided by the factory (factory wins)
    prompt_params = prompt_loader.get_parameters("backlog_generation_agent") or {}

    def _as_int(v, d):
        try:
            i = int(v)
            return i if i > 0 else d
        except Exception:
            return d

    MAX_ADO = _as_int(prompt_params.get("max_ado_in_prompt", 6), 6)
    MAX_ARCH = _as_int(prompt_params.get("max_arch_in_prompt", 6), 6)
    ADO_DESC_LEN = _as_int(prompt_params.get("ado_desc_len", 400), 400)
    ARCH_TEXT_LEN = _as_int(prompt_params.get("arch_text_len", 600), 600)
    
    # Determine effective max tokens
    # Priority: Agent Config > App Config > Model Default (None)
    
    # 1. Agent Config
    agent_max_tokens = prompt_params.get("max_completion_tokens") or prompt_params.get("max_tokens")
    
    # 2. App Config
    # We need to check if there is an app level max token setting. 
    # The user mentioned "application level maximum" but we removed 'generation' section.
    # Assuming there might be a global setting in 'openai' section or similar, but looking at config.poc.yaml, there isn't one explicitly for max tokens in 'openai' section.
    # However, let's check if _cfg has anything relevant or if we should just rely on agent config.
    # The user said "if max token is not defined in both configuration, will use the model default".
    # So we check _cfg for a fallback if agent_max_tokens is None.
    # Since we removed 'generation' section, maybe it's in 'openai' section?
    # The current config.poc.yaml 'openai' section doesn't have it.
    # So effectively, if not in agent config, it's None (Model Default).
    
    app_max_tokens = _cfg.get("openai", {}).get("max_tokens") # Hypothetical app level config
    
    if agent_max_tokens is not None:
        eff_max_tokens = int(agent_max_tokens)
    elif app_max_tokens is not None:
        eff_max_tokens = int(app_max_tokens)
    else:
        eff_max_tokens = None # Use model default
        
    # Prepare model params: whitelist only valid OpenAI model parameters
    allowed_model_param_keys = {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        "n",
    }
    model_params: Dict[str, Any] = {}
    if prompt_params:
        for k, v in prompt_params.items():
            if k in allowed_model_param_keys:
                model_params[k] = v
    if eff_max_tokens is not None:
        model_params["max_completion_tokens"] = eff_max_tokens

    # Create the Strands OpenAIModel via ModelFactory
    try:
        model = ModelFactory.create_openai_model(config_path=config_path, model_params=model_params)
        model_name = getattr(model, "model_id", None) or ModelFactory.get_default_model_id(config_path)
        logger.debug("Initialized Strands OpenAIModel: %s", model_name)
    except Exception as e:
        logger.exception("Failed to create Strands OpenAIModel: %s", e)
        model = None
        model_name = ModelFactory.get_default_model_id(config_path)

    # Initialize Strands Agent with system prompt and model
    agent = None
    if model is not None:
        try:
            agent = Agent(model=model, system_prompt=system_prompt)
        except Exception as e:
            logger.exception("Failed to initialize Strands Agent: %s", e)
            agent = None
    
    # Removed _safe_json_extract: we now rely on strict JSON parsing and Pydantic validation

    @tool
    def generate_backlog(
        segment_data: Union[str, Dict[str, Any]] = None,
        segment_id: int = None,
        segment_text: str = None,
        intent_labels: List[str] = None,
        dominant_intent: str = None,
        retrieved_context: Dict[str, Any] = None,
    ) -> str:
        """
        Generate backlog items (epics, features, stories) from a segment with retrieved context.
        
        Args:
            segment_data: JSON string containing:
                - segment_id: The segment identifier
                - segment_text: The original segment text
                - intent_labels: List of intent labels
                - dominant_intent: The dominant intent
                - retrieved_context: Retrieved ADO items and architecture constraints
            
        Returns:
            JSON string containing generated backlog items (epics, features, stories)
        """

        try:
            # Parse input (support structured tool calls and legacy JSON string)
            if segment_data is not None and (segment_id is None and segment_text is None):
                # Accept dict (already structured) or JSON string strictly
                if isinstance(segment_data, (dict, list)):
                    data = segment_data
                else:
                    data = json.loads(segment_data)
                segment_id = data.get("segment_id", 0)
                segment_text = data.get("segment_text", "")
                intent_labels = data.get("intent_labels", [])
                dominant_intent = data.get("dominant_intent", "")
                retrieved_context = data.get("retrieved_context", {})
            else:
                # Structured args path
                segment_id = segment_id or 0
                segment_text = segment_text or ""
                intent_labels = intent_labels or []
                dominant_intent = dominant_intent or ""
                retrieved_context = retrieved_context or {}
            
            logger.info("Backlog Generation Agent: Processing segment %s (run_id: %s)", segment_id, run_id)
            
            # Build generation prompt from template
            ado_items = retrieved_context.get("ado_items", []) or []
            arch_constraints = retrieved_context.get("architecture_constraints", []) or []

            def _safe_score(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0

            # Sort by score desc when available
            if ado_items:
                ado_items = sorted(ado_items, key=lambda i: _safe_score(i.get("score")), reverse=True)[:MAX_ADO]
            if arch_constraints:
                arch_constraints = sorted(arch_constraints, key=lambda i: _safe_score(i.get("score")), reverse=True)[:MAX_ARCH]
            
            # Format ADO items
            ado_formatted = "No relevant existing ADO items found.\n" if not ado_items else ""
            for item in ado_items:
                try:
                    score_val = _safe_score(item.get('score', 0))
                    ado_formatted += f"\n## {item.get('type', 'Item')} (ID: {item.get('work_item_id', 'N/A')}, Similarity: {score_val:.2f})\n"
                    ado_formatted += f"**Title:** {item.get('title', 'Untitled')}\n"
                    desc = item.get('description', '') or ''
                    if len(desc) > ADO_DESC_LEN:
                        desc = desc[:ADO_DESC_LEN] + "…"
                    ado_formatted += f"**Description:** {desc or 'No description'}\n"
                except Exception:
                    logger.debug("Skipping malformed ADO item during formatting: %s", item)
                    # Skip any malformed item
                    continue
            
            # Format architecture constraints
            arch_formatted = "No relevant architecture constraints found.\n" if not arch_constraints else ""
            for constraint in arch_constraints:
                try:
                    score_val = _safe_score(constraint.get('score', 0))
                    arch_formatted += f"\n## From {constraint.get('source', 'Unknown')} - {constraint.get('section', '')} (Similarity: {score_val:.2f})\n"
                    textv = constraint.get('text', '') or ''
                    if len(textv) > ARCH_TEXT_LEN:
                        textv = textv[:ARCH_TEXT_LEN] + "…"
                    arch_formatted += f"{textv or 'No text'}\n"
                except Exception:
                    logger.debug("Skipping malformed architecture constraint: %s", constraint)
                    continue
            
            prompt = prompt_loader.format_user_prompt(
                "backlog_generation_agent",
                segment_text=segment_text,
                intent_labels=", ".join(intent_labels),
                dominant_intent=dominant_intent,
                ado_items_formatted=ado_formatted,
                architecture_constraints_formatted=arch_formatted
            )

            # Approx token counts for debugging
            sys_tok = estimate_tokens(system_prompt)
            usr_tok = estimate_tokens(prompt)
            approx_total = sys_tok + usr_tok
            logger.debug("Backlog Generation Agent: tokens approx — system=%s, user=%s, total≈%s", sys_tok, usr_tok, approx_total)
            
            # Ensure we can call the agent
            if not api_key or agent is None:
                logger.warning("Backlog Generation Agent: Using MOCK mode (missing OPENAI_API_KEY or agent init failed)")
                return _mock_generation(segment_id, segment_text, intent_labels, run_id)

            logger.info("Backlog Generation Agent: Calling Strands Agent (%s) with structured output...", model_name)
            try:
                result = agent(
                    prompt,
                    structured_output_model=BacklogResponseIn,
                )
                # Strands returns validated structured output; ensure correct type
                validated: BacklogResponseIn = result.structured_output  # type: ignore[assignment]
            except (StructuredOutputException, ValidationError) as e:
                logger.warning("Backlog Generation Agent: Structured output failed, using fallback. Reason: %s", e)
                return _mock_generation(segment_id, segment_text, intent_labels, run_id)
            except Exception as e:
                logger.exception("Backlog Generation Agent: Agent invocation failed: %s", e)
                return _mock_generation(segment_id, segment_text, intent_labels, run_id)

            backlog_items_models = validated.backlog_items
            # Convert to plain dicts for normalization / writing
            backlog_items = []
            if hasattr(backlog_items_models, "__iter__"):
                for m in backlog_items_models:
                    try:
                        item_dict = m.model_dump() if hasattr(m, "model_dump") else m.dict()
                    except Exception:
                        item_dict = dict(m)
                    backlog_items.append(item_dict)
            
            # Assign internal IDs
            item_counter = {"epic": 1, "feature": 1, "story": 1}
            for item in backlog_items:
                # Normalize type and keep counters consistent
                orig_type = str(item.get("type", "story")).strip().lower()
                if orig_type in ("story", "user story", "user_story", "user-story"):
                    norm_key = "story"
                elif orig_type in ("feature", "features"):
                    norm_key = "feature"
                elif orig_type in ("epic", "epics"):
                    norm_key = "epic"
                else:
                    norm_key = orig_type

                if norm_key == "story":
                    display_type = "User Story"
                elif norm_key == "feature":
                    display_type = "Feature"
                elif norm_key == "epic":
                    display_type = "Epic"
                else:
                    display_type = item.get("type", "Story")
                item["type"] = display_type
                if norm_key in item_counter:
                    item["internal_id"] = f"{norm_key}_{segment_id}_{item_counter[norm_key]}"
                    item_counter[norm_key] += 1
                item["segment_id"] = segment_id
                item["run_id"] = run_id
            
            # Ensure output directory exists
            output_dir = Path(f"runs/{run_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Append to generated_backlog.jsonl
            backlog_file = output_dir / "generated_backlog.jsonl"
            with open(backlog_file, "a") as f:
                for item in backlog_items:
                    f.write(json.dumps(item) + "\n")
            
            logger.info("Backlog Generation Agent: Generated %s backlog items", len(backlog_items))
            
            # Prepare summary
            summary = {
                "status": "success",
                "run_id": run_id,
                "segment_id": segment_id,
                "items_generated": len(backlog_items),
                "backlog_file": str(backlog_file),
                "item_counts": {
                    "epics": sum(1 for item in backlog_items if str(item.get("type", "")).lower() in ("epic", "epics")),
                    "features": sum(1 for item in backlog_items if str(item.get("type", "")).lower() in ("feature", "features")),
                    "stories": sum(1 for item in backlog_items if str(item.get("type", "")).lower() in ("story", "user story"))
                },
                "backlog_items": backlog_items
            }
            
            return json.dumps(summary, indent=2)
            
        except json.JSONDecodeError as e:
            logger.warning("Backlog Generation Agent: JSON parse failed (legacy path), using fallback. Reason: %s", str(e))
            return _mock_generation(segment_id, segment_text, intent_labels, run_id)
        
        except Exception as e:
            logger.exception("Backlog generation failed for segment %s: %s", segment_id, e)
            error_msg = {
                "status": "error",
                "error": f"Backlog generation failed: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
    
    return generate_backlog


# Note: Prompt building now handled by prompt_loader from prompts/backlog_generation_agent.yaml


def _mock_generation(segment_id: int, segment_text: str, intent_labels: list, run_id: str) -> str:
    """Generate simple mock backlog: 1 epic, 1 feature, 1 user story."""

    epic_title = "Sample Epic: Improve User Onboarding"
    feature_title = "Sample Feature: Guided Onboarding Flow"
    story_title = "User Story: As a new user, I want a guided setup"

    mock_items = [
        {
            "type": "Epic",
            "title": epic_title,
            "description": f"High-level initiative derived from segment {segment_id}. {segment_text[:200]}...",
            "acceptance_criteria": [
                "Epic goals are defined",
                "Stakeholders aligned on scope"
            ],
            "parent_reference": "",
            "rationale": "Mock epic for local testing",
            "internal_id": f"epic_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id,
        },
        {
            "type": "Feature",
            "title": feature_title,
            "description": "Provide a step-by-step onboarding experience with helpful tips.",
            "acceptance_criteria": [
                "Onboarding covers account setup and first action",
                "Completion rate tracked via analytics"
            ],
            "parent_reference": epic_title,
            "rationale": "Supports the onboarding epic",
            "internal_id": f"feature_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id,
        },
        {
            "type": "User Story",
            "title": story_title,
            "description": "Guide users through profile creation and initial configuration.",
            "acceptance_criteria": [
                "User completes profile setup in under 3 minutes",
                "Progress is saved between steps"
            ],
            "parent_reference": feature_title,
            "rationale": "Delivers immediate value in onboarding",
            "internal_id": f"story_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id,
        },
    ]

    # Save to file
    output_dir = Path(f"runs/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    backlog_file = output_dir / "generated_backlog.jsonl"
    
    with open(backlog_file, "a") as f:
        for item in mock_items:
            f.write(json.dumps(item) + "\n")
    
    summary = {
        "status": "success_mock",
        "run_id": run_id,
        "segment_id": segment_id,
        "items_generated": len(mock_items),
        "backlog_file": str(backlog_file),
        "note": "Mock data - set OPENAI_API_KEY for real generation",
        "item_counts": {
            "epics": sum(1 for item in mock_items if item.get("type", "").lower() == "epic"),
            "features": sum(1 for item in mock_items if item.get("type", "").lower() == "feature"),
            "stories": sum(1 for item in mock_items if item.get("type", "").lower() in ("story", "user story")),
        },
        "backlog_items": mock_items
    }
    
    return json.dumps(summary, indent=2)

# Note: System prompt and user prompt template now loaded from prompts/backlog_generation_agent.yaml
