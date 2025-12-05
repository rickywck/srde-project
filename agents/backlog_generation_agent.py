"""
Backlog Generation Agent - Specialized agent for generating backlog items from segments
"""

import json
import logging
from typing import Dict, Any, List, Union
from pathlib import Path
from pydantic import BaseModel, ValidationError
from strands import tool, Agent
from strands.types.exceptions import StructuredOutputException
from .prompt_loader import get_prompt_loader
from tools.utils.token_utils import estimate_tokens
from .model_factory import ModelFactory
from .utils.backlog_helper import BacklogHelper

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

    # No direct config or api key handling in the agent. Use ModelFactory.

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
    
    # Create the Strands OpenAIModel via ModelFactory helper
    try:
        model = ModelFactory.create_openai_model_for_agent(agent_params=prompt_params)
        model_name = getattr(model, "model_id", None) or ModelFactory.get_default_model_id()
        logger.debug("Initialized Strands OpenAIModel: %s", model_name)
    except Exception as e:
        logger.exception("Failed to create Strands OpenAIModel: %s", e)
        model = None
        model_name = ModelFactory.get_default_model_id()

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
        segment_id: int = None,
        segment_text: str = None,
        intent_labels: List[str] = None,
        dominant_intent: str = None,
        retrieved_context: Dict[str, Any] = None,
    ) -> str:
        """
        Generate backlog items (epics, features, stories) from a segment with retrieved context.
        
        Args:
            segment_id: The segment identifier
            segment_text: The original segment text
            intent_labels: List of intent labels
            dominant_intent: The dominant intent
            retrieved_context: Retrieved ADO items and architecture constraints
            
        Returns:
            JSON string containing generated backlog items (epics, features, stories)
        """
        ado_count = 0
        arch_count = 0
        if retrieved_context:
            ado_count = len(retrieved_context.get("ado_items", []) or [])
            arch_count = len(retrieved_context.get("architecture_constraints", []) or [])
        logger.debug("generate_backlog called with: run_id=%r, segment_id=%r, segment_text=%s..., intent_labels=%r, dominant_intent=%r, ado_items=%d, arch_constraints=%d",
                     run_id, segment_id, segment_text[:100] if segment_text else None, intent_labels, dominant_intent, ado_count, arch_count)

        try:
            # Normalize and validate inputs
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
            if agent is None:
                error_msg = {
                    "status": "error",
                    "error": "Backlog Generation Agent not initialized. No model available.",
                    "run_id": run_id
                }
                return json.dumps(error_msg, indent=2)

            logger.info("Backlog Generation Agent: Calling Strands Agent (%s) with structured output...", model_name)
            try:
                result = agent(
                    prompt,
                    structured_output_model=BacklogResponseIn,
                )
                # Strands returns validated structured output; ensure correct type
                validated: BacklogResponseIn = result.structured_output  # type: ignore[assignment]
            except (StructuredOutputException, ValidationError) as e:
                logger.error("Backlog Generation Agent: Structured output failed: %s", e)
                error_msg = {
                    "status": "error",
                    "error": f"Backlog generation structured output failed: {str(e)}",
                    "run_id": run_id
                }
                return json.dumps(error_msg, indent=2)
            except Exception as e:
                logger.exception("Backlog Generation Agent: Agent invocation failed: %s", e)
                error_msg = {
                    "status": "error",
                    "error": f"Backlog generation agent invocation failed: {str(e)}",
                    "run_id": run_id
                }
                return json.dumps(error_msg, indent=2)

            backlog_items_models = validated.backlog_items
            # Convert to plain dicts
            raw_items = []
            if hasattr(backlog_items_models, "__iter__"):
                for m in backlog_items_models:
                    try:
                        raw_items.append(m.model_dump() if hasattr(m, "model_dump") else m.dict())
                    except Exception:
                        raw_items.append(dict(m))

            # Normalize and annotate
            processed = BacklogHelper.normalize_items(
                raw_items, run_id=run_id, segment_id=segment_id, id_mode="segment"
            )

            # Persist
            output_dir = Path(f"runs/{run_id}")
            backlog_file = output_dir / "generated_backlog.jsonl"
            BacklogHelper.write_jsonl(processed, backlog_file, mode="a")

            logger.info("Backlog Generation Agent: Generated %s backlog items", len(processed))

            # Summary
            summary = BacklogHelper.summarize(
                run_id=run_id, backlog_file=backlog_file, items=processed, segment_id=segment_id
            )
            return json.dumps(summary, indent=2)
            
        except json.JSONDecodeError as e:
            logger.warning("Backlog Generation Agent: JSON parse failed (legacy path). Reason: %s", str(e))
            error_msg = {
                "status": "error",
                "error": f"Invalid JSON payload: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
        
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
# Note: System prompt and user prompt template now loaded from prompts/backlog_generation_agent.yaml
