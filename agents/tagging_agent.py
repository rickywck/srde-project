"""Tagging Agent (reference): classifies a generated user story relative to a backlog.
Tool/system prompt and usage live in `prompts/tagging_agent.yaml`. This module
provides the `tag_story` tool and helper logic; the YAML defines the LLM prompt.
"""

import json
import os
import re
from typing import List, Dict, Any, Union
from pathlib import Path
from strands import Agent, tool
from .prompt_loader import get_prompt_loader
from .utils.similar_story_retriever import SimilarStoryRetriever
from .model_factory import ModelFactory
from .tagging_helper import TaggingInputResolver, finalize_tagging_result
import logging
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from strands.types.exceptions import StructuredOutputException

# Module logger
logger = logging.getLogger(__name__)

# Constants
GENERATED_BACKLOG_FILENAME = "generated_backlog.jsonl"
USER_STORY_TYPES = {"user story", "story", "user_story"}
DEFAULT_MIN_SIMILARITY_THRESHOLD = 0.5

# Simple JSON extractor to tolerate minor formatting issues from LLM outputs
def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first top-level JSON object from text.

    Falls back to empty dict if not found or parsing fails.
    """
    try:
        # Fast path: direct parse
        return json.loads(text)
    except Exception:
        pass
    try:
        # Regex to find the first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return {}

# Reference Pydantic model for the agent's structured output (informational only)
class TaggingDecisionOut(BaseModel):
    decision_tag: str = Field(description="new | gap | duplicate | conflict")
    related_ids: List[Union[int, str]] = Field(default_factory=list)
    reason: str = ""
    model_config = ConfigDict(extra="allow")


class TaggingStoryIn(BaseModel):
    title: str = Field(default="")
    description: str = Field(default="")
    acceptance_criteria: List[str] = Field(default_factory=list)
    internal_id: Union[str, int, None] = None
    model_config = ConfigDict(extra="ignore")


# Note: Prompt building now handled by prompt_loader from prompts/tagging_agent.yaml


def create_tagging_agent(run_id: str, default_similarity_threshold: float = None):
    """Create a tagging agent tool for a specific run."""

    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("tagging_agent")
    params = prompt_loader.get_parameters("tagging_agent") or {}
    # Ensure tagging agent avoids function/tool-calling to prevent 400 errors in chat mode
    # Do not set function_call when no functions are defined; avoid invalid API params
    # Prefer deterministic responses; avoid unsupported params
    try:
        params.setdefault("temperature", 0)
        if "function_call" in params:
            params.pop("function_call", None)
    except Exception:
        pass

    # Similarity threshold from params
    similarity_threshold = float(params.get("min_similarity_threshold", DEFAULT_MIN_SIMILARITY_THRESHOLD))
    if default_similarity_threshold is not None:
        try:
            similarity_threshold = float(default_similarity_threshold)
        except Exception:
            logger.debug("Invalid default_similarity_threshold provided; falling back to configured threshold")

    # Build model via ModelFactory helper; no direct config or API key access here
    try:
        model = ModelFactory.create_openai_model_for_agent(agent_params=params)
        model_id = getattr(model, "model_id", None) or ModelFactory.get_default_model_id()
        logger.debug("Tagging agent model initialized: %s", model_id)
    except Exception as e:
        logger.exception("Failed to create model for tagging agent: %s", e)
        model = None
        model_id = ModelFactory.get_default_model_id()

    # Instantiate a Strands Agent at factory scope (long-lived within supervisor/runtime),
    # so that it can be introspected similarly to the segmentation agent.
    agent = None
    if model is not None:
        try:
            agent = Agent(model=model, system_prompt=system_prompt)
            logger.debug("Tagging Agent instance created at factory scope")
        except Exception as e:
            logger.exception("Failed to instantiate Tagging Agent at factory scope: %s", e)
            agent = None

    @tool
    def tag_story(
        story: Any
    ) -> str:
        """Classify ONE generated user story relative to the current run's backlog.

        Required input:
        - `story`: JSON object with `title` (str), `description` (str), and `acceptance_criteria` (list)

        Behaviour:
        - Accepts ONLY a single `story` argument. No paths, lists, or overrides.
        - Uses the agent's current `run_id` to retrieve existing backlog for similarity.

        Output (STRICT JSON ONLY):
        - `decision_tag`: "new"|"gap"|"duplicate"|"conflict"
        - `related_ids`: list of relevant existing `work_item_id` values
        - `reason`: single short sentence (<= 20 words)
        """
        # Validate and normalize input strictly via Pydantic
        normalized_story: Dict[str, Any] = {}
        try:
            if isinstance(story, str):
                story_obj = json.loads(story)
            else:
                story_obj = story
            ts = TaggingStoryIn(**(story_obj or {}))
            normalized_story = {
                "title": ts.title or "",
                "description": ts.description or "",
                "acceptance_criteria": ts.acceptance_criteria or [],
                "internal_id": ts.internal_id,
            }
        except Exception as e:
            return json.dumps({
                "status": "error",
                "run_id": run_id,
                "decision_tag": "new",
                "related_ids": [],
                "reason": f"Invalid story payload: {e}",
                "early_exit": True,
                "similarity_threshold": similarity_threshold,
                "similar_count": 0,
                "model_used": model_id
            })

        story_title = normalized_story.get('title', 'N/A') if isinstance(normalized_story, dict) else 'N/A'
        desc = normalized_story.get('description', '') if isinstance(normalized_story, dict) else ''
        story_keys = list(normalized_story.keys()) if normalized_story else []
        logger.debug(
            "tag_story called with: run_id=%r, story_title=%r, story_description=%s..., story_keys=%r",
            run_id, story_title, desc[:100] if desc else None, story_keys,
        )

        if not isinstance(normalized_story, dict) or not normalized_story:
            return json.dumps({
                "status": "error",
                "run_id": run_id,
                "decision_tag": "new",
                "related_ids": [],
                "reason": "Invalid input: could not resolve story",
                "early_exit": True,
                "similarity_threshold": similarity_threshold,
                "similar_count": 0,
                "model_used": model_id
            })

        current_out_dir: Path = Path(f"runs/{run_id}")
        effective_run_id: str = run_id
        global_similar: List[Dict[str, Any]] = []

        def _tag_one(story_obj: Dict[str, Any]) -> Dict[str, Any]:
            internal_id = story_obj.get("internal_id")
            title = story_obj.get("title")
            #threshold = default_threshold_local
            # Debug: show the similarity threshold used for tagging this story
            try:
                logger.debug(
                    "Tagging Agent: Using similarity threshold=%.4f for story internal_id=%s title=%s",
                    float(similarity_threshold),
                    internal_id,
                    title,
                )
            except Exception:
                logger.debug("Tagging Agent: Using similarity threshold (could not format values): %s", str(similarity_threshold))
            similar_local = list(global_similar) if isinstance(global_similar, list) else []

            # Log payload
            try:
                raw_story_json = json.dumps(story_obj, ensure_ascii=False)
            except Exception:
                raw_story_json = str(story_obj)
            logger.info("Tagging Agent: Story payload: %s", raw_story_json[:1000])
            logger.info("Tagging Agent: Processing story title='%s' | description='%s…'", title, (story_obj.get('description', '') or '')[:120])

            # Retrieve similar stories for current run
            if not similar_local:
                try:
                    logger.info("Tagging Agent: Retrieving similar stories from configured index…")
                    retriever = SimilarStoryRetriever(config=None, min_similarity=similarity_threshold)
                    similar_local = retriever.find_similar_stories({
                        "title": story_obj.get("title", ""),
                        "description": story_obj.get("description", ""),
                        "acceptance_criteria": story_obj.get("acceptance_criteria", []),
                    })
                    logger.info("Tagging Agent: Retrieval found %s similar stories", len(similar_local or []))
                except Exception as e:
                    logger.warning("Tagging Agent: Similarity retrieval failed: %s", e)
            else:
                # Debug: similar stories were supplied in input, list them
                try:
                    logger.debug("Tagging Agent: Using provided similar stories (count=%d):", len(similar_local))
                    for s in similar_local:
                        sid = s.get("work_item_id") if isinstance(s, dict) else None
                        sim = float(s.get("similarity", 0.0)) if isinstance(s, dict) else 0.0
                        stitle = (s.get("title") or "")[:200] if isinstance(s, dict) else str(s)
                        logger.debug(" - %s | %.4f | %s", sid, sim, stitle)
                except Exception:
                    logger.debug("Tagging Agent: Could not enumerate provided similar stories for debug output")

            above_threshold = [s for s in (similar_local or []) if s.get("similarity", 0.0) >= similarity_threshold]
            if not above_threshold:
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": "new",
                    "related_ids": [],
                    "reason": "No similar existing stories found (all below threshold)",
                    "early_exit": True,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": 0,
                    "model_used": model_id
                }
                finalize_tagging_result(result, current_out_dir, internal_id, title)
                return result

            # Build prompt for LLM using template
            ac_list = story_obj.get("acceptance_criteria", []) or []
            ac_text = "\n- " + "\n- ".join(ac_list) if ac_list else " (none)"
            similar_lines: List[str] = []
            for s in above_threshold:
                similar_lines.append(
                    f"ID: {s.get('work_item_id')} | similarity: {round(s.get('similarity', 0.0), 4)}\nTitle: {s.get('title')}\nDesc: {s.get('description','')[:300]}"
                )
            similar_formatted = "\n\n".join(similar_lines)

            user_prompt = prompt_loader.format_user_prompt(
                "tagging_agent",
                story_title=story_obj.get("title"),
                story_description=story_obj.get("description"),
                story_acceptance_criteria=ac_text,
                similarity_threshold=similarity_threshold,
                similar_stories_formatted=similar_formatted
            )

            if model is None:
                logger.error("Tagging Agent: No model available; skipping LLM evaluation for tagging")
                return {
                    "status": "error",
                    "run_id": effective_run_id,
                    "decision_tag": "new",
                    "related_ids": [],
                    "reason": "Model unavailable for tagging",
                    "early_exit": True,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                }

            try:
                # Call long-lived agent instance
                resp_raw = agent(user_prompt)

                # Normalize response to plain text for logging/parsing
                if isinstance(resp_raw, (str, bytes)):
                    resp_text = resp_raw
                else:
                    resp_text = getattr(resp_raw, "text", None) or str(resp_raw)

                logger.info("Tagging Agent: Received response from LLM: %s", resp_text)

                # Extract JSON from text
                parsed_json = _extract_json(resp_text)
                if not isinstance(parsed_json, dict) or not parsed_json:
                    raise ValueError("Failed to extract JSON from agent response")
                parsed = TaggingDecisionOut(**parsed_json)
                decision = (parsed.decision_tag or "new").lower()
                if decision not in {"new", "gap", "duplicate", "conflict"}:
                    decision = "new"
                related_ids = parsed.related_ids or []
                reason = (parsed.reason or "")[:200]
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": decision,
                    "related_ids": related_ids,
                    "reason": reason,
                    "early_exit": False,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": False
                }
                finalize_tagging_result(result, current_out_dir, internal_id, title)
                return result
            except (StructuredOutputException, ValidationError) as e:
                logger.error("Tagging Agent: Structured output failed. Reason: %s", e)
                return {
                    "status": "error",
                    "run_id": effective_run_id,
                    "decision_tag": "new",
                    "related_ids": [],
                    "reason": "LLM structured output failed",
                    "early_exit": True,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                }
            except Exception as e:
                logger.exception("Tagging Agent: Agent invocation failed: %s", e)
                return {
                    "status": "error",
                    "run_id": effective_run_id,
                    "decision_tag": "new",
                    "related_ids": [],
                    "reason": "LLM invocation failed",
                    "early_exit": True,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                }

        result_obj = _tag_one(normalized_story)
        return json.dumps(result_obj)

    return tag_story

