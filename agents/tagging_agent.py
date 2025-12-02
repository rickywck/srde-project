"""
Tagging Agent - Generates classification (new | gap | conflict) for a generated
user story relative to existing backlog stories. Implements early-exit logic,
robust JSON parsing of LLM output, and lightweight rule-based fallbacks.

IMPORTANT: When invoking this tool, prefer to pass a story object (not a
filesystem path) in the `story` field. The agent accepts these shapes:

1) Single story (recommended):
     {"story": {"title": "...", "description": "...", "acceptance_criteria": ["..."]}}

2) Reference a run/backlog file (fallback):
     {"run_id": "<run-id>"}
     or
     {"backlog_path": "runs/<run-id>/generated_backlog.jsonl"}

Robustness: If the orchestrator (Strands) or the LLM accidentally supplies a
filesystem path (e.g. "runs/.../generated_backlog.jsonl") in the `story` field,
this agent will now treat that string as a `backlog_path` and load the backlog
file (fallback behavior) rather than erroring out. This keeps the tool tolerant
to varying call shapes while encouraging the explicit, correct input shapes.

Output contract (JSON string):
{
    "status": "ok"|"error",
    "run_id": str,
    "decision_tag": "new"|"gap"|"conflict",
    "related_ids": [str|int],
    "reason": str,
    "early_exit": bool,
    "similarity_threshold": float,
    "similar_count": int,
    "model_used": str,
    "story_internal_id": str,
    "story_title": str
}
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
from .tagging_helper import TaggingInputResolver, _rule_based_fallback, finalize_tagging_result
import logging
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from strands.types.exceptions import StructuredOutputException

# Module logger
logger = logging.getLogger(__name__)

# Constants
GENERATED_BACKLOG_FILENAME = "generated_backlog.jsonl"
USER_STORY_TYPES = {"user story", "story", "user_story"}
DEFAULT_MIN_SIMILARITY_THRESHOLD = 0.5

# Note: System prompt now loaded from prompts/tagging_agent.yaml


class TaggingDecisionOut(BaseModel):
    decision_tag: str = Field(description="new | gap | conflict")
    related_ids: List[Union[int, str]] = Field(default_factory=list)
    reason: str = ""
    model_config = ConfigDict(extra="allow")


# Note: Prompt building now handled by prompt_loader from prompts/tagging_agent.yaml


def create_tagging_agent(run_id: str, default_similarity_threshold: float = None):
    """Create a tagging agent tool for a specific run."""

    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("tagging_agent")
    params = prompt_loader.get_parameters("tagging_agent") or {}

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

    @tool
    def tag_story(
        story_data: Any = None,
        story: Dict[str, Any] = None,
        similar_existing_stories: List[Dict[str, Any]] = None,
        run_id_override: str = None,
        backlog_path: str = None,
    ) -> str:
        """Tag a user story relative to existing backlog (new/gap/conflict).

        Accepts a single story. If a filesystem path (run directory or backlog file)
        is provided, the helper resolves it and returns one or more stories to tag.
        """

        # Quick validation: if raw string provided, ensure it's JSON or a path
        if isinstance(story_data, str):
            s = story_data.strip()
            is_path_like = s.endswith(".json") or s.endswith(".jsonl") or ("/" in s) or ("\\" in s)
            if not is_path_like:
                try:
                    json.loads(s)
                except Exception:
                    return json.dumps({
                        "status": "error",
                        "run_id": run_id,
                        "decision_tag": "new",
                        "related_ids": [],
                        "reason": "Invalid input JSON: non-JSON string provided",
                        "early_exit": True,
                        "similarity_threshold": similarity_threshold,
                        "similar_count": 0,
                        "model_used": model_id
                    })

        # Resolve input into normalized payload (single story or list if path)
        resolver = TaggingInputResolver(default_run_id=run_id, default_threshold=similarity_threshold)
        try:
            resolved = resolver.resolve(
                story_data=story_data,
                story=story,
                similar_existing_stories=similar_existing_stories,
                similarity_threshold=similarity_threshold,
                run_id=run_id_override,
                backlog_path=backlog_path,
            )
        except Exception as e:
            return json.dumps({
                "status": "error",
                "run_id": run_id,
                "decision_tag": "new",
                "related_ids": [],
                "reason": f"Invalid input: {e}",
                "early_exit": True,
                "similarity_threshold": similarity_threshold,
                "similar_count": 0,
                "model_used": model_id
            })

        # Output dir and processed key cache
        current_out_dir: Path = resolved["out_dir"]
        processed_story_keys: set = set()
        try:
            current_out_dir.mkdir(parents=True, exist_ok=True)
            # seed de-dupe set from existing file
            tag_file = current_out_dir / "tagging.jsonl"
            if tag_file.exists():
                with open(tag_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            k = str(obj.get("story_internal_id") or obj.get("story_title") or "")
                            if k:
                                processed_story_keys.add(k)
                        except Exception:
                            continue
        except Exception:
            processed_story_keys = set()

        # Finalize/persist behavior moved to `finalize_tagging_result` in tagging_helper

        stories: List[Dict[str, Any]] = resolved.get("stories") or []
        effective_run_id: str = resolved.get("run_id") or run_id
        #default_threshold_local: float = float(resolved.get("threshold", default_similarity_threshold))
        global_similar: List[Dict[str, Any]] = resolved.get("similar_existing_stories") or []

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
            logger.debug("Tagging Agent: Story payload: %s", raw_story_json[:1000])
            logger.info("Tagging Agent: Processing story title='%s' | description='%s…'", title, (story_obj.get('description', '') or '')[:120])

            # Retrieve similar if none provided
            if not similar_local:
                try:
                    logger.info("Tagging Agent: No similar stories provided; performing internal retrieval…")
                    retriever = SimilarStoryRetriever(config=None, min_similarity=similarity_threshold)
                    similar_local = retriever.find_similar_stories(
                        {
                            "title": story_obj.get("title", ""),
                            "description": story_obj.get("description", ""),
                            "acceptance_criteria": story_obj.get("acceptance_criteria", []),
                        }
                    )
                    logger.info("Tagging Agent: Internal retrieval found %s similar stories", len(similar_local or []))
                    # Debug: list retrieved stories and their similarity scores
                    try:
                        logger.debug("Tagging Agent: Retrieved similar stories (id | similarity | title):")
                        for s in (similar_local or []):
                            sid = s.get("work_item_id") if isinstance(s, dict) else None
                            sim = float(s.get("similarity", 0.0)) if isinstance(s, dict) else 0.0
                            stitle = (s.get("title") or "")[:200] if isinstance(s, dict) else str(s)
                            logger.debug(" - %s | %.4f | %s", sid, sim, stitle)
                    except Exception:
                        logger.debug("Tagging Agent: Could not enumerate retrieved similar stories for debug output")
                except Exception as e:
                    logger.exception("Tagging Agent: Internal retrieval failed: %s", e)
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
                finalize_tagging_result(result, current_out_dir, processed_story_keys, internal_id, title)
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
                logger.warning("Tagging Agent: No model available; using rule-based fallback")
                fallback = _rule_based_fallback(story_obj, above_threshold, similarity_threshold)
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": fallback.get("decision_tag", "new"),
                    "related_ids": fallback.get("related_ids", []),
                    "reason": fallback.get("reason", "Fallback applied"),
                    "early_exit": False,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": True
                }
                finalize_tagging_result(result, current_out_dir, processed_story_keys, internal_id, title)
                return result

            try:
                agent = Agent(model=model, system_prompt=system_prompt)
                result_obj = agent(
                    user_prompt,
                    structured_output_model=TaggingDecisionOut,
                )
                parsed: TaggingDecisionOut = result_obj.structured_output  # type: ignore[assignment]
                decision = (parsed.decision_tag or "new").lower()
                if decision not in {"new", "gap", "conflict"}:
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
                finalize_tagging_result(result, current_out_dir, processed_story_keys, internal_id, title)
                return result
            except (StructuredOutputException, ValidationError) as e:
                logger.warning("Tagging Agent: Structured output failed, using rule-based fallback. Reason: %s", e)
                fallback = _rule_based_fallback(story_obj, above_threshold, similarity_threshold)
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": fallback.get("decision_tag", "new"),
                    "related_ids": fallback.get("related_ids", []),
                    "reason": fallback.get("reason", "Fallback applied"),
                    "early_exit": False,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": True
                }
                finalize_tagging_result(result, current_out_dir, processed_story_keys, internal_id, title)
                return result
            except Exception as e:
                logger.exception("Tagging Agent: Agent invocation failed, using rule-based fallback: %s", e)
                fallback = _rule_based_fallback(story_obj, above_threshold, similarity_threshold)
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": fallback.get("decision_tag", "new"),
                    "related_ids": fallback.get("related_ids", []),
                    "reason": fallback.get("reason", "Fallback applied"),
                    "early_exit": False,
                    "similarity_threshold": similarity_threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": True
                }
                finalize_tagging_result(result, current_out_dir, processed_story_keys, internal_id, title)
                return result

        # Process stories (single normally; multiple only when path provided)
        results: List[Dict[str, Any]] = []
        for s in stories:
            if not isinstance(s, dict):
                continue
            results.append(_tag_one(s))

        # Return single result directly, else summary
        if len(results) == 1:
            return json.dumps(results[0])
        return json.dumps({
            "status": "ok",
            "run_id": resolved.get("run_id") or run_id,
            "processed": len(results),
            "model_used": model_id,
        })

    return tag_story

