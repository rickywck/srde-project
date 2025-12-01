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
from services.similar_story_retriever import SimilarStoryRetriever
from .model_factory import ModelFactory
from .tagging_helper import TaggingInputResolver
import logging
from pydantic import BaseModel, Field, ValidationError
from strands.types.exceptions import StructuredOutputException

# Module logger
logger = logging.getLogger(__name__)

# Constants
GENERATED_BACKLOG_FILENAME = "generated_backlog.jsonl"
USER_STORY_TYPES = {"user story", "story", "user_story"}

# Note: System prompt now loaded from prompts/tagging_agent.yaml


class TaggingDecisionOut(BaseModel):
    decision_tag: str = Field(description="new | gap | conflict")
    related_ids: List[Union[int, str]] = Field(default_factory=list)
    reason: str = ""

    class Config:
        extra = "allow"


def _rule_based_fallback(story: Dict[str, Any], similar: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Apply simple deterministic rules if LLM JSON invalid."""
    if not similar:
        return {"decision_tag": "new", "related_ids": [], "reason": "No similar items"}
    # Consider only items above threshold
    considered = [s for s in similar if s.get("similarity", 0.0) >= threshold]
    if not considered:
        return {"decision_tag": "new", "related_ids": [], "reason": "None above similarity threshold"}
    # Duplication: highest similarity > 0.85 and title overlap > 70%
    top = max(considered, key=lambda s: s.get("similarity", 0.0))
    def _norm(t: str) -> List[str]:
        return [w for w in re.split(r"\W+", t.lower()) if w]
    story_title_tokens = set(_norm(story.get("title", "")))
    top_title_tokens = set(_norm(top.get("title", "")))
    overlap_ratio = len(story_title_tokens & top_title_tokens) / (len(story_title_tokens) or 1)
    if top.get("similarity", 0.0) > 0.85 and overlap_ratio >= 0.7:
        return {"decision_tag": "conflict", "related_ids": [top.get("work_item_id")], "reason": "High duplication signal"}
    # Otherwise treat as gap if at least one considered
    return {"decision_tag": "gap", "related_ids": [c.get("work_item_id") for c in considered[:3]], "reason": "Partial overlap suggests extension"}


# Note: Prompt building now handled by prompt_loader from prompts/tagging_agent.yaml


def create_tagging_agent(run_id: str, default_similarity_threshold: float = None):
    """Create a tagging agent tool for a specific run."""

    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("tagging_agent")
    params = prompt_loader.get_parameters("tagging_agent") or {}

    # Similarity threshold from params unless caller overrides
    if default_similarity_threshold is None:
        default_similarity_threshold = float(params.get("min_similarity_threshold", 0.5))

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
        similarity_threshold: float = None,
        run_id_override: str = None,
        backlog_path: str = None,
    ) -> str:
        """Tag a user story relative to existing backlog (new/gap/conflict).

        Accepts a single story. If a filesystem path (run directory or backlog file)
        is provided, the helper resolves it and returns one or more stories to tag.
        """

        # Resolve input into normalized payload (single story or list if path)
        resolver = TaggingInputResolver(default_run_id=run_id, default_threshold=default_similarity_threshold)
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
                "similarity_threshold": default_similarity_threshold,
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

        def _finalize(result: Dict[str, Any], internal_id: Any = None, title: str = None) -> None:
            if internal_id:
                result["story_internal_id"] = internal_id
            if title:
                result["story_title"] = title
            try:
                current_out_dir.mkdir(parents=True, exist_ok=True)
                tag_file = current_out_dir / "tagging.jsonl"
                key = None
                try:
                    key = str(internal_id) if internal_id is not None else (str(title) if title is not None else None)
                except Exception:
                    key = None
                if key is not None and key in processed_story_keys:
                    return
                with open(tag_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass
                if key is not None:
                    processed_story_keys.add(key)
            except Exception:
                pass

        stories: List[Dict[str, Any]] = resolved.get("stories") or []
        effective_run_id: str = resolved.get("run_id") or run_id
        default_threshold_local: float = float(resolved.get("threshold", default_similarity_threshold))
        global_similar: List[Dict[str, Any]] = resolved.get("similar_existing_stories") or []

        def _tag_one(story_obj: Dict[str, Any]) -> Dict[str, Any]:
            internal_id = story_obj.get("internal_id")
            title = story_obj.get("title")
            threshold = default_threshold_local
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
                    retriever = SimilarStoryRetriever(config=None, min_similarity=threshold)
                    similar_local = retriever.find_similar_stories(
                        {
                            "title": story_obj.get("title", ""),
                            "description": story_obj.get("description", ""),
                            "acceptance_criteria": story_obj.get("acceptance_criteria", []),
                        },
                        min_similarity=threshold,
                    )
                    logger.info("Tagging Agent: Internal retrieval found %s similar stories", len(similar_local or []))
                except Exception as e:
                    logger.exception("Tagging Agent: Internal retrieval failed: %s", e)

            above_threshold = [s for s in (similar_local or []) if s.get("similarity", 0.0) >= threshold]
            if not above_threshold:
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": "new",
                    "related_ids": [],
                    "reason": "No similar existing stories found (all below threshold)",
                    "early_exit": True,
                    "similarity_threshold": threshold,
                    "similar_count": 0,
                    "model_used": model_id
                }
                _finalize(result, internal_id, title)
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
                similarity_threshold=threshold,
                similar_stories_formatted=similar_formatted
            )

            if model is None:
                logger.warning("Tagging Agent: No model available; using rule-based fallback")
                fallback = _rule_based_fallback(story_obj, above_threshold, threshold)
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": fallback.get("decision_tag", "new"),
                    "related_ids": fallback.get("related_ids", []),
                    "reason": fallback.get("reason", "Fallback applied"),
                    "early_exit": False,
                    "similarity_threshold": threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": True
                }
                _finalize(result, internal_id, title)
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
                    "similarity_threshold": threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": False
                }
                _finalize(result, internal_id, title)
                return result
            except (StructuredOutputException, ValidationError) as e:
                logger.warning("Tagging Agent: Structured output failed, using rule-based fallback. Reason: %s", e)
                fallback = _rule_based_fallback(story_obj, above_threshold, threshold)
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": fallback.get("decision_tag", "new"),
                    "related_ids": fallback.get("related_ids", []),
                    "reason": fallback.get("reason", "Fallback applied"),
                    "early_exit": False,
                    "similarity_threshold": threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": True
                }
                _finalize(result, internal_id, title)
                return result
            except Exception as e:
                logger.exception("Tagging Agent: Agent invocation failed, using rule-based fallback: %s", e)
                fallback = _rule_based_fallback(story_obj, above_threshold, threshold)
                result = {
                    "status": "ok",
                    "run_id": effective_run_id,
                    "decision_tag": fallback.get("decision_tag", "new"),
                    "related_ids": fallback.get("related_ids", []),
                    "reason": fallback.get("reason", "Fallback applied"),
                    "early_exit": False,
                    "similarity_threshold": threshold,
                    "similar_count": len(above_threshold),
                    "model_used": model_id,
                    "fallback_used": True
                }
                _finalize(result, internal_id, title)
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

