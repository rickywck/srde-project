"""
Tagging Agent - Generates classification (new | gap | conflict) for a generated
user story relative to existing backlog stories. Implements early-exit logic,
robust JSON parsing of LLM output, and lightweight rule-based fallbacks.

IMPORTANT: When invoking this tool, prefer to pass a story object (not a
filesystem path) in the `story` field. The agent accepts these shapes:

1) Single story (recommended):
     {"story": {"title": "...", "description": "...", "acceptance_criteria": ["..."]}}

2) Multiple stories (batch):
     {"stories": [ {"title":..., "description":...}, {...} ]}

3) Reference a run/backlog file (fallback):
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
    def tag_story(story_data: Any = None, story: Dict[str, Any] = None, similar_existing_stories: List[Dict[str, Any]] = None, similarity_threshold: float = None) -> str:
        """Tag a user story relative to existing backlog (new/gap/conflict)."""

        # Default output directory uses the agent's run_id, but may be overridden
        # per-call when caller supplies a backlog path or a run path string.
        current_out_dir = Path(f"runs/{run_id}")
        is_subcall = False
        # Track whether this invocation has initialized (overwritten) the tag file.
        out_file_initialized = False
        # Track which stories have been written in this agent invocation to avoid duplicates.
        processed_story_keys = set()

        def _load_existing_processed_keys(out_dir: Path) -> set:
            keys = set()
            try:
                tag_file = out_dir / "tagging.jsonl"
                if tag_file.exists():
                    with open(tag_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                k = str(obj.get("story_internal_id") or obj.get("story_title") or "")
                                if k:
                                    keys.add(k)
                            except Exception:
                                continue
            except Exception:
                pass
            return keys

        def _finalize(result: Dict[str, Any], internal_id: Any = None, title: str = None) -> str:
            nonlocal out_file_initialized
            nonlocal processed_story_keys
            if internal_id:
                result["story_internal_id"] = internal_id
            if title:
                result["story_title"] = title
            
            # Persist result to file
            try:
                out_dir = current_out_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                tag_file = out_dir / "tagging.jsonl"
                # Avoid writing duplicate entries for the same story within
                # a single agent invocation. Use internal_id if available,
                # otherwise use title as the key.
                key = None
                try:
                    key = str(internal_id) if internal_id is not None else (str(title) if title is not None else None)
                except Exception:
                    key = None
                if key is not None and key in processed_story_keys:
                    # Already written this story in this run; skip writing again.
                    return json.dumps(result)

                # If this is a top-level invocation and we haven't yet initialized
                # the tagging file for this run, open in write mode to overwrite
                # previous contents. Subsequent writes append.
                mode = "a"
                if not is_subcall and not out_file_initialized:
                    mode = "w"
                    out_file_initialized = True
                with open(tag_file, mode) as f:
                    f.write(json.dumps(result) + "\n")
                    try:
                        f.flush()
                        os.fsync(f.fileno())
                    except Exception:
                        pass

                if key is not None:
                    processed_story_keys.add(key)
            except Exception:
                pass # Silently fail on logging errors to preserve flow
                
            return json.dumps(result)

        # Accept both legacy JSON string and structured arguments
        payload: Dict[str, Any] = {}
        if story_data is not None and story is None and similar_existing_stories is None:
            if isinstance(story_data, (dict, list)):
                if isinstance(story_data, dict):
                    payload = story_data
                else:
                    # Allow passing a raw list of stories
                    payload = {"story": story_data}
            else:
                try:
                    loaded = json.loads(story_data)
                    if isinstance(loaded, list):
                        payload = {"story": loaded}
                    else:
                        payload = loaded
                except Exception as e:
                    return _finalize({
                        "status": "error",
                        "run_id": run_id,
                        "decision_tag": "new",
                        "related_ids": [],
                        "reason": f"Invalid input JSON: {e}",
                        "early_exit": True,
                        "similarity_threshold": default_similarity_threshold,
                        "similar_count": 0,
                        "model_used": model_id
                    })
        else:
            payload = {
                "story": story or {},
                "similar_existing_stories": similar_existing_stories or [],
                "similarity_threshold": similarity_threshold if similarity_threshold is not None else default_similarity_threshold,
            }

        # Small validator: detect when callers accidentally pass filesystem paths
        # inside the `story` field (common when orchestration passes run paths).
        def _looks_like_filesystem_path(v: Any) -> bool:
            try:
                if not v:
                    return False
                if isinstance(v, str):
                    s = v.strip()
                    # obvious file extensions or path separators
                    if s.endswith('.jsonl') or s.endswith('.json'):
                        return True
                    if '/' in s or '\\' in s:
                        return True
                    return False
                return False
            except Exception:
                return False

        story_candidate = payload.get("story")
        # If caller passed a filesystem path string in `story_data`, treat it as
        # an explicit backlog_path (robustness for Strands/LLM call shapes).
        if isinstance(story_candidate, str) and _looks_like_filesystem_path(story_candidate):
            payload = {"backlog_path": story_candidate, "run_id": payload.get("run_id") or run_id}
            story_candidate = payload.get("story")

        # If story is a dict but the title/description are path-like and description is missing,
        # assume caller mistakenly put a path into the title field and return a helpful error.
        # If a dict was passed but the title looks like a filesystem path and no
        # description was provided, treat the title as backlog_path (fallback).
        if isinstance(story_candidate, dict):
            title_val = story_candidate.get("title") or ""
            desc_val = story_candidate.get("description") or ""
            if _looks_like_filesystem_path(title_val) and not desc_val:
                payload = {"backlog_path": title_val, "run_id": payload.get("run_id") or run_id}
                story_candidate = payload.get("story")

        # Helper: Resolve backlog file path and output directory from payload arguments.
        def _resolve_io_context(_payload: Dict[str, Any]):
            # Accepts run_id that may be a plain id, a directory path, or a direct file path.
            # Also supports explicit 'backlog_path'.
            candidate = _payload.get("backlog_path") or _payload.get("run_id")
            target_run_id_local = _payload.get("run_id") or run_id
            # Default values
            out_dir_local = Path(f"runs/{target_run_id_local}")
            backlog_path_local = out_dir_local / GENERATED_BACKLOG_FILENAME
            try:
                if candidate:
                    cand_path = Path(str(candidate))
                    if str(candidate).endswith(".jsonl") or cand_path.suffix.lower() == ".jsonl":
                        backlog_path_local = cand_path
                        out_dir_local = cand_path.parent
                    elif cand_path.exists() and cand_path.is_dir():
                        out_dir_local = cand_path
                        backlog_path_local = cand_path / GENERATED_BACKLOG_FILENAME
                    elif "/" in str(candidate) or "\\" in str(candidate):
                        # Treat as a path even if it doesn't exist (yet)
                        if str(candidate).endswith(".jsonl"):
                            backlog_path_local = cand_path
                            out_dir_local = cand_path.parent
                        else:
                            out_dir_local = cand_path
                            backlog_path_local = cand_path / GENERATED_BACKLOG_FILENAME
                    else:
                        # Looks like a plain run id
                        out_dir_local = Path(f"runs/{candidate}")
                        backlog_path_local = out_dir_local / GENERATED_BACKLOG_FILENAME
            except Exception:
                pass
            return out_dir_local, backlog_path_local, target_run_id_local

        # Determine if this is a recursive subcall (skip cleanup if so)
        is_subcall = bool(payload.get("__internal_subcall"))

        # Resolve I/O context and possibly override current_out_dir
        try:
            resolved_out_dir, resolved_backlog_path, target_run_id_local = _resolve_io_context(payload)
            current_out_dir = resolved_out_dir
        except Exception:
            resolved_backlog_path = Path(f"runs/{run_id}") / GENERATED_BACKLOG_FILENAME
            current_out_dir = Path(f"runs/{run_id}")

        # Always seed from existing file to ensure idempotency across multiple
        # invocations (e.g., when the chatbot triggers tagging more than once).
        # Do NOT truncate the file here; instead, skip duplicates on write.
        try:
            current_out_dir.mkdir(parents=True, exist_ok=True)
            processed_story_keys = _load_existing_processed_keys(current_out_dir)
            # Ensure we append by default for top-level as well
            if not is_subcall:
                out_file_initialized = True
        except Exception:
            processed_story_keys = set()

        # No general batch mode: require single story per invocation
        if (
            story_data is None
            and story is None
            and similar_existing_stories is None
            and (not payload.get("story") and not payload.get("similar_existing_stories"))
        ):
            return json.dumps({
                "status": "error",
                "run_id": run_id,
                "message": "No story provided. Call the tool with a single 'story' payload.",
            })

        story = payload.get("story", {})
        similar = payload.get("similar_existing_stories", [])
        threshold = float(payload.get("similarity_threshold", default_similarity_threshold))

        # Normalize similar stories structure
        if isinstance(similar, dict):
            similar = [similar]
        elif not isinstance(similar, list):
            similar = []

        # Disallow multi-story inputs to simplify the agent (enforced by workflow)
        if isinstance(story, list) or isinstance(payload.get("stories"), list):
            return json.dumps({
                "status": "error",
                "run_id": run_id,
                "message": "Multiple stories not supported. Call the tool once per story.",
            })

        # Multi-story handling: Only when no explicit single story is provided.
        def _has_explicit_single_story(s: Dict[str, Any]) -> bool:
            return isinstance(s, dict) and (
                bool(s.get("title")) or bool(s.get("description")) or bool(s.get("internal_id")) or bool(s.get("segment_id"))
            )

        # Continue with single-story flow

        # If incoming story is empty (common when Strands only passes run_id/segment_id),
        # fallback to reading the generated backlog for this run and select the user story
        # matching the provided segment_id (if any). This makes the tool robust when the
        # caller doesn't know how to construct the full story payload.
        def _fallback_load_story_from_generated_backlog(_payload: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Prefer an explicitly provided run_id in payload; otherwise use agent's run_id
                target_run_id = _payload.get("run_id") or run_id
                target_segment_id = (
                    _payload.get("segment_id")
                    or (_payload.get("story") or {}).get("segment_id")
                )
                run_dir = Path(f"runs/{target_run_id}")
                backlog_file = run_dir / "generated_backlog.jsonl"
                if not backlog_file.exists():
                    return {}
                items: List[Dict[str, Any]] = []
                with open(backlog_file, "r") as bf:
                    for line in bf:
                        if line.strip():
                            try:
                                items.append(json.loads(line))
                            except Exception:
                                pass
                # Filter user stories only
                def _is_user_story(i: Dict[str, Any]) -> bool:
                    t = (i.get("type") or i.get("work_item_type") or "").lower()
                    return t in {"user story", "story", "user_story"}
                stories = [i for i in items if _is_user_story(i)]
                if not stories:
                    return {}
                if target_segment_id is not None:
                    try:
                        # Normalize to int for comparison if possible
                        seg_val = int(target_segment_id)
                    except Exception:
                        seg_val = target_segment_id
                    stories = [s for s in stories if s.get("segment_id") == seg_val]
                    if not stories:
                        return {}
                # Pick the first matching story
                selected = stories[0]
                title = (
                    selected.get("title")
                    or selected.get("name")
                    or selected.get("summary")
                    or selected.get("headline")
                    or ""
                )
                desc = (
                    selected.get("description")
                    or selected.get("details")
                    or selected.get("body")
                    or ""
                )
                ac = (
                    selected.get("acceptance_criteria")
                    or selected.get("acceptanceCriteria")
                    or selected.get("ac")
                    or []
                )
                if isinstance(ac, str):
                    ac = [s.strip() for s in ac.replace("\r", "").split("\n") if s.strip()]
                internal_id_fallback = selected.get("internal_id") or selected.get("id") or selected.get("uid")
                # Fallback title if missing
                if not title:
                    if isinstance(desc, str) and desc.strip():
                        words = desc.strip().split()
                        title = " ".join(words[:12]) + ("…" if len(words) > 12 else "")
                    elif internal_id_fallback:
                        title = f"Story {internal_id_fallback}"
                    else:
                        title = "Untitled Story"
                return {
                    "title": title,
                    "description": desc or "",
                    "acceptance_criteria": ac,
                    "internal_id": internal_id_fallback,
                }
            except Exception:
                return {}

        def _is_empty_story(s: Dict[str, Any]) -> bool:
            if not s:
                return True
            return not (s.get("title") or s.get("description"))

        # No batch mode supported

        if _is_empty_story(story):
            loaded = _fallback_load_story_from_generated_backlog(payload)
            if loaded:
                story = loaded
            else:
                # Continue with empty story; will early-exit as "new" with reason
                pass

        if not isinstance(story, dict):
            story = {}
        internal_id = story.get("internal_id")
        title = story.get("title")

        # Print the raw incoming story payload for troubleshooting (trim long output)
        try:
            raw_story_json = json.dumps(story, ensure_ascii=False)
        except Exception:
            raw_story_json = str(story)
        logger.debug("Tagging Agent: Received story payload: %s", raw_story_json[:1000])

        # Log story title and description for troubleshooting
        logger.info("Tagging Agent: Processing story title='%s' | description='%s…'", title, (story.get('description', '') or '')[:120])

        # If caller didn't provide similar stories, perform internal retrieval (no merging or size checks)
        if not similar:
            try:
                logger.info("Tagging Agent: No similar stories provided; performing internal retrieval…")
                retriever = SimilarStoryRetriever(config=None, min_similarity=threshold)
                similar = retriever.find_similar_stories(
                    {
                        "title": story.get("title", ""),
                        "description": story.get("description", ""),
                        "acceptance_criteria": story.get("acceptance_criteria", []),
                    },
                    min_similarity=threshold,
                )
                logger.info("Tagging Agent: Internal retrieval found %s similar stories", len(similar or []))
            except Exception as e:
                logger.exception("Tagging Agent: Internal retrieval failed: %s", e)

        # Early exit if no similar above threshold
        # Ensure we evaluate against all available similar stories above threshold
        above_threshold = [s for s in (similar or []) if s.get("similarity", 0.0) >= threshold]
        if not above_threshold:
            return _finalize({
                "status": "ok",
                "run_id": run_id,
                "decision_tag": "new",
                "related_ids": [],
                "reason": "No similar existing stories found (all below threshold)",
                "early_exit": True,
                "similarity_threshold": threshold,
                "similar_count": 0,
                "model_used": model_id
            }, internal_id, title)

        # Build prompt for LLM using template
        ac_list = story.get("acceptance_criteria", []) or []
        ac_text = "\n- " + "\n- ".join(ac_list) if ac_list else " (none)"
        
        similar_lines = []
        for s in above_threshold:
            similar_lines.append(
                f"ID: {s.get('work_item_id')} | similarity: {round(s.get('similarity', 0.0), 4)}\nTitle: {s.get('title')}\nDesc: {s.get('description','')[:300]}"
            )
        similar_formatted = "\n\n".join(similar_lines)
        
        user_prompt = prompt_loader.format_user_prompt(
            "tagging_agent",
            story_title=story.get("title"),
            story_description=story.get("description"),
            story_acceptance_criteria=ac_text,
            similarity_threshold=threshold,
            similar_stories_formatted=similar_formatted
        )

        if model is None:
            logger.warning("Tagging Agent: No model available; using rule-based fallback")
            fallback = _rule_based_fallback(story, above_threshold, threshold)
            return _finalize({
                "status": "ok",
                "run_id": run_id,
                "decision_tag": fallback.get("decision_tag", "new"),
                "related_ids": fallback.get("related_ids", []),
                "reason": fallback.get("reason", "Fallback applied"),
                "early_exit": False,
                "similarity_threshold": threshold,
                "similar_count": len(above_threshold),
                "model_used": model_id,
                "fallback_used": True
            }, internal_id, title)

        try:
            agent = Agent(model=model, system_prompt=system_prompt)
            result = agent(
                user_prompt,
                structured_output_model=TaggingDecisionOut,
            )
            parsed: TaggingDecisionOut = result.structured_output  # type: ignore[assignment]
            decision = (parsed.decision_tag or "new").lower()
            if decision not in {"new", "gap", "conflict"}:
                decision = "new"
            related_ids = parsed.related_ids or []
            reason = (parsed.reason or "")[:200]

            return _finalize({
                "status": "ok",
                "run_id": run_id,
                "decision_tag": decision,
                "related_ids": related_ids,
                "reason": reason,
                "early_exit": False,
                "similarity_threshold": threshold,
                "similar_count": len(above_threshold),
                "model_used": model_id,
                "fallback_used": False
            }, internal_id, title)
        except (StructuredOutputException, ValidationError) as e:
            logger.warning("Tagging Agent: Structured output failed, using rule-based fallback. Reason: %s", e)
            fallback = _rule_based_fallback(story, above_threshold, threshold)
            return _finalize({
                "status": "ok",
                "run_id": run_id,
                "decision_tag": fallback.get("decision_tag", "new"),
                "related_ids": fallback.get("related_ids", []),
                "reason": fallback.get("reason", "Fallback applied"),
                "early_exit": False,
                "similarity_threshold": threshold,
                "similar_count": len(above_threshold),
                "model_used": model_id,
                "fallback_used": True
            }, internal_id, title)
        except Exception as e:
            logger.exception("Tagging Agent: Agent invocation failed, using rule-based fallback: %s", e)
            fallback = _rule_based_fallback(story, above_threshold, threshold)
            return _finalize({
                "status": "ok",
                "run_id": run_id,
                "decision_tag": fallback.get("decision_tag", "new"),
                "related_ids": fallback.get("related_ids", []),
                "reason": fallback.get("reason", "Fallback applied"),
                "early_exit": False,
                "similarity_threshold": threshold,
                "similar_count": len(above_threshold),
                "model_used": model_id,
                "fallback_used": True
            }, internal_id, title)

    return tag_story

