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
from typing import List, Dict, Any
from pathlib import Path
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from .prompt_loader import get_prompt_loader
import yaml
from services.similar_story_retriever import SimilarStoryRetriever
from .model_factory import ModelFactory
import logging

# Module logger
logger = logging.getLogger(__name__)

# Constants
GENERATED_BACKLOG_FILENAME = "generated_backlog.jsonl"
USER_STORY_TYPES = {"user story", "story", "user_story"}

# Note: System prompt now loaded from prompts/tagging_agent.yaml


def _safe_json_extract(text: str) -> Dict[str, Any]:
    """Attempt to extract JSON object from LLM text response."""
    # Direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Regex to find first { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


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

    # Load app config via ModelFactory
    config_path = "config.poc.yaml"
    try:
        _cfg = ModelFactory._load_config(config_path)
        logger.debug("Loaded config for tagging agent: %s", {k: v for k, v in (_cfg or {}).items()})
    except Exception as e:
        logger.exception("Error loading config via ModelFactory: %s", e)
        _cfg = {}

    if default_similarity_threshold is None:
        default_similarity_threshold = float(_cfg.get("retrieval", {}).get("tagging", {}).get("min_similarity_threshold", 0.5))

    # Determine effective max tokens (priority: agent prompt params -> app config -> model default)
    agent_max_tokens = params.get("max_completion_tokens") or params.get("max_tokens")
    app_max_tokens = _cfg.get("openai", {}).get("max_tokens")
    if agent_max_tokens is not None:
        eff_max_tokens = int(agent_max_tokens)
    elif app_max_tokens is not None:
        eff_max_tokens = int(app_max_tokens)
    else:
        eff_max_tokens = None

    # Build model via ModelFactory to centralize defaults and param mapping
    model = None
    model_id = None
    model_params = {}
    if eff_max_tokens is not None:
        model_params["max_completion_tokens"] = eff_max_tokens
    try:
        model_descriptor = ModelFactory.create_openai_model(config_path=config_path, model_params=model_params)
        model = model_descriptor
        model_id = getattr(model_descriptor, "model_id", None) or ModelFactory.get_default_model_id(config_path)
        logger.debug("Tagging agent model descriptor: model_id=%s params=%s", model_id, getattr(model_descriptor, "params", {}))
    except Exception as e:
        logger.exception("ModelFactory.create_openai_model failed for tagging agent: %s", e)
        # Fallback: use simple OpenAIModel with default model id
        model_id = ModelFactory.get_default_model_id(config_path)
        try:
            model = OpenAIModel(model_id=model_id, params={"max_completion_tokens": max_comp_tokens})
        except Exception:
            # Last resort: set model to None; Agent() may still accept None depending on strands implementation
            model = None

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

        # For top-level calls (not internal subcalls) we overwrite the tagging file
        # so each tagging run starts fresh. For internal subcalls, seed processed
        # keys from the existing file to avoid duplicate writes within the same run.
        if not is_subcall:
            try:
                current_out_dir.mkdir(parents=True, exist_ok=True)
                tag_file_path = current_out_dir / "tagging.jsonl"
                # Truncate/overwrite previous results for a fresh run
                with open(tag_file_path, "w") as f:
                    f.write("")
                processed_story_keys = set()
                # Mark file as initialized so _finalize will append subsequent writes
                out_file_initialized = True
            except Exception:
                processed_story_keys = set()
        else:
            try:
                processed_story_keys = _load_existing_processed_keys(current_out_dir)
            except Exception:
                processed_story_keys = set()

        # If no payload content was supplied at all (chat invoked without args),
        # attempt batch tagging from the current run's generated backlog file.
        if (
            story_data is None
            and story is None
            and similar_existing_stories is None
            and (not payload.get("story") and not payload.get("similar_existing_stories"))
        ):
            try:
                backlog_file = resolved_backlog_path
                if not backlog_file.exists():
                    return json.dumps({
                        "status": "error",
                        "run_id": str(payload.get("run_id") or run_id),
                        "message": "No generated backlog found for this run. Provide a story payload or generate backlog first.",
                    })
                items = []
                with open(backlog_file, "r") as bf:
                    for line in bf:
                        if line.strip():
                            try:
                                items.append(json.loads(line))
                            except Exception:
                                pass
                def _normalize_story_fields(item: Dict[str, Any]) -> Dict[str, Any]:
                    title = (
                        item.get("title")
                        or item.get("name")
                        or item.get("summary")
                        or item.get("headline")
                        or ""
                    )
                    desc = (
                        item.get("description")
                        or item.get("details")
                        or item.get("body")
                        or ""
                    )
                    ac = (
                        item.get("acceptance_criteria")
                        or item.get("acceptanceCriteria")
                        or item.get("ac")
                        or []
                    )
                    if isinstance(ac, str):
                        ac = [s.strip() for s in ac.replace("\r", "").split("\n") if s.strip()]
                    internal_id = item.get("internal_id") or item.get("id") or item.get("uid")
                    # Fallback title if missing: derive from description or internal id
                    if not title:
                        if isinstance(desc, str) and desc.strip():
                            words = desc.strip().split()
                            title = " ".join(words[:12]) + ("…" if len(words) > 12 else "")
                        elif internal_id:
                            title = f"Story {internal_id}"
                        else:
                            title = "Untitled Story"
                    return {
                        "title": title,
                        "description": desc or "",
                        "acceptance_criteria": ac,
                        "internal_id": internal_id,
                    }

                stories = [i for i in items if ((i.get("type") or i.get("work_item_type") or "").lower() in USER_STORY_TYPES)]
                if not stories:
                    return json.dumps({
                        "status": "error",
                        "run_id": run_id,
                        "message": "No user stories found in generated backlog for this run.",
                    })

                # Use provided threshold or default
                batch_threshold = float(payload.get("similarity_threshold", default_similarity_threshold))
                tag_counts = {"new": 0, "gap": 0, "conflict": 0}
                processed = 0
                errors = 0

                for s in stories:
                    norm = _normalize_story_fields(s)
                    single_payload = {
                        "story": norm,
                        # Leave similar stories empty to trigger internal retrieval
                        "similar_existing_stories": [],
                        "similarity_threshold": batch_threshold,
                        "__internal_subcall": True,
                        "run_id": payload.get("run_id"),
                    }
                    try:
                        single_result_str = tag_story(json.dumps(single_payload))
                        single_obj = json.loads(single_result_str)
                        decision = (single_obj.get("decision_tag") or "").lower()
                        if decision in tag_counts:
                            tag_counts[decision] += 1
                        processed += 1
                    except Exception:
                        errors += 1
                        continue

                return json.dumps({
                    "status": "ok",
                    "run_id": run_id,
                    "mode": "batch",
                    "processed": processed,
                    "errors": errors,
                    "stories_found": len(stories),
                    "tag_counts": tag_counts,
                    "message": f"Tagged {processed} stories from generated_backlog.jsonl",
                })
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "run_id": run_id,
                    "message": f"Batch tagging failed: {e}",
                })

        story = payload.get("story", {})
        similar = payload.get("similar_existing_stories", [])
        threshold = float(payload.get("similarity_threshold", default_similarity_threshold))

        # Normalize similar stories structure
        if isinstance(similar, dict):
            similar = [similar]
        elif not isinstance(similar, list):
            similar = []

        # Helper to merge two similar-story lists into a unique set by work_item_id or title/desc
        def _merge_similar_sets(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            merged: Dict[str, Dict[str, Any]] = {}
            def _make_key(item: Dict[str, Any]) -> str:
                wid = item.get("work_item_id")
                if wid is not None:
                    return f"id:{wid}"
                # fallback key using title + snippet of desc
                t = (item.get("title") or "").strip().lower()
                d = (item.get("description") or "").strip().lower()[:80]
                return f"td:{t}|{d}"
            def _merge(dst: Dict[str, Any], src: Dict[str, Any]):
                # keep max similarity
                try:
                    dst["similarity"] = max(dst.get("similarity", 0.0), src.get("similarity", 0.0))
                except Exception:
                    pass
                # prefer non-empty title/desc from either side
                if not dst.get("title") and src.get("title"):
                    dst["title"] = src.get("title")
                if not dst.get("description") and src.get("description"):
                    dst["description"] = src.get("description")
                if not dst.get("work_item_id") and src.get("work_item_id"):
                    dst["work_item_id"] = src.get("work_item_id")

            for item in (a or []) + (b or []):
                if not isinstance(item, dict):
                    continue
                key = _make_key(item)
                if key in merged:
                    _merge(merged[key], item)
                else:
                    merged[key] = dict(item)
            # sort by similarity desc for nicer prompt ordering
            return sorted(merged.values(), key=lambda x: x.get("similarity", 0.0), reverse=True)

        # Multi-story handling: if the incoming story is a list or payload has a list of backlog items,
        # process each story individually and aggregate results. Support multiple common keys used by LLMs.
        if (
            isinstance(story, list)
            or isinstance(payload.get("stories"), list)
            or isinstance(payload.get("user_stories"), list)
            or isinstance(payload.get("items"), list)
            or isinstance(payload.get("backlog_items"), list)
            or isinstance(payload.get("work_items"), list)
            or isinstance(payload.get("backlog"), list)
        ):
            raw_list = (
                story if isinstance(story, list)
                else payload.get("stories")
                or payload.get("user_stories")
                or payload.get("items")
                or payload.get("backlog_items")
                or payload.get("work_items")
                or payload.get("backlog")
            )
            # Filter: tag only explicit User Story types; if type is missing, keep (assume caller supplied stories list)
            stories_list = []
            for s in raw_list:
                if not isinstance(s, dict):
                    continue
                t = (s.get("type") or s.get("work_item_type") or "").lower()
                if t and t not in USER_STORY_TYPES:
                    continue
                stories_list.append(s)
            processed = 0
            errors = 0
            results = []
            for s in stories_list:
                if not isinstance(s, dict):
                    errors += 1
                    continue
                single_payload = {
                    "story": s,
                    # leave similar empty to trigger internal retrieval
                    "similar_existing_stories": [],
                    "similarity_threshold": threshold,
                    # allow outer payload to carry run_id/segment_id for fallback if needed
                    "run_id": payload.get("run_id"),
                    "segment_id": payload.get("segment_id"),
                    "__internal_subcall": True,
                }
                try:
                    res_str = tag_story(json.dumps(single_payload))
                    res = json.loads(res_str)
                    results.append(res)
                    processed += 1
                except Exception:
                    errors += 1
            return json.dumps({
                "status": "ok",
                "run_id": run_id,
                "mode": "multi",
                "processed": processed,
                "errors": errors,
                "results": results,
                "message": f"Tagged {processed} of {len(stories_list)} stories",
            })

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

        # If story fields are empty but caller supplied run_id/segment_id or a backlog_path,
        # run batch tagging over all user stories in the generated backlog file
        # for the provided run/path (ignoring segment_id), mirroring the multi-story behavior.
        if _is_empty_story(story) and (payload.get("run_id") or payload.get("segment_id") or payload.get("backlog_path")):
            try:
                # Reuse resolved context
                backlog_file = resolved_backlog_path
                if not backlog_file.exists():
                    return json.dumps({
                        "status": "error",
                        "run_id": str(payload.get("run_id") or run_id),
                        "message": "No generated backlog found for this run.",
                    })
                items = []
                with open(backlog_file, "r") as bf:
                    for line in bf:
                        if line.strip():
                            try:
                                items.append(json.loads(line))
                            except Exception:
                                pass
                stories = [i for i in items if ((i.get("type") or i.get("work_item_type") or "").lower() in USER_STORY_TYPES)]
                if not stories:
                    return json.dumps({
                        "status": "error",
                        "run_id": str(payload.get("run_id") or run_id),
                        "message": "No user stories found in generated backlog for this run.",
                    })
                batch_threshold = float(payload.get("similarity_threshold", default_similarity_threshold))
                tag_counts = {"new": 0, "gap": 0, "conflict": 0}
                processed = 0
                errors = 0
                results = []
                for s in stories:
                    single_payload = {
                        "story": {
                            "title": s.get("title"),
                            "description": s.get("description"),
                            "acceptance_criteria": s.get("acceptance_criteria", []),
                            "internal_id": s.get("internal_id")
                        },
                        "similar_existing_stories": [],
                        "similarity_threshold": batch_threshold,
                        "__internal_subcall": True,
                        "run_id": payload.get("run_id"),
                    }
                    try:
                        single_result_str = tag_story(json.dumps(single_payload))
                        single_obj = json.loads(single_result_str)
                        results.append(single_obj)
                        decision = (single_obj.get("decision_tag") or "").lower()
                        if decision in tag_counts:
                            tag_counts[decision] += 1
                        processed += 1
                    except Exception:
                        errors += 1
                        continue
                return json.dumps({
                    "status": "ok",
                    "run_id": str(payload.get("run_id") or run_id),
                    "mode": "batch",
                    "processed": processed,
                    "errors": errors,
                    "stories_found": len(stories),
                    "tag_counts": tag_counts,
                    "results": results,
                    "message": f"Tagged {processed} stories from generated_backlog.jsonl",
                })
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "run_id": run_id,
                    "message": f"Batch tagging (run_id only) failed: {e}",
                })

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

        # If caller didn't provide similar stories or provided too few, perform internal retrieval,
        # then merge with any provided list to ensure we always tag against the full set.
        need_retrieval = (not similar) or (isinstance(similar, list) and len(similar) < 2)
        if need_retrieval:
            try:
                logger.info("Tagging Agent: No similar stories provided; performing internal retrieval…")
                retriever = SimilarStoryRetriever(config=_cfg, min_similarity=threshold)
                retrieved = retriever.find_similar_stories(
                    {
                        "title": story.get("title", ""),
                        "description": story.get("description", ""),
                        "acceptance_criteria": story.get("acceptance_criteria", []),
                    },
                    min_similarity=threshold,
                )
                # Merge provided and retrieved for completeness
                similar = _merge_similar_sets(similar, retrieved)
                logger.info("Tagging Agent: Internal retrieval found %s similar stories (after merge)", len(similar))
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

        agent = Agent(model=model, system_prompt=system_prompt, tools=[], callback_handler=None)
        llm_response = agent(user_prompt)
        llm_text = str(llm_response)

        parsed = _safe_json_extract(llm_text)
        if not parsed or not isinstance(parsed, dict) or "decision_tag" not in parsed:
            # Fallback to rule-based
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

        # Clean / normalize parsed output
        decision = parsed.get("decision_tag", "new").lower()
        if decision not in {"new", "gap", "conflict"}:
            decision = "new"
        related_ids = parsed.get("related_ids", [])
        if not isinstance(related_ids, list):
            related_ids = []
        reason = parsed.get("reason", "")[:200]

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

    return tag_story

