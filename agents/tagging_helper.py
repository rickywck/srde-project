"""
TaggingInputResolver - Helper to normalize inputs for tagging agent.
Handles scenarios where callers provide a filesystem path (run directory or
explicit backlog file), and converts them into a list of story dicts to tag.

It returns a normalized payload:
- stories: List[Dict[str, Any]]  (single or multiple if file path)
- out_dir: Path                  (where tagging.jsonl should be written)
- run_id: str                    (effective run id)
- similar_existing_stories: List[Dict[str, Any]]
- threshold: float
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)

GENERATED_BACKLOG_FILENAME = "generated_backlog.jsonl"
USER_STORY_TYPES = {"user story", "story", "user_story"}


def _rule_based_fallback(story: Dict[str, Any], similar: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Apply simple deterministic rules if LLM JSON invalid.

    This was previously defined inside `tagging_agent.py`. Moving it here
    centralizes tagging helpers and allows reuse from other modules.
    """
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


def finalize_tagging_result(
    result: Dict[str, Any],
    out_dir: Path,
    processed_keys: set,
    internal_id: Any = None,
    title: str = None,
) -> None:
    """Persist a tagging `result` to `out_dir/tagging.jsonl` and update the
    `processed_keys` de-duplication set.

    Behaviour mirrors the previous `_finalize` closure in `tagging_agent.py`:
    - Attaches `story_internal_id` and `story_title` to `result` when provided.
    - Builds a key from `internal_id` or `title` and skips writing if already seen.
    - Appends the JSON line to `tagging.jsonl`, attempts to flush/fsync for
      durability, and adds the key to `processed_keys` on success.
    - Swallows filesystem errors to avoid failing the overall tagging flow.
    """
    if internal_id:
        result["story_internal_id"] = internal_id
    if title:
        result["story_title"] = title
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag_file = out_dir / "tagging.jsonl"
        key = None
        try:
            key = str(internal_id) if internal_id is not None else (str(title) if title is not None else None)
        except Exception:
            key = None
        if key is not None and key in processed_keys:
            return
        with open(tag_file, "a") as f:
            f.write(json.dumps(result) + "\n")
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
        if key is not None:
            processed_keys.add(key)
    except Exception:
        pass


class TaggingInputResolver:
    def __init__(
        self,
        default_run_id: str,
        default_threshold: float,
        generated_filename: str = GENERATED_BACKLOG_FILENAME,
    ) -> None:
        """Initialize the resolver.

        Args:
            default_run_id: fallback run id used when none is provided.
            default_threshold: default similarity threshold.
            generated_filename: name of the generated backlog file to look for.
        """
        self.default_run_id = default_run_id
        self.default_threshold = float(default_threshold)
        self.generated_filename = generated_filename

    def resolve(
        self,
        story_data: Any = None,
        story: Optional[Dict[str, Any]] = None,
        similar_existing_stories: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: Optional[float] = None,
        run_id: Optional[str] = None,
        backlog_path: Optional[str] = None,
        segment_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Normalize inputs for the tagging agent.

        This method accepts several convenient input shapes (raw JSON string,
        dicts, explicit backlog paths, or run ids) and returns a normalized
        payload containing:
            - `stories`: list of story dicts to tag
            - `out_dir`: Path where outputs should be written
            - `run_id`: effective run id
            - `similar_existing_existing_stories`: any pre-supplied similar stories
            - `threshold`: effective similarity threshold
        """
        # Normalize incoming payload
        payload: Dict[str, Any] = {}
        if story_data is not None and story is None and similar_existing_stories is None and backlog_path is None:
            if isinstance(story_data, (dict, list)):
                # If story_data is already a story-like dict, wrap under "story"
                if isinstance(story_data, dict) and self._looks_like_story_dict(story_data):
                    payload = {"story": story_data}
                else:
                    payload = story_data if isinstance(story_data, dict) else {"story": story_data}
            else:
                try:
                    loaded = json.loads(story_data)
                    if isinstance(loaded, dict) and self._looks_like_story_dict(loaded) and "story" not in loaded:
                        payload = {"story": loaded}
                    else:
                        payload = loaded if isinstance(loaded, dict) else {"story": loaded}
                except Exception:
                    payload = {"story": story_data}
        else:
            payload = {
                "story": story,
                "similar_existing_stories": similar_existing_stories,
                "similarity_threshold": similarity_threshold,
                "run_id": run_id,
                "backlog_path": backlog_path,
                "segment_id": segment_id,
            }

        eff_run_id = (payload.get("run_id") or self.default_run_id) or self.default_run_id
        eff_threshold = float(
            payload.get("similarity_threshold") if payload.get("similarity_threshold") is not None else self.default_threshold
        )

        # If an explicit backlog_path present, treat as path mode
        if payload.get("backlog_path"):
            out_dir, backlog_file = self._resolve_io_context(eff_run_id, str(payload.get("backlog_path")))
            stories = self._load_stories_from_backlog(backlog_file, payload)
            return {
                "stories": stories,
                "out_dir": out_dir,
                "run_id": eff_run_id,
                "similar_existing_stories": [],
                "threshold": eff_threshold,
            }

        # Detect accidental path passed via story or its fields
        st = payload.get("story")
        if isinstance(st, str) and self._looks_like_filesystem_path(st):
            out_dir, backlog_file = self._resolve_io_context(eff_run_id, st)
            stories = self._load_stories_from_backlog(backlog_file, payload)
            return {
                "stories": stories,
                "out_dir": out_dir,
                "run_id": eff_run_id,
                "similar_existing_stories": [],
                "threshold": eff_threshold,
            }

        if isinstance(st, dict):
            title_val = st.get("title") or ""
            desc_val = st.get("description") or ""
            if self._looks_like_filesystem_path(title_val) and not desc_val:
                out_dir, backlog_file = self._resolve_io_context(eff_run_id, title_val)
                stories = self._load_stories_from_backlog(backlog_file, payload)
                return {
                    "stories": stories,
                    "out_dir": out_dir,
                    "run_id": eff_run_id,
                    "similar_existing_stories": [],
                    "threshold": eff_threshold,
                }

        # Non-path mode: return single story list (or empty)
        # If payload itself is a story dict at top-level, wrap it
        if not isinstance(st, (dict, list)) and isinstance(payload, dict) and self._looks_like_story_dict(payload):
            st = payload
        stories: List[Dict[str, Any]] = []
        if isinstance(st, dict):
            stories = [st]
        elif isinstance(st, list):
            # Not expected anymore, but be tolerant and pass through
            stories = [s for s in st if isinstance(s, dict)]
        else:
            stories = [dict()]

        out_dir = Path(f"runs/{eff_run_id}")
        return {
            "stories": stories,
            "out_dir": out_dir,
            "run_id": eff_run_id,
            "similar_existing_stories": payload.get("similar_existing_stories") or [],
            "threshold": eff_threshold,
        }

    def _looks_like_filesystem_path(self, v: Any) -> bool:
        """Return True if `v` resembles a filesystem path or filename.

        Heuristics consider `.json`, `.jsonl` extensions or the presence of
        path separators.
        """
        try:
            if not v or not isinstance(v, str):
                return False
            s = v.strip()
            if s.endswith(".jsonl") or s.endswith(".json"):
                return True
            if "/" in s or "\\" in s:
                return True
            return False
        except Exception:
            return False

    def _resolve_io_context(self, run_id: str, candidate: str) -> Tuple[Path, Path]:
        """Determine the output directory and backlog file path for a given
        `candidate` string which may be a file, directory, or run-relative path.

        Returns a tuple `(out_dir, backlog_path)` where `backlog_path` points to
        the file to be read (often `generated_backlog.jsonl`).
        """
        out_dir_local = Path(f"runs/{run_id}")
        backlog_path_local = out_dir_local / self.generated_filename
        try:
            cand_path = Path(str(candidate))
            if str(candidate).endswith(".jsonl") or cand_path.suffix.lower() == ".jsonl":
                # Special case: if segments.jsonl is passed, redirect to generated_backlog.jsonl in same folder
                if cand_path.name.lower() == "segments.jsonl":
                    out_dir_local = cand_path.parent
                    backlog_path_local = out_dir_local / self.generated_filename
                else:
                    backlog_path_local = cand_path
                    out_dir_local = cand_path.parent
            elif cand_path.exists() and cand_path.is_dir():
                out_dir_local = cand_path
                backlog_path_local = cand_path / self.generated_filename
            elif "/" in str(candidate) or "\\" in str(candidate):
                if str(candidate).endswith(".jsonl"):
                    backlog_path_local = cand_path
                    out_dir_local = cand_path.parent
                else:
                    out_dir_local = cand_path
                    backlog_path_local = cand_path / self.generated_filename
        except Exception:
            pass
        return out_dir_local, backlog_path_local

    def _load_stories_from_backlog(self, backlog_file: Path, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load and normalize user stories from a backlog JSONL file.

        Reads the provided `backlog_file`, filters to user-story work items,
        optionally filters by `segment_id` (if present in `payload`), and
        normalizes fields to a predictable dict shape:
            {"title","description","acceptance_criteria","internal_id"}

        Returns an empty list on errors or when the file is not found.
        """
        try:
            if not backlog_file.exists():
                logger.warning("TaggingInputResolver: Backlog file not found: %s", backlog_file)
                return []
            rows: List[Dict[str, Any]] = []
            with open(backlog_file, "r") as bf:
                for line in bf:
                    if line.strip():
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            # Filter to user stories only, optionally by provided segment_id
            def _is_user_story(i: Dict[str, Any]) -> bool:
                t = (i.get("type") or i.get("work_item_type") or "").lower()
                return t in USER_STORY_TYPES

            # payload.get('story') may be a string (path) or a dict; guard accordingly
            story_field = payload.get("story")
            segment_id = payload.get("segment_id")
            if segment_id is None and isinstance(story_field, dict):
                segment_id = story_field.get("segment_id")
            stories = [i for i in rows if _is_user_story(i)]
            if segment_id is not None:
                try:
                    seg_val = int(segment_id)
                except Exception:
                    seg_val = segment_id
                stories = [s for s in stories if s.get("segment_id") == seg_val]

            normalized: List[Dict[str, Any]] = []
            for s in stories:
                title = s.get("title") or s.get("name") or s.get("summary") or s.get("headline") or ""
                desc = s.get("description") or s.get("details") or s.get("body") or ""
                ac = s.get("acceptance_criteria") or s.get("acceptanceCriteria") or s.get("ac") or []
                if isinstance(ac, str):
                    ac = [t.strip() for t in ac.replace("\r", "").split("\n") if t.strip()]
                internal_id = s.get("internal_id") or s.get("id") or s.get("uid")
                if not title:
                    if isinstance(desc, str) and desc.strip():
                        words = desc.strip().split()
                        title = " ".join(words[:12]) + ("…" if len(words) > 12 else "")
                    elif internal_id:
                        title = f"Story {internal_id}"
                    else:
                        title = "Untitled Story"
                normalized.append({
                    "title": title,
                    "description": desc or "",
                    "acceptance_criteria": ac or [],
                    "internal_id": internal_id,
                })
            return normalized
        except Exception as e:
            logger.exception("TaggingInputResolver: Failed to load stories from backlog: %s", e)
            return []

    def _looks_like_story_dict(self, obj: Dict[str, Any]) -> bool:
        """Heuristic to determine whether `obj` resembles a story dict.

        Returns True if keys associated with user stories are present.
        """
        try:
            if not isinstance(obj, dict):
                return False
            keys = set(obj.keys())
            story_keys = {"title", "description", "acceptance_criteria", "internal_id", "name", "summary", "headline"}
            return bool(keys & story_keys)
        except Exception:
            return False
