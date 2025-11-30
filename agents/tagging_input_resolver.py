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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

GENERATED_BACKLOG_FILENAME = "generated_backlog.jsonl"
USER_STORY_TYPES = {"user story", "story", "user_story"}


class TaggingInputResolver:
    def __init__(
        self,
        default_run_id: str,
        default_threshold: float,
        generated_filename: str = GENERATED_BACKLOG_FILENAME,
    ) -> None:
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

            segment_id = (payload.get("segment_id") or (payload.get("story") or {}).get("segment_id"))
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
        try:
            if not isinstance(obj, dict):
                return False
            keys = set(obj.keys())
            story_keys = {"title", "description", "acceptance_criteria", "internal_id", "name", "summary", "headline"}
            return bool(keys & story_keys)
        except Exception:
            return False
