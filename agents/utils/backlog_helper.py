"""
Shared utilities for backlog item normalization, persistence, and summarization.

This module centralizes logic used by multiple agents (generation and regeneration)
to keep agents focused on orchestration and model interaction rather than IO and
post-processing details.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _normalize_type(value: str) -> tuple[str, str]:
    """Map various type spellings to canonical key and display label.

    Returns a tuple of (key, display) where:
      - key is one of "story" | "feature" | "epic"
      - display is one of "User Story" | "Feature" | "Epic" (or original fallback)
    """
    v = (value or "").strip().lower()
    if v in ("story", "user story", "user_story", "user-story"):
        return "story", "User Story"
    if v in ("feature", "features"):
        return "feature", "Feature"
    if v in ("epic", "epics"):
        return "epic", "Epic"
    # Fallback: keep as-is but default to Story classification
    return "story", value or "User Story"


class BacklogHelper:
    """Helper methods for processing and storing backlog items.

    Responsibilities:
      - Normalize LLM output into consistent item shape
      - Assign deterministic internal IDs per agent mode
      - Persist items to JSONL files
      - Provide standard summary payloads for responses
    """

    @staticmethod
    def normalize_items(
        items: List[Dict[str, Any]],
        run_id: str,
        segment_id: Optional[int] = None,
        id_mode: str = "segment",
    ) -> List[Dict[str, Any]]:
        """Normalize item types, assign internal IDs, and annotate with run/segment.

        id_mode:
          - "segment": internal_id pattern "{key}_{segment_id}_{n}"
          - "regen":   internal_id pattern "regen_{key}_{n}"
        """
        counters = {"epic": 1, "feature": 1, "story": 1}
        out: List[Dict[str, Any]] = []
        for item in items:
            # shallow copy to avoid mutating caller structures
            obj = dict(item) if isinstance(item, dict) else {}
            key, display = _normalize_type(str(obj.get("type", "story")))
            obj["type"] = display
            if key in counters:
                if id_mode == "regen":
                    obj["internal_id"] = f"regen_{key}_{counters[key]}"
                else:
                    seg = 0 if segment_id is None else segment_id
                    obj["internal_id"] = f"{key}_{seg}_{counters[key]}"
                counters[key] += 1
            if segment_id is not None:
                obj["segment_id"] = segment_id
            obj["run_id"] = run_id
            out.append(obj)
        return out

    @staticmethod
    def write_jsonl(items: List[Dict[str, Any]], file_path: Path, mode: str = "a") -> None:
        """Write backlog items to a JSONL file.

        mode:
          - "a": Append new items to the file (used by generation)
          - "w": Overwrite the file with provided items (used by regeneration)
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, mode) as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    @staticmethod
    def count_items(items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute counts per type with semantics matching agent summaries."""
        def is_epic(t: str) -> bool:
            l = (t or "").lower()
            return l in ("epic", "epics")

        def is_feature(t: str) -> bool:
            l = (t or "").lower()
            return l in ("feature", "features")

        def is_story(t: str) -> bool:
            l = (t or "").lower()
            return l in ("story", "user story")

        return {
            "epics": sum(1 for it in items if is_epic(str(it.get("type", "")))),
            "features": sum(1 for it in items if is_feature(str(it.get("type", "")))),
            "stories": sum(1 for it in items if is_story(str(it.get("type", "")))),
        }

    @staticmethod
    def summarize(
        run_id: str,
        backlog_file: Path,
        items: List[Dict[str, Any]],
        segment_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build the standard summary payload returned by agents."""
        return {
            "status": "success",
            "run_id": run_id,
            "segment_id": segment_id if segment_id is not None else 0,
            "items_generated": len(items),
            "backlog_file": str(backlog_file),
            "item_counts": BacklogHelper.count_items(items),
            "backlog_items": items,
        }
