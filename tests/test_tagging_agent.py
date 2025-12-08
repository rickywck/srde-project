#!/usr/bin/env python3
"""Unit tests for Tagging Agent.

These tests exercise the simplified `tag_story(story)` API in two modes:
- Direct hard-coded story tagging
- Tagging all user stories from an explicit generated backlog path
"""

import os
import json
import sys
from pathlib import Path

import pytest

# Ensure repository root on path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import tagging_agent  # noqa: E402  pylint: disable=wrong-import-position

def test_tagging_agent_hardcoded_story_tags_ok():
    """Tagging a single hard-coded story returns a well-formed result."""

    # Use a synthetic run id; retrieval may return empty, which is fine
    run_id = "unit-test-run-single-story"
    tag_tool = tagging_agent.create_tagging_agent(run_id=run_id, default_similarity_threshold=0.5)

    story = {
        "title": "Capture uptime metrics",
        "description": "Collect service uptime stats for weekly reports",
        "acceptance_criteria": [
            "Dashboard shows uptime %",
            "Alerts fire below 99.5%",
        ],
        "internal_id": "T-1",
    }

    result = json.loads(tag_tool(story))

    # Print result for visibility when running pytest with -s
    print("HARD-CODED STORY TAG RESULT:", json.dumps(result, indent=2))

    assert result["status"] in {"ok", "error"}
    assert result["decision_tag"] in {"new", "gap", "duplicate", "conflict"}
    assert "related_ids" in result
    assert "reason" in result


def test_tagging_agent_tags_latest_run_backlog():
    """Tag all user stories from a specified generated backlog file.

    Users must set TAGGING_TEST_BACKLOG_PATH to point at
    runs/<run_id>/generated_backlog.jsonl. If not set, this test is skipped.
    """

    backlog_env = os.getenv("TAGGING_TEST_BACKLOG_PATH")
    if not backlog_env:
        pytest.skip("TAGGING_TEST_BACKLOG_PATH not set; skipping backlog tagging test")

    backlog = Path(backlog_env)
    if not backlog.exists():
        pytest.skip(f"Backlog file {backlog} does not exist")

    # Derive run_id from parent directory name
    run_id = backlog.parent.name

    # Load generated backlog items
    items: list[dict] = []
    with backlog.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            items.append(item)

    # Filter user stories using the same heuristic as BacklogSynthesisWorkflow
    def _is_user_story(it: dict) -> bool:
        t = (it.get("type") or it.get("work_item_type") or "").lower()
        return t in {"user story", "story", "user_story"}

    stories = [it for it in items if _is_user_story(it)]
    if not stories:
        pytest.skip("No user stories found in generated_backlog.jsonl")

    tag_tool = tagging_agent.create_tagging_agent(run_id=run_id, default_similarity_threshold=0.5)

    results = []
    for s in stories:
        payload = {
            "title": s.get("title") or "",
            "description": s.get("description") or "",
            "acceptance_criteria": s.get("acceptance_criteria", []) or [],
            "internal_id": s.get("internal_id"),
        }
        res = json.loads(tag_tool(payload))
        results.append(res)

    # Print results for visibility when running pytest with -s
    print("BACKLOG TAG RESULTS:")
    for r in results:
        print(json.dumps(r, indent=2))

    # Basic sanity checks: at least one result and each has the expected shape
    assert results, "Expected at least one tagging result"
    for res in results:
        assert res["status"] in {"ok", "error"}
        assert res["decision_tag"] in {"new", "gap", "duplicate", "conflict"}
        assert "related_ids" in res
        assert "reason" in res
