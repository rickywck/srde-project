#!/usr/bin/env python3
"""Unit tests for Section 7 Tagging Agent."""

import json
import sys
from pathlib import Path

import pytest

# Ensure repository root on path for direct imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import tagging_agent  # noqa: E402  pylint: disable=wrong-import-position


@pytest.fixture(autouse=True)
def patch_openai_model(monkeypatch):
    """Replace OpenAIModel with a lightweight stub for tests."""

    class DummyModel:
        def __init__(self, model_id: str, params: dict):
            self.model_id = model_id
            self.params = params

    monkeypatch.setattr(tagging_agent, "OpenAIModel", DummyModel)


def test_tagging_agent_returns_new_when_no_similar(monkeypatch):
    """Stories with no similar matches should early-exit as 'new'."""

    class FailAgent:
        def __init__(self, *args, **kwargs):  # pragma: no cover - should never run
            raise AssertionError("Agent should not be instantiated on early exit")

    monkeypatch.setattr(tagging_agent, "Agent", FailAgent)

    tag_tool = tagging_agent.create_tagging_agent(run_id="test-run", default_similarity_threshold=0.8)
    payload = {
        "story": {
            "title": "Capture uptime metrics",
            "description": "Collect service uptime stats for weekly reports",
            "acceptance_criteria": ["Dashboard shows uptime %", "Alerts fire below 99.5%"]
        },
        "similar_existing_stories": [
            {"work_item_id": 123, "title": "Legacy reporting", "similarity": 0.5}
        ]
    }

    result = json.loads(tag_tool(json.dumps(payload)))

    assert result["status"] == "ok"
    assert result["decision_tag"] == "new"
    assert result["early_exit"] is True
    assert result["similar_count"] == 0
    assert result["related_ids"] == []


def test_tagging_agent_rule_based_fallback(monkeypatch):
    """Fallback logic should classify duplicates as conflicts when LLM output invalid."""

    captured_prompt = {}

    class InvalidJSONAgent:
        def __init__(self, model, system_prompt, tools, callback_handler):
            captured_prompt["system_prompt"] = system_prompt

        def __call__(self, prompt):
            captured_prompt["prompt"] = prompt
            return "This is not JSON"

    monkeypatch.setattr(tagging_agent, "Agent", InvalidJSONAgent)

    tag_tool = tagging_agent.create_tagging_agent(run_id="run-fallback", default_similarity_threshold=0.7)
    payload = {
        "story": {
            "title": "Implement MFA login",
            "description": "Add multi-factor auth with SMS codes",
            "acceptance_criteria": ["Send SMS", "Fallback to email"]
        },
        "similar_existing_stories": [
            {
                "work_item_id": "ADO-42",
                "title": "Implement MFA login",
                "description": "Users must confirm via SMS before login",
                "similarity": 0.92
            }
        ]
    }

    result = json.loads(tag_tool(json.dumps(payload)))

    assert result["status"] == "ok"
    assert result["decision_tag"] == "conflict"
    assert result["fallback_used"] is True
    assert result["early_exit"] is False
    assert result["similar_count"] == 1
    assert result["related_ids"] == ["ADO-42"]
    assert "Implement MFA login" in captured_prompt["prompt"]
    assert "new|gap|conflict" in captured_prompt["prompt"]


def test_tagging_agent_llm_success(monkeypatch):
    """Agent should use LLM JSON when valid and avoid fallback."""

    class GoodJSONAgent:
        def __init__(self, *args, **kwargs):  # pragma: no cover
            pass

        def __call__(self, prompt):  # pragma: no cover
            # Return valid JSON indicating gap classification
            return json.dumps({
                "decision_tag": "gap",
                "related_ids": ["WK-11", "WK-12"],
                "reason": "Extends existing stories with missing criteria"
            })

    monkeypatch.setattr(tagging_agent, "Agent", GoodJSONAgent)

    tag_tool = tagging_agent.create_tagging_agent(run_id="run-llm-success", default_similarity_threshold=0.7)
    payload = {
        "story": {
            "title": "Add password strength meter",
            "description": "Provide real-time feedback for chosen password complexity",
            "acceptance_criteria": ["Shows strength bar", "Suggest improvements"]
        },
        "similar_existing_stories": [
            {"work_item_id": "WK-11", "title": "Improve password UX", "description": "Simplify password creation flow", "similarity": 0.75},
            {"work_item_id": "WK-12", "title": "Password validation rules", "description": "Validate against reused passwords", "similarity": 0.72},
            {"work_item_id": "WK-99", "title": "Legacy auth cleanup", "description": "Refactor old auth code", "similarity": 0.4}
        ]
    }

    result = json.loads(tag_tool(json.dumps(payload)))

    assert result["status"] == "ok"
    assert result["decision_tag"] == "gap"
    assert result["fallback_used"] is False
    assert result["early_exit"] is False
    assert result["similar_count"] == 2  # Only above threshold counted
    assert result["related_ids"] == ["WK-11", "WK-12"]
    assert len(result["reason"]) > 0


def test_tagging_agent_invalid_input_json():
    """Invalid JSON input should return error status and early exit."""

    tag_tool = tagging_agent.create_tagging_agent(run_id="run-invalid-json", default_similarity_threshold=0.7)

    result = json.loads(tag_tool("not valid json"))

    assert result["status"] == "error"
    assert result["decision_tag"] == "new"  # default in error path
    assert result["early_exit"] is True
    assert result["similar_count"] == 0
    assert result["related_ids"] == []
    assert "Invalid input JSON" in result["reason"]
