import os
import json
import uuid
import pytest
from pathlib import Path

# Ensure project root import path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.evaluation_agent import create_evaluation_agent

@pytest.mark.parametrize("mode", ["live", "batch"])
def test_evaluation_agent_mock(mode):
    os.environ["EVALUATION_AGENT_MOCK"] = "1"
    run_id = str(uuid.uuid4())
    agent_tool = create_evaluation_agent(run_id)

    # Minimal synthetic payload
    payload = {
        "segment_text": "User authentication enhancement and performance issues discussed.",
        "retrieved_context": {
            "ado_items": [
                {"work_item_id": 123, "title": "Existing Auth Epic", "description": "Improve multi-factor auth."}
            ],
            "architecture_constraints": [
                {"file_name": "security.md", "text": "MFA required for high privilege actions."}
            ]
        },
        "generated_backlog": [
            {
                "type": "Story",
                "title": "Implement MFA with authenticator app",
                "description": "Add support for TOTP based authenticator app.",
                "acceptance_criteria": [
                    "User can enroll authenticator app",
                    "User can remove MFA device"
                ]
            },
            {
                "type": "Story",
                "title": "Reduce search API latency",
                "description": "Optimize indexing for dashboard search queries.",
                "acceptance_criteria": [
                    "P95 latency under 800ms",
                    "Search results relevance unchanged"
                ]
            }
        ],
        "evaluation_mode": mode
    }

    result = json.loads(agent_tool(json.dumps(payload)))
    assert result["status"] in ("success_mock", "success")
    assert "evaluation" in result
    ev = result["evaluation"]
    for k in ["completeness", "relevance", "quality"]:
        assert k in ev and "score" in ev[k]
    assert "overall_score" in ev
    assert "summary" in ev
    assert "suggestions" not in ev

    # Live mode should persist evaluation.jsonl
    if mode == "live":
        eval_file = Path(f"runs/{run_id}/evaluation.jsonl")
        assert eval_file.exists(), "live mode should write evaluation file"
        content = eval_file.read_text().strip()
        assert content, "evaluation file should not be empty"

    # Cleanup mock env var
    del os.environ["EVALUATION_AGENT_MOCK"]
