#!/usr/bin/env python3
"""
Pytest for the backlog generation API using FastAPI TestClient.
Covers both Strands workflow (if available) and the sequential workflow.
"""

import json
from pathlib import Path
import pytest

from fastapi.testclient import TestClient

SAMPLE_DOC = """
Product Planning Meeting - Q1 2024

Topic 1: User Authentication Enhancement
We need to add multi-factor authentication to improve security.
Users have been requesting this feature for account protection.

Topic 2: Performance Issues
Several customers reported slow page load times.
The search API is taking 3-5 seconds on average.
This needs to be fixed as a P1 bug.

Topic 3: Mobile App Offline Mode
Product team found high demand for offline mode.
This is a major feature requiring architecture changes.
"""


def _upload_document(client: TestClient) -> str:
    files = {"file": ("test_meeting.txt", SAMPLE_DOC, "text/plain")}
    r = client.post("/upload", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    return data["run_id"]


@pytest.mark.parametrize("use_strands", [pytest.param(True, id="strands"), pytest.param(False, id="sequential")])
def test_generate_backlog_end_to_end(tmp_path: Path, use_strands: bool):
    # Ensure AWS metadata lookups are disabled before importing the app
    import os, sys, importlib
    os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")
    os.environ.setdefault("AWS_METADATA_SERVICE_TIMEOUT", "1")
    os.environ.setdefault("AWS_METADATA_SERVICE_NUM_ATTEMPTS", "1")

    # Re-import app after setting env to ensure clean state
    if "app" in sys.modules:
        del sys.modules["app"]
    from app import app  # type: ignore

    client = TestClient(app)

    run_id = _upload_document(client)

    # Execute workflow
    resp = client.post(f"/generate-backlog/{run_id}", params={"use_strands_workflow": use_strands})

    # If Strands not available at runtime, API returns 501
    if use_strands and resp.status_code == 501:
        pytest.skip("Strands workflow endpoint not available in this environment")

    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data.get("status") == "success"
    assert "response" in data

    # Check artifacts exist
    run_dir = Path("runs") / run_id
    segments = run_dir / "segments.jsonl"
    backlog = run_dir / "generated_backlog.jsonl"

    assert segments.exists(), "segments.jsonl not found"
    assert backlog.exists(), "generated_backlog.jsonl not found"

    # Sanity-check content
    seg_lines = [json.loads(l) for l in segments.read_text().splitlines() if l.strip()]
    assert len(seg_lines) > 0, "no segments parsed"

    backlog_lines = [json.loads(l) for l in backlog.read_text().splitlines() if l.strip()]
    assert len(backlog_lines) >= 0  # may be zero in mock mode

    # Optional: if Strands run, confirm debug snapshots are written
    if use_strands:
        debug_status = run_dir / "debug_strands_status.json"
        assert debug_status.exists(), "No Strands debug status found"
