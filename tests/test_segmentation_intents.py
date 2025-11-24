"""Intent quality validation for segmentation agent (mock mode supported).

Run with SEGMENTATION_AGENT_MOCK=1 to avoid LLM calls:
  export SEGMENTATION_AGENT_MOCK=1
  python -m pytest tests/test_segmentation_intents.py -q
"""

import os
import json
import uuid
from pathlib import Path

from supervisor import SupervisorAgent


SAMPLE_DOC = """
Product Planning Meeting - Q1 2024

Discussion Topic 1: User Authentication Enhancement
The team discussed adding multi-factor authentication to improve security. 
Users have been requesting this feature for enhanced account protection.
We need to support SMS, email, and authenticator app options.
Technical consideration: This will require changes to our auth service and user database schema.

Discussion Topic 2: Performance Issues
Several customers reported slow page load times on the dashboard.
Our monitoring shows the search API is taking 3-5 seconds on average.
Root cause analysis points to inefficient database queries and missing indexes.
This needs to be prioritized as a P1 bug fix.

Discussion Topic 3: Mobile App Feature Request
Product team presented findings from user research showing high demand for offline mode.
Users want to access their documents even without internet connectivity.
This would be a major feature requiring significant architecture changes.
We should create an epic to track this work with multiple phases.

Discussion Topic 4: API Documentation
Developers are struggling with our API documentation being outdated.
We need to implement automatic API doc generation from code comments.
This is a technical debt item that's blocking external partner integrations.

Open Questions:
- What's the timeline for the authentication enhancement?
- Do we have budget for the mobile offline feature?
- Who will own the documentation tooling setup?
"""

GENERIC_BANNED = {
    "feature_request", "bug_report", "enhancement", "discussion", "decision", "question", "user_story"
}


def _validate_intent_labels(intents):
    assert isinstance(intents, list), "intent_labels must be a list"
    assert 1 <= len(intents) <= 6, "intent_labels must have between 1 and 6 items"
    for label in intents:
        assert label == label.lower(), f"Intent label '{label}' must be lowercase"
        assert " " not in label, f"Intent label '{label}' must use underscores, no spaces"
        assert label not in GENERIC_BANNED, f"Generic label '{label}' is banned"
        assert len(label) >= 8, f"Intent label '{label}' too short / likely generic"
        assert any(ch == '_' for ch in label), f"Intent label '{label}' must be multi-word (use underscores)"


def test_segmentation_intent_quality():
    # Force mock mode to ensure deterministic offline behavior
    os.environ.setdefault("SEGMENTATION_AGENT_MOCK", "1")

    supervisor = SupervisorAgent()
    run_id = str(uuid.uuid4())
    result = os.getenv("PYTEST_SUPPRESS_SEGMENTATION") or supervisor.segment_document

    # supervisor.segment_document is async; create a lightweight wrapper if needed
    import asyncio
    segments_result = asyncio.run(supervisor.segment_document(run_id, SAMPLE_DOC))

    assert segments_result["status"] == "success", f"Segmentation did not succeed: {segments_result}" 
    segments = segments_result["segments"]
    assert segments, "No segments returned"

    for seg in segments:
        intents = seg.get("intent_labels", [])
        _validate_intent_labels(intents)
        dominant = seg.get("dominant_intent")
        assert dominant in intents, "dominant_intent must be one of intent_labels"

    # Ensure banned labels absent globally
    all_intents = {i for seg in segments for i in seg.get("intent_labels", [])}
    assert not (all_intents & GENERIC_BANNED), f"Banned generic intents present: {all_intents & GENERIC_BANNED}"

    # Persist for manual inspection
    out_path = Path(f"runs/{run_id}/segments_intent_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"segments": segments}, f, indent=2)

    print(f"Intent quality test artifacts written to {out_path}")
