import json
import uuid
from pathlib import Path

import pytest

from agents.backlog_generation_agent import create_backlog_generation_agent


def _long_text(n: int) -> str:
    return ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. ") * ((n // 56) + 1)


def test_backlog_generation_with_synthetic_context(monkeypatch, tmp_path):
    # Force mock mode to avoid real LLM calls
    #monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    run_id = str(uuid.uuid4())
    tool = create_backlog_generation_agent(run_id)

    # Build synthetic retrieved context with many items and long text to exercise caps
    ado_items = []
    for i in range(10):
        ado_items.append({
            "id": f"ado_{i}",
            "score": 0.9 - i * 0.05,
            "type": "Story" if i % 2 else "Feature",
            "title": f"Synthetic ADO Item {i}",
            "description": _long_text(1200),
            "work_item_id": 1000 + i,
        })

    arch_items = []
    for i in range(10):
        arch_items.append({
            "id": f"arch_{i}",
            "score": 0.88 - i * 0.04,
            "source": f"doc_{i}.md",
            "section": f"Section {i}",
            "text": _long_text(2000),
        })

    payload = {
        "segment_id": 42,
        "segment_text": "User wants a merchant portal with multilingual support and fraud alerts.",
        "intent_labels": ["merchant_portal", "multilingual_support", "fraud_alerts"],
        "dominant_intent": "merchant_portal",
        "retrieved_context": {
            "ado_items": ado_items,
            "architecture_constraints": arch_items,
        },
    }

    raw = tool(
        segment_id=payload["segment_id"],
        segment_text=payload["segment_text"],
        intent_labels=payload["intent_labels"],
        dominant_intent=payload["dominant_intent"],
        retrieved_context=payload["retrieved_context"],
    )
    res = json.loads(raw)

    assert res.get("status") in ("success", "success_mock")
    items = res.get("backlog_items") or res.get("backlog_items", [])
    # In mock mode, backlog_items is included in the response summary
    assert isinstance(items, list)
    assert len(items) > 0

    # Verify file output exists and has content
    out_file = Path(f"runs/{run_id}/generated_backlog.jsonl")
    assert out_file.exists()
    with out_file.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) > 0
