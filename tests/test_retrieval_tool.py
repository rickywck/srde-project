import json
import uuid
from typing import Any, Dict, List

from tools.retrieval_tool import create_retrieval_tool


def _print_retrieved(segment_id: int, result: Dict[str, Any]) -> None:
    summary = result.get("retrieval_summary", {}) or {}
    ado_items: List[Dict[str, Any]] = result.get("ado_items", []) or []
    arch_items: List[Dict[str, Any]] = result.get("architecture_constraints", []) or []

    print("\n" + "=" * 80)
    print(f"Segment {segment_id} - Retrieval Summary")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if ado_items:
        print("\nADO Items:")
        for i, item in enumerate(ado_items, 1):
            title = item.get("title") or item.get("name") or item.get("work_item_title") or "Untitled"
            wid = item.get("id") or item.get("work_item_id") or item.get("key")
            desc = item.get("description") or item.get("content") or item.get("text") or ""
            desc_snippet = (desc[:200] + "…") if isinstance(desc, str) and len(desc) > 200 else desc
            print(f"  {i}. [{wid}] {title}")
            if desc_snippet:
                print(f"     - {desc_snippet}")

    if arch_items:
        print("\nArchitecture Constraints:")
        for i, item in enumerate(arch_items, 1):
            name = item.get("name") or item.get("title") or "Constraint"
            cid = item.get("id") or item.get("key")
            body = item.get("content") or item.get("description") or item.get("text") or ""
            body_snippet = (body[:200] + "…") if isinstance(body, str) and len(body) > 200 else body
            print(f"  {i}. [{cid}] {name}")
            if body_snippet:
                print(f"     - {body_snippet}")


def test_retrieval_tool_prints_results(segments):
    # Fresh run_id for isolation; retrieval tool may persist per-run outputs.
    run_id = str(uuid.uuid4())
    tool = create_retrieval_tool(run_id)

    for seg in segments:
        payload = {
            "segment_text": seg.get("raw_text", ""),
            "intent_labels": seg.get("intent_labels", []),
            "dominant_intent": seg.get("dominant_intent", ""),
            "segment_id": seg.get("segment_id"),
        }
        result_raw = tool(json.dumps(payload))
        try:
            result = json.loads(result_raw)
        except Exception:
            # If tool already returns a dict, accept it; otherwise fail.
            if isinstance(result_raw, dict):
                result = result_raw
            else:
                raise

        # Print retrieved content for this segment
        _print_retrieved(seg.get("segment_id"), result)

        # Minimal sanity checks (kept loose to avoid flakiness)
        assert isinstance(result, dict)
        assert "retrieval_summary" in result
