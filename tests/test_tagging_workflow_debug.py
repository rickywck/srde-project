import os
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from agents.tagging_agent import create_tagging_agent


def _find_latest_backlog_jsonl(root: Path) -> Optional[Path]:
    runs_dir = root / "runs"
    if not runs_dir.exists():
        return None
    candidates: List[Path] = list(runs_dir.glob("**/generated_backlog.jsonl"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_config(root: Path) -> Dict[str, Any]:
    cfg_path = root / "config.poc.yaml"
    if cfg_path.exists():
        try:
            import yaml  # type: ignore
            with cfg_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def _print_similar(segment_title: str, similar: List[Dict[str, Any]], threshold: float) -> None:
    print("\n" + "=" * 80)
    print(f"Story: {segment_title}")
    print("Similar Existing Stories (filtered to type includes 'story'):")
    if not similar:
        print("  - None found")
        return
    above = [s for s in similar if s.get("similarity", 0.0) >= threshold]
    print(f"  Total: {len(similar)} | Above threshold {threshold}: {len(above)}")
    for i, s in enumerate(similar[:10], 1):
        wid = s.get("work_item_id")
        sim = round(float(s.get("similarity", 0.0)), 4)
        title = s.get("title", "")
        desc = s.get("description", "")
        desc_snip = (desc[:200] + "…") if isinstance(desc, str) and len(desc) > 200 else desc
        print(f"  {i}. [{wid}] sim={sim} | {title}")
        if desc_snip:
            print(f"     - {desc_snip}")


def _get_clients(cfg: Dict[str, Any]):
    index_name = cfg.get("pinecone", {}).get("index_name", "rde-lab")
    namespace = (cfg.get("pinecone", {}).get("project") or "").strip()
    try:
        from openai import OpenAI  # type: ignore
        from pinecone import Pinecone  # type: ignore
    except Exception as e:
        pytest.skip(f"Missing client libraries: {e}")
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        pytest.skip("Requires OPENAI_API_KEY and PINECONE_API_KEY to run embedding + vector search")
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pinecone_client.Index(index_name)
    return openai_client, index, namespace


def _load_backlog_items(backlog_path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with backlog_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _normalize_matches_to_similar(matches: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in matches or []:
        if isinstance(m, dict):
            score = float(m.get("score", 0.0))
            md = m.get("metadata", {}) or {}
            mid = m.get("id")
        else:
            score = float(getattr(m, "score", 0.0))
            md = getattr(m, "metadata", {}) or {}
            mid = getattr(m, "id", None)
        item_type = (md.get("type") or md.get("work_item_type") or "").lower()
        if "story" in item_type:
            out.append(
                {
                    "work_item_id": md.get("work_item_id") or mid,
                    "title": md.get("title", ""),
                    "description": (md.get("description", "") or "")[:500],
                    "similarity": score,
                }
            )
    return out


def _retrieve_similar_for_story(
    openai_client,
    index,
    namespace: str,
    embedding_model: str,
    embedding_dimensions: int,
    threshold: float,
    story: Dict[str, Any],
) -> List[Dict[str, Any]]:
    ac = story.get("acceptance_criteria", []) or []
    story_text = (
        (story.get("title", "") or "")
        + "\n"
        + (story.get("description", "") or "")
        + "\n"
        + "\n".join(ac)
    )
    emb = openai_client.embeddings.create(
        model=embedding_model,
        input=story_text[:3000],
        dimensions=embedding_dimensions,
    )
    vec = emb.data[0].embedding
    query_kwargs = {
        "vector": vec,
        "top_k": 10,
        "filter": {"doc_type": "ado_backlog"},
        "include_metadata": True,
    }
    if namespace:
        query_kwargs["namespace"] = namespace
    query_res = index.query(**query_kwargs)
    matches = query_res.get("matches", []) if isinstance(query_res, dict) else getattr(query_res, "matches", [])
    return _normalize_matches_to_similar(matches)


def _print_tag_result(result: Dict[str, Any]) -> None:
    print("\nTagging Result:")
    try:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception:
        print(result)


def _is_user_story(item: Dict[str, Any]) -> bool:
    return str(item.get("type", "")).lower() == "user story"


def _get_threshold(cfg: Dict[str, Any]) -> float:
    return float(cfg.get("retrieval", {}).get("min_similarity_threshold", 0.5))


@pytest.mark.skipif(False, reason="enabled")
def test_tagging_workflow_debug():
    project_root = Path(__file__).parent.parent
    backlog_path = _find_latest_backlog_jsonl(project_root)
    if not backlog_path or not backlog_path.exists():
        pytest.skip("No generated_backlog.jsonl found under runs/**")

    cfg = _load_config(project_root)
    min_similarity = _get_threshold(cfg)
    embedding_model = cfg.get("openai", {}).get("embedding_model", "text-embedding-3-small")
    embedding_dimensions = int(cfg.get("openai", {}).get("embedding_dimensions", 512))
    openai_client, index, namespace = _get_clients(cfg)

    backlog_items = _load_backlog_items(backlog_path)
    stories = [i for i in backlog_items if _is_user_story(i)]
    if not stories:
        pytest.skip("No user stories found in latest generated_backlog.jsonl")

    run_id = str(uuid.uuid4())
    tag_tool = create_tagging_agent(run_id)

    max_cases = int(os.getenv("TAGGING_DEBUG_MAX_STORIES", "5"))
    stories = stories[:max_cases]

    for story in stories:
        similar_stories = _retrieve_similar_for_story(
            openai_client,
            index,
            namespace,
            embedding_model,
            embedding_dimensions,
            min_similarity,
            story,
        )
        _print_similar(story.get("title", "(untitled)"), similar_stories, float(min_similarity))

        payload = {
            "story": {
                "title": story.get("title"),
                "description": story.get("description"),
                "acceptance_criteria": story.get("acceptance_criteria", []),
                "internal_id": story.get("internal_id"),
            },
            "similar_existing_stories": similar_stories,
            "similarity_threshold": float(min_similarity),
        }

        raw = tag_tool(json.dumps(payload))
        try:
            result = json.loads(raw)
        except Exception:
            result = raw if isinstance(raw, dict) else {"status": "error", "raw": str(raw)}

        _print_tag_result(result)

        assert isinstance(result, dict)
        assert "decision_tag" in result
        assert result.get("status") in {"ok", "error"}
