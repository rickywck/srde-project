"""
Tagging Agent - Generates classification (new | gap | conflict) for a generated
user story relative to existing backlog stories. Implements early-exit logic,
robust JSON parsing of LLM output, and lightweight rule-based fallbacks.

Input contract (JSON passed as string to tool):
{
  "story": {
    "title": str,
    "description": str,
    "acceptance_criteria": [str],
    "internal_id": str (optional, for tracking)
  },
  "similar_existing_stories": [
     {"work_item_id": str|int, "title": str, "description": str, "similarity": float}, ...
  ],
  "similarity_threshold": float (optional overrides default)
}

Output contract (JSON string):
{
  "status": "ok"|"error",
  "run_id": str,
  "decision_tag": "new"|"gap"|"conflict",
  "related_ids": [str|int],
  "reason": str,
  "early_exit": bool,
  "similarity_threshold": float,
  "similar_count": int,
  "model_used": str,
  "story_internal_id": str,
  "story_title": str
}
"""

import json
import os
import re
from typing import List, Dict, Any
from pathlib import Path
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from .prompt_loader import get_prompt_loader
import yaml


# Note: System prompt now loaded from prompts/tagging_agent.yaml
TAGGING_AGENT_SYSTEM_PROMPT_LEGACY = """You are a backlog tagging specialist.
Classify ONE generated user story relative to existing backlog items.
Return STRICT JSON ONLY with fields: decision_tag, related_ids, reason.

Taxonomy:
- new: No meaningful overlap; introduces novel scope.
- gap: Extends or complements existing work; fills missing acceptance criteria or edge cases.
- conflict: Duplicates (substantial overlap) OR contradicts direction/constraints of an existing story.

Heuristics guidance (you may reference internally but output must stay concise):
- Strong duplication signals: Very similar title wording AND >0.82 similarity.
- Conflict signals: Existing story implies alternative approach / mutually exclusive requirement.
- Gap signals: Existing stories partially cover scope but miss explicit ACs or user value described.

YOU MUST choose exactly one decision_tag.
related_ids should list the most relevant existing work_item_id values (empty if new).
reason should be a single short sentence.
Output JSON ONLY. No markdown, no prose outside JSON.
"""


def _safe_json_extract(text: str) -> Dict[str, Any]:
    """Attempt to extract JSON object from LLM text response."""
    # Direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass
    # Regex to find first { ... }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        snippet = match.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            return {}
    return {}


def _rule_based_fallback(story: Dict[str, Any], similar: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Apply simple deterministic rules if LLM JSON invalid."""
    if not similar:
        return {"decision_tag": "new", "related_ids": [], "reason": "No similar items"}
    # Consider only items above threshold
    considered = [s for s in similar if s.get("similarity", 0.0) >= threshold]
    if not considered:
        return {"decision_tag": "new", "related_ids": [], "reason": "None above similarity threshold"}
    # Duplication: highest similarity > 0.85 and title overlap > 70%
    top = max(considered, key=lambda s: s.get("similarity", 0.0))
    def _norm(t: str) -> List[str]:
        return [w for w in re.split(r"\W+", t.lower()) if w]
    story_title_tokens = set(_norm(story.get("title", "")))
    top_title_tokens = set(_norm(top.get("title", "")))
    overlap_ratio = len(story_title_tokens & top_title_tokens) / (len(story_title_tokens) or 1)
    if top.get("similarity", 0.0) > 0.85 and overlap_ratio >= 0.7:
        return {"decision_tag": "conflict", "related_ids": [top.get("work_item_id")], "reason": "High duplication signal"}
    # Otherwise treat as gap if at least one considered
    return {"decision_tag": "gap", "related_ids": [c.get("work_item_id") for c in considered[:3]], "reason": "Partial overlap suggests extension"}


# Note: Prompt building now handled by prompt_loader from prompts/tagging_agent.yaml


def create_tagging_agent(run_id: str, default_similarity_threshold: float = None):
    """Create a tagging agent tool for a specific run."""

    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("tagging_agent")
    params = prompt_loader.get_parameters("tagging_agent")
    
    # Prepare lightweight model instance (reads OPENAI_API_KEY env)
    # Load default model from config, allow env override
    config_path = "config.poc.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            _cfg = yaml.safe_load(f) or {}
    else:
        _cfg = {"openai": {"chat_model": "gpt-4o"}}

    if default_similarity_threshold is None:
        default_similarity_threshold = float(_cfg.get("retrieval", {}).get("tagging", {}).get("min_similarity_threshold", 0.5))

    model_id = os.getenv("OPENAI_CHAT_MODEL", _cfg.get("openai", {}).get("chat_model", "gpt-4o"))
    model = OpenAIModel(model_id=model_id, params={"temperature": params.get("temperature", 0.2), "max_tokens": params.get("max_tokens", 500)})

    @tool
    def tag_story(story_data: Any = None, story: Dict[str, Any] = None, similar_existing_stories: List[Dict[str, Any]] = None, similarity_threshold: float = None) -> str:
        """Tag a user story relative to existing backlog (new/gap/conflict)."""
        
        def _finalize(result: Dict[str, Any], internal_id: Any = None, title: str = None) -> str:
            if internal_id:
                result["story_internal_id"] = internal_id
            if title:
                result["story_title"] = title
            
            # Persist result to file
            try:
                out_dir = Path(f"runs/{run_id}")
                out_dir.mkdir(parents=True, exist_ok=True)
                tag_file = out_dir / "tagging.jsonl"
                with open(tag_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
            except Exception:
                pass # Silently fail on logging errors to preserve flow
                
            return json.dumps(result)

        # Accept both legacy JSON string and structured arguments
        payload: Dict[str, Any] = {}
        if story_data is not None and story is None and similar_existing_stories is None:
            if isinstance(story_data, (dict, list)):
                payload = story_data if isinstance(story_data, dict) else {}
            else:
                try:
                    payload = json.loads(story_data)
                except Exception as e:
                    return _finalize({
                        "status": "error",
                        "run_id": run_id,
                        "decision_tag": "new",
                        "related_ids": [],
                        "reason": f"Invalid input JSON: {e}",
                        "early_exit": True,
                        "similarity_threshold": default_similarity_threshold,
                        "similar_count": 0,
                        "model_used": model_id
                    })
        else:
            payload = {
                "story": story or {},
                "similar_existing_stories": similar_existing_stories or [],
                "similarity_threshold": similarity_threshold if similarity_threshold is not None else default_similarity_threshold,
            }

        story = payload.get("story", {})
        similar = payload.get("similar_existing_stories", [])
        threshold = float(payload.get("similarity_threshold", default_similarity_threshold))
        internal_id = story.get("internal_id")
        title = story.get("title")

        # Log story title and description for troubleshooting
        print(f"Tagging Agent: Processing story title='{title}' | description='{story.get('description', '')[:120]}…'")

        # Early exit if no similar above threshold
        above_threshold = [s for s in similar if s.get("similarity", 0.0) >= threshold]
        if not above_threshold:
            return _finalize({
                "status": "ok",
                "run_id": run_id,
                "decision_tag": "new",
                "related_ids": [],
                "reason": "No similar existing stories found (all below threshold)",
                "early_exit": True,
                "similarity_threshold": threshold,
                "similar_count": 0,
                "model_used": model_id
            }, internal_id, title)

        # Build prompt for LLM using template
        ac_list = story.get("acceptance_criteria", []) or []
        ac_text = "\n- " + "\n- ".join(ac_list) if ac_list else " (none)"
        
        similar_lines = []
        for s in above_threshold:
            similar_lines.append(
                f"ID: {s.get('work_item_id')} | similarity: {round(s.get('similarity', 0.0), 4)}\nTitle: {s.get('title')}\nDesc: {s.get('description','')[:300]}"
            )
        similar_formatted = "\n\n".join(similar_lines)
        
        user_prompt = prompt_loader.format_user_prompt(
            "tagging_agent",
            story_title=story.get("title"),
            story_description=story.get("description"),
            story_acceptance_criteria=ac_text,
            similarity_threshold=threshold,
            similar_stories_formatted=similar_formatted
        )

        agent = Agent(model=model, system_prompt=system_prompt, tools=[], callback_handler=None)
        llm_response = agent(user_prompt)
        llm_text = str(llm_response)

        parsed = _safe_json_extract(llm_text)
        if not parsed or not isinstance(parsed, dict) or "decision_tag" not in parsed:
            # Fallback to rule-based
            fallback = _rule_based_fallback(story, above_threshold, threshold)
            return _finalize({
                "status": "ok",
                "run_id": run_id,
                "decision_tag": fallback.get("decision_tag", "new"),
                "related_ids": fallback.get("related_ids", []),
                "reason": fallback.get("reason", "Fallback applied"),
                "early_exit": False,
                "similarity_threshold": threshold,
                "similar_count": len(above_threshold),
                "model_used": model_id,
                "fallback_used": True
            }, internal_id, title)

        # Clean / normalize parsed output
        decision = parsed.get("decision_tag", "new").lower()
        if decision not in {"new", "gap", "conflict"}:
            decision = "new"
        related_ids = parsed.get("related_ids", [])
        if not isinstance(related_ids, list):
            related_ids = []
        reason = parsed.get("reason", "")[:200]

        return _finalize({
            "status": "ok",
            "run_id": run_id,
            "decision_tag": decision,
            "related_ids": related_ids,
            "reason": reason,
            "early_exit": False,
            "similarity_threshold": threshold,
            "similar_count": len(above_threshold),
            "model_used": model_id,
            "fallback_used": False
        }, internal_id, title)

    return tag_story

