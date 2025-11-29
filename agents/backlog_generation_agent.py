"""
Backlog Generation Agent - Specialized agent for generating backlog items from segments
"""

import os
import json
import yaml
from typing import Dict, Any, List, Union
from pathlib import Path
from openai import OpenAI
from strands import tool
from .prompt_loader import get_prompt_loader
from tools.token_utils import estimate_tokens


def create_backlog_generation_agent(run_id: str):
    """
    Create a backlog generation agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    # Get OpenAI configuration
    api_key = os.getenv("OPENAI_API_KEY")
    # Load defaults from config, allow env override
    config_path = "config.poc.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            _cfg = yaml.safe_load(f) or {}
    else:
        _cfg = {"openai": {"chat_model": "gpt-4.1-mini"}}
    model_name = os.getenv("OPENAI_CHAT_MODEL", _cfg.get("openai", {}).get("chat_model", "gpt-4.1-mini"))

    # Generation limits from config
    gen_cfg = _cfg.get("generation", {}) if isinstance(_cfg, dict) else {}
    def _as_int(v, d):
        try:
            i = int(v)
            return i if i > 0 else d
        except Exception:
            return d
    MAX_ADO = _as_int(gen_cfg.get("max_ado_in_prompt", 6), 6)
    MAX_ARCH = _as_int(gen_cfg.get("max_arch_in_prompt", 6), 6)
    ADO_DESC_LEN = _as_int(gen_cfg.get("ado_desc_len", 400), 400)
    ARCH_TEXT_LEN = _as_int(gen_cfg.get("arch_text_len", 600), 600)
    # Prefer new Responses-style token key; fall back to legacy if present
    CFG_MAX_TOKENS = _as_int(
        gen_cfg.get("max_completion_tokens", gen_cfg.get("max_tokens", 1500)),
        1500,
    )
    
    openai_client = OpenAI(api_key=api_key) if api_key else None
    
    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("backlog_generation_agent")
    params = prompt_loader.get_parameters("backlog_generation_agent")
    
    def _safe_json_extract(text: str) -> Dict[str, Any]:
        """Attempt to parse JSON, falling back to extracting first {...} block."""
        if text is None:
            return {}
        if isinstance(text, (dict, list)):
            return text  # already parsed
        try:
            return json.loads(text)
        except Exception:
            pass
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

    @tool
    def generate_backlog(
        segment_data: Union[str, Dict[str, Any]] = None,
        segment_id: int = None,
        segment_text: str = None,
        intent_labels: List[str] = None,
        dominant_intent: str = None,
        retrieved_context: Dict[str, Any] = None,
    ) -> str:
        """
        Generate backlog items (epics, features, stories) from a segment with retrieved context.
        
        Args:
            segment_data: JSON string containing:
                - segment_id: The segment identifier
                - segment_text: The original segment text
                - intent_labels: List of intent labels
                - dominant_intent: The dominant intent
                - retrieved_context: Retrieved ADO items and architecture constraints
            
        Returns:
            JSON string containing generated backlog items (epics, features, stories)
        """
        
        try:
            # Parse input (support structured tool calls and legacy JSON string)
            if segment_data is not None and (segment_id is None and segment_text is None):
                data = _safe_json_extract(segment_data)
                segment_id = data.get("segment_id", 0)
                segment_text = data.get("segment_text", "")
                intent_labels = data.get("intent_labels", [])
                dominant_intent = data.get("dominant_intent", "")
                retrieved_context = data.get("retrieved_context", {})
            else:
                # Structured args path
                segment_id = segment_id or 0
                segment_text = segment_text or ""
                intent_labels = intent_labels or []
                dominant_intent = dominant_intent or ""
                retrieved_context = retrieved_context or {}
            
            print(f"Backlog Generation Agent: Processing segment {segment_id} (run_id: {run_id})")
            
            # Build generation prompt from template
            ado_items = retrieved_context.get("ado_items", []) or []
            arch_constraints = retrieved_context.get("architecture_constraints", []) or []

            def _safe_score(x):
                try:
                    return float(x)
                except Exception:
                    return 0.0

            # Sort by score desc when available
            if ado_items:
                ado_items = sorted(ado_items, key=lambda i: _safe_score(i.get("score")), reverse=True)[:MAX_ADO]
            if arch_constraints:
                arch_constraints = sorted(arch_constraints, key=lambda i: _safe_score(i.get("score")), reverse=True)[:MAX_ARCH]
            
            # Format ADO items
            ado_formatted = "No relevant existing ADO items found.\n" if not ado_items else ""
            for item in ado_items:
                try:
                    score_val = _safe_score(item.get('score', 0))
                    ado_formatted += f"\n## {item.get('type', 'Item')} (ID: {item.get('work_item_id', 'N/A')}, Similarity: {score_val:.2f})\n"
                    ado_formatted += f"**Title:** {item.get('title', 'Untitled')}\n"
                    desc = item.get('description', '') or ''
                    if len(desc) > ADO_DESC_LEN:
                        desc = desc[:ADO_DESC_LEN] + "…"
                    ado_formatted += f"**Description:** {desc or 'No description'}\n"
                except Exception:
                    # Skip any malformed item
                    continue
            
            # Format architecture constraints
            arch_formatted = "No relevant architecture constraints found.\n" if not arch_constraints else ""
            for constraint in arch_constraints:
                try:
                    score_val = _safe_score(constraint.get('score', 0))
                    arch_formatted += f"\n## From {constraint.get('source', 'Unknown')} - {constraint.get('section', '')} (Similarity: {score_val:.2f})\n"
                    textv = constraint.get('text', '') or ''
                    if len(textv) > ARCH_TEXT_LEN:
                        textv = textv[:ARCH_TEXT_LEN] + "…"
                    arch_formatted += f"{textv or 'No text'}\n"
                except Exception:
                    continue
            
            prompt = prompt_loader.format_user_prompt(
                "backlog_generation_agent",
                segment_text=segment_text,
                intent_labels=", ".join(intent_labels),
                dominant_intent=dominant_intent,
                ado_items_formatted=ado_formatted,
                architecture_constraints_formatted=arch_formatted
            )

            # Approx token counts for debugging
            sys_tok = estimate_tokens(system_prompt)
            usr_tok = estimate_tokens(prompt)
            approx_total = sys_tok + usr_tok
            print(f"Backlog Generation Agent: tokens approx — system={sys_tok}, user={usr_tok}, total≈{approx_total}")
            
            # Check if we have OpenAI client
            if not openai_client:
                print("Backlog Generation Agent: Using MOCK mode (missing OPENAI_API_KEY)")
                return _mock_generation(segment_id, segment_text, intent_labels, run_id)
            
            print("Backlog Generation Agent: Calling LLM to generate backlog items...")

            # Call LLM with resilience to model/format issues
            # Respect prompt parameter defaults, cap by config if provided.
            # Map across possible config keys to the modern "max_completion_tokens".
            eff_max_tokens = (
                min(params.get("max_completion_tokens", params.get("max_tokens", 2000)), CFG_MAX_TOKENS)
                if CFG_MAX_TOKENS else params.get("max_completion_tokens", params.get("max_tokens", 2000))
            )

            def _try_llm_call(use_response_format: bool, mdl: str):
                return openai_client.chat.completions.create(
                    model=mdl,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=params.get("temperature", 0.7),
                    max_tokens=eff_max_tokens,
                    **({"response_format": {"type": params.get("response_format", "json_object")}} if use_response_format else {})
                )

            response = None
            last_err = None
            # Disable response_format for models that commonly reject it
            prefer_no_format = any(k in (model_name or "").lower() for k in ["gpt-4.1", "gpt-5"]) 
            for attempt, (use_fmt, mdl) in enumerate([
                (not prefer_no_format, model_name),
                (False, model_name),
                (False, "gpt-4o-mini"),
            ], start=1):
                try:
                    response = _try_llm_call(use_fmt, mdl)
                    if mdl != model_name:
                        print(f"Backlog Generation Agent: Fallback model in use: {mdl}")
                    break
                except Exception as e:
                    last_err = e
                    print(f"Backlog Generation Agent: LLM call attempt {attempt} failed: {e}")
                    continue

            if response is None:
                raise RuntimeError(f"LLM call failed after retries: {last_err}")

            # Parse response
            result_text = response.choices[0].message.content if hasattr(response.choices[0].message, "content") else str(response)
            # First try strict JSON, then fall back to best-effort extraction
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                result = _safe_json_extract(result_text)
            
            # Validate structure
            if not isinstance(result, dict) or "backlog_items" not in result:
                # If we still didn't get a valid structure, fall back to mock path
                raise json.JSONDecodeError("Missing 'backlog_items' key in response", result_text or "", 0)
            
            backlog_items = result["backlog_items"]
            
            # Assign internal IDs
            item_counter = {"epic": 1, "feature": 1, "story": 1}
            for item in backlog_items:
                # Normalize type and keep counters consistent
                orig_type = str(item.get("type", "story")).strip().lower()
                if orig_type in ("story", "user story", "user_story", "user-story"):
                    norm_key = "story"
                elif orig_type in ("feature", "features"):
                    norm_key = "feature"
                elif orig_type in ("epic", "epics"):
                    norm_key = "epic"
                else:
                    norm_key = orig_type

                if norm_key == "story":
                    display_type = "User Story"
                elif norm_key == "feature":
                    display_type = "Feature"
                elif norm_key == "epic":
                    display_type = "Epic"
                else:
                    display_type = item.get("type", "Story")
                item["type"] = display_type
                if norm_key in item_counter:
                    item["internal_id"] = f"{norm_key}_{segment_id}_{item_counter[norm_key]}"
                    item_counter[norm_key] += 1
                item["segment_id"] = segment_id
                item["run_id"] = run_id
            
            # Ensure output directory exists
            output_dir = Path(f"runs/{run_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Append to generated_backlog.jsonl
            backlog_file = output_dir / "generated_backlog.jsonl"
            with open(backlog_file, "a") as f:
                for item in backlog_items:
                    f.write(json.dumps(item) + "\n")
            
            print(f"Backlog Generation Agent: Generated {len(backlog_items)} backlog items")
            
            # Prepare summary
            summary = {
                "status": "success",
                "run_id": run_id,
                "segment_id": segment_id,
                "items_generated": len(backlog_items),
                "backlog_file": str(backlog_file),
                "item_counts": {
                    "epics": sum(1 for item in backlog_items if str(item.get("type", "")).lower() in ("epic", "epics")),
                    "features": sum(1 for item in backlog_items if str(item.get("type", "")).lower() in ("feature", "features")),
                    "stories": sum(1 for item in backlog_items if str(item.get("type", "")).lower() in ("story", "user story"))
                },
                "backlog_items": backlog_items
            }
            
            return json.dumps(summary, indent=2)
            
        except json.JSONDecodeError as e:
            # Fallback: use mock generation if LLM JSON is truncated or invalid
            print(f"Backlog Generation Agent: JSON parse failed, using fallback. Reason: {str(e)}")
            return _mock_generation(segment_id, segment_text, intent_labels, run_id)
        
        except Exception as e:
            error_msg = {
                "status": "error",
                "error": f"Backlog generation failed: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
    
    return generate_backlog


# Note: Prompt building now handled by prompt_loader from prompts/backlog_generation_agent.yaml


def _mock_generation(segment_id: int, segment_text: str, intent_labels: list, run_id: str) -> str:
    """Generate mock backlog items for testing"""
    
    # Simple heuristic-based generation
    mock_items = []
    
    # Check intents to decide what to generate
    has_auth = any("auth" in label.lower() for label in intent_labels)
    has_performance = any("performance" in label.lower() or "latency" in label.lower() or "optimize" in label.lower() for label in intent_labels)
    has_offline = any("offline" in label.lower() for label in intent_labels)
    
    if has_auth:
        mock_items.append({
            "type": "Feature",
            "title": "Multi-Factor Authentication Implementation",
            "description": f"Implement multi-factor authentication based on requirements identified in segment analysis. {segment_text[:100]}...",
            "acceptance_criteria": [],
            "parent_reference": "Security & Authentication Improvements Epic",
            "rationale": "Addresses authentication security requirements identified in segment",
            "internal_id": f"feature_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
        mock_items.append({
            "type": "Story",
            "title": "As a user, I want to enable MFA with authenticator app",
            "description": "Allow users to enable multi-factor authentication using TOTP authenticator apps",
            "acceptance_criteria": [
                "User can scan QR code to add account to authenticator app",
                "User must enter verification code to complete MFA setup",
                "User is prompted for MFA code on subsequent logins",
                "User can generate backup codes for account recovery"
            ],
            "parent_reference": "Multi-Factor Authentication Implementation Feature",
            "rationale": "Provides secure MFA option using industry-standard TOTP protocol",
            "internal_id": f"story_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    if has_performance:
        mock_items.append({
            "type": "Story",
            "title": "As a developer, I want optimized database queries with proper indexes",
            "description": "Add database indexes and optimize slow queries identified in performance analysis",
            "acceptance_criteria": [
                "Identify top 10 slowest queries using query analyzer",
                "Add appropriate indexes to relevant tables",
                "Query response time improves by at least 50%",
                "95th percentile API response time is under 200ms"
            ],
            "parent_reference": "API Performance Optimization Feature",
            "rationale": "Addresses performance issues and latency concerns identified in segment",
            "internal_id": f"story_{segment_id}_2",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    if has_offline:
        mock_items.append({
            "type": "Epic",
            "title": "Mobile Offline Mode Support",
            "description": "Enable users to access and work with documents without internet connectivity",
            "acceptance_criteria": [],
            "parent_reference": "",
            "rationale": "Major architectural initiative identified from user research in segment",
            "internal_id": f"epic_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    # Default story if no specific intents matched
    if not mock_items:
        mock_items.append({
            "type": "Story",
            "title": f"Implement requirements from segment {segment_id}",
            "description": f"Address requirements identified in segment: {segment_text[:200]}...",
            "acceptance_criteria": [
                "Requirements are clearly defined",
                "Implementation meets acceptance criteria",
                "Changes are tested and reviewed"
            ],
            "parent_reference": "",
            "rationale": "Generated from segment analysis",
            "internal_id": f"story_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    # Save to file
    output_dir = Path(f"runs/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    backlog_file = output_dir / "generated_backlog.jsonl"
    
    with open(backlog_file, "a") as f:
        for item in mock_items:
            f.write(json.dumps(item) + "\n")
    
    summary = {
        "status": "success_mock",
        "run_id": run_id,
        "segment_id": segment_id,
        "items_generated": len(mock_items),
        "backlog_file": str(backlog_file),
        "note": "Mock data - set OPENAI_API_KEY for real generation",
        "item_counts": {
            "epics": sum(1 for item in mock_items if item.get("type", "").lower() == "epic"),
            "features": sum(1 for item in mock_items if item.get("type", "").lower() == "feature"),
            "stories": sum(1 for item in mock_items if item.get("type", "").lower() == "story")
        },
        "backlog_items": mock_items
    }
    
    return json.dumps(summary, indent=2)

# Note: System prompt and user prompt template now loaded from prompts/backlog_generation_agent.yaml
