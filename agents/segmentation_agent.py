"""
Segmentation Agent - Specialized agent for document segmentation with intent detection
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from strands import Agent, tool
from strands.models.openai import OpenAIModel


def create_segmentation_agent(run_id: str):
    """
    Create a segmentation agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    # Get OpenAI configuration from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")
    
    openai_client = OpenAI(api_key=api_key) if api_key else None
    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    
    @tool
    def segment_document(document_text: str) -> str:
        """
        Segments a document into coherent chunks with intent detection.
        
        Args:
            document_text: The full text of the document to segment
            
        Returns:
            JSON string containing segmentation results with segment_id, raw_text, intent_labels, and dominant_intent
        """
        
        # Build improved segmentation + semantic intent prompt
        segmentation_prompt = (
    """You are an expert in document segmentation AND fine‑grained semantic intent extraction for software/product engineering artifacts (meeting notes, transcripts, requirement docs, architecture discussions).

GOAL
1. Split the document into coherent, semantically focused segments (≈500–1000 tokens each) using topic / responsibility / problem / decision boundaries.
2. For EACH segment produce SPECIFIC, DOMAIN RICH intent labels that capture the concrete subject matter, not generic categories.
3. Return strictly valid JSON (response_format json_object) matching the schema below.

SEGMENTATION GUIDELINES
- Create a new segment when the primary subject, problem, feature area, architectural concern, or decision focus changes.
- Keep each segment self‑contained: a reader should grasp the core idea without reading neighbors.
- Preserve important technical detail (components, services, protocols, metrics, constraints).

INTENT EXTRACTION GUIDELINES (CRITICAL)
DO NOT use overly generic labels like: feature_request, bug_report, enhancement, discussion, decision, question, user_story.
INSTEAD produce 3–6 HIGH SPECIFICITY, multi‑word, lower_snake_case intent labels describing:
    - Core feature / capability (e.g. multi_factor_authentication_security_upgrade)
    - Concrete problem / issue (e.g. dashboard_search_api_latency_optimization)
    - Architectural / technical change (e.g. offline_sync_local_cache_strategy)
    - Constraint or requirement (e.g. auth_service_schema_migration_requirement)
    - User value / outcome (e.g. secure_account_access_for_users)

Intent Label Rules:
    - Each label MUST contain at least one domain entity (feature, component, service, metric, artifact) AND an action/theme (implement, optimize, migrate, enable, reduce, automate, support, improve, generate, index, cache, sync, etc.).
    - Use lower_snake_case (no spaces, no punctuation besides underscores).
    - Avoid filler words: generic_discussion, general_context, etc. (never include).
    - Be specific: "api_documentation_automation_tooling" > "documentation".
    - If a question segment exists, convert it into concrete intent labels capturing what is sought, e.g. timeline_for_auth_enhancement_delivery.

dominant_intent:
- Choose the SINGLE most representative label from intent_labels (verbatim copy of one label) – DO NOT introduce a new abstraction.

OPTIONAL CONTENT THAT STILL BELONGS IN intent_labels WHEN PRESENT
- Performance metrics (e.g. reduce_search_api_p95_latency)
- Data / schema changes (e.g. user_db_schema_index_additions)
- Security / compliance aspects (e.g. mfa_fraud_prevention_requirements)
- Technical debt remediation (e.g. api_docs_code_comment_generation_adoption)
- Decision outcomes (e.g. adopt_authenticator_app_mfa_channel) – still specific.

EXAMPLES (for style only; do NOT reuse verbatim unless applicable):
    Raw discussion about adding MFA → ["multi_factor_authentication_security_upgrade", "sms_email_totp_channel_support", "auth_service_schema_migration_requirement", "secure_account_access_for_users"]
    Performance issue dashboard search → ["dashboard_search_api_latency_optimization", "database_query_index_additions", "improve_page_load_time_dashboard", "p95_response_time_reduction_goal"]

DOCUMENT TO SEGMENT
--- START ---
""" + document_text + """
--- END ---

OUTPUT JSON SCHEMA (STRICT):
{{
    "segments": [
        {{
            "segment_id": <int starting at 1>,
            "segment_order": <same as segment_id>,
            "raw_text": "<exact segment text>",
            "intent_labels": ["lower_snake_case_specific_intent", "..."],
            "dominant_intent": "<one label copied from intent_labels>"
        }}
    ]
}}

VALIDATION REQUIREMENTS BEFORE YOU ANSWER
- Every segment has 3–6 intent_labels unless genuinely only 1–2 topics (rare). 
- No generic labels (feature_request, bug_report, enhancement, discussion, decision, question, user_story) appear anywhere.
- dominant_intent matches EXACTLY one of the intent_labels.
- JSON parses without error.

Return ONLY the JSON object with the "segments" array – no commentary."""
    )
        
        try:
            print(f"Segmentation Agent: Processing document (run_id: {run_id})")

            # Mock mode for offline / no network testing
            if os.getenv("SEGMENTATION_AGENT_MOCK") == "1":
                print("Segmentation Agent: Using MOCK mode (SEGMENTATION_AGENT_MOCK=1)")
                # Simple segmentation by double newlines
                raw_segments = [s.strip() for s in document_text.split("\n\n") if s.strip()]

                def _generate_intents(text: str):
                    lower = text.lower()
                    intents = []
                    patterns = [
                        ("multi-factor authentication", ["multi_factor_authentication_security_upgrade", "sms_email_totp_channel_support", "auth_service_schema_migration_requirement", "secure_account_access_for_users"]),
                        ("performance", ["dashboard_search_api_latency_optimization", "database_query_index_additions", "p95_response_time_reduction_goal"]),
                        ("offline mode", ["offline_sync_local_cache_strategy", "offline_document_access_user_value"]),
                        ("api documentation", ["api_docs_code_comment_generation_adoption", "developer_integration_experience_improvement", "technical_debt_documentation_modernization"]),
                        ("open questions", ["timeline_for_auth_enhancement_delivery", "budget_for_offline_mode_initiative", "ownership_api_doc_tooling_setup"]),
                    ]
                    for trigger, labels in patterns:
                        if trigger in lower:
                            intents.extend(labels)
                    # Fallback heuristic: take top keywords
                    if not intents:
                        import re
                        words = re.findall(r"[a-zA-Z_]{4,}", lower)
                        stop = {"this", "that", "with", "from", "have", "will", "need", "users", "been", "were", "also", "even", "such", "into", "then", "them", "they", "high", "mode", "docs", "api", "page"}
                        filtered = [w for w in words if w not in stop]
                        unique = []
                        for w in filtered:
                            if w not in unique:
                                unique.append(w)
                        base = unique[:4] if unique else ["general", "segment"]
                        intents.append("_".join(base) + "_topic_focus")
                    # Trim to 3–6 intents
                    if len(intents) > 6:
                        intents = intents[:6]
                    dominant = intents[0]
                    return intents, dominant

                segments = []
                for i, seg_text in enumerate(raw_segments, 1):
                    intents, dominant = _generate_intents(seg_text)
                    segments.append({
                        "segment_id": i,
                        "segment_order": i,
                        "raw_text": seg_text,
                        "intent_labels": intents,
                        "dominant_intent": dominant
                    })
                result = {"status": "success_mock", "segments": segments}
            else:
                if not openai_client:
                    raise ValueError("OpenAI client not initialized and mock mode not enabled. Set OPENAI_API_KEY or SEGMENTATION_AGENT_MOCK=1.")
                # Call LLM for segmentation
                response = openai_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": segmentation_prompt}],
                    response_format={"type": "json_object"}
                )
                # Parse response
                result_text = response.choices[0].message.content
                result = json.loads(result_text)
            
            # Validate structure
            if "segments" not in result:
                raise ValueError("Response missing 'segments' key")
            
            segments = result["segments"]
            
            # Ensure output directory exists
            output_dir = Path(f"runs/{run_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write segments to JSONL file
            segments_file = output_dir / "segments.jsonl"
            with open(segments_file, "w") as f:
                for segment in segments:
                    f.write(json.dumps(segment) + "\n")
            
            print(f"Segmentation Agent: Created {len(segments)} segments")
            
            # Prepare summary for display
            summary = {
                "status": "success",
                "run_id": run_id,
                "total_segments": len(segments),
                "segments_file": str(segments_file),
                "segments": segments
            }
            
            return json.dumps(summary, indent=2)
            
        except json.JSONDecodeError as e:
            error_msg = {
                "status": "error",
                "error": f"Failed to parse LLM response as JSON: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
        
        except Exception as e:
            error_msg = {
                "status": "error",
                "error": f"Segmentation failed: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
    
    return segment_document


# System prompt for segmentation agent (for documentation)
SEGMENTATION_AGENT_SYSTEM_PROMPT = """
You are a document segmentation specialist. Your role is to:

1. Analyze meeting notes, transcripts, and requirement documents
2. Split documents into coherent, semantically independent segments
3. Identify intents and themes within each segment
4. Maintain context while creating logical boundaries

Your output will be used for:
- High precision context retrieval (vector similarity depends on semantic richness)
- Backlog item generation (epics, features, stories)
- Intent-based processing, prioritization, and architectural alignment

INTENT QUALITY MANDATE
- Avoid generic category tags (feature_request, bug_report, enhancement, decision, discussion, question, user_story)
- Produce specific, lower_snake_case multi-word phrases embedding domain entities + action/theme
- 3–6 intents per segment (unless truly singular)
- dominant_intent must be one of intent_labels verbatim
Example style: multi_factor_authentication_security_upgrade, dashboard_search_api_latency_optimization, offline_sync_local_cache_strategy, api_docs_code_comment_generation_adoption
"""
