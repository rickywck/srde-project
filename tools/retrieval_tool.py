"""
Retrieval Tool - Tool for querying Pinecone for relevant context
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, List
from openai import OpenAI
from pinecone import Pinecone
from strands import tool

# Module logger
logger = logging.getLogger(__name__)


def create_retrieval_tool(run_id: str):
    """
    Create a retrieval tool for a specific run.
    
    Args:
        run_id: The run identifier for tracking
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    # Load configuration
    config_path = "config.poc.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "openai": {"embedding_model": "text-embedding-3-small"},
            "pinecone": {"index_name": "rde-lab"},
            "retrieval": {"min_similarity_threshold": 0.5}
        }
    
    # Initialize clients
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
    pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
    
    embedding_model = config.get("openai", {}).get("embedding_model", "text-embedding-3-small")
    # Sanitize numeric config values
    def _as_int(v, default):
        try:
            i = int(v)
            return i if i > 0 else default
        except Exception:
            return default
    def _as_float(v, default):
        try:
            return float(v)
        except Exception:
            return default
    embedding_dimensions = _as_int(config.get("openai", {}).get("embedding_dimensions", 512), 512)
    index_name = config.get("pinecone", {}).get("index_name", "rde-lab")
    namespace = (config.get("pinecone", {}).get("project") or "").strip()
    # Per-source retrieval settings with back-compat fallback
    ado_top_k = _as_int(
        config.get("retrieval", {}).get("ado", {}).get("top_k", 10), 10
    )
    arch_top_k = _as_int(
        config.get("retrieval", {}).get("architecture", {}).get("top_k", 5), 5
    )
    ado_min_similarity = _as_float(
        config.get("retrieval", {}).get("ado", {}).get(
            "min_similarity_threshold",
            config.get("retrieval", {}).get("min_similarity_threshold", 0.5)
        ), 0.5
    )
    arch_min_similarity = _as_float(
        config.get("retrieval", {}).get("architecture", {}).get(
            "min_similarity_threshold",
            config.get("retrieval", {}).get("min_similarity_threshold", 0.5)
        ), 0.5
    )
    
    # Legacy JSON input extraction removed; callers provide structured args

    @tool
    def retrieve_context(
        segment_text: str = None,
        intent_labels: List[str] = None,
        dominant_intent: str = None,
        segment_id: int = None,
    ) -> str:
        """
        Retrieve relevant context from Pinecone (ADO backlog items and architecture constraints).

        Args:
            segment_text: The segment text to embed and query with
            intent_labels: List of intent labels
            dominant_intent: The dominant intent
            segment_id: The segment identifier

        Returns:
            JSON string containing retrieved ADO items and architecture constraints
        """
        logger.debug("retrieve_context called with: segment_text=%s..., intent_labels=%r, dominant_intent=%r, segment_id=%r, run_id=%r",
                 segment_text[:100] if segment_text else None, intent_labels, dominant_intent, segment_id, run_id)
        
        try:
            # Normalize structured inputs
            segment_text = segment_text or ""
            intent_labels = intent_labels or []
            dominant_intent = dominant_intent or ""
            segment_id = int(segment_id or 0)
            
            print(f"Retrieval Tool: Processing segment {segment_id} (run_id: {run_id})")
            
            # Check if we have required clients
            if not openai_client or not pc:
                print("Retrieval Tool: Using MOCK mode (missing API keys)")
                return _mock_retrieval(segment_text, intent_labels, dominant_intent, segment_id)
            
            # Build intent query string
            # Combine dominant intent + intent labels + first ~300 chars of segment text
            intent_query_parts = [dominant_intent] + intent_labels
            intent_query = " ".join(intent_query_parts) + " " + segment_text[:300]
            
            print("Retrieval Tool: Generating embedding for intent query...")
            
            # Generate embedding
            embedding_response = openai_client.embeddings.create(
                model=embedding_model,
                input=intent_query,
                dimensions=embedding_dimensions
            )
            query_vector = embedding_response.data[0].embedding
            
            index = pc.Index(index_name)

            def _coerce_matches(resp):
                if hasattr(resp, "matches"):
                    return resp.matches or []
                if isinstance(resp, dict):
                    return resp.get("matches", []) or []
                return []

            def _coerce_fields(m):
                if isinstance(m, dict):
                    return m.get("id"), m.get("score", 0.0), m.get("metadata", {}) or {}
                mid = getattr(m, "id", None)
                mscore = getattr(m, "score", 0.0)
                mmeta = getattr(m, "metadata", {}) or {}
                return mid, mscore, mmeta

            def _query_ado():
                query_kwargs = {
                    "vector": query_vector,
                    "top_k": ado_top_k,
                    "include_metadata": True,
                    "filter": {"doc_type": {"$eq": "ado_backlog"}}
                }
                if namespace:
                    query_kwargs["namespace"] = namespace
                res = index.query(**query_kwargs)
                items = []
                for match in _coerce_matches(res):
                    mid, mscore, mmeta = _coerce_fields(match)
                    if (mscore or 0) >= ado_min_similarity:
                        items.append({
                            "id": mid,
                            "score": mscore,
                            "type": mmeta.get("work_item_type", mmeta.get("type", "unknown")),
                            "title": mmeta.get("title", ""),
                            "description": (mmeta.get("description", "") or "")[:1000],
                            "acceptance_criteria": mmeta.get("acceptance_criteria"),
                            "work_item_id": mmeta.get("work_item_id")
                        })
                return items

            print(f"Retrieval Tool: Querying Pinecone for ADO items... (namespace='{namespace or 'default'}')")
            ado_items = _query_ado()
            if not ado_items:
                print(f"Retrieval Tool: No ADO matches (doc_type='ado_backlog', threshold={ado_min_similarity}).")

            print(f"Retrieval Tool: Found {len(ado_items)} relevant ADO items")

            print("Retrieval Tool: Querying Pinecone for architecture constraints...")
            arch_query_kwargs = {
                "vector": query_vector,
                "top_k": arch_top_k,
                "include_metadata": True,
                "filter": {"doc_type": {"$eq": "architecture"}}
            }
            if namespace:
                arch_query_kwargs["namespace"] = namespace
            arch_results = index.query(**arch_query_kwargs)

            architecture_items = []
            for match in _coerce_matches(arch_results):
                mid, mscore, mmeta = _coerce_fields(match)
                if (mscore or 0) >= arch_min_similarity:
                    architecture_items.append({
                        "id": mid,
                        "score": mscore,
                        "source": mmeta.get("file_name", mmeta.get("source", "")),
                        "text": (mmeta.get("chunk_text", mmeta.get("text", "")) or "")[:1000],
                        "section": mmeta.get("section", ""),
                        "project": mmeta.get("project"),
                        "chunk_index": mmeta.get("chunk_index")
                    })

            print(f"Retrieval Tool: Found {len(architecture_items)} relevant architecture constraints")
            
            # Build result
            result = {
                "status": "success",
                "run_id": run_id,
                "segment_id": segment_id,
                "retrieval_summary": {
                    "ado_items_count": len(ado_items),
                    "architecture_items_count": len(architecture_items),
                    "ado_min_similarity_threshold": ado_min_similarity,
                    "arch_min_similarity_threshold": arch_min_similarity
                },
                "ado_items": ado_items,
                "architecture_constraints": architecture_items,
                "query_info": {
                    "dominant_intent": dominant_intent,
                    "intent_labels": intent_labels
                }
            }
            
            return json.dumps(result, indent=2)
            
        except json.JSONDecodeError as e:
            # Fail open: proceed with empty context but include warning
            warn = f"Failed to parse query_data JSON: {str(e)}"
            fallback = {
                "status": "success",
                "run_id": run_id,
                "segment_id": 0,
                "retrieval_summary": {
                    "ado_items_count": 0,
                    "architecture_items_count": 0,
                    "ado_min_similarity_threshold": ado_min_similarity,
                    "arch_min_similarity_threshold": arch_min_similarity,
                    "warning": warn
                },
                "ado_items": [],
                "architecture_constraints": [],
                "query_info": {}
            }
            return json.dumps(fallback, indent=2)
        
        except Exception as e:
            # Fail open: proceed with empty context but include warning
            warn = f"Retrieval failed: {str(e)}"
            fallback = {
                "status": "success",
                "run_id": run_id,
                "segment_id": 0,
                "retrieval_summary": {
                    "ado_items_count": 0,
                    "architecture_items_count": 0,
                    "ado_min_similarity_threshold": ado_min_similarity,
                    "arch_min_similarity_threshold": arch_min_similarity,
                    "warning": warn
                },
                "ado_items": [],
                "architecture_constraints": [],
                "query_info": {}
            }
            return json.dumps(fallback, indent=2)
    
    return retrieve_context


def _mock_retrieval(segment_text: str, intent_labels: List[str], dominant_intent: str, segment_id: int) -> str:
    text_lower = f"{segment_text or ''} {' '.join(intent_labels or [])} {dominant_intent or ''}".lower()
    force_populate = os.getenv("RDE_MOCK_ALWAYS_POPULATE", "0").lower() in ("1", "true", "yes")
    mock_ado_items: List[Dict[str, Any]] = []
    mock_architecture: List[Dict[str, Any]] = []

    if "authentication" in (dominant_intent or "").lower() or any("auth" in (label or "").lower() for label in (intent_labels or [])):
        mock_ado_items.extend([
            {"id": "ado_mock_1", "score": 0.85, "type": "Epic", "title": "Security & Authentication Improvements", "description": "Enhance security posture across the platform", "work_item_id": 12345},
            {"id": "ado_mock_2", "score": 0.78, "type": "Feature", "title": "Implement OAuth2 Authentication", "description": "Add OAuth2 support for third-party integrations", "work_item_id": 12346},
        ])
        mock_architecture.append({"id": "arch_mock_1", "score": 0.80, "source": "security-guidelines.md", "text": "All authentication mechanisms must support industry-standard protocols (OAuth2, SAML, OpenID Connect). Multi-factor authentication should be available for all user types.", "section": "Authentication Requirements"})

    if "performance" in (dominant_intent or "").lower() or any(("latency" in (l or "").lower()) or ("optimiz" in (l or "").lower()) for l in (intent_labels or [])):
        mock_ado_items.append({"id": "ado_mock_3", "score": 0.82, "type": "Feature", "title": "API Performance Optimization", "description": "Reduce API response times through caching and indexing", "work_item_id": 12347})
        mock_architecture.append({"id": "arch_mock_2", "score": 0.75, "source": "performance-standards.md", "text": "API endpoints must respond within 200ms for 95th percentile. Database queries must use proper indexes and limit result sets.", "section": "Performance Standards"})

    if any(k in text_lower for k in ("language", "multilingual", "spanish", "french", "mandarin")):
        mock_ado_items.append({"id": "ado_mock_lang", "score": 0.81, "type": "Feature", "title": "Multi-language UI Support", "description": "Add i18n and l10n for web, mobile, and chatbot flows", "work_item_id": 22001})
        mock_architecture.append({"id": "arch_mock_lang", "score": 0.72, "source": "i18n-guidelines.md", "text": "All UI strings must be externalized and locale-aware.", "section": "Internationalization"})

    if any(k in text_lower for k in ("merchant", "portal")):
        mock_ado_items.append({"id": "ado_mock_portal", "score": 0.79, "type": "Epic", "title": "Merchant Self-Service Portal", "description": "Portal for status, evidence upload, and agent messaging", "work_item_id": 22002})

    if any(k in text_lower for k in ("fraud", "alert")):
        mock_ado_items.append({"id": "ado_mock_fraud", "score": 0.77, "type": "Feature", "title": "Real-Time Fraud Alerts", "description": "Streaming detection and notifications", "work_item_id": 22003})

    if any(k in text_lower for k in ("explainab", "transparent", "ai", "model")):
        mock_ado_items.append({"id": "ado_mock_ai", "score": 0.8, "type": "Feature", "title": "AI Decision Explainability", "description": "Provide rationale for automated approvals/denials", "work_item_id": 22004})

    if any(k in text_lower for k in ("upload", "drag-and-drop", "bulk")):
        mock_ado_items.append({"id": "ado_mock_upload", "score": 0.76, "type": "Story", "title": "Improve Document Upload UX", "description": "Drag-and-drop, bulk upload, and validation", "work_item_id": 22005})

    if any(k in text_lower for k in ("notification", "whatsapp", "in-app", "chat")):
        mock_ado_items.append({"id": "ado_mock_notify", "score": 0.74, "type": "Story", "title": "Expanded Notification Channels", "description": "Add WhatsApp and in-app chat notifications", "work_item_id": 22006})

    if any(k in text_lower for k in ("provisional credit", "timeline", "debit card")):
        mock_architecture.append({"id": "arch_mock_reg", "score": 0.7, "source": "bank-policy.md", "text": "Provisional credit timeline reduced to 5 days for debit card disputes.", "section": "Regulatory"})

    if any(k in text_lower for k in ("chargeback", "pdf", "template", "network")):
        mock_ado_items.append({"id": "ado_mock_cb", "score": 0.73, "type": "Story", "title": "Standardize Chargeback PDF Template", "description": "Unified evidence package formatting across networks", "work_item_id": 22007})

    if any(k in text_lower for k in ("duplicate", "rules", "fuzzy")):
        mock_ado_items.append({"id": "ado_mock_dup", "score": 0.73, "type": "Story", "title": "Rules-based Duplicate Detection", "description": "Replace fuzzy matching with deterministic rules", "work_item_id": 22008})

    if any(k in text_lower for k in ("requirements", "design", "devops", "compliance", "ux", "sprint", "planning")):
        mock_ado_items.append({"id": "ado_mock_pm", "score": 0.72, "type": "Task", "title": "Planning and Documentation Updates", "description": "Update docs, plan infra, and coordinate stakeholders", "work_item_id": 22009})

    if force_populate or not mock_ado_items:
        mock_ado_items.append({"id": "ado_mock_generic", "score": 0.7, "type": "Story", "title": "Generic Related Work Item", "description": (segment_text or "")[:500], "work_item_id": 99999})
    if force_populate or not mock_architecture:
        mock_architecture.append({"id": "arch_mock_generic", "score": 0.7, "source": "architecture.md", "text": "General architectural considerations apply. Validate NFRs for new capabilities.", "section": "General"})

    result = {
        "status": "success_mock",
        "run_id": "mock",
        "segment_id": segment_id,
        "retrieval_summary": {
            "ado_items_count": len(mock_ado_items),
            "architecture_items_count": len(mock_architecture),
            "min_similarity_threshold": 0.5,
            "note": "Mock data - set OPENAI_API_KEY and PINECONE_API_KEY for real retrieval"
        },
        "ado_items": mock_ado_items,
        "architecture_constraints": mock_architecture,
        "query_info": {
            "dominant_intent": dominant_intent,
            "intent_labels": intent_labels
        }
    }
    return json.dumps(result, indent=2)


# Documentation for retrieval tool
RETRIEVAL_TOOL_DESCRIPTION = """
Retrieval Tool - Context Retrieval from Pinecone

Purpose:
- Query Pinecone vector store for relevant existing backlog items
- Retrieve architecture constraints and requirements
- Provide context for backlog generation and story tagging

Inputs:
- Segment text with intent labels
- Query parameters (top_k, similarity threshold)

Outputs:
- Relevant ADO backlog items (Epics, Features, Stories)
- Architecture constraints and technical requirements
- Similarity scores and metadata

Configuration:
- Pinecone index and namespace from config
- Embedding model: text-embedding-3-small
- Minimum similarity threshold: configurable (default 0.7)
"""
