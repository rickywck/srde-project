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
                warn = "Missing API keys for OpenAI or Pinecone; returning empty retrieval."
                print(f"Retrieval Tool: {warn}")
                fallback = {
                    "status": "success_no_api",
                    "run_id": run_id,
                    "segment_id": segment_id,
                    "retrieval_summary": {
                        "ado_items_count": 0,
                        "architecture_items_count": 0,
                        "ado_min_similarity_threshold": ado_min_similarity,
                        "arch_min_similarity_threshold": arch_min_similarity,
                        "warning": warn
                    },
                    "ado_items": [],
                    "architecture_constraints": [],
                    "query_info": {
                        "dominant_intent": dominant_intent,
                        "intent_labels": intent_labels
                    }
                }
                return json.dumps(fallback, indent=2)
            
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


# Mock retrieval removed: callers must provide real API keys for production retrieval.


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
- Minimum similarity threshold: configurable (default 0.5)
"""
