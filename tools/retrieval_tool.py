"""
Retrieval Tool - Tool for querying Pinecone for relevant context
"""

import os
import json
import yaml
from typing import Dict, Any, List
from openai import OpenAI
from pinecone import Pinecone
from strands import tool


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
    index_name = config.get("pinecone", {}).get("index_name", "rde-lab")
    min_similarity = config.get("retrieval", {}).get("min_similarity_threshold", 0.5)
    
    @tool
    def retrieve_context(query_data: str) -> str:
        """
        Retrieve relevant context from Pinecone (ADO backlog items and architecture constraints).
        
        Args:
            query_data: JSON string containing query information with fields:
                - segment_text: The segment text
                - intent_labels: List of intent labels
                - dominant_intent: The dominant intent
                - segment_id: The segment identifier
            
        Returns:
            JSON string containing retrieved ADO items and architecture constraints
        """
        
        try:
            # Parse input
            query_info = json.loads(query_data)
            segment_text = query_info.get("segment_text", "")
            intent_labels = query_info.get("intent_labels", [])
            dominant_intent = query_info.get("dominant_intent", "")
            segment_id = query_info.get("segment_id", 0)
            
            print(f"Retrieval Tool: Processing segment {segment_id} (run_id: {run_id})")
            
            # Check if we have required clients
            if not openai_client or not pc:
                print("Retrieval Tool: Using MOCK mode (missing API keys)")
                return _mock_retrieval(segment_text, intent_labels, dominant_intent, segment_id)
            
            # Build intent query string
            # Combine dominant intent + intent labels + first ~300 chars of segment text
            intent_query_parts = [dominant_intent] + intent_labels
            intent_query = " ".join(intent_query_parts) + " " + segment_text[:300]
            
            print(f"Retrieval Tool: Generating embedding for intent query...")
            
            # Generate embedding
            embedding_response = openai_client.embeddings.create(
                model=embedding_model,
                input=intent_query
            )
            query_vector = embedding_response.data[0].embedding
            
            # Get Pinecone index
            index = pc.Index(index_name)
            
            # Query for ADO items (namespace: "ado_items")
            print(f"Retrieval Tool: Querying Pinecone for ADO items...")
            ado_results = index.query(
                vector=query_vector,
                top_k=10,
                namespace="ado_items",
                include_metadata=True
            )
            
            # Filter by similarity threshold
            ado_items = []
            for match in ado_results.get("matches", []):
                if match.get("score", 0) >= min_similarity:
                    metadata = match.get("metadata", {})
                    ado_items.append({
                        "id": match.get("id"),
                        "score": match.get("score"),
                        "type": metadata.get("type", "unknown"),
                        "title": metadata.get("title", ""),
                        "description": metadata.get("description", "")[:500],  # Truncate long descriptions
                        "work_item_id": metadata.get("work_item_id")
                    })
            
            print(f"Retrieval Tool: Found {len(ado_items)} relevant ADO items")
            
            # Query for architecture constraints (namespace: "architecture")
            print(f"Retrieval Tool: Querying Pinecone for architecture constraints...")
            arch_results = index.query(
                vector=query_vector,
                top_k=5,
                namespace="architecture",
                include_metadata=True
            )
            
            # Filter by similarity threshold
            architecture_items = []
            for match in arch_results.get("matches", []):
                if match.get("score", 0) >= min_similarity:
                    metadata = match.get("metadata", {})
                    architecture_items.append({
                        "id": match.get("id"),
                        "score": match.get("score"),
                        "source": metadata.get("source", ""),
                        "text": metadata.get("text", "")[:1000],  # Truncate long text
                        "section": metadata.get("section", "")
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
                    "min_similarity_threshold": min_similarity
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
            error_result = {
                "status": "error",
                "error": f"Failed to parse query_data JSON: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_result, indent=2)
        
        except Exception as e:
            error_result = {
                "status": "error",
                "error": f"Retrieval failed: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_result, indent=2)
    
    return retrieve_context


def _mock_retrieval(segment_text: str, intent_labels: List[str], dominant_intent: str, segment_id: int) -> str:
    """Mock retrieval for testing without Pinecone"""
    
    # Generate mock ADO items based on intents
    mock_ado_items = []
    
    if "authentication" in dominant_intent.lower() or any("auth" in label.lower() for label in intent_labels):
        mock_ado_items.extend([
            {
                "id": "ado_mock_1",
                "score": 0.85,
                "type": "Epic",
                "title": "Security & Authentication Improvements",
                "description": "Enhance security posture across the platform",
                "work_item_id": 12345
            },
            {
                "id": "ado_mock_2",
                "score": 0.78,
                "type": "Feature",
                "title": "Implement OAuth2 Authentication",
                "description": "Add OAuth2 support for third-party integrations",
                "work_item_id": 12346
            }
        ])
    
    if "performance" in dominant_intent.lower() or any("latency" in label.lower() or "optimize" in label.lower() for label in intent_labels):
        mock_ado_items.extend([
            {
                "id": "ado_mock_3",
                "score": 0.82,
                "type": "Feature",
                "title": "API Performance Optimization",
                "description": "Reduce API response times through caching and indexing",
                "work_item_id": 12347
            }
        ])
    
    # Generate mock architecture constraints
    mock_architecture = []
    
    if "authentication" in dominant_intent.lower():
        mock_architecture.append({
            "id": "arch_mock_1",
            "score": 0.80,
            "source": "security-guidelines.md",
            "text": "All authentication mechanisms must support industry-standard protocols (OAuth2, SAML, OpenID Connect). Multi-factor authentication should be available for all user types.",
            "section": "Authentication Requirements"
        })
    
    if "performance" in dominant_intent.lower():
        mock_architecture.append({
            "id": "arch_mock_2",
            "score": 0.75,
            "source": "performance-standards.md",
            "text": "API endpoints must respond within 200ms for 95th percentile. Database queries must use proper indexes and limit result sets.",
            "section": "Performance Standards"
        })
    
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
