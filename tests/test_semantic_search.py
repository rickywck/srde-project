"""
Semantic Search Tests - Tests for retrieval logic based on section 5.2 of the POC plan.

Tests the retrieval of ADO backlog items and architecture constraints from Pinecone
using intent embeddings and similarity thresholds.
"""

import os
import pytest
from typing import List, Dict, Any
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class SemanticSearcher:
    """Handles semantic search operations against Pinecone."""
    
    def __init__(
        self,
        pinecone_client: Pinecone,
        openai_client: OpenAI,
        index_name: str,
        embedding_model: str = "text-embedding-3-small",
        min_similarity: float = 0.7
    ):
        self.pinecone_client = pinecone_client
        self.openai_client = openai_client
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.min_similarity = min_similarity
        self.index = pinecone_client.Index(index_name)
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
            dimensions=512  # Match Pinecone index dimension
        )
        embedding = response.data[0].embedding
        return embedding
    
    def search_ado_items(
        self,
        query_text: str,
        namespace: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for ADO backlog items.
        
        Args:
            query_text: Query text to search for
            namespace: Pinecone namespace (project name)
            top_k: Number of results to return
        
        Returns:
            List of matching items with metadata and scores
        """
        print("\n" + "="*80)
        print("🔍 ADO BACKLOG SEARCH")
        print("="*80)
        print(f"Query Text: '{query_text}'")
        print(f"\nPinecone Query:")
        print(f"  index: {self.index_name}")
        print(f"  namespace: {namespace}")
        print(f"  filter: {{doc_type: 'ado_backlog'}}")
        print(f"  top_k: {top_k}")
        print(f"  similarity_threshold: {self.min_similarity}")
        
        # Generate query embedding
        query_embedding = self.embed_text(query_text)
        
        # Query Pinecone with ADO filter
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter={"doc_type": "ado_backlog"},
            include_metadata=True
        )
        
        print(f"\nPinecone Results: {len(results.matches)} matches")
        for i, match in enumerate(results.matches, 1):
            embedded_preview = (match.metadata.get('embedded_text','')[:80] + '…') if match.metadata.get('embedded_text') else ''
            print(f"  {i}. score={match.score:.4f} | {match.metadata.get('title', match.id)} {embedded_preview}")
        
        # Filter by similarity threshold
        filtered_matches = [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
            if match.score >= self.min_similarity
        ]
        
        print(f"\nAfter Threshold Filter (>= {self.min_similarity}): {len(filtered_matches)} matches")
        if len(filtered_matches) == 0 and len(results.matches) > 0:
            best_score = max(m.score for m in results.matches)
            print(f"⚠️  All {len(results.matches)} results filtered out (best score: {best_score:.4f})")
        print("="*80)
        
        return filtered_matches
    
    def search_architecture(
        self,
        query_text: str,
        namespace: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for architecture constraint documents.
        
        Args:
            query_text: Query text to search for
            namespace: Pinecone namespace (project name)
            top_k: Number of results to return
        
        Returns:
            List of matching items with metadata and scores
        """
        print("\n" + "="*80)
        print("🔍 ARCHITECTURE SEARCH")
        print("="*80)
        print(f"Query Text: '{query_text}'")
        print(f"\nPinecone Query:")
        print(f"  index: {self.index_name}")
        print(f"  namespace: {namespace}")
        print(f"  filter: {{doc_type: 'architecture'}}")
        print(f"  top_k: {top_k}")
        print(f"  similarity_threshold: {self.min_similarity}")
        
        # Generate query embedding
        query_embedding = self.embed_text(query_text)
        
        # Query Pinecone with architecture filter
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter={"doc_type": "architecture"},
            include_metadata=True
        )
        
        print(f"\nPinecone Results: {len(results.matches)} matches")
        for i, match in enumerate(results.matches, 1):
            print(f"  {i}. score={match.score:.4f} | {match.metadata.get('file_name', match.id)} (chunk {match.metadata.get('chunk_index', 'N/A')})")
        
        # Filter by similarity threshold
        filtered_matches = [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
            if match.score >= self.min_similarity
        ]
        
        print(f"\nAfter Threshold Filter (>= {self.min_similarity}): {len(filtered_matches)} matches")
        if len(filtered_matches) == 0 and len(results.matches) > 0:
            best_score = max(m.score for m in results.matches)
            print(f"⚠️  All {len(results.matches)} results filtered out (best score: {best_score:.4f})")
        print("="*80)
        
        return filtered_matches
    
    def search_combined(
        self,
        query_text: str,
        namespace: str,
        top_k_ado: int = 5,
        top_k_arch: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search both ADO items and architecture constraints.
        
        Args:
            query_text: Query text to search for
            namespace: Pinecone namespace (project name)
            top_k_ado: Number of ADO results to return
            top_k_arch: Number of architecture results to return
        
        Returns:
            Dictionary with 'ado_items' and 'architecture' keys
        """
        ado_items = self.search_ado_items(query_text, namespace, top_k_ado)
        architecture = self.search_architecture(query_text, namespace, top_k_arch)
        
        return {
            "ado_items": ado_items,
            "architecture": architecture
        }


# Test fixtures
@pytest.fixture
def config():
    """Load test configuration."""
    load_dotenv()
    with open('config.poc.yaml', 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def searcher(config):
    """Create SemanticSearcher instance."""
    pinecone_api_key = os.getenv(config['pinecone']['api_key_env_var'])
    openai_api_key = os.getenv(config['openai']['api_key_env_var'])
    
    if not pinecone_api_key or not openai_api_key:
        pytest.skip("API keys not configured")
    
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    
    # Use ADO-specific similarity threshold for ADO-focused searches used in tests.
    ado_min_sim = config.get('retrieval', {}).get('ado', {}).get('min_similarity_threshold', 0.7)
    return SemanticSearcher(
        pinecone_client=pinecone_client,
        openai_client=openai_client,
        index_name=config['pinecone']['index_name'],
        embedding_model=config['openai']['embedding_model'],
        min_similarity=ado_min_sim
    )


# Tests
class TestSemanticSearch:
    """Tests for semantic search functionality."""
    
    def test_embed_text(self, searcher):
        """Test text embedding generation."""
        text = "User authentication and authorization"
        embedding = searcher.embed_text(text)
        
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
    
    def test_search_ado_items(self, searcher, config):
        """Test searching for ADO backlog items."""
        query = "user login authentication"
        namespace = config['pinecone']['project']
        
        results = searcher.search_ado_items(query, namespace, top_k=5)
        
        # Validate result structure
        assert isinstance(results, list)
        for result in results:
            assert "id" in result
            assert "score" in result
            assert "metadata" in result
            assert result["score"] >= searcher.min_similarity
            assert result["metadata"]["doc_type"] == "ado_backlog"
    
    def test_search_architecture(self, searcher, config):
        """Test searching for architecture constraints."""
        query = "security requirements authentication"
        namespace = config['pinecone']['project']
        
        results = searcher.search_architecture(query, namespace, top_k=5)
        
        # Validate result structure
        assert isinstance(results, list)
        for result in results:
            assert "id" in result
            assert "score" in result
            assert "metadata" in result
            assert result["score"] >= searcher.min_similarity
            assert result["metadata"]["doc_type"] == "architecture"
    
    def test_search_combined(self, searcher, config):
        """Test combined search for both ADO and architecture."""
        query = "implement user authentication with OAuth2"
        namespace = config['pinecone']['project']
        
        results = searcher.search_combined(query, namespace)
        
        assert "ado_items" in results
        assert "architecture" in results
        assert isinstance(results["ado_items"], list)
        assert isinstance(results["architecture"], list)
    
    def test_similarity_threshold_filtering(self, searcher, config):
        """Test that results below similarity threshold are filtered out."""
        # Use a very specific query unlikely to have high similarity matches
        query = "xyz123 nonexistent feature abc789"
        namespace = config['pinecone']['project']
        
        results = searcher.search_ado_items(query, namespace, top_k=10)
        
        # All results should be above threshold
        for result in results:
            assert result["score"] >= searcher.min_similarity
    
    def test_empty_query(self, searcher, config):
        """Test handling of empty query."""
        query = ""
        namespace = config['pinecone']['project']
        
        # Should not raise an error
        results = searcher.search_ado_items(query, namespace)
        assert isinstance(results, list)
    
    def test_intent_based_query(self, searcher, config):
        """Test query construction based on intent (as in section 5.2)."""
        # Simulate segment with intent labels
        dominant_intent = "feature_request"
        intent_labels = ["authentication", "user_management"]
        segment_text = "We need to implement single sign-on functionality for enterprise users"
        
        # Build intent query (as described in 5.2)
        intent_query = f"{dominant_intent} {' '.join(intent_labels)} {segment_text[:300]}"
        
        namespace = config['pinecone']['project']
        results = searcher.search_combined(intent_query, namespace)
        
        assert "ado_items" in results
        assert "architecture" in results


# Manual test runner for development
def manual_test():
    """Manual test for development/debugging."""
    load_dotenv()
    
    with open('config.poc.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    pinecone_api_key = os.getenv(config['pinecone']['api_key_env_var'])
    openai_api_key = os.getenv(config['openai']['api_key_env_var'])
    
    if not pinecone_api_key or not openai_api_key:
        print("Error: API keys not configured")
        return
    
    pinecone_client = Pinecone(api_key=pinecone_api_key)
    openai_client = OpenAI(api_key=openai_api_key)
    
    searcher = SemanticSearcher(
        pinecone_client=pinecone_client,
        openai_client=openai_client,
        index_name=config['pinecone']['index_name'],
        embedding_model=config['openai']['embedding_model'],
        min_similarity=config.get('retrieval', {}).get('ado', {}).get('min_similarity_threshold', 0.7)
    )
    
    # Test query
    query = "Customer communication"
    namespace = config['project']['name']
    
    print("\n" + "="*80)
    print(f"🔎 Testing search for: '{query}'")
    print(f"Config: namespace='{namespace}', threshold={searcher.min_similarity}, index='{config['pinecone']['index_name']}'")
    print("="*80)
    
    results = searcher.search_combined(query, namespace)
    
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print(f"ADO Items: {len(results['ado_items'])} results")
    for item in results['ado_items']:
        preview = (item['metadata'].get('embedded_text','')[:100] + '…') if item['metadata'].get('embedded_text') else ''
        print(f"  [{item['score']:.3f}] {item['metadata'].get('title', 'N/A')} :: {preview}")
    
    print(f"\nArchitecture: {len(results['architecture'])} results")
    for item in results['architecture']:
        print(f"  [{item['score']:.3f}] {item['metadata'].get('file_name', 'N/A')} (chunk {item['metadata'].get('chunk_index', 'N/A')})")
    print("="*80 + "\n")


if __name__ == "__main__":
    manual_test()
