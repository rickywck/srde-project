"""
Shared Similar Story Retriever
---------------------------------
Encapsulates the logic to find similar existing backlog stories using
OpenAI embeddings and Pinecone vector search.

This class is designed to be reusable by both the BacklogSynthesisWorkflow
and agents (e.g., tagging_agent) to avoid duplication of retrieval logic.

Behavior mirrors the logic previously implemented in
BacklogSynthesisWorkflow._find_similar_stories.
"""

from typing import Any, Dict, List, Optional, Callable
import os

from openai import OpenAI
from pinecone import Pinecone


class SimilarStoryRetriever:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config or {}
        self.min_similarity = float(min_similarity)
        self._openai_client: Optional[OpenAI] = None
        self._pinecone_client: Optional[Pinecone] = None
        self._index = None
        self.log_fn = log_fn

        # Configuration
        self.embedding_model = (
            self.config.get("openai", {}).get("embedding_model", "text-embedding-3-small")
        )
        self.embedding_dimensions = int(
            self.config.get("openai", {}).get("embedding_dimensions", 512)
        )
        self.index_name = self.config.get("pinecone", {}).get("index_name", "rde-lab")
        self.pinecone_namespace = (self.config.get("pinecone", {}).get("project") or "").strip()

    # ---- Lazy clients
    @property
    def openai_client(self) -> Optional[OpenAI]:
        if self._openai_client is None and os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._openai_client

    @property
    def pinecone_client(self) -> Optional[Pinecone]:
        if self._pinecone_client is None and os.getenv("PINECONE_API_KEY"):
            self._pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        return self._pinecone_client

    @property
    def index(self):
        if self._index is None and self.pinecone_client:
            self._index = self.pinecone_client.Index(self.index_name)
        return self._index

    # ---- Public API
    def find_similar_stories(
        self, story: Dict[str, Any], min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Find similar existing stories using vector similarity search.

        Returns a list of dicts:
        [{work_item_id, title, description, similarity}, ...]
        """
        sim_threshold = float(min_similarity) if min_similarity is not None else self.min_similarity

        if not self.openai_client or not self.index:
            self._log("SimilarStoryRetriever: Missing OpenAI or Pinecone client, returning empty list")
            return []

        # Build story text for embedding
        ac = story.get("acceptance_criteria", []) or []
        story_text = (
            (story.get("title", "") or "")
            + "\n"
            + (story.get("description", "") or "")
            + "\n"
            + "\n".join(ac)
        )

        try:
            # Generate embedding
            emb_resp = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=story_text[:3000],
                dimensions=self.embedding_dimensions,
            )
            vec = emb_resp.data[0].embedding

            # Query Pinecone (use namespace if configured) with metadata filter
            query_kwargs: Dict[str, Any] = {
                "vector": vec,
                "top_k": 10,
                "filter": {"doc_type": "ado_backlog"},
                "include_metadata": True,
            }
            if self.pinecone_namespace:
                query_kwargs["namespace"] = self.pinecone_namespace

            query_res = self.index.query(**query_kwargs)

            # Filter by similarity threshold and work item type
            similar_stories: List[Dict[str, Any]] = []
            for match in query_res.get("matches", []):
                score = match.get("score", 0)
                if score >= sim_threshold:
                    md = match.get("metadata", {})
                    item_type = (md.get("type") or md.get("work_item_type") or "").lower()
                    if "story" in item_type:
                        similar_stories.append(
                            {
                                "work_item_id": md.get("work_item_id") or match.get("id"),
                                "title": md.get("title", ""),
                                "description": (md.get("description", "") or "")[:500],
                                "similarity": score,
                            }
                        )

            return similar_stories

        except Exception as e:
            self._log(f"SimilarStoryRetriever: Retrieval failed: {e}")
            return []

    # ---- Helpers
    def _log(self, message: str) -> None:
        try:
            if self.log_fn:
                self.log_fn(message)
            else:
                # Best-effort logging without raising
                print(message)
        except Exception:
            pass
