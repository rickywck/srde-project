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
import logging

from openai import OpenAI
from pinecone import Pinecone

from .config_loader import ConfigLoader

# Module logger for debug/troubleshooting
logger = logging.getLogger(__name__)


class SimilarStoryRetriever:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.min_similarity = float(min_similarity)
        self._openai_client: Optional[OpenAI] = None
        self._pinecone_client: Optional[Pinecone] = None
        self._index = None
        self.log_fn = log_fn

        # Load configuration (prefer provided, else disk)
        if config is None:
            config = ConfigLoader.load()
        self.config = config or {}

        # Configuration
        self.embedding_model = (
            self.config.get("openai", {}).get("embedding_model", "text-embedding-3-small")
        )
        self.embedding_dimensions = int(
            self.config.get("openai", {}).get("embedding_dimensions", 512)
        )
        self.index_name = self.config.get("pinecone", {}).get("index_name", "rde-lab")
        self.pinecone_namespace = (self.config.get("pinecone", {}).get("project") or "").strip()
        logger.debug(
            "SimilarStoryRetriever: Initialized with index_name=%s, namespace=%s, embedding_model=%s, embedding_dimensions=%s, min_similarity=%s",
            self.index_name,
            self.pinecone_namespace,
            self.embedding_model,
            self.embedding_dimensions,
            self.min_similarity,
        )

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
    def find_similar_stories(self, story: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find similar existing stories using vector similarity search.

        Uses the `min_similarity` configured when the retriever was instantiated.

        Returns a list of dicts:
        [{work_item_id, title, description, similarity}, ...]
        """
        sim_threshold = float(self.min_similarity)

        logger.debug(
            "SimilarStoryRetriever: find_similar_stories called for title=%s threshold=%s",
            story.get("title"),
            sim_threshold,
        )

        if not self.openai_client or not self.index:
            logger.debug(
                "SimilarStoryRetriever: Missing OpenAI or Pinecone client, returning empty list"
            )
            try:
                if self.log_fn:
                    self.log_fn("Missing OpenAI or Pinecone client, returning empty list")
            except Exception:
                pass
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
            logger.debug(
                "SimilarStoryRetriever: Pinecone query returned %s matches",
                len(query_res.get("matches", [])),
            )

            # Filter by similarity threshold and work item type
            similar_stories: List[Dict[str, Any]] = []
            for match in query_res.get("matches", []):
                score = match.get("score", 0)
                logger.debug(
                    "SimilarStoryRetriever: Match score=%s for id=%s",
                    score,
                    match.get("id"),
                )
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
            logger.exception("SimilarStoryRetriever: Retrieval failed")
            try:
                if self.log_fn:
                    self.log_fn(f"Retrieval failed: {e}")
            except Exception:
                pass
            return []

    # ---- Helpers
    def _log(self, message: str) -> None:
        """Backward-compatible helper that prefers a logger and also calls an
        optional provided log function. Kept for external callers that may
        have used this previously.
        """
        try:
            logger.debug(message)
            if self.log_fn:
                try:
                    self.log_fn(message)
                except Exception:
                    logger.debug(
                        "SimilarStoryRetriever: user log_fn raised an exception"
                    )
        except Exception:
            pass
