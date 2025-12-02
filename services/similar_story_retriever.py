"""Deprecated module; use `agents.utils.similar_story_retriever` instead."""
import logging
from agents.utils.similar_story_retriever import SimilarStoryRetriever  # noqa: F401

logging.getLogger(__name__).warning(
    "services.similar_story_retriever is deprecated; import from agents.utils.similar_story_retriever"
)
