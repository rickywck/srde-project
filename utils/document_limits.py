"""Utilities for document size and token limit calculations used by segmentation and agents.

This module exposes helpers to read segmentation prompt configuration and
resolve sensible token limits for downstream model calls.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

import yaml

from tools.utils.token_utils import estimate_tokens


_SEGMENTATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "segmentation_agent.yaml"
_DEFAULT_MAX_COMPLETION = 5000
_LIMIT_RATIO = 0.5


class DocumentLimitUtils:
    """Helper for shared document size validation based on segmentation settings."""

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_segmentation_max_completion_tokens() -> int:
        if not _SEGMENTATION_PROMPT_PATH.exists():
            return _DEFAULT_MAX_COMPLETION

        try:
            with _SEGMENTATION_PROMPT_PATH.open("r", encoding="utf-8") as config_file:
                data = yaml.safe_load(config_file) or {}
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "DocumentLimitUtils: Failed to read segmentation config: %s",
                exc,
            )
            return _DEFAULT_MAX_COMPLETION

        parameters = data.get("parameters", {})
        value = parameters.get("max_completion_tokens")
        if isinstance(value, int) and value > 0:
            return value
        return _DEFAULT_MAX_COMPLETION

    @classmethod
    def get_segmentation_max_completion_tokens(cls) -> int:
        return cls._load_segmentation_max_completion_tokens()

    @classmethod
    def get_allowed_token_threshold(cls) -> int:
        max_completion = cls.get_segmentation_max_completion_tokens()
        return int(max_completion * _LIMIT_RATIO)

    @classmethod
    def analyze_document(cls, document_text: str) -> Dict[str, int]:
        token_count = estimate_tokens(document_text or "")
        max_completion = cls.get_segmentation_max_completion_tokens()
        max_allowed = int(max_completion * _LIMIT_RATIO)
        return {
            "token_count": token_count,
            "max_allowed_tokens": max_allowed,
            "max_completion_tokens": max_completion,
        }

    @classmethod
    def build_over_limit_message(cls, metrics: Dict[str, int]) -> str:
        return (
            "Document is too large for processing. Token count: "
            f"{metrics['token_count']} exceeds limit of {metrics['max_allowed_tokens']} tokens "
            f"(50% of {metrics['max_completion_tokens']} max_completion_tokens). Please provide a smaller document."
        )

    @classmethod
    def ensure_within_limit(cls, document_text: str) -> Dict[str, int]:
        metrics = cls.analyze_document(document_text)
        if metrics["token_count"] > metrics["max_allowed_tokens"]:
            raise ValueError(cls.build_over_limit_message(metrics))
        return metrics
