import os
import yaml
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Centralized application configuration loader.

    - Reads YAML from a default path (`config.poc.yaml`) unless overridden
    - Returns a dictionary (empty when file missing or unreadable)
    - Handles errors internally and logs diagnostics
    """

    DEFAULT_CONFIG_PATH = "config.poc.yaml"

    @staticmethod
    def load(config_path: Optional[str] = None) -> Dict[str, Any]:
        path = config_path or ConfigLoader.DEFAULT_CONFIG_PATH
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = yaml.safe_load(f) or {}
                    logger.debug("ConfigLoader: Loaded config from %s", path)
                    return data
            logger.debug("ConfigLoader: Config file %s not found; returning empty config", path)
            return {}
        except Exception as e:
            logger.exception("ConfigLoader: Failed to load config from %s: %s", path, e)
            return {}
