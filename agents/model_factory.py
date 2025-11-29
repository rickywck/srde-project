import os
import yaml
from typing import Dict, Any, Optional
from strands.models.openai import OpenAIModel

class ModelFactory:
    """
    Factory class for creating OpenAIModel instances with consistent configuration.
    Centralizes logic for:
    - Configuration loading
    - Environment variable handling
    - Parameter mapping (e.g. max_tokens -> max_completion_tokens)
    """

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {"openai": {"chat_model": "gpt-4.1-mini"}}

    @staticmethod
    def create_openai_model(
        config_path: str = "config.poc.yaml",
        model_params: Optional[Dict[str, Any]] = None,
        model_id_override: Optional[str] = None
    ) -> OpenAIModel:
        """
        Create and configure an OpenAIModel instance.

        Args:
            config_path: Path to the configuration YAML file.
            model_params: Dictionary of parameters to pass to the model (e.g. max_tokens).
                          Will be sanitized/mapped automatically.
            model_id_override: Optional override for the model ID.

        Returns:
            Configured OpenAIModel instance.
        """
        cfg = ModelFactory._load_config(config_path)
        
        # Determine model ID
        # Priority: override > env var > config > default
        default_model = cfg.get("openai", {}).get("chat_model", "gpt-4.1-mini")
        model_id = model_id_override or os.getenv("OPENAI_CHAT_MODEL", default_model)
        
        # Ensure API key is set (OpenAIModel likely reads this from env, but good to check)
        api_key_var = cfg.get("openai", {}).get("api_key_env_var", "OPENAI_API_KEY")
        if not os.getenv(api_key_var) and not os.getenv("OPENAI_API_KEY"):
             # We don't raise here to allow mock modes to proceed if they handle it,
             # but we warn or let OpenAIModel fail if it needs the key.
             pass

        # Prepare parameters
        final_params = {}
        if model_params:
            # Map token parameter for newer OpenAI Responses API
            max_tokens = (
                model_params.get("max_completion_tokens")
                or model_params.get("max_output_tokens")
                or model_params.get("max_tokens")
            )
            if max_tokens:
                val = int(max_tokens)
                final_params["max_completion_tokens"] = val
                # Also set max_tokens for compatibility with frameworks/models that check it
                final_params["max_tokens"] = val
            
            # Pass through other relevant parameters
            for k, v in model_params.items():
                if k not in ["max_completion_tokens", "max_output_tokens", "max_tokens"]:
                    final_params[k] = v

        return OpenAIModel(
            model_id=model_id,
            params=final_params
        )
