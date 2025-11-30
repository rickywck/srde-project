import os
import yaml
import logging
from typing import Dict, Any, Optional
from strands.models.openai import OpenAIModel

# Module logger
logger = logging.getLogger(__name__)

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
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                    logger.debug("Loaded config from %s: %s", config_path, {k: v for k, v in (data or {}).items()})
                    return data
            # File missing: return sensible default
            logger.debug("Config file %s not found, using defaults", config_path)
            return {"openai": {"chat_model": "gpt-4.1-mini"}}
        except Exception as e:
            # Log exception and return a safe fallback config
            logger.exception("Failed to load config from %s: %s", config_path, e)
            return {"openai": {"chat_model": "gpt-4.1-mini"}}

    @staticmethod
    def get_default_model_id(config_path: str = "config.poc.yaml", model_id_override: Optional[str] = None) -> str:
        """
        Resolve the effective default model id using the same priority rules as
        `create_openai_model` without instantiating any model objects.

        Priority: override > env var > config default
        """
        try:
            cfg = ModelFactory._load_config(config_path)
        except Exception:
            cfg = {"openai": {"chat_model": "gpt-4o-mini"}}
        default_model = cfg.get("openai", {}).get("chat_model", "gpt-4o-mini")
        return model_id_override or os.getenv("OPENAI_CHAT_MODEL", default_model)

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
        try:
            cfg = ModelFactory._load_config(config_path)
        except Exception as e:
            # _load_config already logs, but be defensive here
            logger.debug("_load_config raised exception, using empty config: %s", e)
            cfg = {"openai": {"chat_model": "gpt-4.1-mini"}}
        
        # Determine model ID
        # Priority: override > env var > config > default
        default_model = cfg.get("openai", {}).get("chat_model", "gpt-4.1-mini")
        model_id = model_id_override or os.getenv("OPENAI_CHAT_MODEL", default_model)
        logger.debug("Selected model_id=%s (override=%s, env=%s, config_default=%s)", model_id, model_id_override, os.getenv("OPENAI_CHAT_MODEL"), default_model)
        
        # Ensure API key is set (OpenAIModel likely reads this from env, but good to check)
        api_key_var = cfg.get("openai", {}).get("api_key_env_var", "OPENAI_API_KEY")
        if not os.getenv(api_key_var) and not os.getenv("OPENAI_API_KEY"):
            # Do not raise here (caller may operate in mock mode), but log clearly for debugging
            logger.warning("OpenAI API key not found in environment (checked %s and OPENAI_API_KEY)", api_key_var)

        # Prepare parameters
        final_params = {}
        if model_params:
            # Map token parameter for newer OpenAI Responses API; be defensive about types
            max_tokens = (
                model_params.get("max_completion_tokens")
                or model_params.get("max_output_tokens")
                or model_params.get("max_tokens")
            )
            if max_tokens is not None:
                try:
                    val = int(max_tokens)
                    # Use the newer Responses API parameter `max_completion_tokens`.
                    # Do NOT set both `max_completion_tokens` and `max_tokens` together
                    # to avoid invalid parameter combination errors from the API.
                    final_params["max_completion_tokens"] = val
                except (ValueError, TypeError) as e:
                    logger.exception("Invalid max_tokens value: %s", max_tokens)

            # Pass through other relevant parameters, skipping token aliases already handled
            for k, v in model_params.items():
                if k not in ["max_completion_tokens", "max_output_tokens", "max_tokens"]:
                    final_params[k] = v

        logger.debug("Final model params to pass to OpenAIModel: %s", final_params)
        # Determine effective token cap from config when model_params did not provide one.
        # Agent-level cap (from generation.* in config)
        try:
            gen_cfg = cfg.get("generation", {}) or {}
            agent_max = gen_cfg.get("max_completion_tokens", gen_cfg.get("max_tokens"))
            agent_max_val = int(agent_max) if agent_max is not None else None
        except Exception:
            agent_max_val = None

        # Model-level cap (optional) that may be specified under openai or generation
        try:
            # Model-level cap MUST be specified under `openai.model_max_tokens`.
            model_cfg_max = cfg.get("openai", {}).get("model_max_tokens")
            model_cfg_max_val = int(model_cfg_max) if model_cfg_max is not None else None
        except Exception:
            model_cfg_max_val = None

        # Decide effective cap: if model-level cap present -> min(model_cap, agent_cap) when agent cap present.
        # If model-level cap absent -> use agent cap. Do not override explicit params provided by caller.
        if "max_completion_tokens" not in final_params and "max_tokens" not in final_params:
            eff_cap = None
            if model_cfg_max_val is not None and agent_max_val is not None:
                eff_cap = min(model_cfg_max_val, agent_max_val)
            elif model_cfg_max_val is not None:
                eff_cap = model_cfg_max_val
            elif agent_max_val is not None:
                eff_cap = agent_max_val

            if eff_cap is not None:
                final_params["max_completion_tokens"] = int(eff_cap)
                logger.debug("Injected effective token cap into final_params: %s", eff_cap)

        try:
            model = OpenAIModel(
                model_id=model_id,
                params=final_params
            )
            # Ensure callers can inspect mapped params and model id
            try:
                setattr(model, "params", final_params)
                setattr(model, "model_id", model_id)
            except Exception:
                logger.debug("Could not attach fallback attrs to OpenAIModel instance")
            logger.debug("Created OpenAIModel instance for model_id=%s", model_id)
            return model
        except Exception as e:
            # Log and raise a descriptive error so callers can decide how to proceed
            logger.exception("Failed to instantiate OpenAIModel for model_id=%s: %s", model_id, e)
            raise RuntimeError(f"Failed to create OpenAIModel for '{model_id}': {e}") from e
