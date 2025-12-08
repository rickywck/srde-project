"""
Backlog Regeneration Agent - Updates existing backlog items based on user instructions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
from pydantic import BaseModel, ValidationError

from strands import tool, Agent
from strands.types.exceptions import StructuredOutputException

from .prompt_loader import get_prompt_loader
from .model_factory import ModelFactory
from tools.utils.token_utils import estimate_tokens
from .utils.backlog_helper import BacklogHelper

logger = logging.getLogger(__name__)


BacklogInput = Union[Dict[str, Any], List[Dict[str, Any]], None]


class BacklogItemOut(BaseModel):
    type: str
    title: str
    description: str | None = None
    acceptance_criteria: List[str] | None = None
    parent_reference: str | None = None
    rationale: str | None = None

    class Config:
        extra = "allow"


class BacklogResponseOut(BaseModel):
    backlog_items: List[BacklogItemOut]

    class Config:
        extra = "allow"


def _as_int(value, default):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _load_backlog_from_file(path: Path) -> List[Dict[str, Any]]:
    """Load backlog entries from a JSONL file."""
    items: List[Dict[str, Any]] = []
    if not path.exists():
        return items
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    items.append(data)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed backlog line in %s", path)
    return items


def create_backlog_regeneration_agent(run_id: str, default_backlog_file: str | None = None):
    """Create a backlog regeneration agent tool for a specific run."""
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("backlog_regeneration_agent")
    prompt_params = prompt_loader.get_parameters("backlog_regeneration_agent") or {}

    MAX_ITEMS = _as_int(prompt_params.get("max_backlog_items_in_prompt", 200), 200)

    # Initialize the model
    try:
        model = ModelFactory.create_openai_model_for_agent(agent_params=prompt_params)
        model_name = getattr(model, "model_id", None) or ModelFactory.get_default_model_id()
        logger.debug("Initialized OpenAIModel for backlog regeneration: %s", model_name)
    except Exception as exc:
        logger.exception("Failed to create model for backlog regeneration: %s", exc)
        model = None
        model_name = ModelFactory.get_default_model_id()

    agent = None
    if model is not None:
        try:
            agent = Agent(model=model, system_prompt=system_prompt)
        except Exception as exc:
            logger.exception("Failed to initialize Strands Agent for backlog regeneration: %s", exc)
            agent = None

    default_file_path = Path(default_backlog_file or f"runs/{run_id}/generated_backlog.jsonl")

    @tool
    def regenerate_backlog(
        user_instructions: str,
        existing_backlog: BacklogInput = None,
        backlog_file_path: str | None = None,
    ) -> str:
        """Regenerate backlog items using explicit user instructions.

        Args:
            user_instructions: Free-form text describing how the existing backlog should change.
            existing_backlog: Parsed JSON payload describing the backlog to revise. Accepts a
                list of backlog objects or a single backlog object. Pass `null` to make the tool
                load the latest backlog from disk automatically.
            backlog_file_path: Optional override path for the backlog file when `existing_backlog`
                is `null`. Defaults to `runs/<run_id>/generated_backlog.jsonl`.

        Returns:
            JSON string summarizing the regeneration operation with keys: status, run_id,
            items_generated, backlog_file, item_counts, and backlog_items.

        Notes:
            - Callers MUST supply parsed JSON objects (dict or list of dict). Strings and
              filesystem paths are not supported.
            - When `existing_backlog` is `null`, the tool loads the backlog from
              `backlog_file_path` (or the default run file).
            - The tool overwrites the target backlog file with the LLM's response, so only use it
              after an initial backlog has been generated.
        """
        instructions = (user_instructions or "").strip()
        if not instructions:
            error = {
                "status": "error",
                "error": "User instructions are required to regenerate backlog items.",
                "run_id": run_id,
            }
            return json.dumps(error, indent=2)

        target_path = Path(backlog_file_path or default_file_path)
        existing_items: List[Dict[str, Any]] = []
        if existing_backlog is None:
            existing_items = _load_backlog_from_file(target_path)
        elif isinstance(existing_backlog, list):
            if not all(isinstance(item, dict) for item in existing_backlog):
                error = {
                    "status": "error",
                    "error": "existing_backlog must be a list of JSON objects or null.",
                    "run_id": run_id,
                }
                return json.dumps(error, indent=2)
            existing_items = existing_backlog
        elif isinstance(existing_backlog, dict):
            existing_items = [existing_backlog]
        else:
            error = {
                "status": "error",
                "error": "existing_backlog must be a dict, list[dict], or null.",
                "run_id": run_id,
            }
            return json.dumps(error, indent=2)

        if not existing_items:
            error = {
                "status": "error",
                "error": "No existing backlog items found to regenerate.",
                "run_id": run_id,
                "backlog_file": str(target_path),
            }
            return json.dumps(error, indent=2)

        trimmed_items = existing_items[:MAX_ITEMS]
        prompt = prompt_loader.format_user_prompt(
            "backlog_regeneration_agent",
            existing_backlog_formatted=json.dumps(trimmed_items, indent=2),
            shown_item_count=len(trimmed_items),
            total_item_count=len(existing_items),
            user_instructions=instructions,
        )

        sys_tok = estimate_tokens(system_prompt)
        usr_tok = estimate_tokens(prompt)
        logger.debug(
            "Backlog Regeneration Agent tokens: system=%s user=%s total≈%s",
            sys_tok,
            usr_tok,
            sys_tok + usr_tok,
        )

        if agent is None:
            error = {
                "status": "error",
                "error": "Backlog Regeneration Agent not initialized. No model available.",
                "run_id": run_id,
            }
            return json.dumps(error, indent=2)

        logger.info(
            "Backlog Regeneration Agent: Updating %s items for run %s", len(existing_items), run_id
        )
        try:
            # Call agent WITHOUT structured_output_model since the prompt already
            # specifies response_format: json_object
            result = agent(prompt)
            
            # Extract the text response
            response_text = ""
            if hasattr(result, 'output'):
                response_text = result.output
            elif hasattr(result, 'text'):
                response_text = result.text
            elif hasattr(result, 'content'):
                response_text = result.content
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)
            
            logger.info("Backlog Regeneration Agent: Received response length: %d", len(response_text))
            
            # Parse JSON response
            response_data = json.loads(response_text)
            
            # Validate with Pydantic
            validated: BacklogResponseOut = BacklogResponseOut(**response_data)
            
            # Convert to plain dicts
            raw_items: List[Dict[str, Any]] = []
            for model_item in validated.backlog_items:
                try:
                    raw_items.append(
                        model_item.model_dump() if hasattr(model_item, "model_dump") else model_item.dict()
                    )
                except Exception:
                    raw_items.append(dict(model_item))
            
            logger.info("Backlog Regeneration Agent: Validated and extracted %s backlog items", len(raw_items))
            
        except json.JSONDecodeError as exc:
            logger.error("Backlog Regeneration Agent: Failed to parse JSON response: %s", exc)
            error = {
                "status": "error",
                "error": f"Failed to parse response as JSON: {exc}",
                "run_id": run_id,
            }
            return json.dumps(error, indent=2)
        except ValidationError as exc:
            logger.error("Backlog Regeneration Agent: Pydantic validation failed: %s", exc)
            error = {
                "status": "error",
                "error": f"Backlog regeneration validation failed: {exc}",
                "run_id": run_id,
            }
            return json.dumps(error, indent=2)
        except Exception as exc:
            logger.exception("Backlog Regeneration Agent invocation failed: %s", exc)
            error = {
                "status": "error",
                "error": f"Backlog regeneration agent invocation failed: {exc}",
                "run_id": run_id,
            }
            return json.dumps(error, indent=2)

        if not raw_items:
            logger.warning("Backlog Regeneration Agent returned no items; existing backlog left untouched")
            summary = {
                "status": "warning",
                "message": "Regeneration returned no backlog items; original backlog retained.",
                "run_id": run_id,
                "backlog_file": str(target_path),
                "items_generated": 0,
            }
            return json.dumps(summary, indent=2)

        # Normalize and annotate (regen id mode)
        processed = BacklogHelper.normalize_items(raw_items, run_id=run_id, segment_id=None, id_mode="regen")

        # Overwrite file with new content
        BacklogHelper.write_jsonl(processed, target_path, mode="w")

        summary = BacklogHelper.summarize(
            run_id=run_id, backlog_file=target_path, items=processed, segment_id=0
        )
        return json.dumps(summary, indent=2)

    return regenerate_backlog
