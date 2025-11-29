"""
Evaluation Agent - LLM-as-a-judge for generated backlog quality
Implements live and batch evaluation modes.
"""
import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from .prompt_loader import get_prompt_loader
from .model_factory import ModelFactory
import logging

EVALUATION_SCHEMA = {
    "completeness": {"score": "int (1-5)", "reasoning": "string"},
    "relevance": {"score": "int (1-5)", "reasoning": "string"},
    "quality": {"score": "int (1-5)", "reasoning": "string"},
    "overall_score": "float",
    "summary": "string",
}

def _mock_evaluation() -> Dict[str, Any]:
    return {
        "status": "success_mock",
        "evaluation": {
            "completeness": {"score": 4, "reasoning": "Most key points captured in backlog."},
            "relevance": {"score": 5, "reasoning": "Items are tightly linked to segment intents."},
            "quality": {"score": 4, "reasoning": "Titles and ACs are clear with minor improvement room."},
            "overall_score": 4.33,
            "summary": "Backlog items are generally strong and actionable."
        }
    }


def create_evaluation_agent(run_id: str):
    """Create an evaluation agent tool for a specific run.

    Returns a tool function evaluate_backlog_quality that accepts a JSON string with:
    {
        "segment_text": str,
        "retrieved_context": {"ado_items": [...], "architecture_constraints": [...]},
        "generated_backlog": [...],
        "evaluation_mode": "live" | "batch"
    }
    """
    # Module logger
    logger = logging.getLogger(__name__)

    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    evaluation_system_prompt = prompt_loader.get_system_prompt("evaluation_agent")
    params = prompt_loader.get_parameters("evaluation_agent") or {}
    eval_config = prompt_loader.load_prompt("evaluation_agent")
    evaluation_schema = eval_config.get("evaluation_schema", EVALUATION_SCHEMA)

    # Load app config via ModelFactory
    config_path = "config.poc.yaml"
    try:
        _cfg = ModelFactory._load_config(config_path)
        logger.debug("Loaded config for evaluation agent: %s", {k: v for k, v in (_cfg or {}).items()})
    except Exception as e:
        logger.exception("Error loading config via ModelFactory: %s", e)
        _cfg = {}

    # Determine effective max tokens (priority: agent prompt params -> app config -> model default)
    agent_max_tokens = params.get("max_completion_tokens") or params.get("max_output_tokens") or params.get("max_tokens")
    app_max_tokens = _cfg.get("openai", {}).get("max_tokens")
    if agent_max_tokens is not None:
        eff_max_tokens = int(agent_max_tokens)
    elif app_max_tokens is not None:
        eff_max_tokens = int(app_max_tokens)
    else:
        eff_max_tokens = None

    # Build model via ModelFactory to centralize defaults and param mapping
    model = None
    model_id = None
    model_params = {}
    if eff_max_tokens is not None:
        model_params["max_completion_tokens"] = eff_max_tokens
    try:
        model_descriptor = ModelFactory.create_openai_model(config_path=config_path, model_params=model_params)
        model = model_descriptor
        model_id = getattr(model_descriptor, "model_id", None) or ModelFactory.get_default_model_id(config_path)
        logger.debug("Evaluation agent model descriptor: model_id=%s params=%s", model_id, getattr(model_descriptor, "params", {}))
    except Exception as e:
        logger.exception("ModelFactory.create_openai_model failed for evaluation agent: %s", e)
        model_id = ModelFactory.get_default_model_id(config_path)
        try:
            model = OpenAIModel(model_id=model_id, params={"max_completion_tokens": eff_max_tokens} if eff_max_tokens else None)
        except Exception:
            model = None

    @tool
    def evaluate_backlog_quality(input_json: str) -> str:
        try:
            payload = json.loads(input_json)
        except json.JSONDecodeError as e:
            return json.dumps({"status": "error", "error": f"Invalid JSON payload: {e}"})

        segment_text = payload.get("segment_text", "")
        retrieved_context = payload.get("retrieved_context", {})
        generated_backlog = payload.get("generated_backlog", [])
        evaluation_mode = payload.get("evaluation_mode", "live")

        # Mock mode
        if os.getenv("EVALUATION_AGENT_MOCK") == "1":
            mock = _mock_evaluation()
            mock.update({
                "run_id": run_id,
                "segment_length": len(segment_text),
                "items_evaluated": len(generated_backlog),
                "mode": evaluation_mode,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Persist if live mode (even in mock mode for testing)
            if evaluation_mode == "live":
                out_dir = Path(f"runs/{run_id}")
                out_dir.mkdir(parents=True, exist_ok=True)
                eval_file = out_dir / "evaluation.jsonl"
                with open(eval_file, "a") as f:
                    f.write(json.dumps(mock) + "\n")
            return json.dumps(mock, indent=2)

        if not model:
            return json.dumps({"status": "error", "error": "No model available (OPENAI_API_KEY not set or ModelFactory failed)"}, indent=2)

        # Build evaluation prompt using template
        ado_items_fmt = []
        for item in retrieved_context.get("ado_items", [])[:5]:
            ado_items_fmt.append(f"- [{item.get('work_item_id','?')}] {item.get('title','')} :: {item.get('description','')[:120]}")
        arch_fmt = []
        for c in retrieved_context.get("architecture_constraints", [])[:5]:
            arch_fmt.append(f"- {c.get('file_name','constraint')} :: {c.get('text','')[:120]}")
        backlog_fmt = []
        for bi in generated_backlog[:15]:
            acs = bi.get("acceptance_criteria", [])
            backlog_fmt.append(
                f"- {bi.get('type','?')}: {bi.get('title','')}\n  Desc: {bi.get('description','')[:160]}\n  ACs: " + "; ".join(acs[:5])
            )

        user_prompt = prompt_loader.format_user_prompt(
            "evaluation_agent",
            segment_text=segment_text[:4000],
            ado_items_formatted=os.linesep.join(ado_items_fmt) if ado_items_fmt else "None",
            architecture_constraints_formatted=os.linesep.join(arch_fmt) if arch_fmt else "None",
            backlog_items_formatted=os.linesep.join(backlog_fmt) if backlog_fmt else "None",
            evaluation_schema=json.dumps(evaluation_schema, indent=2)
        )

        try:
            agent = Agent(model=model, system_prompt=evaluation_system_prompt, tools=[], callback_handler=None)
            llm_response = agent(user_prompt)
            llm_text = str(llm_response)
            eval_obj = json.loads(llm_text)
            # Basic validation
            for key in ["completeness", "relevance", "quality"]:
                if key not in eval_obj or "score" not in eval_obj[key]:
                    raise ValueError(f"Evaluation missing {key}.score")
            if "overall_score" not in eval_obj:
                scores = [eval_obj[k]["score"] for k in ["completeness", "relevance", "quality"]]
                eval_obj["overall_score"] = round(sum(scores) / len(scores), 2)
            result = {
                "status": "success",
                "run_id": run_id,
                "segment_length": len(segment_text),
                "items_evaluated": len(generated_backlog),
                "evaluation": eval_obj,
                "mode": evaluation_mode,
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": model_id,
            }
            # Persist if live mode
            if evaluation_mode == "live":
                out_dir = Path(f"runs/{run_id}")
                out_dir.mkdir(parents=True, exist_ok=True)
                eval_file = out_dir / "evaluation.jsonl"
                with open(eval_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.exception("Evaluation agent failed: %s", e)
            return json.dumps({"status": "error", "error": f"Evaluation failed: {e}", "run_id": run_id}, indent=2)

    return evaluate_backlog_quality
