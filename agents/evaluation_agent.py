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
from openai import OpenAI
from .prompt_loader import get_prompt_loader

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
    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=api_key) if api_key else None
    # Load default model from config, allow env override
    config_path = "config.poc.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            _cfg = yaml.safe_load(f) or {}
    else:
        _cfg = {"openai": {"chat_model": "gpt-4.1-mini"}}
    model_name = os.getenv("OPENAI_CHAT_MODEL", _cfg.get("openai", {}).get("chat_model", "gpt-4.1-mini"))

    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    evaluation_system_prompt = prompt_loader.get_system_prompt("evaluation_agent")
    params = prompt_loader.get_parameters("evaluation_agent")
    eval_config = prompt_loader.load_prompt("evaluation_agent")
    evaluation_schema = eval_config.get("evaluation_schema", EVALUATION_SCHEMA)

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

        if not openai_client:
            return json.dumps({"status": "error", "error": "OPENAI_API_KEY not set and mock mode not enabled"}, indent=2)

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
            # Normalize token parameter and omit temperature (some models only allow default=1)
            max_comp = (
                params.get("max_completion_tokens")
                or params.get("max_output_tokens")
                or params.get("max_tokens")
                or 1000
            )
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": evaluation_system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_comp,
                response_format={"type": params.get("response_format", "json_object")}
            )
            content = response.choices[0].message.content
            eval_obj = json.loads(content)
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
                "timestamp": datetime.utcnow().isoformat()
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
            return json.dumps({"status": "error", "error": f"Evaluation failed: {e}", "run_id": run_id}, indent=2)

    return evaluate_backlog_quality
