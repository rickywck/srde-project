"""
Evaluation Agent - LLM-as-a-judge for generated backlog quality
Implements live and batch evaluation modes.
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field, ValidationError
from strands import Agent, tool
from strands.types.exceptions import StructuredOutputException
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


# NOTE: We do not read cached evaluations. Persist only the latest result by overwriting the file.


class ScoreReason(BaseModel):
    score: int = Field(ge=1, le=5, description="Score from 1 to 5")
    reasoning: str


class EvaluationOut(BaseModel):
    completeness: ScoreReason
    relevance: ScoreReason
    quality: ScoreReason
    overall_score: Optional[float] = None
    summary: str

    class Config:
        extra = "allow"


def create_evaluation_agent(run_id: str):
    """Create an evaluation agent tool for a specific run.

    Returns a tool function evaluate_backlog_quality that accepts a JSON string with:
    {
        "segment_text": str,
        "generated_backlog": [...],
        "evaluation_mode": "live" | "batch"
    }
    
    Evaluates generated backlog items against the original document text only.
    """
    # Module logger
    logger = logging.getLogger(__name__)

    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    evaluation_system_prompt = prompt_loader.get_system_prompt("evaluation_agent")
    params = prompt_loader.get_parameters("evaluation_agent") or {}
    eval_config = prompt_loader.load_prompt("evaluation_agent")
    evaluation_schema = eval_config.get("evaluation_schema", EVALUATION_SCHEMA)
    
    # Build model via ModelFactory helper; no direct config or API key access here
    try:
        model = ModelFactory.create_openai_model_for_agent(agent_params=params)
        model_id = getattr(model, "model_id", None) or ModelFactory.get_default_model_id()
        logger.debug("Evaluation agent model initialized: %s", model_id)
    except Exception as e:
        logger.exception("Failed to create model for evaluation agent: %s", e)
        model = None
        model_id = ModelFactory.get_default_model_id()

    @tool
    def evaluate_backlog_quality(input_json: str) -> str:
        """Evaluate the quality of generated backlog items against the original document.
        
        This tool performs LLM-as-a-judge evaluation to assess backlog quality across
        three dimensions: completeness, relevance, and quality.
        
        Args:
            input_json: A JSON string containing the evaluation request with the following structure:
                {
                    "segment_text": "<full original document text to evaluate against>",
                    "generated_backlog": [
                        {
                            "type": "Epic|Feature|User Story",
                            "title": "<item title>",
                            "description": "<item description>",
                            "acceptance_criteria": ["<criterion 1>", "<criterion 2>", ...]
                        },
                        ...
                    ],
                    "evaluation_mode": "live|batch"  (optional, defaults to "live")
                }
        
        Returns:
            JSON string with evaluation results:
            {
                "status": "success|error",
                "run_id": "<run_id>",
                "evaluation": {
                    "completeness": {"score": 1-5, "reasoning": "..."},
                    "relevance": {"score": 1-5, "reasoning": "..."},
                    "quality": {"score": 1-5, "reasoning": "..."},
                    "overall_score": <float>,
                    "summary": "..."
                },
                "segment_length": <int>,
                "items_evaluated": <int>,
                "mode": "live|batch",
                "timestamp": "<ISO timestamp>",
                "model_used": "<model_id>"
            }
        
        Example:
            input_json = json.dumps({
                "segment_text": "The system shall support user authentication...",
                "generated_backlog": [
                    {
                        "type": "User Story",
                        "title": "User Login",
                        "description": "As a user, I want to log in...",
                        "acceptance_criteria": ["User can enter credentials", "System validates input"]
                    }
                ],
                "evaluation_mode": "live"
            })
            result = evaluate_backlog_quality(input_json)
        """
        logger.debug("evaluate_backlog_quality called with: run_id=%r, input_json=%s...",
                     run_id, input_json[:200] if input_json else None)
        
        try:
            payload = json.loads(input_json)
        except json.JSONDecodeError as e:
            return json.dumps({"status": "error", "error": f"Invalid JSON payload: {e}"})

        segment_text = payload.get("segment_text", "")
        generated_backlog = payload.get("generated_backlog", [])
        evaluation_mode = payload.get("evaluation_mode", "live")

        if not model:
            return json.dumps({"status": "error", "error": "No model available for evaluation.", "run_id": run_id}, indent=2)

        logger.debug("Evaluation agent model in live mode.")
        try:
            # Build evaluation prompt using template
            def _safe_str(val, max_len=None) -> str:
                """Coerce any value to a safe string and optionally truncate."""
                try:
                    s = "" if val is None else (val if isinstance(val, str) else str(val))
                except Exception:
                    s = ""
                return s[:max_len] if (max_len is not None) else s

            # Generated backlog context
            backlog_fmt = []
            for bi in (generated_backlog or [])[:15]:
                if not isinstance(bi, dict):
                    # Skip non-dict items to avoid runtime errors
                    continue
                # Acceptance criteria may be None, string, or list
                acs_raw = bi.get("acceptance_criteria")
                if acs_raw is None:
                    acs_list = []
                elif isinstance(acs_raw, (list, tuple)):
                    acs_list = [_safe_str(x, 160) for x in acs_raw if x is not None][:5]
                else:
                    acs_list = [_safe_str(acs_raw, 160)]

                bi_type = _safe_str(bi.get("type", "?"))
                bi_title = _safe_str(bi.get("title"), 200)
                bi_desc = _safe_str(bi.get("description"), 160)
                backlog_fmt.append(
                    f"- {bi_type}: {bi_title}\n  Desc: {bi_desc}\n  ACs: " + "; ".join(acs_list)
                )
            logger.debug("Evaluation agent model in input context prepared.")
        except Exception as e:
            logger.exception("Failed preparing evaluation input context: %s", e)
            raise

        user_prompt = prompt_loader.format_user_prompt(
            "evaluation_agent",
            segment_text=segment_text[:4000],
            backlog_items_formatted=os.linesep.join(backlog_fmt) if backlog_fmt else "None",
            evaluation_schema=json.dumps(evaluation_schema, indent=2)
        )
        logger.debug("Evaluation agent model input prompt: %s", user_prompt)

        try:
            agent = Agent(model=model, system_prompt=evaluation_system_prompt)
            agent_result = agent(
                user_prompt,
                structured_output_model=EvaluationOut,
            )
            logger.debug("Evaluation agent model output: %s", agent_result)
            evaluation: EvaluationOut = agent_result.structured_output  # type: ignore[assignment]

            # Ensure overall_score present; compute mean if missing
            if evaluation.overall_score is None:
                try:
                    scores = [
                        evaluation.completeness.score,
                        evaluation.relevance.score,
                        evaluation.quality.score,
                    ]
                    evaluation.overall_score = round(sum(scores) / len(scores), 2)
                except Exception:
                    pass

            logger.debug("Evaluation agent model output parsed: %s", evaluation)
            # Convert to plain dict for persistence/return
            try:
                eval_obj = evaluation.model_dump()
            except Exception:
                eval_obj = json.loads(evaluation.json())
            logger.debug("Evaluation agent model output dict: %s", eval_obj)

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
            # Persist if live mode — overwrite with latest evaluation
            if evaluation_mode == "live":
                out_dir = Path(f"runs/{run_id}")
                out_dir.mkdir(parents=True, exist_ok=True)
                eval_file = out_dir / "evaluation.jsonl"
                with open(eval_file, "w") as f:
                    f.write(json.dumps(result) + "\n")
            return json.dumps(result, indent=2)
        except (StructuredOutputException, ValidationError) as e:
            logger.exception("Evaluation structured output failed: %s", e)
            return json.dumps({
                "status": "error",
                "error": f"Evaluation structured output failed: {e}",
                "run_id": run_id,
                "model_used": model_id,
            }, indent=2)
        except Exception as e:
            logger.exception("Evaluation agent failed: %s", e)
            return json.dumps({
                "status": "error",
                "error": f"Evaluation failed: {e}",
                "run_id": run_id,
                "model_used": model_id,
            }, indent=2)

    return evaluate_backlog_quality
