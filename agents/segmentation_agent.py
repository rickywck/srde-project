"""
Segmentation Agent - Specialized agent for document segmentation with intent detection
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, ValidationError
from strands import Agent, tool
from strands.types.exceptions import StructuredOutputException
from .prompt_loader import get_prompt_loader
from .model_factory import ModelFactory

# Module logger
logger = logging.getLogger(__name__)


class SegmentOut(BaseModel):
    segment_id: int
    segment_order: int
    raw_text: str
    intent_labels: List[str]
    dominant_intent: str

    class Config:
        extra = "allow"


class SegmentationResponseOut(BaseModel):
    segments: List[SegmentOut]

    class Config:
        extra = "allow"


def create_segmentation_agent(run_id: str):
    """
    Create a segmentation agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    # Load prompts from external configuration
    prompt_loader = get_prompt_loader()
    prompt_loader.load_prompt("segmentation_agent")
    system_prompt = prompt_loader.get_system_prompt("segmentation_agent")
    prompt_loader.get_user_prompt_template("segmentation_agent")
    prompt_params = prompt_loader.get_parameters("segmentation_agent") or {}

    # Create a Strands OpenAIModel using factory helper (agent does not access config or API key directly)
    try:
        model = ModelFactory.create_openai_model_for_agent(agent_params=prompt_params)
        model_name = getattr(model, "model_id", None) or ModelFactory.get_default_model_id()
        logger.debug("Initialized Strands OpenAIModel for segmentation: %s", model_name)
    except Exception as e:
        logger.exception("Failed to create Strands OpenAIModel for segmentation: %s", e)
        model = None
        model_name = ModelFactory.get_default_model_id()

    # Initialize Strands Agent
    agent = None
    if model is not None:
        try:
            # Prevent tool/function calls; ensure pure structured JSON responses
            try:
                agent = Agent(
                    model=model,
                    system_prompt=system_prompt,
                    disable_tool_calls=True,
                    max_tool_calls=0,
                )
            except TypeError:
                agent = Agent(model=model, system_prompt=system_prompt)
        except Exception as e:
            logger.exception("Failed to initialize Strands Agent (segmentation): %s", e)
            agent = None
    
    @tool
    def segment_document(document_text: str) -> str:
        """
        Segments a document into coherent chunks with intent detection.
        
        Args:
            document_text: The full text of the document to segment
            
        Returns:
            JSON string containing segmentation results with segment_id, raw_text, intent_labels, and dominant_intent
        """
        logger.debug("segment_document called with: run_id=%r, document_text_length=%d, document_text_preview=%s...",
                     run_id, len(document_text) if document_text else 0, document_text[:100] if document_text else None)
        
        # Build segmentation prompt from template
        segmentation_prompt = prompt_loader.format_user_prompt(
            "segmentation_agent",
            document_text=document_text
        )
        
        try:
            logger.info("Segmentation Agent: Processing document (run_id: %s)", run_id)
            # Require live LLM; no mock mode
            if agent is None:
                raise ValueError("Segmentation agent not initialized. No model available.")

            # Call agent WITHOUT structured_output_model since the prompt already
            # specifies response_format: json_object and returns a complete segments array
            try:
                agent_result = agent(segmentation_prompt)
                
                # Extract the text response
                response_text = ""
                if hasattr(agent_result, 'output'):
                    response_text = agent_result.output
                elif hasattr(agent_result, 'text'):
                    response_text = agent_result.text
                elif hasattr(agent_result, 'content'):
                    response_text = agent_result.content
                elif isinstance(agent_result, str):
                    response_text = agent_result
                else:
                    response_text = str(agent_result)
                
                logger.info("Segmentation Agent: Received response length: %d", len(response_text))
                
                # Parse JSON response
                result = json.loads(response_text)
                
                # Validate with Pydantic
                validated: SegmentationResponseOut = SegmentationResponseOut(**result)
                
                # Convert to plain dicts
                segments_list = []
                for seg in validated.segments:
                    seg_dict = seg.model_dump() if hasattr(seg, "model_dump") else seg.dict()
                    segments_list.append(seg_dict)
                
                result = {"segments": segments_list}
                logger.info("Segmentation Agent: Validated and extracted %s segments", len(segments_list))
                
            except json.JSONDecodeError as e:
                logger.error("Segmentation Agent: Failed to parse JSON response: %s", e)
                raise ValueError(f"Failed to parse response as JSON: {e}")
            except ValidationError as e:
                logger.error("Segmentation Agent: Pydantic validation failed: %s", e)
                raise ValueError(f"Response validation failed: {e}")
            
            # Validate structure
            if "segments" not in result:
                raise ValueError("Response missing 'segments' key")
            
            segments = result["segments"]
            
            # Ensure output directory exists
            output_dir = Path(f"runs/{run_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write segments to JSONL file
            segments_file = output_dir / "segments.jsonl"
            with open(segments_file, "w") as f:
                for segment in segments:
                    f.write(json.dumps(segment) + "\n")
            
            logger.info("Segmentation Agent: Created %s segments", len(segments))
            
            # Prepare summary for display
            summary = {
                "status": "success",
                "run_id": run_id,
                "total_segments": len(segments),
                "segments_file": str(segments_file),
                "segments": segments
            }
            
            return json.dumps(summary, indent=2)
            
        except json.JSONDecodeError as e:
            error_msg = {
                "status": "error",
                "error": f"Failed to parse response as JSON: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)

        except Exception as e:
            error_msg = {
                "status": "error",
                "error": f"Segmentation failed: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
    
    return segment_document


# Note: System prompt now loaded from prompts/segmentation_agent.yaml
# This ensures consistency and easier prompt management
