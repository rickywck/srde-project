"""
Backlog Synthesizer Supervisor Agent using AWS Strands Framework
Orchestrates specialized agents and tools for backlog synthesis workflow
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry

# Import specialized agents and tools
from agents.segmentation_agent import create_segmentation_agent
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.tagging_agent import create_tagging_agent
from tools.retrieval_tool import create_retrieval_tool
from tools.ado_writer_tool import create_ado_writer_tool
from agents.evaluation_agent import create_evaluation_agent
from agents.prompt_loader import get_prompt_loader


class SupervisorAgent:
    """
    Supervisor agent that orchestrates the backlog synthesis workflow.
    Uses AWS Strands framework to coordinate specialized agents for:
    - Document segmentation with intent detection
    - Context retrieval from vector stores
    - Backlog item generation (Epics, Features, Stories)
    - Story tagging and classification
    - Quality evaluation
    """
    
    def __init__(self, config_path: str = "config.poc.yaml"):
        """Initialize supervisor with configuration"""
        self.config = self._load_config(config_path)
        
        # Initialize OpenAI configuration
        api_key = os.getenv(self.config["openai"]["api_key_env_var"])
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {self.config['openai']['api_key_env_var']}")
        
        # Ensure OpenAI API key is set in environment for Strands and child agents
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_CHAT_MODEL"] = self.config["openai"]["chat_model"]
        
        # Configure OpenTelemetry for observability (optional)
        # Uncomment and configure if using Langfuse or other OTEL endpoint
        # os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
        # self.telemetry = StrandsTelemetry().setup_otlp_exporter()
        
        # Load prompts from external configuration
        prompt_loader = get_prompt_loader()
        prompt_config = prompt_loader.load_prompt("supervisor_agent")
        self.system_prompt = prompt_loader.get_system_prompt("supervisor_agent")
        model_params = prompt_loader.get_parameters("supervisor_agent")
        
        # Initialize OpenAI model for Strands
        self.model = OpenAIModel(
            model_id=self.config["openai"]["chat_model"],
            params=model_params
        )
        
        # Track current run_id for tools
        self.current_run_id = None
        
        # Agent will be initialized per-run with appropriate tools
        self.agent = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            # Return default config if file doesn't exist
            return {
                "openai": {
                    "api_key_env_var": "OPENAI_API_KEY",
                    "chat_model": "gpt-4o"
                }
            }
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    async def process_message(
        self,
        run_id: str,
        message: str,
        instruction_type: Optional[str] = None,
        document_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message using Strands agent orchestration.
        
        Args:
            run_id: Unique identifier for this run
            message: User's message
            instruction_type: Optional type hint for the instruction
            document_text: Optional document content for context
        
        Returns:
            Dict with response and status information
        """
        
        # Update current run_id
        self.current_run_id = run_id
        
        # Create specialized agents and tools for this run
        # Following the teachers_assistant pattern where each agent/tool is created as needed
        segmentation_agent = create_segmentation_agent(run_id)
        backlog_generation_agent = create_backlog_generation_agent(run_id)
        tagging_agent = create_tagging_agent(run_id)
        retrieval_tool = create_retrieval_tool(run_id)
        evaluation_agent = create_evaluation_agent(run_id)
        ado_writer_tool = create_ado_writer_tool(run_id)
        
        # Create/update the Strands agent with all specialized agents as tools
        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            callback_handler=None,
            tools=[
                segmentation_agent,
                backlog_generation_agent,
                tagging_agent,
                retrieval_tool,
                evaluation_agent,
                ado_writer_tool
            ],
            trace_attributes={
                "service.name": "backlog-synthesizer",
                "agent.type": "supervisor",
                "run.id": run_id,
                "langfuse.tags": [
                    "Backlog-Synthesizer-POC",
                    "Strands-Agent",
                    "OpenAI"
                ]
            }
        )
        
        # Build context-aware query for the agent
        query_parts = []
        
        # Add document context if available
        if document_text:
            context_msg = f"""[CONTEXT] The user has uploaded a document. Here's the content:

--- DOCUMENT START ---
{document_text[:3000]}{"..." if len(document_text) > 3000 else ""}
--- DOCUMENT END ---
"""
            query_parts.append(context_msg)
        
        # Add instruction type hint if provided
        if instruction_type:
            query_parts.append(f"[INSTRUCTION TYPE: {instruction_type}]")
        
        # Add user message
        query_parts.append(f"[USER QUERY] {message}")
        
        # Combine all parts
        full_query = "\n\n".join(query_parts)
        
        try:
            # Call Strands agent - it will orchestrate tools and sub-agents as needed
            response = self.agent(full_query)
            
            # Extract response content
            assistant_message = str(response)
            
            return {
                "response": assistant_message,
                "status": {
                    "run_id": run_id,
                    "model": self.config["openai"]["chat_model"],
                    "has_document": document_text is not None,
                    "mode": "strands_orchestration",
                    "framework": "aws_strands",
                    "agents_available": [
                        "segment_document",
                        "generate_backlog",
                        "tag_story",
                        "retrieve_context",
                        "evaluate_backlog_quality"
                    ],
                    "tools_invoked": getattr(response, 'tool_calls', []) if hasattr(response, 'tool_calls') else []
                }
            }
        
        except Exception as e:
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "status": {
                    "run_id": run_id,
                    "error": str(e),
                    "mode": "strands_orchestration",
                    "framework": "aws_strands"
                }
            }
