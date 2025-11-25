"""
Backlog Synthesizer Supervisor Agent using AWS Strands Framework
Orchestrates specialized agents and tools for backlog synthesis workflow
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import base64
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry

# Import specialized agents and tools
from agents.segmentation_agent import create_segmentation_agent
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.tagging_agent import create_tagging_agent
from tools.retrieval_tool import create_retrieval_tool
from agents.evaluation_agent import create_evaluation_agent


class SupervisorAgent:
    """
    Supervisor agent that orchestrates the backlog synthesis workflow.
    Currently implements passthrough to LLM for POC.
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
        
        # Initialize OpenAI model for Strands
        # Note: Strands OpenAIModel reads API key from OPENAI_API_KEY environment variable
        model_config = {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9,
        }
        
        self.model = OpenAIModel(
            model_id=self.config["openai"]["chat_model"],
            params=model_config
        )
        
        # System prompt for the supervisor
        self.system_prompt = """You are BacklogSynthAI, a sophisticated orchestrator for software backlog synthesis.

Your role is to:
1. Analyze incoming user requests about meeting notes and documents
2. Coordinate specialized agents and tools to fulfill user requests
3. Guide users through the backlog synthesis workflow
4. Orchestrate document segmentation, context retrieval, backlog generation, and tagging

Available specialized agents and tools:
- segment_document: Splits documents into coherent segments with intent detection
- generate_backlog: Creates epics, features, and user stories from segments
- tag_story: Tags stories relative to existing backlog as new/gap/conflict
- retrieve_context: Retrieves relevant ADO items and architecture constraints
- evaluate_backlog_quality: Evaluates the quality of generated backlog items

Workflow:
1. User uploads document → Use segment_document tool
2. For each segment → Use retrieve_context to get relevant existing work
3. Generate backlog items → Use generate_backlog tool
4. Tag each story → Use tag_story tool
5. Optionally write to ADO

Always route requests to the appropriate specialized agent or tool. Be helpful, clear, and focused on completing the user's workflow."""
        
        # Track current run_id for tools
        self.current_run_id = None
        
        # Agent will be initialized per-run with appropriate tools
        # (similar to teachers_assistant pattern)
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
                evaluation_agent
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

You have access to a 'segment_document' tool that can split this document into coherent segments with intent detection. Use it when the user asks to analyze, segment, or process the document.
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
    
    async def segment_document(self, run_id: str, document_text: str) -> Dict[str, Any]:
        """
        Segment document into coherent chunks with intent labels.
        This is now implemented as a specialized agent in segmentation_agent.py
        
        Args:
            run_id: Unique identifier for this run
            document_text: Full text of document to segment
            
        Returns:
            Dict with segmentation results
        """
        # Create and invoke segmentation agent directly
        segmentation_agent = create_segmentation_agent(run_id)
        result_json = segmentation_agent(document_text)
        
        # Parse and return result
        result = json.loads(result_json)
        return result
    
    async def generate_backlog(self, run_id: str, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate backlog items from a segment with retrieved context.
        To be implemented in future iteration.
        """
        return {
            "status": "not_implemented",
            "message": "Backlog generation will be implemented in next iteration"
        }
    
    async def tag_story(self, run_id: str, story: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tag a user story relative to existing backlog.
        To be implemented in future iteration.
        """
        return {
            "status": "not_implemented",
            "message": "Story tagging will be implemented in next iteration"
        }
