"""
Backlog Synthesizer Supervisor Agent using AWS Strands Framework
Orchestrates specialized agents and tools for backlog synthesis workflow
"""

import os
from typing import Dict, Any, Optional
import yaml
import base64
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry

class SupervisorAgent:
    """
    Supervisor agent that orchestrates the backlog synthesis workflow.
    Currently implements passthrough to LLM for POC.
    """
    
    def __init__(self, config_path: str = "config.poc.yaml"):
        """Initialize supervisor with configuration"""
        self.config = self._load_config(config_path)
        
        # Initialize OpenAI client
        api_key = os.getenv(self.config["openai"]["api_key_env_var"])
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {self.config['openai']['api_key_env_var']}")
        
        # Ensure OpenAI API key is set in environment for Strands
        os.environ["OPENAI_API_KEY"] = api_key
        
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
2. Help users understand their uploaded documents
3. Guide them through the backlog generation process
4. Coordinate specialized agents and tools when needed

Available capabilities (being implemented in iterations):
- Segment documents into coherent chunks with intent detection
- Generate epics, features, and user stories from segments
- Tag stories relative to existing backlog (new/gap/conflict)
- Retrieve relevant context from ADO backlog and architecture constraints
- Write items to Azure DevOps

Current mode: Intelligent assistant providing guidance and context-aware responses.
Always be helpful, clear, and focused on the user's needs."""
        
        # Initialize the Strands agent
        # Tools will be added in future iterations (segmentation, generation, tagging, retrieval, ADO writer)
        self.agent = Agent(
            model=self.model,
            system_prompt=self.system_prompt,
            callback_handler=None,
            tools=[],  # Will add tools in future iterations
            trace_attributes={
                "service.name": "backlog-synthesizer",
                "agent.type": "supervisor",
                "langfuse.tags": [
                    "Backlog-Synthesizer-POC",
                    "Strands-Agent",
                    "OpenAI"
                ]
            }
        )
    
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
                    "framework": "aws_strands"
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
        To be implemented in future iteration.
        """
        return {
            "status": "not_implemented",
            "message": "Document segmentation will be implemented in next iteration"
        }
    
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
