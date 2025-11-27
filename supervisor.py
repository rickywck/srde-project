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
from strands.session.file_session_manager import FileSessionManager

# Import specialized agents and tools
from agents.segmentation_agent import create_segmentation_agent
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.tagging_agent import create_tagging_agent
from tools.retrieval_tool import create_retrieval_tool
from tools.ado_writer_tool import create_ado_writer_tool
from agents.evaluation_agent import create_evaluation_agent
from agents.prompt_loader import get_prompt_loader
import base64

# Build Basic Auth header.
LANGFUSE_AUTH = base64.b64encode(
    f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
).decode()
 
# Configure OpenTelemetry endpoint & headers
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_BASE_URL") + "/api/public/otel"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"
#os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"

# Configure the telemetry
# (Creates new tracer provider and sets it as global)
strands_telemetry = StrandsTelemetry().setup_otlp_exporter()

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
    
    def __init__(self, config_path: str = "config.poc.yaml", sessions_dir: str = "sessions"):
        """Initialize supervisor with configuration and session management"""
        self.config = self._load_config(config_path)

        # Initialize OpenAI configuration
        api_key = os.getenv(self.config["openai"]["api_key_env_var"])
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {self.config['openai']['api_key_env_var']}")

        # Ensure OpenAI API key is set in environment for Strands and child agents
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_CHAT_MODEL"] = self.config["openai"]["chat_model"]

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

        # Session management
        self.sessions_dir = sessions_dir
        Path(self.sessions_dir).mkdir(exist_ok=True)
        self.agents_cache: Dict[str, Agent] = {}

        # Track current run_id for tools
        self.current_run_id = None
    
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
        Process a user message using Strands agent orchestration with session management.
        """
        import asyncio

        self.current_run_id = run_id

        # Track backlog file state before processing to detect new generation
        runs_dir = Path("runs")
        run_dir = runs_dir / run_id
        backlog_file = run_dir / "generated_backlog.jsonl"
        ado_result_file = run_dir / "ado_export_last.json"
        backlog_existed_before = backlog_file.exists()
        backlog_mtime_before = backlog_file.stat().st_mtime if backlog_existed_before else None
        backlog_size_before = backlog_file.stat().st_size if backlog_existed_before else None
        ado_existed_before = ado_result_file.exists()
        ado_mtime_before = ado_result_file.stat().st_mtime if ado_existed_before else None
        ado_size_before = ado_result_file.stat().st_size if ado_existed_before else None

        # Setup session manager for this run
        session_manager = FileSessionManager(
            session_id=run_id,
            storage_dir=self.sessions_dir
        )

        # Reuse agent if cached, else create and cache
        if run_id in self.agents_cache:
            agent = self.agents_cache[run_id]
        else:
            segmentation_agent = create_segmentation_agent(run_id)
            backlog_generation_agent = create_backlog_generation_agent(run_id)
            tagging_agent = create_tagging_agent(run_id)
            retrieval_tool = create_retrieval_tool(run_id)
            evaluation_agent = create_evaluation_agent(run_id)
            ado_writer_tool = create_ado_writer_tool(run_id)

            agent = Agent(
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
                session_manager=session_manager,
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
            self.agents_cache[run_id] = agent

        # Build context-aware query for the agent
        query_parts = []
        if document_text:
            context_msg = f"""[CONTEXT] The user has uploaded a document. Here's the content:

--- DOCUMENT START ---
{document_text[:3000]}{"..." if len(document_text) > 3000 else ""}
--- DOCUMENT END ---
"""
            query_parts.append(context_msg)
        if instruction_type:
            query_parts.append(f"[INSTRUCTION TYPE: {instruction_type}]")
        query_parts.append(f"[USER QUERY] {message}")
        full_query = "\n\n".join(query_parts)

        try:
            # Run agent in thread for async compatibility
            response = await asyncio.to_thread(agent, full_query)
            assistant_message = str(response)
            conversation_length = len(agent.messages) if hasattr(agent, "messages") else None
            result: Dict[str, Any] = {
                "response": assistant_message,
                "status": {
                    "run_id": run_id,
                    "model": self.config["openai"]["chat_model"],
                    "has_document": document_text is not None,
                    "mode": "strands_orchestration",
                    "framework": "aws_strands",
                    "session_managed": True,
                    "conversation_length": conversation_length,
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

            # Detect if ADO export result was produced during this message handling
            try:
                if ado_result_file.exists():
                    am_after = ado_result_file.stat().st_mtime
                    as_after = ado_result_file.stat().st_size
                    if (not ado_existed_before) or (am_after != ado_mtime_before) or (as_after != ado_size_before):
                        result["response_type"] = "ado_export"
                # If not ADO, detect backlog generation/update
                if "response_type" not in result and backlog_file.exists():
                    mtime_after = backlog_file.stat().st_mtime
                    size_after = backlog_file.stat().st_size
                    if (not backlog_existed_before) or (mtime_after != backlog_mtime_before) or (size_after != backlog_size_before):
                        result["response_type"] = "backlog_generated"
            except Exception:
                # Non-fatal; ignore detection errors
                pass

            return result
        except Exception as e:
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "status": {
                    "run_id": run_id,
                    "error": str(e),
                    "mode": "strands_orchestration",
                    "framework": "aws_strands",
                    "session_managed": True
                }
            }

    def get_conversation_history(self, run_id: str) -> Optional[List[Any]]:
        """Return conversation history for a session (if available)"""
        agent = self.agents_cache.get(run_id)
        if agent and hasattr(agent, "messages"):
            return agent.messages
        return None

    def clear_session(self, run_id: str):
        """Remove agent from cache and optionally delete session files"""
        if run_id in self.agents_cache:
            del self.agents_cache[run_id]
        session_path = Path(self.sessions_dir) / f"session_{run_id}"
        if session_path.exists():
            for file in session_path.glob("*"):
                file.unlink()
            session_path.rmdir()
