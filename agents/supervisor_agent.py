"""
Backlog Synthesizer Supervisor Agent using AWS Strands Framework
Orchestrates specialized agents and tools for backlog synthesis workflow
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.telemetry import StrandsTelemetry
from strands.session.file_session_manager import FileSessionManager
from .model_factory import ModelFactory
import logging
from agents.supervisor_helper import SupervisorRunHelper

# Import specialized agents and tools
from agents.segmentation_agent import create_segmentation_agent
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.backlog_regeneration_agent import create_backlog_regeneration_agent
from agents.tagging_agent import create_tagging_agent
from tools.ado_writer_tool import create_ado_writer_tool
from agents.evaluation_agent import create_evaluation_agent
from agents.prompt_loader import get_prompt_loader
from tools.retrieval_backlog_tool import create_retrieval_backlog_tool
import base64
from utils.document_limits import DocumentLimitUtils
from strands.tools.mcp import MCPClient

from tools.mcp_ado import mcp_ado_tool


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
        """Initialize supervisor with session management and prompt configuration only."""
        # Module logger
        self.logger = logging.getLogger(__name__)

        # Load prompts from external configuration (agent's own YAML)
        prompt_loader = get_prompt_loader()
        prompt_loader.load_prompt("supervisor_agent")
        self.system_prompt = prompt_loader.get_system_prompt("supervisor_agent")
        self.prompt_params = prompt_loader.get_parameters("supervisor_agent") or {}
        self.logger.debug("Supervisor: Loaded prompt parameters: %s", list(self.prompt_params.keys()))

        # Session management
        self.sessions_dir = sessions_dir
        Path(self.sessions_dir).mkdir(exist_ok=True)
        self.agents_cache: Dict[str, Agent] = {}
        self.agent_models: Dict[str, str] = {}

        # Track current run_id for tools
        self.current_run_id = None
    
    async def process_message(
        self,
        run_id: str,
        message: str,
        instruction_type: Optional[str] = None,
        document_text: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message using Strands agent orchestration with session management.
        """
        import asyncio

        self.current_run_id = run_id

        # Snapshot file state before processing to detect new generation later
        helper = SupervisorRunHelper(self.logger)
        before_snapshot = helper.snapshot_before(run_id)

        # Determine model to use (override > factory default)
        model_id = model_override or ModelFactory.get_default_model_id()
        # Ensure env var aligns for child tools created below (compatibility)
        os.environ["OPENAI_CHAT_MODEL"] = model_id
        self.logger.info("Supervisor: Using model_id=%s", model_id)

        # Setup session manager for this run
        session_manager = FileSessionManager(
            session_id=run_id,
            storage_dir=self.sessions_dir
        )

        # Reuse agent if cached, else create and cache
        agent = None
        if run_id in self.agents_cache and self.agent_models.get(run_id) == model_id:
            agent = self.agents_cache[run_id]
            self.logger.debug("Supervisor: Reusing cached agent for run_id=%s", run_id)
        else:
            self.logger.info("Supervisor: Creating agent for run_id=%s with model_id=%s", run_id, model_id)
            segmentation_agent = create_segmentation_agent(run_id)
            backlog_generation_agent = create_backlog_generation_agent(run_id)
            backlog_regeneration_agent = create_backlog_regeneration_agent(run_id)
            tagging_agent = create_tagging_agent(run_id)
            # Pass the factory-scoped backlog generation tool into the combined tool
            retrieval_backlog_tool = create_retrieval_backlog_tool(run_id, backlog_fn=backlog_generation_agent)
            evaluation_agent = create_evaluation_agent(run_id)
            ado_writer_tool = create_ado_writer_tool(run_id)

            # Create a model instance for this session via ModelFactory only
            try:
                session_model = ModelFactory.create_openai_model_for_agent(
                    agent_params=self.prompt_params,
                    model_id_override=model_id,
                )
                self.logger.debug(
                    "Supervisor: Session model initialized (model_id=%s, class=%s)",
                    getattr(session_model, "model_id", model_id),
                    type(session_model).__name__ if session_model else "None",
                )
            except Exception as e:
                self.logger.exception("Supervisor: Failed to create session model: %s", e)
                return {
                    "response": f"Initialization error: {str(e)}",
                    "status": {
                        "run_id": run_id,
                        "error": str(e),
                        "mode": "strands_orchestration",
                        "framework": "aws_strands",
                        "session_managed": True
                    }
                }

            agent = Agent(
                model=session_model,
                system_prompt=self.system_prompt,
                callback_handler=None,
                tools=[
                    segmentation_agent,
                    backlog_generation_agent,
                    backlog_regeneration_agent,
                    tagging_agent,
                    retrieval_backlog_tool,
                    evaluation_agent,
                    ado_writer_tool,
                    mcp_ado_tool
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
            self.agent_models[run_id] = model_id
            self.logger.info("Supervisor: Agent created with %d tools", len(agent.tools) if hasattr(agent, "tools") else 0)


        # Build context-aware query for the agent
        query_parts = []
        if document_text:
            metrics = DocumentLimitUtils.analyze_document(document_text)
            token_count = metrics["token_count"]
            max_completion_tokens = metrics["max_completion_tokens"]
            max_allowed_tokens = metrics["max_allowed_tokens"]

            if token_count > max_allowed_tokens:
                self.logger.warning(
                    "Supervisor: Document too large for segmentation (tokens=%d limit=%d run_id=%s)",
                    token_count,
                    max_allowed_tokens,
                    run_id,
                )
                error_message = DocumentLimitUtils.build_over_limit_message(metrics)
                return {
                    "response": error_message,
                    "status": {
                        "run_id": run_id,
                        "error": "document_too_large",
                        "token_count": token_count,
                        "token_limit": max_allowed_tokens,
                        "mode": "strands_orchestration",
                        "framework": "aws_strands",
                        "session_managed": True,
                    },
                }

            context_msg = f"""[CONTEXT] The user has uploaded a document. Here's the content:

{document_text}
"""
            query_parts.append(context_msg)
        if instruction_type:
            query_parts.append(f"[INSTRUCTION TYPE: {instruction_type}]")
        query_parts.append(f"[USER QUERY] {message}")
        full_query = "\n\n".join(query_parts)

        try:
            # Run agent in thread for async compatibility
            # Log approximate input tokens for debugging
            helper.log_input_tokens(full_query)
            # Ask the agent once; Strands interprets tool calls internally
            response = await asyncio.to_thread(agent, full_query)
            assistant_message = str(response)
            tool_calls = getattr(response, 'tool_calls', []) if hasattr(response, 'tool_calls') else []
            if tool_calls:
                self.logger.info("Supervisor: Strands tool_calls reported: %s", tool_calls)
            else:
                self.logger.debug("Supervisor: No tool_calls reported by Strands")
            conversation_length = len(agent.messages) if hasattr(agent, "messages") else None
            result: Dict[str, Any] = {
                "response": assistant_message,
                "status": {
                    "run_id": run_id,
                    "model": model_id,
                    "has_document": document_text is not None,
                    "mode": "strands_orchestration",
                    "framework": "aws_strands",
                    "session_managed": True,
                    "conversation_length": conversation_length,
                    "agents_available": [
                        "segment_document",
                        "generate_backlog",
                        "regenerate_backlog",
                        "tag_story",
                        "generate_backlog_with_retrieval",
                        "evaluate_backlog_quality",
                        "write_to_ado"
                    ],
                    "tools_invoked": tool_calls
                }
            }

            # Detect side-effects to classify response and optional auto-tagging
            detection = helper.detect_response_type(
                run_id=run_id,
                before=before_snapshot,
                enable_auto_tagging=(os.getenv("SUPERVISOR_AUTO_TAGGING") == "1")
            )
            if detection.get("response_type"):
                result["response_type"] = detection["response_type"]
            if detection.get("status_updates"):
                result["status"].update(detection["status_updates"]) 

            return result
        except Exception as e:
            self.logger.exception("Supervisor: Error during processing: %s", e)
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

    def get_dashboard_snapshot(self, run_id: str) -> Dict[str, Any]:
        """Return aggregated stats for the active supervisor session."""
        helper = SupervisorRunHelper(self.logger)
        agent = self.agents_cache.get(run_id)
        model_id = self.agent_models.get(run_id)
        return helper.build_dashboard(run_id, self.sessions_dir, model_id, agent)

    def clear_session(self, run_id: str):
        """Remove agent from cache and optionally delete session files"""
        if run_id in self.agents_cache:
            del self.agents_cache[run_id]
        session_path = Path(self.sessions_dir) / f"session_{run_id}"
        if session_path.exists():
            for file in session_path.glob("*"):
                file.unlink()
            session_path.rmdir()
