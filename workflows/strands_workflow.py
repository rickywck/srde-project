"""
Strands Workflow-based orchestration for backlog synthesis
Alternative implementation using Strands built-in workflow tool
Provides task management, dependency resolution, and parallel execution

Note: Requires strands-tools package:
    pip install strands-tools
"""

from typing import Dict, Any, Optional
from pathlib import Path

try:
    from strands import Agent
    from strands_tools import workflow
    STRANDS_WORKFLOW_AVAILABLE = True
except ImportError:
    STRANDS_WORKFLOW_AVAILABLE = False
    Agent = None
    workflow = None


class StrandsBacklogWorkflow:
    """
    Strands Workflow implementation for backlog synthesis.
    
    Uses Strands' built-in workflow tool for:
    - Automatic dependency resolution
    - Parallel execution where possible
    - State management and monitoring
    - Retry logic and error handling
    
    Requires: strands-tools package
    """
    
    def __init__(self, run_id: str, run_dir: Path):
        if not STRANDS_WORKFLOW_AVAILABLE:
            raise ImportError(
                "Strands workflow tools not available. Install with: pip install strands-tools"
            )
        
        self.run_id = run_id
        self.run_dir = run_dir
        self.workflow_id = f"backlog_synthesis_{run_id}"
        
        # Create workflow coordinator agent
        self.coordinator = Agent(
            system_prompt="You coordinate multi-agent backlog synthesis workflows.",
            tools=[workflow]
        )
    
    def create_workflow(self, document_text: str) -> Dict[str, Any]:
        """
        Create workflow with task definitions and dependencies.
        
        Task dependency graph:
        
        segmentation (priority 5)
            ↓
        retrieval (priority 4, depends on segmentation)
            ↓
        generation (priority 3, depends on retrieval)
            ↓
        tagging (priority 2, depends on generation)
            ↓
        evaluation (priority 1, depends on tagging, optional)
        """
        
        # Define workflow tasks with dependencies
        tasks = [
            {
                "task_id": "segmentation",
                "description": f"Segment the document into logical sections with intent detection. Document: {document_text[:500]}...",
                "system_prompt": """You are a document segmentation specialist. Analyze the document and break it into 
                logical sections. For each section, identify semantic intent labels (functional, non-functional, 
                technical, business) and determine the dominant intent. Return structured JSON with segments.""",
                "priority": 5,
                "dependencies": []
            },
            {
                "task_id": "retrieval",
                "description": "Retrieve relevant ADO items and architecture constraints for each segment using vector similarity search.",
                "system_prompt": """You retrieve relevant context from vector stores. For each segment, search Pinecone 
                for similar existing ADO backlog items and relevant architecture constraints. Filter by similarity 
                threshold and intent alignment.""",
                "priority": 4,
                "dependencies": ["segmentation"]
            },
            {
                "task_id": "generation",
                "description": "Generate hierarchical backlog items (Epics, Features, Stories) from segments using retrieved context.",
                "system_prompt": """You generate structured backlog items. Create Epics (high-level themes), Features 
                (functional capabilities), and User Stories (actionable work with acceptance criteria). Use retrieved 
                context to inform generation and ensure traceability.""",
                "priority": 3,
                "dependencies": ["retrieval"]
            },
            {
                "task_id": "tagging",
                "description": "Classify generated stories as gap/conflict/new relative to existing backlog.",
                "system_prompt": """You classify stories against existing backlog. For each story, determine if it's a 
                'gap' (extends existing functionality), 'conflict' (contradicts existing items), or 'new' (novel capability). 
                Provide justification and related item references.""",
                "priority": 2,
                "dependencies": ["generation"]
            }
        ]
        
        # Create workflow
        result = self.coordinator.tool.workflow(
            action="create",
            workflow_id=self.workflow_id,
            tasks=tasks
        )
        
        return result
    
    def start_workflow(self) -> Dict[str, Any]:
        """Start workflow execution (parallel processing where possible)"""
        result = self.coordinator.tool.workflow(
            action="start",
            workflow_id=self.workflow_id
        )
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed workflow status with progress and task metrics"""
        result = self.coordinator.tool.workflow(
            action="status",
            workflow_id=self.workflow_id
        )
        return result
    
    def pause_workflow(self) -> Dict[str, Any]:
        """Pause workflow execution"""
        result = self.coordinator.tool.workflow(
            action="pause",
            workflow_id=self.workflow_id
        )
        return result
    
    def resume_workflow(self) -> Dict[str, Any]:
        """Resume paused workflow"""
        result = self.coordinator.tool.workflow(
            action="resume",
            workflow_id=self.workflow_id
        )
        return result
    
    def delete_workflow(self) -> Dict[str, Any]:
        """Delete workflow and cleanup resources"""
        result = self.coordinator.tool.workflow(
            action="delete",
            workflow_id=self.workflow_id
        )
        return result
    
    async def execute(self, document_text: str) -> Dict[str, Any]:
        """
        Execute complete workflow: create → start → monitor → results.
        
        This wraps the Strands workflow tool for end-to-end execution.
        """
        # Create workflow definition
        create_result = self.create_workflow(document_text)
        if create_result.get("status") != "success":
            raise ValueError(f"Workflow creation failed: {create_result.get('message')}")
        
        # Start execution
        start_result = self.start_workflow()
        if start_result.get("status") != "success":
            raise ValueError(f"Workflow start failed: {start_result.get('message')}")
        
        # Monitor until completion
        # Note: In production, use async polling or webhooks
        status = self.get_status()
        
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "status": status.get("status"),
            "progress": status.get("progress"),
            "tasks": status.get("tasks"),
            "message": "Workflow executed using Strands orchestration"
        }


def list_all_workflows() -> Dict[str, Any]:
    """List all active workflows (utility function)"""
    if not STRANDS_WORKFLOW_AVAILABLE:
        raise ImportError(
            "Strands workflow tools not available. Install with: pip install strands-tools"
        )
    
    coordinator = Agent(
        system_prompt="Workflow coordinator",
        tools=[workflow]
    )
    return coordinator.tool.workflow(action="list")
