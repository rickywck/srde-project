"""
Workflow orchestration package for RDE system.

Provides two workflow implementations:
1. BacklogSynthesisWorkflow: Custom sequential workflow with explicit stage management
2. StrandsBacklogWorkflow: Strands-native workflow using built-in workflow tool

Both implement the same backlog synthesis pipeline:
segment → retrieve → generate → tag → evaluate
"""

from .backlog_synthesis_workflow import BacklogSynthesisWorkflow

__all__ = [
    "BacklogSynthesisWorkflow",
    "StrandsBacklogWorkflow",
    "list_all_workflows"
]
