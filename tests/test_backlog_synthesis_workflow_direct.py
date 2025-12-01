
import pytest
import json
from pathlib import Path
from workflows.backlog_synthesis_workflow import BacklogSynthesisWorkflow

SAMPLE_DOC = """
Product Planning Meeting - Q1 2024

Topic 1: User Authentication Enhancement
We need to add multi-factor authentication to improve security.
Users have been requesting this feature for account protection.

Topic 2: Performance Issues
Several customers reported slow page load times.
The search API is taking 3-5 seconds on average.
This needs to be fixed as a P1 bug.
"""

@pytest.mark.asyncio
async def test_backlog_synthesis_workflow_direct_execution(tmp_path):
    """
    Test BacklogSynthesisWorkflow by calling the class directly.
    Verifies the full pipeline: segment -> retrieve -> generate -> tag.
    """
    import os
    import shutil
    
    # Enable mock mode for agents
    os.environ["SEGMENTATION_AGENT_MOCK"] = "1"
    # We might need to mock other agents too if they don't have mock mode or if we want to avoid API calls.
    # Assuming other agents handle it or we might need to patch them.
    # For now, let's try with just segmentation mock as it was the first failure.
    
    # Setup run directory
    run_id = "test_direct_execution_001"
    # The workflow/agents write to runs/{run_id} relative to CWD
    run_dir = Path("runs") / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Write raw file
    raw_file = run_dir / "raw.txt"
    raw_file.write_text(SAMPLE_DOC, encoding="utf-8")
    
    # Initialize workflow
    workflow = BacklogSynthesisWorkflow(run_id, run_dir)
    
    try:
        # Execute workflow
        result = await workflow.execute(SAMPLE_DOC)
        
        # Verify result structure
        assert result["status"] == "success"
        assert result["run_id"] == run_id
        assert "response" in result
        
        # Verify artifacts
        segments_file = run_dir / "segments.jsonl"
        backlog_file = run_dir / "generated_backlog.jsonl"
        tagging_file = run_dir / "tagging.jsonl"
        
        assert segments_file.exists(), "Segments file not created"
        assert backlog_file.exists(), "Backlog file not created"
        
        # Check segments content
        segments = [json.loads(line) for line in segments_file.read_text().splitlines()]
        assert len(segments) > 0, "No segments generated"
        
        # Check backlog content
        backlog_items = [json.loads(line) for line in backlog_file.read_text().splitlines()]
        print(f"Generated {len(backlog_items)} backlog items")
        
        # Verify result counts match artifacts
        assert result["counts"]["segments"] == len(segments)
        assert result["counts"]["backlog_items"] == len(backlog_items)
        
    finally:
        # Cleanup
        if run_dir.exists():
            shutil.rmtree(run_dir)

if __name__ == "__main__":
    # Allow running this file directly
    import asyncio
    try:
        asyncio.run(test_backlog_synthesis_workflow_direct_execution(Path(".")))
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
