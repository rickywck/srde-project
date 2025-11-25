"""
Quick test to verify workflow refactoring
Tests that the BacklogSynthesisWorkflow can be instantiated and structured correctly
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from workflows import BacklogSynthesisWorkflow

def test_workflow_instantiation():
    """Test that workflow can be instantiated"""
    run_id = "test_123"
    run_dir = Path("test_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    workflow = BacklogSynthesisWorkflow(run_id, run_dir)
    
    assert workflow.run_id == run_id
    assert workflow.run_dir == run_dir
    assert workflow.min_similarity >= 0
    assert workflow.embedding_model is not None
    
    print("✓ Workflow instantiation successful")
    print(f"  - Run ID: {workflow.run_id}")
    print(f"  - Min similarity: {workflow.min_similarity}")
    print(f"  - Embedding model: {workflow.embedding_model}")
    print(f"  - Index name: {workflow.index_name}")

def test_workflow_structure():
    """Test workflow has all required methods"""
    run_id = "test_456"
    run_dir = Path("test_runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    workflow = BacklogSynthesisWorkflow(run_id, run_dir)
    
    # Check required methods exist
    assert hasattr(workflow, 'execute')
    assert hasattr(workflow, 'evaluate')
    assert hasattr(workflow, '_stage_segmentation')
    assert hasattr(workflow, '_stage_retrieval_and_generation')
    assert hasattr(workflow, '_stage_tagging')
    assert hasattr(workflow, '_find_similar_stories')
    assert hasattr(workflow, 'log_progress')
    
    print("✓ Workflow structure verification successful")
    print("  - All required methods present")
    print("  - Workflow stages defined")

def test_imports():
    """Test all workflow imports"""
    from workflows import BacklogSynthesisWorkflow, StrandsBacklogWorkflow
    
    print("✓ Workflow imports successful")
    print(f"  - BacklogSynthesisWorkflow: {BacklogSynthesisWorkflow.__name__}")
    print(f"  - StrandsBacklogWorkflow: {StrandsBacklogWorkflow.__name__}")

if __name__ == "__main__":
    print("=" * 60)
    print("WORKFLOW REFACTORING VERIFICATION")
    print("=" * 60)
    print()
    
    try:
        test_imports()
        print()
        test_workflow_instantiation()
        print()
        test_workflow_structure()
        print()
        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ TEST FAILED: {str(e)}")
        print("=" * 60)
        sys.exit(1)
