# Workflow Refactoring: Externalized Logic and Strands Integration

## Overview

The backlog generation logic has been refactored to:
1. **Externalize workflow orchestration** from FastAPI endpoints into dedicated workflow modules
2. **Leverage Strands Workflow Tool** for improved multi-agent coordination
3. **Separate concerns** between UI (FastAPI), orchestration (workflows), and agents

## Architecture Changes

### Before Refactoring

```
app.py (590 lines)
├── FastAPI endpoints
├── Workflow orchestration logic (200+ lines inline)
│   ├── Segmentation stage
│   ├── Retrieval stage
│   ├── Generation stage
│   ├── Tagging stage (with inline similarity search)
│   └── Result compilation
└── Chat interface integration
```

**Issues:**
- Tight coupling between UI and business logic
- Difficult to test workflow independently
- Hard to maintain and extend
- No clear separation of concerns
- Inline orchestration logic mixed with HTTP handling

### After Refactoring

```
app.py (335 lines, -43% reduction)
├── FastAPI endpoints (thin controllers)
└── Delegates to workflow modules

workflows/
├── __init__.py
├── backlog_synthesis_workflow.py (480 lines)
│   └── BacklogSynthesisWorkflow
│       ├── Sequential stage execution
│       ├── Explicit dependency management
│       ├── State tracking
│       └── Progress logging
└── strands_workflow.py (180 lines)
    └── StrandsBacklogWorkflow
        ├── Strands workflow tool integration
        ├── Automatic dependency resolution
        ├── Parallel execution where possible
        └── Built-in retry and monitoring
```

**Benefits:**
- Clean separation: UI ↔ Orchestration ↔ Agents
- Testable workflow logic in isolation
- Reusable orchestration modules
- Two workflow implementations for flexibility
- Easier to extend and maintain

## Workflow Implementations

### 1. BacklogSynthesisWorkflow (Custom Sequential)

**Purpose:** Explicit control over workflow stages with clear dependency management.

**Features:**
- Sequential execution: segment → retrieve → generate → tag → evaluate
- Explicit state management in `self.results`
- Progress logging and chat history integration
- Lazy initialization of expensive resources (OpenAI, Pinecone clients)
- Detailed error handling per stage

**Usage:**
```python
from workflows import BacklogSynthesisWorkflow

workflow = BacklogSynthesisWorkflow(run_id, run_dir)
result = await workflow.execute(document_text)

# Optional: Run evaluation separately
evaluation = await workflow.evaluate()
```

**Key Methods:**
- `execute(document_text)` - Full pipeline: segment → retrieve → generate → tag
- `evaluate()` - Quality assessment stage (can be called independently)
- `_stage_segmentation()` - Document segmentation with intent detection
- `_stage_retrieval_and_generation()` - RAG-enhanced generation per segment
- `_stage_tagging()` - Story classification with similarity search
- `_find_similar_stories()` - Vector similarity search in Pinecone

**When to Use:**
- Need explicit control over execution order
- Want to inspect intermediate results
- Require custom logging/monitoring
- Debugging workflow stages
- Production use with well-understood dependencies

### 2. StrandsBacklogWorkflow (Strands Native)

**Purpose:** Leverage Strands' built-in workflow tool for automatic orchestration.

**Features:**
- Automatic dependency resolution
- Parallel execution of independent tasks
- Built-in state management and persistence
- Retry logic with exponential backoff
- Progress monitoring and metrics
- Pause/resume capabilities

**Usage:**
```python
from workflows import StrandsBacklogWorkflow

workflow = StrandsBacklogWorkflow(run_id, run_dir)
result = await workflow.execute(document_text)

# Advanced: Monitor, pause, resume
status = workflow.get_status()
workflow.pause_workflow()
workflow.resume_workflow()
```

**Task Dependency Graph:**
```
segmentation (priority 5)
    ↓
retrieval (priority 4, depends on segmentation)
    ↓
generation (priority 3, depends on retrieval)
    ↓
tagging (priority 2, depends on generation)
    ↓
evaluation (priority 1, depends on tagging, optional)
```

**When to Use:**
- Want automatic optimization of execution order
- Need parallel execution where possible
- Require robust error handling and retries
- Long-running workflows with pause/resume
- Production scale with monitoring needs

## API Changes

### Endpoint: POST /generate-backlog/{run_id}

**New Query Parameter:**
```
use_strands_workflow: bool = False
```

**Examples:**

```bash
# Use custom sequential workflow (default)
curl -X POST http://localhost:8000/generate-backlog/123e4567-e89b-12d3-a456-426614174000

# Use Strands native workflow
curl -X POST "http://localhost:8000/generate-backlog/123e4567-e89b-12d3-a456-426614174000?use_strands_workflow=true"
```

**Response:** Same structure for both implementations
```json
{
  "run_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "success",
  "message": "Workflow completed",
  "response": "🎯 Backlog Synthesis Complete\n...",
  "counts": {
    "segments": 5,
    "backlog_items": 23,
    "stories": 15,
    "tags": {"new": 10, "gap": 3, "conflict": 2}
  },
  "files": {
    "segments": "/path/to/segments.jsonl",
    "backlog": "/path/to/generated_backlog.jsonl",
    "tagging": "/path/to/tagging.jsonl"
  },
  "workflow_steps": { ... },
  "timestamp": "2025-11-25T12:34:56.789Z"
}
```

### Endpoint: POST /evaluate/{run_id}

**Changes:**
- Delegates to `BacklogSynthesisWorkflow.evaluate()`
- Simplified from 60+ lines to ~20 lines
- Same response structure

## File Structure

```
workflows/
├── __init__.py                           # Package exports
├── backlog_synthesis_workflow.py         # Custom sequential workflow
└── strands_workflow.py                   # Strands native workflow

app.py                                    # FastAPI application (simplified)
├── Imports workflows module
├── /generate-backlog/{run_id}           # Delegates to workflow
└── /evaluate/{run_id}                   # Delegates to workflow.evaluate()
```

## Migration Guide

### For Developers

**Before:**
```python
# All logic inline in app.py
@app.post("/generate-backlog/{run_id}")
async def generate_backlog(run_id: str):
    # 200+ lines of orchestration logic
    segmentation_tool = create_segmentation_agent(run_id)
    seg_result = json.loads(segmentation_tool(document_text))
    # ... more inline logic
```

**After:**
```python
# Clean delegation to workflow module
@app.post("/generate-backlog/{run_id}")
async def generate_backlog(run_id: str, use_strands_workflow: bool = False):
    workflow = BacklogSynthesisWorkflow(run_id, run_dir)
    result = await workflow.execute(document_text)
    return result
```

### For API Users

**No Breaking Changes** - Existing API calls work without modification.

**New Capability:**
```bash
# Try Strands workflow for automatic optimization
curl -X POST "http://localhost:8000/generate-backlog/{run_id}?use_strands_workflow=true"
```

## Testing

### Unit Testing Workflows

```python
import pytest
from workflows import BacklogSynthesisWorkflow

@pytest.mark.asyncio
async def test_workflow_execution():
    run_dir = Path("test_runs/test_123")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    workflow = BacklogSynthesisWorkflow("test_123", run_dir)
    result = await workflow.execute("Sample document text...")
    
    assert result["status"] == "success"
    assert result["counts"]["segments"] > 0
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_full_pipeline():
    # Upload document
    response = client.post("/upload", files={"file": ("test.txt", "content")})
    run_id = response.json()["run_id"]
    
    # Execute workflow
    response = client.post(f"/generate-backlog/{run_id}")
    assert response.status_code == 200
    
    # Verify artifacts
    assert Path(f"runs/{run_id}/segments.jsonl").exists()
    assert Path(f"runs/{run_id}/generated_backlog.jsonl").exists()
```

## Performance Considerations

### BacklogSynthesisWorkflow
- **Execution:** Sequential, predictable timing
- **Resource Usage:** Lazy initialization, low memory overhead
- **Parallelism:** None (explicit sequential stages)
- **Best For:** Simple documents, debugging, development

### StrandsBacklogWorkflow
- **Execution:** Automatic parallelization where possible
- **Resource Usage:** More aggressive resource utilization
- **Parallelism:** Automatic based on dependency graph
- **Best For:** Large documents, production, complex workflows

## Strands Workflow Features

### Task Management
```python
workflow.get_status()
# Returns: task progress, execution times, dependencies
```

### Pause/Resume
```python
workflow.pause_workflow()
# ... do other work ...
workflow.resume_workflow()
```

### Error Handling
- Automatic retries with exponential backoff
- Task-level error isolation
- Detailed error reporting

### Monitoring
```python
status = workflow.get_status()
print(f"Progress: {status['progress']}%")
print(f"Tasks: {status['tasks']}")
```

## Future Enhancements

### Planned
1. **Webhook notifications** for long-running workflows
2. **Incremental execution** - resume from last successful stage
3. **Workflow templates** for common patterns
4. **Metrics collection** - execution times, success rates
5. **Workflow versioning** - track changes over time

### Possible
1. **Dynamic task injection** - add tasks during execution
2. **Conditional branching** - different paths based on results
3. **Human-in-the-loop** - approval gates between stages
4. **Distributed execution** - scale across multiple workers

## References

- [Strands Workflow Documentation](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/multi-agent/workflow/)
- [Strands Tools Repository](https://github.com/strands-agents/tools/blob/main/src/strands_tools/workflow.py)
- [Workflow Examples](https://strandsagents.com/latest/documentation/docs/examples/python/agents_workflows/)

## Summary

This refactoring achieves:
- ✅ **Separation of concerns:** UI ↔ Orchestration ↔ Agents
- ✅ **Testability:** Workflow logic independent of FastAPI
- ✅ **Flexibility:** Two workflow implementations for different needs
- ✅ **Maintainability:** Clear module boundaries, easier to extend
- ✅ **Clarity:** Strands Workflow for transparent dependency management
- ✅ **No breaking changes:** Existing API remains compatible

**Result:** A cleaner, more maintainable, and extensible architecture that leverages modern multi-agent orchestration patterns.
