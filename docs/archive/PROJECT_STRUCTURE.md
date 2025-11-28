# Project Structure

This document describes the reorganized project structure for better maintainability and readability.

## Directory Organization

```
v2/
├── agents/              # Specialized agent implementations
│   ├── __init__.py
│   ├── segmentation_agent.py          ✅ Implemented
│   ├── backlog_generation_agent.py    📋 Placeholder
│   └── tagging_agent.py               📋 Placeholder
│
├── tools/               # Tool implementations
│   ├── __init__.py
│   └── retrieval_tool.py              📋 Placeholder
│
├── tests/               # Test scripts and demos
│   ├── __init__.py
│   ├── demo_segmentation.py           # Simple demo
│   ├── test_segmentation.py           # Comprehensive tests
│   ├── test_semantic_search.py
│   ├── test_setup.py
│   ├── test_strands_supervisor.py
│   └── show_architecture.py           # Architecture visualization
│
├── evaluate/            # Evaluation scripts and datasets
│   ├── __init__.py
│   ├── evaluate_tagging.py
│   └── generate_eval_dataset.py
│
├── datasets/            # Evaluation datasets
│   ├── eval_dataset.json
│   └── tagging_test.json
│
├── runs/                # Runtime outputs (per run_id)
│   └── {run_id}/
│       └── segments.jsonl
│
├── static/              # Web interface assets
│   ├── index.html
│   └── app.js
│
├── architecture/        # Architecture documentation
│   └── architecture-constraints.md
│
├── strands-multi-agent/ # Reference examples
│   └── ...
│
├── supervisor.py        # Main orchestrator
├── app.py              # FastAPI backend
├── config.poc.yaml     # Configuration
├── requirements.txt
└── README.md
```

## Module Descriptions

### Core Modules

**`supervisor.py`**
- Main orchestrator for the backlog synthesis workflow
- Creates and coordinates specialized agents per run
- Handles chat interface and API integration

**`app.py`**
- FastAPI backend for HTTP API
- Provides endpoints for upload, chat, and results retrieval

**`config.poc.yaml`**
- Configuration for ADO, Pinecone, OpenAI
- Project settings and thresholds

### Agents Directory (`agents/`)

Contains specialized agent implementations. Each agent is responsible for a single capability.

**`segmentation_agent.py`** ✅
- Splits documents into coherent segments
- Detects intents per segment
- Outputs to `runs/{run_id}/segments.jsonl`

**`backlog_generation_agent.py`** 📋
- Generates epics, features, and user stories
- Uses segment + retrieved context
- Creates acceptance criteria

**`tagging_agent.py`** 📋
- Tags stories as new/gap/conflict
- Compares against existing backlog
- Provides reasoning for decisions

### Tools Directory (`tools/`)

Contains tool implementations that can be used by agents or the supervisor.

**`retrieval_tool.py`** 📋
- Queries Pinecone vector store
- Retrieves ADO backlog items
- Fetches architecture constraints

### Tests Directory (`tests/`)

Contains all test scripts and demos.

**`demo_segmentation.py`**
- Simple demonstration of segmentation agent
- Shows clean output with example document

**`test_segmentation.py`**
- Comprehensive test suite for segmentation
- Tests direct method calls and chat interface

**`show_architecture.py`**
- Displays ASCII art visualization of architecture
- Shows patterns and workflow

**Other test files:**
- `test_semantic_search.py` - Tests for semantic search
- `test_setup.py` - Setup validation tests
- `test_strands_supervisor.py` - Strands integration tests

### Evaluate Directory (`evaluate/`)

Contains evaluation scripts and dataset generation tools.

**`evaluate_tagging.py`**
- Evaluates tagging agent accuracy
- Computes precision/recall/F1 metrics
- Uses gold-labeled test dataset

**`generate_eval_dataset.py`**
- Generates evaluation datasets
- Creates test cases for validation

## Running Tests and Demos

All tests and demos should be run from the project root directory:

```bash
# From project root: /Users/ricky.c.wong/poc/rde/v2

# Run segmentation demo
python tests/demo_segmentation.py

# Run comprehensive tests
python tests/test_segmentation.py

# Show architecture visualization
python tests/show_architecture.py
```

## Import Patterns

### For main modules (e.g., `supervisor.py`)
```python
from agents.segmentation_agent import create_segmentation_agent
from agents.backlog_generation_agent import create_backlog_generation_agent
from agents.tagging_agent import create_tagging_agent
from tools.retrieval_tool import create_retrieval_tool
```

### For test files (in `tests/` directory)
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.supervisor_agent import SupervisorAgent
```

## Benefits of This Structure

✅ **Clear Separation**: Agents, tools, tests, and evaluation are clearly separated

✅ **Easy Navigation**: Find files based on their purpose

✅ **Scalability**: Easy to add new agents or tools without cluttering root

✅ **Testability**: All test files in one place

✅ **Maintainability**: Smaller, focused directories

✅ **Professional**: Follows Python project best practices

## Key Principles

1. **Agents** = Specialized capabilities with LLM reasoning
2. **Tools** = Utility functions (retrieval, ADO writer, etc.)
3. **Tests** = All validation and demonstration code
4. **Evaluate** = Metrics and evaluation workflows
5. **Root** = Only core orchestration files (supervisor, app, config)

## Next Steps for Development

When implementing new features:

1. **New Agent**: Add to `agents/` directory
2. **New Tool**: Add to `tools/` directory
3. **New Test**: Add to `tests/` directory
4. **New Evaluation**: Add to `evaluate/` directory
5. **Update imports**: Update `supervisor.py` imports as needed

## Migration Notes

The refactoring maintains backward compatibility:
- Same API surface for `SupervisorAgent`
- Same output formats and file locations
- All existing tests pass without changes (except import paths)
