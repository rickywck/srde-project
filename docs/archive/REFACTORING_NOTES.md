# Refactored Architecture - Specialized Agents Pattern

## Overview

The supervisor has been refactored to follow the **teachers_assistant pattern** from AWS Strands, where specialized agents and tools are implemented in separate files and passed to the main supervisor agent as tools.

## Architecture

```
supervisor.py (Orchestrator)
├── segmentation_agent.py (Implemented ✓)
├── backlog_generation_agent.py (Placeholder)
├── tagging_agent.py (Placeholder)
└── retrieval_tool.py (Placeholder)
```

## File Structure

### Core Components

**`supervisor.py`** - Main orchestrator
- Initializes the Strands supervisor agent
- Creates specialized agents per run
- Passes all agents as tools to the supervisor
- Handles chat interface and workflow orchestration

**`segmentation_agent.py`** - Document segmentation specialist ✓
- Splits documents into coherent segments (500-1000 tokens)
- Identifies intents for each segment
- Saves output to `runs/{run_id}/segments.jsonl`
- Fully implemented and tested

**`backlog_generation_agent.py`** - Backlog item generator (placeholder)
- Will generate epics, features, and user stories
- Uses segment + retrieved context as input
- Creates structured backlog items with acceptance criteria

**`tagging_agent.py`** - Story tagging specialist (placeholder)
- Will tag stories as new/gap/conflict
- Compares against existing backlog
- Provides reasoning for tagging decisions

**`retrieval_tool.py`** - Context retrieval tool (placeholder)
- Will query Pinecone for relevant context
- Retrieves ADO backlog items and architecture constraints
- Applies similarity thresholds

## Key Design Patterns

### 1. Factory Pattern
Each agent/tool is created via a factory function that binds the `run_id`:

```python
def create_segmentation_agent(run_id: str):
    @tool
    def segment_document(document_text: str) -> str:
        # Implementation with access to run_id
        ...
    return segment_document
```

### 2. Strands Tool Decorator
Each agent function uses the `@tool` decorator to be callable by the supervisor:

```python
from strands import tool

@tool
def segment_document(document_text: str) -> str:
    """
    Docstring becomes the tool description for the LLM.
    """
    ...
```

### 3. Per-Run Agent Initialization
Agents are created fresh for each run, ensuring proper isolation:

```python
# In supervisor.process_message()
segmentation_agent = create_segmentation_agent(run_id)
backlog_generation_agent = create_backlog_generation_agent(run_id)
tagging_agent = create_tagging_agent(run_id)
retrieval_tool = create_retrieval_tool(run_id)

self.agent = Agent(
    model=self.model,
    system_prompt=self.system_prompt,
    tools=[
        segmentation_agent,
        backlog_generation_agent,
        tagging_agent,
        retrieval_tool
    ],
    ...
)
```

## Benefits of This Architecture

### ✅ Separation of Concerns
- Each agent has a single, well-defined responsibility
- Easy to understand and maintain individual components
- Clear boundaries between different capabilities

### ✅ Testability
- Each agent can be tested independently
- Mock implementations can be easily swapped
- Unit tests can focus on specific agent logic

### ✅ Extensibility
- New agents can be added without modifying existing code
- Easy to version and update individual agents
- Simple to enable/disable agents per run

### ✅ Code Organization
- Clear file structure mirrors logical architecture
- Each file is focused and manageable in size
- Easy to navigate and locate functionality

### ✅ Reusability
- Agents can be used independently or by other supervisors
- Common patterns can be extracted to base classes
- Tools can be composed in different ways

## Comparison with teachers_assistant.py

### Similarities
1. **Specialized agents as tools**: Both use `@tool` decorator on specialized functions
2. **Agent composition**: Main agent receives list of specialized agents/tools
3. **Clear separation**: Each capability in its own file
4. **Factory pattern**: Agents created via factory functions (our implementation)

### Differences
1. **Per-run agents**: Our agents are created per-run with run_id binding
2. **State management**: Our agents maintain run-specific state (file outputs)
3. **Async interface**: Our supervisor uses async methods
4. **Configuration**: Our agents read from centralized config and environment

## Usage Example

```python
from agents.supervisor_agent import SupervisorAgent
import uuid

# Initialize supervisor
supervisor = SupervisorAgent()

# Create a run
run_id = str(uuid.uuid4())

# Direct method call
result = await supervisor.segment_document(run_id, document_text)

# Or via chat interface (supervisor orchestrates automatically)
chat_result = await supervisor.process_message(
    run_id=run_id,
    message="Please segment this document",
    document_text=document_text
)
```

## Testing

```bash
# Test individual agent
python demo_segmentation.py

# Comprehensive test suite
python test_segmentation.py
```

## Next Steps for Implementation

### Phase 2: Retrieval Tool
- Implement Pinecone integration in `retrieval_tool.py`
- Add embedding generation for segments
- Apply similarity thresholds

### Phase 3: Backlog Generation Agent
- Implement LLM-based backlog item generation
- Create prompts for epic/feature/story generation
- Add acceptance criteria generation

### Phase 4: Tagging Agent
- Implement story comparison logic
- Add tagging decision prompts
- Create tagging evaluation metrics

## Environment Variables Required

```bash
export OPENAI_API_KEY="your-openai-key"
export OPENAI_CHAT_MODEL="gpt-4o"  # Optional, defaults to gpt-4o
export PINECONE_API_KEY="your-pinecone-key"  # For future retrieval tool
```

## Files Modified

- ✅ `supervisor.py` - Refactored to use specialized agents pattern
- ✅ `segmentation_agent.py` - Extracted from supervisor, fully functional
- ✅ `backlog_generation_agent.py` - Created placeholder
- ✅ `tagging_agent.py` - Created placeholder
- ✅ `retrieval_tool.py` - Created placeholder

## Backward Compatibility

All existing tests and demos continue to work:
- ✅ `demo_segmentation.py` - Works with refactored code
- ✅ `test_segmentation.py` - All tests pass
- ✅ Same API for `segment_document()` method
- ✅ Same output format and file structure
