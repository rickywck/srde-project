# Upgrading to AWS Strands Agentic Framework

## What Changed

The Supervisor Agent has been upgraded to use the **AWS Strands Agentic Framework**, which provides:
- Better agent orchestration capabilities
- Built-in observability with OpenTelemetry
- Modular tool and sub-agent integration
- Improved tracing and debugging

## Installation

### 1. Install Strands Framework

```bash
pip install strands strands-tools
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "from strands import Agent; print('✓ Strands installed successfully')"
```

## Key Changes in supervisor.py

### Before (Direct OpenAI API)
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=api_key)
response = await client.chat.completions.create(...)
```

### After (AWS Strands Framework)
```python
from strands import Agent
from strands.models.openai import OpenAIModel

model = OpenAIModel(model_id="gpt-4o", params={...}, api_key=api_key)
agent = Agent(model=model, system_prompt="...", tools=[])
response = agent(query)
```

## Benefits

1. **Agent Orchestration**: The supervisor can now easily coordinate multiple specialized agents
2. **Tool Integration**: Built-in support for tools that agents can invoke
3. **Observability**: Optional OpenTelemetry integration for tracing and monitoring
4. **Modularity**: Easy to add new agents and tools as the system evolves

## Future Additions

With Strands, we can easily add:

### Specialized Agents (as tools)
- **Segmentation Agent**: `@tool` decorated function that segments documents
- **Generation Agent**: Creates backlog items from segments
- **Tagging Agent**: Classifies stories relative to existing backlog

### Tools
- **Retrieval Tool**: Search Pinecone for relevant ADO items and architecture
- **ADO Writer Tool**: Write items to Azure DevOps
- **File Tools**: Read/write segments and backlog items

### Example Pattern (from teachers_assistant.py)
```python
from strands import Agent, tool

@tool
def segmentation_agent(document: str) -> str:
    """Segment a document into coherent chunks with intent labels."""
    agent = Agent(
        model=model,
        system_prompt=SEGMENTATION_PROMPT,
        tools=[...]
    )
    return str(agent(document))

# Add to supervisor
supervisor_agent = Agent(
    model=model,
    system_prompt=SUPERVISOR_PROMPT,
    tools=[segmentation_agent, generation_agent, tagging_agent]
)
```

## Testing

After installation, test the supervisor:

```bash
python test_setup.py
```

Then restart the server:
```bash
python app.py
```

The chat interface should work exactly as before, but now using Strands under the hood!

## Observability (Optional)

To enable OpenTelemetry tracing, uncomment these lines in `supervisor.py`:

```python
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:4318"
self.telemetry = StrandsTelemetry().setup_otlp_exporter()
```

This enables tracing to Langfuse or any OTEL-compatible backend.

## Architecture

```
User Request
    ↓
FastAPI Endpoint (/chat)
    ↓
SupervisorAgent (Strands Agent)
    ↓
[Future] Orchestrates:
    - Segmentation Tool
    - Retrieval Tool
    - Generation Agent
    - Tagging Agent
    - ADO Writer Tool
    ↓
Response to User
```

## Compatibility

- ✅ All existing functionality preserved
- ✅ Same API endpoints and responses
- ✅ Same chat interface behavior
- ✅ Ready for future agent/tool additions
- ✅ Optional observability features

The upgrade is **transparent to users** - the chat interface works identically, but the backend is now powered by a more sophisticated agentic framework.
