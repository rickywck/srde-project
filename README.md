# Requirements Document Elaboration (RDE) System

An intelligent multi-agent system that transforms high-level requirements documents into detailed, actionable Azure DevOps backlog items using AI-powered analysis and generation.

## Overview

The RDE system uses a supervisor-coordinated multi-agent architecture to:

1. **Segment requirements documents** into logical sections with semantic intent detection
2. **Generate structured backlog items** (Epics, Features, Stories) with context-aware RAG
3. **Tag and classify stories** against existing backlog (gap/conflict/new)
4. **Evaluate output quality** using LLM-as-judge methodology
5. **Provide interactive chat interface** for iterative refinement

## Architecture

### Multi-Agent System & Session Management

- **Supervisor Agent**: Orchestrates workflow, manages agent cache by `run_id`, and routes user requests to specialized agents. Ensures conversation continuity via session management.
- **Segmentation Agent**: Analyzes documents and breaks them into semantic sections with intent labels.
- **Backlog Generation Agent**: Creates Epics/Features/Stories from segments using RAG retrieval.
- **Tagging Agent**: Classifies stories as gap/conflict/new relative to existing backlog.
- **Evaluation Agent**: Assesses completeness, relevance, and quality of generated backlog.

#### Session Management Design

The system uses a **FileSessionManager** and agent caching to maintain conversation history and agent state across requests. Each chat session is identified by a unique `run_id`:

- **Single Agent Instance per run_id**: Ensures context continuity for each conversation.
- **FileSessionManager**: Automatically persists agent state and messages to disk (`sessions/session_{run_id}/`).
- **agents_cache**: Supervisor caches agents in memory for efficient reuse.
- **Session Restoration**: Sessions survive server restarts and can be restored from disk.
- **Frontend Integration**: Frontend must maintain and reuse `run_id` for all messages in a session (see `docs/FRONTEND_SESSION_GUIDE.md`).

See [`docs/SESSION_MANAGEMENT.md`](docs/SESSION_MANAGEMENT.md) and [`docs/FRONTEND_SESSION_GUIDE.md`](docs/FRONTEND_SESSION_GUIDE.md) for full design and integration details.

### Tools

- **`retrieval_backlog_tool.py`**: Combined retrieval and generation tool. Performs retrieval and backlog generation in a single call to reduce conversation history.
- **`retrieval_tool.py`**: Standalone retrieval tool for querying Pinecone for ADO items and architecture constraints.
- **`ado_writer_tool.py`**: Writes generated backlog items (Epics, Features, Stories) to Azure DevOps.
- **`file_extractor.py`**: Utility for extracting text content from various file formats.
- **`token_utils.py`**: Utility for counting tokens to manage context window usage.

### Workflow Orchestration

Two workflow implementations for backlog generation:

1. **BacklogSynthesisWorkflow** (Custom Sequential)
  - Explicit control over workflow stages with clear dependency management
  - Sequential execution: segment → retrieve → generate → tag → evaluate
  - Lazy initialization of expensive resources (OpenAI, Pinecone clients)
  - Ideal for debugging, development, and simple documents

2. **StrandsBacklogWorkflow** (Strands Native)
  - Automatic dependency resolution and parallel execution where possible
  - Built-in retry logic with exponential backoff
  - Progress monitoring, pause/resume capabilities
  - Best for production, large documents, and complex workflows

Both workflows are externalized from the FastAPI layer into dedicated `workflows/` modules, providing clean separation between UI, orchestration, and agent logic.

### Technology Stack


- **Framework**: AWS Strands for multi-agent orchestration and session management
- **Backend**: FastAPI with async support
- **LLM**: OpenAI GPT-4o (configurable via UI and backend), text-embedding-3-small for embeddings
- **Vector Store**: Pinecone (serverless, 512-dim embeddings)
- **Frontend**: HTML/CSS/JavaScript chat interface with session-aware chat and model picker
- **Prompt Management**: YAML-based external configuration with centralized loader
- **Workflow Orchestration**: Externalized workflow modules with Strands integration

## Prerequisites

- Python 3.9+
- Azure DevOps account with Personal Access Token (PAT)
- Pinecone account and API key
- OpenAI API key

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment template and configure:
```bash
cp .env.example .env
```

3. Edit `.env` with your credentials:
```
ADO_PAT=your_ado_personal_access_token
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
```

4. Update `config.poc.yaml` with your project details and Pinecone namespace:
```yaml
ado:
  organization: your-org
  project: your-project

pinecone:
  project: your-project  # Pinecone namespace used for retrieval
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your credentials
cat > .env << EOF
ADO_PAT=your_ado_personal_access_token
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
EOF

# Export environment variables (if not using .env file)
export ADO_PAT=your_pat
export PINECONE_API_KEY=your_key
export OPENAI_API_KEY=your_key
```

### 2. Configure Project

Edit `config.poc.yaml` with your project details and Pinecone namespace:

```yaml
ado:
  organization: your-org
  project: your-project

pinecone:
  project: your-project  # Used as Pinecone namespace
```

### 3. Load Data (One-time Setup)

```bash
# Load existing ADO backlog items
python ingestion/ado_loader.py

# Load architecture constraint documents
python ingestion/arch_loader.py --path ~/path/to/architecture/docs
```

### 4. Start the Application

```bash
# Activate your environment (if using conda)
conda activate strands

# Start the FastAPI server
python app.py
```

The application will be available at `http://localhost:8000`

### 5. Use the Chat Interface

Open your browser to `http://localhost:8000` and interact with the system:

- Upload requirements documents for analysis
- Ask questions about segmentation results
- Request backlog generation for specific segments
- Review tagging and classification results
- Request evaluation of generated backlog items

## Usage Examples

### Process a Requirements Document

```
User: "Analyze the requirements in dispute-resolution-architecture-overview.md"
System: [Segments document, identifies intents, provides summary]

User: "Generate backlog items for the workflow management section"
System: [Creates Epics/Features/Stories with RAG-enhanced context]

User: "Tag these stories against existing backlog"
System: [Classifies as gap/conflict/new with justifications]

User: "Evaluate the quality of generated items"
System: [Provides completeness, relevance, and quality scores]
```

### Workflow Execution

```bash
# Use default custom sequential workflow
curl -X POST http://localhost:8000/generate-backlog/123e4567-e89b-12d3-a456-426614174000

# Use Strands native workflow with automatic optimization
curl -X POST "http://localhost:8000/generate-backlog/123e4567-e89b-12d3-a456-426614174000?use_strands_workflow=true"

# Run evaluation separately
curl -X POST http://localhost:8000/evaluate/123e4567-e89b-12d3-a456-426614174000
```

### Data Ingestion

```bash
# Load ADO backlog with custom config
python ingestion/ado_loader.py --config custom_config.yaml

# Load architecture docs with custom chunking
python ingestion/arch_loader.py --path ./docs --chunk-size 1500 --chunk-overlap 300
```

### Evaluation

```bash
# Generate synthetic evaluation dataset
python evaluate/generate_eval_dataset.py --sample-size 50 --output datasets/eval_dataset.json

# Run tagging evaluation
python evaluate/evaluate_tagging.py --dataset datasets/tagging_test.json --threshold 0.6
```

## Configuration

Edit `config.poc.yaml`:

```yaml
retrieval:
  min_similarity_threshold: 0.7  # Adjust similarity threshold (0.0 to 1.0)

openai:
  embedding_model: text-embedding-3-small  # Change embedding model if needed
```

## Project Structure

```
.
├── agents/                    # Specialized AI agents
│   ├── segmentation_agent.py       # Document segmentation with intent detection
│   ├── backlog_generation_agent.py # Backlog item generation with RAG
│   ├── tagging_agent.py            # Story classification (gap/conflict/new)
│   ├── evaluation_agent.py         # Quality assessment (LLM-as-judge)
│   ├── tagging_input_resolver.py   # Helper for tagging agent
│   ├── model_factory.py            # Factory for creating LLM instances
│   └── prompt_loader.py            # Centralized prompt management utility
├── workflows/                 # Workflow orchestration modules
│   ├── __init__.py                 # Package exports
│   ├── backlog_synthesis_workflow.py  # Custom sequential workflow
├── prompts/                   # External YAML prompt configurations
│   ├── segmentation_agent.yaml
│   ├── backlog_generation_agent.yaml
│   ├── tagging_agent.yaml
│   ├── evaluation_agent.yaml
│   └── supervisor_agent.yaml
├── ingestion/                 # Data loading utilities
│   ├── ado_loader.py              # Load ADO backlog into Pinecone
│   ├── arch_loader.py             # Load architecture docs into Pinecone
│   └── chunker.py                 # Document chunking utility
├── evaluate/                  # Evaluation framework
│   ├── evaluate_tagging.py        # Tagging agent evaluation
│   └── generate_eval_dataset.py   # Synthetic dataset generation
├── tests/                     # Test suite
│   └── test_supervisor_integration.py
├── static/                    # Frontend assets (chat interface)
├── runs/                      # Session state storage (git-ignored)
├── docs/                      # Documentation
│   ├── archive/                   # Development notes (archived)
│   └── ...
├── app.py                     # FastAPI application entry point (335 lines, -43% reduction)
├── supervisor.py              # Supervisor agent orchestration

├── config.poc.yaml            # System configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## How It Works

### Workflow Architecture

The system separates concerns into three layers:

1. **UI Layer** (`app.py`): FastAPI endpoints for HTTP handling
2. **Orchestration Layer** (`workflows/`): Workflow coordination and state management
3. **Agent Layer** (`agents/`): Specialized AI agents for specific tasks

### Supervisor Orchestration

The supervisor agent coordinates all specialized agents using AWS Strands framework:

1. User submits request via chat interface (`/chat/{run_id}` endpoint)
2. Supervisor analyzes intent and determines which agent(s) to invoke
3. Agents are executed with appropriate context and prompts
4. Results are aggregated and returned to user
5. Session state persisted in `runs/{run_id}/` directory

### Workflow Execution

Two workflow implementations provide flexibility:

**BacklogSynthesisWorkflow** (Sequential):
- Explicit stage execution: segment → retrieve → generate → tag → evaluate
- State management in `self.results` dictionary
- Lazy initialization of OpenAI and Pinecone clients
- Best for debugging and understanding workflow stages

**StrandsBacklogWorkflow** (Parallel):
- Automatic dependency resolution from task graph
- Parallel execution of independent tasks
- Built-in retry, monitoring, pause/resume
- Best for production and performance optimization

### Document Segmentation

1. User uploads requirements document or provides text
2. Segmentation agent analyzes structure and content
3. Document split into logical sections with:
   - Section number, title, and text
   - Semantic intent labels (functional, non-functional, technical, business)
   - Dominant intent classification
   - Rationale for segmentation decisions

### Backlog Generation (RAG-Enhanced)

1. Takes segmented sections from segmentation agent
2. Uses RAG to retrieve relevant context:
   - Existing ADO backlog items (Pinecone similarity search)
   - Architecture constraints (filtered by intent)
3. Generates hierarchical backlog:
   - Epics (high-level themes)
   - Features (functional capabilities)
   - User Stories (actionable work items with acceptance criteria)
4. Embeds traceability and context from source document

#### New: Combined Retrieval + Generation Tool

- Tool name: `retrieval_backlog_tool` (function: `generate_backlog_with_retrieval`)
- Purpose: Performs retrieval and backlog generation in a single tool call and returns only the generation result (retrieval payload is not returned), reducing conversation history size.
- Typical usage: Provide a segment_id (read from runs/<run_id>/segments.jsonl) and optional intent labels. The tool loads the segment, runs retrieval, then calls the backlog generator with the retrieved context.
- When to use: Prefer this tool for most segment-based generation. Use generate_backlog when you explicitly want to generate directly from input text without RAG.

### Tagging & Classification

1. Analyzes generated stories against existing backlog
1. Analyzes generated stories against existing backlog
2. Performs similarity search to find related items (handles retrieval internally if not provided)
3. Classifies each story as:
   - **gap**: Extends/enhances existing functionality
   - **conflict**: Contradicts/replaces existing items
   - **new**: Introduces novel capability
4. Provides justification and related item references

### Quality Evaluation (LLM-as-Judge)

1. Evaluates generated backlog on three dimensions:
   - **Completeness**: Coverage of requirements (0-10)
   - **Relevance**: Alignment with context (0-10)
   - **Quality**: Clarity and actionability (0-10)
2. Provides overall score and improvement suggestions
3. Uses structured JSON schema for consistent assessment

## Key Features

### Externalized Workflow Orchestration
- Clean separation: UI ↔ Orchestration ↔ Agents
- Two workflow implementations: sequential (explicit control) and parallel (automatic optimization)
- Testable workflow logic independent of FastAPI
- Strands Workflow Tool integration for advanced orchestration
- No breaking changes to existing API

### External Prompt Management
- All agent prompts stored in YAML configuration files (`prompts/`)
- Centralized `prompt_loader.py` utility with LRU caching
- Easy prompt tuning without code changes
- Template variable substitution via `str.format()`

### RAG-Enhanced Generation
- Pinecone vector store for semantic search
- Separate namespaces for ADO items and architecture constraints
- Configurable similarity thresholds
- Intent-based filtering for relevant context

### Session Management

### Session Management (Strands)
- **Conversation Continuity**: Each chat session uses a unique `run_id` for context continuity.
- **Automatic Persistence**: Agent state and messages are auto-saved to `sessions/session_{run_id}/`.
- **Agent Caching**: Supervisor caches agents by `run_id` for efficient reuse.
- **Crash Recovery**: Sessions persist across server restarts and can be restored from disk.
- **Frontend Integration**: Frontend must maintain and reuse `run_id` for all messages in a session. See [`docs/FRONTEND_SESSION_GUIDE.md`](docs/FRONTEND_SESSION_GUIDE.md).
- **Multi-Turn Conversations**: Natural follow-up questions work; tool context is remembered.
- **Per-Session Isolation**: Each `run_id` has independent history.

### Evaluation Framework
- Synthetic dataset generation for tagging validation
- Automated evaluation of tagging accuracy
- LLM-as-judge for backlog quality assessment

## Configuration

Edit `config.poc.yaml` to customize behavior:

```yaml
# Azure DevOps settings
ado:
  organization: your-org
  project: your-project
  pat_env_var: ADO_PAT

# Vector database settings
pinecone:
  api_key_env_var: PINECONE_API_KEY
  index_name: your-index
  environment: us-east-1

# OpenAI settings
openai:
  api_key_env_var: OPENAI_API_KEY
  embedding_model: text-embedding-3-small
  chat_model: gpt-4o

# Retrieval settings
retrieval:
  min_similarity_threshold: 0.5

# Project settings (used as Pinecone namespace)
pinecone:
  project: your-project
```

## Troubleshooting

### Environment Variables Not Found
- Create `.env` file in project root with required credentials
- Or export variables in your shell: `export ADO_PAT=your_token`
- Ensure variables are exported (not just set): `export -p | grep ADO_PAT`


### Session Management Issues
- **Agent doesn't remember previous messages**: Ensure same `run_id` is used for all messages; check `session_managed: true` in response status.
- **Conversation resets unexpectedly**: Verify `run_id` is not overwritten; sessionStorage/localStorage is not cleared.
- **Multiple conversations mixed together**: Use unique `run_id` for each conversation; do not reuse old `run_id`s.

See [`docs/SESSION_MANAGEMENT.md`](docs/SESSION_MANAGEMENT.md) and [`docs/FRONTEND_SESSION_GUIDE.md`](docs/FRONTEND_SESSION_GUIDE.md) for troubleshooting and integration tips.

## API Endpoints

### POST /generate-backlog/{run_id}

Generate complete backlog from uploaded document.

**Query Parameters:**
- `use_strands_workflow` (bool, optional): Use Strands native workflow instead of custom sequential workflow (default: false)

**Response:**
```json
{
  "run_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "success",
  "message": "Workflow completed",
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
  }
}
```

### POST /evaluate/{run_id}

Evaluate quality of generated backlog items.

**Response:**
```json
{
  "run_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "success",
  "evaluation": {
    "completeness": 8.5,
    "relevance": 9.0,
    "quality": 8.0,
    "overall": 8.5,
    "suggestions": ["..."]
  }
}
```

## Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_workflow_refactoring.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run tests matching pattern
pytest tests/ -k "workflow" -v

# Run with output for debugging
pytest tests/ -v -s
```

### Test Organization

- `test_ado_writer.py` - ADO work item writing functionality
- `test_evaluation_agent.py` - Quality evaluation agent
- `test_file_extractor.py` - Document text extraction
- `test_real_files.py` - Real document processing (TelecomBRD)
- `test_semantic_search.py` - Vector search and embedding
- `test_setup.py` - Environment and configuration validation
- `test_strands_supervisor.py` - Supervisor agent integration
- `test_tagging_agent.py` - Story classification
- `test_workflow_api.py` - API endpoint testing
- `test_workflow_refactoring.py` - Workflow module structure

### Mock Mode

Many tests support mock mode to avoid LLM API calls:

```bash
export EVALUATION_AGENT_MOCK=1
export SEGMENTATION_AGENT_MOCK=1
pytest tests/ -v
```

## Development Notes

Development documentation archived in `docs/archive/`:
- `WORKFLOW_REFACTORING.md` - Workflow externalization and Strands integration
- `PROMPT_EXTERNALIZATION.md` - Prompt refactoring details
- `SUPERVISOR_INTEGRATION.md` - Supervisor agent implementation
- `STRANDS_UPGRADE.md` - AWS Strands migration notes
- `WORKFLOW_IMPLEMENTATION.md` - Multi-agent workflow design
- And more...

## License

MIT
