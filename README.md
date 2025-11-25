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

### Multi-Agent System

- **Supervisor Agent**: Orchestrates workflow and routes user requests to specialized agents
- **Segmentation Agent**: Analyzes documents and breaks them into semantic sections with intent labels
- **Backlog Generation Agent**: Creates Epics/Features/Stories from segments using RAG retrieval
- **Tagging Agent**: Classifies stories as gap/conflict/new relative to existing backlog
- **Evaluation Agent**: Assesses completeness, relevance, and quality of generated backlog

### Technology Stack

- **Framework**: AWS Strands for multi-agent orchestration
- **Backend**: FastAPI with async support
- **LLM**: OpenAI GPT-4o for reasoning, text-embedding-3-small for embeddings
- **Vector Store**: Pinecone (serverless, 512-dim embeddings)
- **Frontend**: HTML/CSS/JavaScript chat interface
- **Prompt Management**: YAML-based external configuration with centralized loader

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

4. Update `config.poc.yaml` with your project details:
```yaml
ado:
  organization: your-org
  project: your-project

project:
  name: your-project
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

Edit `config.poc.yaml` with your project details:

```yaml
ado:
  organization: your-org
  project: your-project

project:
  name: your-project  # Used as Pinecone namespace
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
│   └── prompt_loader.py            # Centralized prompt management utility
├── prompts/                   # External YAML prompt configurations
│   ├── segmentation_agent.yaml
│   ├── backlog_generation_agent.yaml
│   ├── tagging_agent.yaml
│   ├── evaluation_agent.yaml
│   └── supervisor_agent.yaml
├── ingestion/                 # Data loading utilities
│   ├── ado_loader.py              # Load ADO backlog into Pinecone
│   └── arch_loader.py             # Load architecture docs into Pinecone
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
├── app.py                     # FastAPI application entry point
├── supervisor.py              # Supervisor agent orchestration
├── chunker.py                 # Document chunking utility
├── config.poc.yaml            # System configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## How It Works

### Supervisor Orchestration

The supervisor agent coordinates all specialized agents using AWS Strands framework:

1. User submits request via chat interface (`/chat/{run_id}` endpoint)
2. Supervisor analyzes intent and determines which agent(s) to invoke
3. Agents are executed with appropriate context and prompts
4. Results are aggregated and returned to user
5. Session state persisted in `runs/{run_id}/` directory

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

### Tagging & Classification

1. Analyzes generated stories against existing backlog
2. Performs similarity search to find related items
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
- Each chat session gets unique `run_id`
- State persisted in `runs/{run_id}/` directory
- Supports iterative refinement and follow-up questions

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
project:
  name: your-project
```

## Troubleshooting

### Environment Variables Not Found
- Create `.env` file in project root with required credentials
- Or export variables in your shell: `export ADO_PAT=your_token`
- Ensure variables are exported (not just set): `export -p | grep ADO_PAT`

### Loader Scripts Fail to Import
- Loaders moved to `ingestion/` directory
- Scripts automatically resolve paths to parent directory for config and modules
- Run from project root: `python ingestion/ado_loader.py`

### No Results from RAG Search
- Verify data loaded: Check Pinecone dashboard for vectors
- Lower similarity threshold in `config.poc.yaml`
- Ensure namespace matches `project.name` in config

### Chat Interface Not Loading
- Check FastAPI is running on `http://localhost:8000`
- Verify static files in `static/` directory
- Check browser console for JavaScript errors

### Agent Errors
- Verify all prompts exist in `prompts/` directory
- Check YAML syntax (indentation, structure)
- Review agent logs in terminal output

## Development Notes

Development documentation archived in `docs/archive/`:
- `PROMPT_EXTERNALIZATION.md` - Prompt refactoring details
- `SUPERVISOR_INTEGRATION.md` - Supervisor agent implementation
- `STRANDS_UPGRADE.md` - AWS Strands migration notes
- `WORKFLOW_IMPLEMENTATION.md` - Multi-agent workflow design
- And more...

## License

MIT
