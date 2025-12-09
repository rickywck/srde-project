# Requirements Document Elaboration (RDE) System

A multi-agent system for synthesizing, refining, and evaluating software backlogs from requirements documents. This system orchestrates specialized AI agents to segment documents, retrieve context, generate backlog items (Epics, Features, Stories), tag them against existing backlogs, and evaluate their quality.

## 🏗 Architecture

The system is built on the **AWS Strands** framework for agent orchestration and uses **Pinecone** for vector storage and **OpenAI** for LLM capabilities.

### Core Components

#### 🤖 Agents (`agents/`)
- **Supervisor Agent** (`supervisor_agent.py`): The orchestrator. Manages session state, routes requests to specialized agents, and handles user interactions.
- **Segmentation Agent** (`segmentation_agent.py`): Splits raw requirement documents into coherent segments and identifies intents.
- **Backlog Generation Agent** (`backlog_generation_agent.py`): Generates structured backlog items (Epics, Features, Stories) from segments using retrieved context.
- **Backlog Regeneration Agent** (`backlog_regeneration_agent.py`): Updates and refines existing backlog items based on user instructions.
- **Tagging Agent** (`tagging_agent.py`): Classifies generated stories as `new`, `gap`, `duplicate`, or `conflict` relative to the existing backlog.
- **Evaluation Agent** (`evaluation_agent.py`): LLM-as-a-judge that assesses backlog quality (completeness, relevance, quality) in both live and batch modes.
- **Model Factory** (`model_factory.py`): Centralized factory for creating configured `OpenAIModel` instances.

#### 🛠 Tools (`tools/`)
- **Retrieval Backlog Tool** (`retrieval_backlog_tool.py`): Combined tool that orchestrates retrieval (from Pinecone) and generation to minimize conversation payload.
- **Retrieval Tool** (`retrieval_tool.py`): Standalone tool for querying Pinecone for ADO items and architecture constraints.
- **ADO Writer Tool** (`ado_writer_tool.py`): Writes generated backlog items to Azure DevOps (creates Epics, Features, Stories with parent links).
- **File Extractor** (`file_extractor.py`): Utilities for extracting text from various file formats.

#### 🔄 Workflows (`workflows/`)
- **Backlog Synthesis Workflow** (`backlog_synthesis_workflow.py`): Externalized orchestration logic for the full pipeline (Segment → Retrieve → Generate → Tag → Evaluate), separating business logic from the API layer.

## Modes of Operation

There are two primary ways to interact with the system:

- **Workflow Mode (Quick Actions / Workflows):** Uses pre-defined workflows such as `BacklogSynthesisWorkflow` to run the full pipeline in a deterministic, batched manner: segment → retrieve → generate → tag → (optional) evaluate → (optional) write. Ideal for reproducible runs and bulk processing.
- **Chat Mode (Supervisor Agent):** Conversational interaction via the `Supervisor Agent` where free-form instructions are interpreted and the agent dynamically orchestrates specialized agents and tools. Ideal for iterative refinement and human-in-the-loop workflows.

Both modes share the same Agents and Tools and persist artifacts under `runs/{run_id}` for consistent outputs.

## Agents vs Tools

To make responsibilities clear, the repo separates LLM-driven components (Agents) from deterministic connectors/helpers (Tools):

- **Agents** include: `Supervisor Agent`, `Segmentation Agent`, `Backlog Generation Agent`, `Backlog Regeneration Agent`, `Tagging Agent`, `Evaluation Agent`.
- **Tools** include: `Retrieval Tool`, `Retrieval Backlog Tool` (combined retrieval+generation helper), `ADO Writer Tool`, and ingestion utilities.


#### 📥 Ingestion (`ingestion/`)
- **ADO Loader** (`ado_loader.py`): CLI script to ingest existing ADO backlogs into Pinecone.
- **Architecture Loader** (`arch_loader.py`): CLI script to ingest architecture documentation into Pinecone.
- **Chunker** (`chunker.py`): Semantic chunking utility for documents.

#### 🧪 Evaluation (`evaluate/`)
- **Tagging Evaluation** (`evaluate_tagging.py`): Script to evaluate tagging accuracy against a gold dataset.
- **Dataset Generation** (`generate_eval_dataset.py`): Utility to generate synthetic evaluation datasets from ADO stories.

#### 📝 Prompts (`prompts/`)
YAML-based prompt templates for all agents, managed via `prompt_loader.py`.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- OpenAI API Key
- Pinecone API Key & Index
- Azure DevOps PAT (for ADO integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd rde/v2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   ADO_PAT=...
   ADO_ORG=...
   ADO_PROJECT=...
   ```

4. **Update Configuration**
   Edit `config.poc.yaml` to match your environment settings.

### Usage

#### 1. Ingest Data
Load your existing backlog and architecture docs into Pinecone:
```bash
# Load ADO Backlog
python ingestion/ado_loader.py --organization my-org --project my-project (both parameters are optional, defaults are taken from config.poc.yaml)

# Load Architecture Docs
python ingestion/arch_loader.py --path ./docs/architecture 
```

#### 2. Run the System
Start the FastAPI backend (if applicable) or run the workflow script directly (depending on entry point).
```bash
python app.py
```

#### 3. Evaluation
Run evaluation scripts to assess performance:
```bash
# Generate test dataset
python evaluate/generate_eval_dataset.py --output eval/datasets/test.jsonl

# Run tagging evaluation (against evaluation data set)
python evaluate/evaluate_tagging.py --threshold 0.6
```

## 📂 Project Structure

```
.
├── agents/                 # Agent implementations
├── tools/                  # Tool implementations
├── workflows/              # Workflow orchestration
├── ingestion/              # Data ingestion scripts
├── evaluate/               # Evaluation scripts
├── prompts/                # YAML prompt templates
├── config.poc.yaml         # Main configuration
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🔌 API & Integration

The system exposes agents and tools that can be integrated into a chat interface or CI/CD pipeline. The `Supervisor Agent` acts as the main entry point for conversational interaction, while `BacklogSynthesisWorkflow` provides a structured execution path.

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v
```

## 📄 License

MIT
