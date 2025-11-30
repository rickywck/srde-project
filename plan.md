# Plan: Backlog Synthesizer POC (Simplified)

## 0. Scope & Goals

This simplified POC keeps the **same high-level architecture and flow** as the full plan, but:
- Focuses on a **single project** and a **single environment**.
- Reduces configuration surface (hard-coded defaults where reasonable).

Primary goals for the POC:
- Ingest existing ADO backlog + architecture constraints into a vector store.
- Upload a meeting note / transcript, segment it, and generate backlog items.
- Tag generated **user stories** as new / gap / conflict / extend relative to existing stories.
- Persist artifacts on disk for inspection (no UI beyond a basic upload form).

---
## 1. Multi-Agent Architecture Overview

### 1.1 Architecture Diagram

The system follows a multi-agent architecture where a Supervisor orchestrates specialized agents to process meeting notes and generate backlog items:

```mermaid
graph TB
    User[User] -->|Upload document & chat| UI[Chat Interface]
    UI -->|Instructions| API[FastAPI Backend]
    
    API -->|Raw document| Supervisor[Supervisor Agent]
    
    Supervisor -->|Document text| SegAgent[Segmentation Agent]
    SegAgent -->|Segments + intents| Supervisor
    
    Supervisor -->|Per segment| Retrieval[Retrieval Tool]
    Retrieval -->|Query| PineconeADO[("Pinecone: ADO Backlog")]
    Retrieval -->|Query| PineconeArch[("Pinecone: Architecture")]
    PineconeADO -->|Relevant items| Retrieval
    PineconeArch -->|Relevant constraints| Retrieval
    Retrieval -->|Context| Supervisor
    
    Supervisor -->|Segment + context| GenAgent[Backlog Generation Agent]
    GenAgent -->|Epics/Features/Stories| Supervisor
    
    Supervisor -->|Per story| TagAgent[Tagging Agent]
    TagAgent -->|Query existing stories| PineconeADO
    PineconeADO -->|Similar stories| TagAgent
    TagAgent -->|Tag: new/gap/conflict| Supervisor
    
    Supervisor -->|Optional: write items| ADOTool[ADO Writer Tool]
    ADOTool -->|Create items| ADO[Azure DevOps]
    
    Supervisor -->|Generated items + context| EvalAgent[Evaluation Agent]
    EvalAgent -->|Quality scores & feedback| Supervisor
    
    Supervisor -->|Results| API
    API -->|Generated backlog| UI
    UI -->|Display results| User
    
    style Supervisor fill:#e1f5ff,stroke:#01579b,stroke-width:3px
    style SegAgent fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style GenAgent fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style TagAgent fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style EvalAgent fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style ADOTool fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style PineconeADO fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style PineconeArch fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
```

### 1.2 Agent Roles

- **Supervisor Agent**: Orchestrates the entire workflow, manages state, and decides when to invoke specialized agents or tools.
- **Segmentation Agent**: Splits documents into coherent segments and identifies intents.
- **Retrieval Tool**: Tool invoked by Supervisor to search Pinecone for relevant ADO backlog items and architecture constraints.
- **Backlog Generation Agent**: Creates epics, features, and user stories from segments with retrieved context.
- **Tagging Agent**: Classifies generated stories relative to existing backlog (new/gap/conflict).
- **Evaluation Agent**: LLM-as-a-judge that evaluates quality of generated backlog items across multiple dimensions (completeness, relevance, quality).
- **ADO Writer Tool**: Optional tool invoked by Supervisor to persist items to Azure DevOps.

### 1.3 Data Flow



1. User uploads document and provides instructions via chat interface
2. Supervisor receives document and orchestrates processing
3. Document is segmented with intent detection
4. For each segment: retrieve relevant context → generate backlog items
5. For each story: retrieve similar stories → assign tag
6. Optionally: Supervisor invokes Evaluation Agent to assess quality of generated backlog items
7. Optionally: Supervisor invokes ADO Writer tool to persist items
8. Results returned to user via chat interface

---
## 2. Minimal Architecture Components
Components (kept from full design, simplified behavior):
- **External Loaders (CLI):**
  - ADO Backlog Loader: one-time/manual load of Epics/Features/Stories.
  - Architecture Loader: one-time/manual load of constraint docs.
- **Upload & Run API:** Simple FastAPI app with 3–4 endpoints.
- **Segmentation Agent:** Single LLM call to segment document + label intents.
- **Supervisor:** Orchestrates segmentation, retrieval, generation, tagging, and evaluation.
- **Backlog Generation Agent:** LLM that takes one segment + retrieved context and outputs epics/features/stories.
- **Tagging Agent:** LLM that takes one generated user story + retrieved existing stories and outputs a tag.
- **Evaluation Agent:** LLM-as-a-judge that evaluates generated backlog quality across multiple dimensions.
- **ADO Writer (optional for POC):** Simple script to create new items only (no modify/patch logic).

Non-goals / out-of-scope for this POC:
- Complex retry policies, cost monitoring, or multi-project routing.
- Custom / fine-grained Langfuse instrumentation (we rely on default Strands traces).

---
## 3. External Loaders (Simplified)

### 3.1 ADO Backlog Loader

**Purpose:** Load existing Epics/Features/Stories from one ADO project into Pinecone.

**Implementation (minimal):**
- CLI script `ado_loader.py` with a single command:
  - Inputs: ADO organization, project, PAT, Pinecone API key.
  - Fetches items of types `[Epic, Feature, Story]` via REST.
  - For each item, build `text = title + '\n' + description + '\n' + acceptanceCriteria`.
  - Embed with `text-embedding-3-small`.
  - Upsert into one Pinecone index, namespace = project name.
  - Metadata (minimal): `work_item_id`, `work_item_type`, `state`, `parent_id`, `project`, `doc_type='ado_backlog'`.

### 3.2 Architecture Loader

**Purpose:** Load a **small set** of architecture markdown/PDF/docs into Pinecone.

**Implementation (minimal):**
- CLI script `arch_loader.py`:
  - Inputs: `--project`, `--path` to folder of docs.
  - For each file, convert to text (support `.md`, `.docx`, `.pdf` using libraries like `python-docx` and `PyPDF2`).
  - Smart chunking using `RecursiveTokenChunker` from `ingestion.chunker`:
    - `chunk_size = 1000` (character length, roughly 500 tokens).
    - `chunk_overlap = 200`.
    - `separators = ["\n\n", "\n", ".", "?", "!", " ", ""]` (optimized for semantic coherence).
  - Embed each chunk with `text-embedding-3-small` (same model as ADO backlog loader).
  - Upsert into same Pinecone index, namespace = project name.
  - Metadata (minimal): `doc_type='architecture'`, `file_name`, `project`.

No retries, no observability, no JSONL artifacts required for POC (optional to add).

---
## 4. Chat Interface & API (Enhanced)

Implement a FastAPI app `app.py` with a **chat-based interface** that allows users to interact conversationally with the system:

### 4.1 Backend Endpoints

- `POST /upload`:
  - Multipart file upload.
  - Create `run_id` (UUID).
  - Save raw file to `runs/{run_id}/raw.txt`.
  - Return `{ run_id }`.

- `POST /chat/{run_id}`:
  - Request body: `{ message: string, instruction_type?: string }`.
  - Processes user instructions via chat (e.g., "Generate backlog items", "Write to ADO", "Show me conflicts").
  - Invokes **Supervisor Agent** which interprets the instruction and orchestrates appropriate agents/tools.
  - Returns: `{ run_id, response: string, status: object }`.

- `GET /backlog/{run_id}`:
  - Returns JSON content of `runs/{run_id}/generated_backlog.jsonl` (parsed as list).

- `GET /tagging/{run_id}`:
  - Returns JSON content of `runs/{run_id}/tagging.jsonl` (parsed as list).

- `GET /chat-history/{run_id}`:
  - Returns conversation history for the run.

### 4.2 Front-end: Chat Interface

- Single-page application with:
  - **File upload area**: Drag-and-drop or button to upload meeting notes/transcripts.
  - **Chat panel**: 
    - Message input box for user instructions.
    - Conversation history display showing user messages and system responses.
    - Quick action buttons: "Generate Backlog", "Show Tagging Results", "Write to ADO".
  - **Results panel**: 
    - Display generated backlog items in expandable cards.
    - Show tagging results with color-coded tags (new=green, gap=blue, conflict=red).
### 4.3 Conversational Capabilities

The chat interface allows users to:
- Request backlog generation: "Analyze this document and create backlog items"
- Query results: "Show me all conflict items", "What gaps were identified?"
- Request quality evaluation: "Evaluate the quality of generated backlog items", "How good are the generated stories?"
- Request modifications: "Regenerate stories for segment 3"
- Trigger ADO write: "Write the new and gap items to Azure DevOps"
- Ask for explanations: "Why was story X tagged as conflict?", "What are the quality issues with the generated items?"

The Supervisor Agent interprets these natural language instructions and orchestrates the appropriate workflow.

The Supervisor Agent interprets these natural language instructions and orchestrates the appropriate workflow.

---
## 5. Segmentation Agent (Simplified)

**Input:** Full document text.

**Output:** `runs/{run_id}/segments.jsonl` records with minimal fields:
- `segment_id` (int, 1..N)
- `segment_order` (same as `segment_id`)
- `raw_text`
- `intent_labels` (short list of strings)
- `dominant_intent` (string)

**Behavior:**
- Single LLM call with a prompt that:
  - Asks for segmentation into coherent chunks (roughly 500–1000 tokens each).
  - Asks to detect high-level intents per segment.
  - Returns structured JSON (array of segments).
- No retries beyond what the SDK provides.

**POC simplifications:**
- No Langfuse instrumentation requirement (optional to add later).
- No advanced quality heuristics, merging, or token metrics; only basic segmentation.

---
## 6. Per-Segment Retrieval & Generation (Simplified)

The Supervisor invokes the **Retrieval Backlog Tool** (combined retrieval + generation) for each segment.

### 6.1 Intent Embedding

For each segment:
- Build an intent query string: `dominant_intent + ' ' + ' '.join(intent_labels)` plus the first ~300 characters of `raw_text`.
- Call OpenAI embeddings (`text-embedding-3-small`) to get an intent vector.

### 6.2 Retrieval & Generation (Combined Tool)

The Supervisor invokes `generate_backlog_with_retrieval` (from `retrieval_backlog_tool`) which internally:
1. Queries Pinecone for ADO items and architecture constraints (using `retrieval_tool` logic).
2. Calls the Backlog Generation Agent with the retrieved context.
3. Returns ONLY the generated backlog items (reducing conversation history).

This combined approach is preferred over separate retrieval and generation steps to minimize token usage in the conversation history.

### 6.3 Prompt Assembly

Build a simple prompt template:

- Section 1: Original segment text.
- Section 2: Retrieved ADO items (id, type, title, short description).
- Section 3: Retrieved architecture chunks (short text + identifiers).
- Section 4: Instructions to generate epics/features/stories + ACs.

### 6.4 Generation Agent Call

- Single LLM call per segment with the above prompt.
- Response format: structured JSON list of backlog items with fields:
  - `type` (Epic|Feature|Story)
  - `title`
  - `description`
  - `acceptance_criteria` (list of strings)
  - `parent_ref` (optional ID or title of parent epic/feature)

Supervisor:
- Parses JSON.
- Assigns internal IDs (`epic_id`, `feature_id`, `story_id` counters within the run).
- Appends all items to `runs/{run_id}/generated_backlog.jsonl`.

POC simplifications:
- No retries beyond one optional re-call if JSON invalid.
- No constraint violation flags or AC delta computation (can be added later).

---
## 7. Per-Story Retrieval & Tagging (Simplified)

Once all segments are processed and backlog items are generated:

### 7.1 Story Selection

- Filter `generated_backlog.jsonl` to only `type == 'Story'`.

### 7.2 Story Embedding & Retrieval

For each story:
- The Tagging Agent can now perform internal retrieval if similar stories are not provided.
- It builds story text: `title + '\n' + description + '\n' + '\n'.join(acceptance_criteria)`.
- Queries Pinecone for existing user stories (using `SimilarStoryRetriever` service).
- Filters results to only include matches with `score >= min_similarity`.

**Early exit optimization:**
- If **all** retrieved stories fall below the threshold (i.e., no relevant matches found), automatically tag the story as `"new"` without calling the Tagging Agent LLM.
- Record this decision in `tagging.jsonl` with `decision_tag="new"`, `related_ids=[]`, and `reason="No similar existing stories found (all below threshold)"`.

### 7.3 Tagging Agent Call

**Only called if at least one existing story passes the similarity threshold** (see 7.2).

Prompt includes:
- Generated story (title, description, ACs).
- List of retrieved existing stories (id, title, short description, similarity score).
- Simple rules for deciding **one** of: `new`, `gap`, `conflict`.

LLM returns structured JSON:
- `decision_tag`: `"new" | "gap" | "conflict"`
- `related_ids`: list of ADO work item IDs considered most relevant.
- Optional `reason` (short text for debugging).

Supervisor:
- Writes tagging records to `runs/{run_id}/tagging.jsonl`.
- Also updates in-memory backlog items with `assigned_tag`.

POC simplifications:
- Threshold filtering happens before LLM call (see 7.2 early exit).
- LLM only reasons over stories that are sufficiently similar.
- Only three tag values: `new` (no similar stories), `gap` (fills a gap or extends existing work), `conflict` (contradicts existing stories).

The tagging output produced here is the main input to the simple evaluation in Section 9.

### 7.4 Direct Dataset Tagging Mode (Evaluation Feed)

To support reproducible tagging evaluation (F1 metrics) without relying on live Pinecone retrieval, the Tagging Agent can run in a **direct dataset mode**. In this mode, the Supervisor supplies a predefined snapshot of existing stories along with a generated story, bypassing embedding-based similarity filtering and retrieval.

#### 7.4.1 Purpose
Enable deterministic tagging on a curated evaluation dataset (see Section 9.1) so predictions can be directly compared with gold labels to compute precision/recall/F1.

#### 7.4.2 Input Record Schema (JSONL)
Each evaluation record provided to the Supervisor (e.g. from `datasets/tagging_test.jsonl`) contains:
- `story_title`
- `story_description`
- `story_acceptance_criteria` (list[str])
- `existing_stories` (list of objects):
  - `title`
  - `description`
  - `acceptance_criteria` (list[str])
  - Optional: `work_item_id`
- `gold_tag` (for evaluation only; not passed to Tagging Agent)
- Optional: `gold_related_ids`

#### 7.4.3 Supervisor Invocation Flow
1. Load evaluation record.
2. Construct tagging prompt directly from:
  - Generated story fields
  - Provided `existing_stories` list (treated as the retrieved context)
3. Skip retrieval & similarity threshold logic (Sections 7.2 early exit rules not applied).
4. Call Tagging Agent once per record.
5. Capture Tagging Agent output:
  - `decision_tag`
  - `related_ids` (if any)
  - `reason` (optional explanatory text)
6. Write prediction to `eval/tagging_predictions.jsonl` with fields:
  - `story_title`, `decision_tag`, `related_ids`, `reason`, `gold_tag` (copied from dataset for downstream metric computation).

#### 7.4.4 Differences vs Standard Mode
- No embedding/vector similarity filtering.
- Entire `existing_stories` snapshot is given verbatim to the Tagging Agent.
- Guarantees consistent inputs across runs → stable evaluation metrics.
- Still uses identical prompt structure & decision taxonomy (`new | gap | conflict`).

#### 7.4.5 Optional Enhancements (Future)
- Add a lightweight consistency check comparing direct-mode tag vs pipeline-mode tag for the same story.
- Include a confidence score (LLM self-assessment) to analyze correlation with errors.

#### 7.4.6 Integration With Evaluation (Section 9)
The file `eval/tagging_predictions.jsonl` is consumed by the tagging evaluation script to compute TP/FP/FN without re-running the agent. If predictions file exists, the evaluation script may skip live inference for faster iterative metric calculation.

This mode ensures clear isolation of **model reasoning quality** from **retrieval quality**, allowing targeted improvements.

---
## 8. ADO Writer Tool (Supervisor-Invoked)

The ADO Writer is implemented as a **tool** that the Supervisor Agent can optionally invoke based on user instructions via chat.

### 8.1 Tool Interface

**Tool Name**: `write_to_ado`

**Input Parameters**:
- `run_id`: The run identifier to read generated backlog from.
- `filter_tags`: List of tags to include (default: `["new", "gap"]`).
- `dry_run`: Boolean flag for preview mode (default: `false`).

**Output**:
- `created_items`: List of created ADO work item IDs.
- `summary`: Object with counts of created epics/features/stories.
- `errors`: List of any errors encountered.

### 8.2 Implementation

Implemented in `ado_writer_tool.py`:

**Behavior**:
1. Reads `runs/{run_id}/generated_backlog.jsonl` and `tagging.jsonl`.
2. Filters items based on `filter_tags` (default includes only items tagged as `new` or `gap`).
3. For each filtered item:
   - Creates new ADO work items (Epics/Features/Stories) via REST API.
   - Maintains parent-child relationships.
   - Tracks created item IDs.
4. Returns structured result with created item IDs and summary.

**POC Limitations**:
- **Only creates new items** (no PATCH/modify of existing items).
- No idempotency hashing or duplicate detection.
- No advanced retry logic beyond basic SDK retries.
- Items tagged as `conflict` are excluded by default (require manual review).

### 8.3 Supervisor Integration

The Supervisor Agent can invoke this tool when:
- User explicitly requests: "Write to ADO", "Create these items in Azure DevOps".
- User asks for dry-run preview: "Show me what would be created in ADO".
- As part of an automated workflow if configured.

The tool invocation is logged and results are displayed in the chat interface.

---
## 9. Minimal Evaluation

For the POC, we add two **lightweight evaluation** approaches to assess system quality:

### 9.1 Tagging Evaluation (Traditional Metrics)

This evaluates whether the Tagging Agent correctly classifies stories relative to existing backlog.

#### 9.1.1 Tagging Test Dataset

- Store a simple JSONL file at `datasets/tagging_test.jsonl` with records:
  - **Generated story (the one being tagged):**
    - `story_title`
    - `story_description`
    - `story_acceptance_criteria` (list of strings)
  - **Existing stories context (snapshot of what was available at tagging time):**
    - `existing_stories` (list of objects with `title`, `description`, `acceptance_criteria`)
  - **Gold label:**
    - `gold_tag` (`"new" | "gap" | "conflict"`)
    - `gold_related_ids` (optional list of identifiers or titles the human labeler considered when deciding the tag)

#### 9.1.2 Tagging Evaluation Script

- CLI script `evaluate_tagging.py`:
  1. Loads `datasets/eval_dataset.jsonl`.
  2. For each record:
     - Takes the **generated story** (title, description, ACs).
     - Uses the **provided `existing_stories`** snapshot (no live retrieval from Pinecone; uses the fixed set from the dataset).
     - Calls the **Tagging Agent LLM** with the same prompt structure used in the main flow.
  3. Compares predicted `decision_tag` with `gold_tag`.
  4. Counts TP/FP/FN per class and computes precision/recall/F1 per tag and macro-average.
  5. Writes a small JSON report to `eval/tagging_f1.json`.

### 9.2 Backlog Generation Evaluation (LLM-as-a-Judge)

This evaluates the overall quality and effectiveness of generated backlog items using an **Evaluation Agent** - an LLM-as-a-judge implemented as a specialized child agent.

#### 9.2.1 Evaluation Agent Architecture

The Evaluation Agent is a specialized agent that can be invoked by the Supervisor in two modes:

**Mode 1: Live Evaluation (Production Mode)**
- Invoked by Supervisor during or after backlog generation workflow
- Evaluates items in real-time as they are generated
- Provides immediate feedback to users via chat interface
- Results stored in `runs/{run_id}/evaluation.jsonl`

**Mode 2: Batch Evaluation (Offline/Testing Mode)**
- CLI script `evaluate_backlog_generation.py` invokes the agent on pre-prepared datasets
- Enables reproducible evaluation on curated test cases
- Results stored in `eval/backlog_generation_judge.json`

#### 9.2.2 Evaluation Agent Interface

**Agent Name**: `evaluate_backlog_quality`

**Input Parameters**:
- `segment_text`: Original meeting note segment
- `retrieved_context`: Object containing:
  - `ado_items`: List of retrieved existing ADO items
  - `architecture_constraints`: List of retrieved architecture chunks
- `generated_backlog`: List of generated epics/features/stories
- `evaluation_mode`: `"live"` or `"batch"` (default: `"live"`)

**Output Structure**:
```json
{
  "status": "success",
  "run_id": "...",
  "segment_id": "...",
  "evaluation": {
    "completeness": {"score": 4, "reasoning": "..."},
    "relevance": {"score": 5, "reasoning": "..."},
    "quality": {"score": 4, "reasoning": "..."},
    "overall_score": 4.3,
    "summary": "The generated backlog effectively captures...",
    "suggestions": ["Consider adding...", "Clarify..."]
  },
  "timestamp": "2025-11-24T..."
}
```

#### 9.2.3 Agent Implementation

Implemented in `agents/evaluation_agent.py`:

**Core Functionality**:
1. Receives segment, context, and generated backlog from Supervisor
2. Constructs evaluation prompt with structured criteria
3. Calls LLM (configurable model: `gpt-4o`, `claude-3-5-sonnet`, etc.) with JSON response format
4. Evaluates on multiple dimensions (1-5 scale):
   
   **Completeness (1-5):**
   - Does the generated backlog capture all key requirements from the segment?
   - Are critical details missing?
   
   **Relevance (1-5):**
   - Are the generated items relevant to the segment content?
   - Is there hallucinated or off-topic content?
   
   **Quality (1-5):**
   - Are titles clear and concise?
   - Are descriptions detailed and actionable?
   - Are acceptance criteria specific and testable?

5. Returns structured evaluation results
6. In live mode: writes to `runs/{run_id}/evaluation.jsonl`
7. In batch mode: aggregates scores and returns summary statistics

**Prompt Template Structure**:

```
You are an expert product manager and backlog quality evaluator. Evaluate the generated backlog items based on the original segment and context provided.

## Original Segment:
{segment_text}

## Retrieved Context:
### Existing ADO Items:
{retrieved_ado_items}

### 9.3 Combined Evaluation Report

A summary script `evaluate/generate_eval_report.py` combines both evaluation results:
- Tagging accuracy metrics (precision/recall/F1) from Section 9.1
- Backlog generation quality scores from Evaluation Agent (Section 9.2)
- Overall system effectiveness assessment
- Recommendations for prompt tuning or architecture changes
- Trend analysis if multiple evaluation runs are available

This dual evaluation approach provides both **objective metrics** (tagging accuracy) and **holistic quality assessment** (Evaluation Agent) to validate the POC's effectiveness.

### 9.4 Evaluation Agent vs. Batch Script

**Key Design Decisions:**

1. **Evaluation Agent** (`agents/evaluation_agent.py`):
   - Reusable component invokable by Supervisor or standalone
   - Implements the core LLM-as-a-judge logic
   - Single responsibility: evaluate one segment's generated backlog
   - Returns structured evaluation JSON
   - Can be used in production for real-time feedback

2. **Batch Evaluation Script** (`evaluate/evaluate_backlog_generation.py`):
   - Thin wrapper that loads dataset and invokes Evaluation Agent repeatedly
   - Handles aggregation, reporting, and metric computation
   - Enables reproducible testing on curated datasets
   - Used for CI/CD validation and regression testing

This separation ensures the evaluation logic is:
- **Consistent** across live and batch modes
- **Testable** in isolation
- **Reusable** for different workflows (e.g., per-segment evaluation vs. bulk assessment)
- **Observable** through standard Strands telemetry
2. Relevance: Are items relevant without hallucinations?
3. Quality: Are titles, descriptions, and ACs well-written?

Provide scores, reasoning, and actionable suggestions for improvement.

Return JSON matching this schema:
{evaluation_schema}
```

#### 9.2.4 Supervisor Integration

The Supervisor can invoke the Evaluation Agent when:
- User explicitly requests: "Evaluate the quality of generated items", "How good are these stories?"
- User asks for feedback: "What could be improved?", "Are there any quality issues?"
- Automated evaluation is enabled in config (for CI/CD validation)
- During batch evaluation runs for system testing

The agent invocation is logged and results are displayed in the chat interface with color-coded scores and actionable suggestions.

#### 9.2.5 Batch Evaluation Dataset

For batch/offline evaluation, store test cases at `datasets/eval_dataset.json` with records:
- **Input:**
  - `segment_text`: The original meeting note segment
  - `retrieved_context`: Snapshot of ADO items and architecture constraints provided to generation agent
- **Generated output:**
  - `generated_backlog`: List of generated epics/features/stories with full details
- **Reference (optional):**
  - `reference_backlog`: Optional human-created backlog items for comparison
  - `expected_score_range`: Optional guidance for validation

#### 9.2.6 Batch Evaluation Script

CLI script `evaluate/evaluate_backlog_generation.py`:
1. Loads `datasets/eval_dataset.json`
2. For each test case, invokes Evaluation Agent in batch mode
3. Collects all evaluation results
4. Aggregates scores across all test cases:
   - Per-dimension averages and distributions
   - Overall average score
   - Detailed per-case breakdowns
   - Common issues and patterns identified
5. Writes comprehensive report to `eval/backlog_generation_judge.json`
6. Optionally displays summary in terminal with visual charts
Provide scores, reasoning, and actionable suggestions for improvement.
```

### 9.3 Combined Evaluation Report

A summary script `generate_eval_report.py` combines both evaluation results:
- Tagging accuracy metrics (precision/recall/F1)
- Backlog generation quality scores (LLM judge)
- Overall system effectiveness assessment
- Recommendations for prompt tuning or architecture changes

This dual evaluation approach provides both **objective metrics** (tagging accuracy) and **holistic quality assessment** (LLM judge) to validate the POC's effectiveness.

---
## 10. Minimal Configuration

For the POC, use a single simple configuration file `config.poc.yaml`:

```yaml
ado:
  organization: your-org
  project: your-project
  pat_env_var: ADO_PAT

pinecone:
  api_key_env_var: PINECONE_API_KEY
  index_name: brd-poc
  environment: us-east-1

openai:
  api_key_env_var: OPENAI_API_KEY
  embedding_model: text-embedding-3-small
  chat_model: gpt-4o

retrieval:
  min_similarity_threshold: 0.7  # Filter out results below this score

project:
  name: your-project
```

No per-feature toggles, no thresholds configuration. Any extra knobs (e.g. `top_k`) can be hard-coded constants in code for this POC.

---
## 11. POC Success Criteria

The POC is considered successful if we can:
- Run loaders once to populate Pinecone with **real** ADO backlog and a small architecture corpus.
- Upload a real meeting note / transcript and:
  - See it segmented into 3–10 reasonable segments.
  - See 5–30 generated backlog items (including user stories) based on those segments.
  - See each story tagged as new/gap/extend/conflict with at least **plausible** reasoning.
- Inspect all artifacts under `runs/{run_id}/` to manually verify quality.

Future iterations can then:
- Add formal evaluation (F1), retries, constraint checks, and richer observability.
- Generalize configuration, support multiple projects/environments, and refine prompts.
