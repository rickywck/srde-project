# Section 6 Implementation: Per-Segment Retrieval & Generation

## Overview

This implementation covers **Section 6** of the plan: Per-Segment Retrieval & Generation workflow. The system now:

1. **Segments documents** with intent detection
2. **Retrieves relevant context** from Pinecone (ADO items + architecture constraints)
3. **Generates structured backlog items** (Epics, Features, Stories) using LLM with retrieved context

## Architecture

### Components Implemented

1. **Retrieval Tool** (`tools/retrieval_tool.py`)
   - Queries Pinecone vector store for relevant context
   - Two-phase retrieval: ADO items + architecture constraints
   - Similarity threshold filtering (configurable, default: 0.5)
   - Mock mode for testing without Pinecone

2. **Backlog Generation Agent** (`agents/backlog_generation_agent.py`)
   - Generates structured backlog items from segments
   - Uses retrieved context to inform generation
   - Creates proper hierarchy (Epic → Feature → Story)
   - Includes acceptance criteria for stories
   - Mock mode for testing without OpenAI

3. **Supervisor Integration**
   - Orchestrates the complete workflow
   - Manages state across tools and agents
   - Tracks run_id for file organization

## Workflow

```
Document
   ↓
[Segmentation Agent]
   ↓ (segments with intents)
For each segment:
   ↓
[Retrieval Tool]
   ├─→ Query Pinecone (ADO items namespace)
   └─→ Query Pinecone (architecture namespace)
   ↓ (retrieved context)
[Backlog Generation Agent]
   ↓ (structured backlog items)
Save to runs/{run_id}/generated_backlog.jsonl
```

## Usage

### Basic Usage

```python
from agents.supervisor_agent import SupervisorAgent
from tools.retrieval_tool import create_retrieval_tool
from agents.backlog_generation_agent import create_backlog_generation_agent

# Initialize
supervisor = SupervisorAgent()
run_id = "my-run-123"

# Step 1: Segment document
result = await supervisor.segment_document(run_id, document_text)
segments = result["segments"]

# Step 2: For each segment, retrieve context and generate backlog
retrieval_tool = create_retrieval_tool(run_id)
generation_agent = create_backlog_generation_agent(run_id)

for segment in segments:
    # Retrieve context
    query_data = json.dumps({
        "segment_text": segment["raw_text"],
        "intent_labels": segment["intent_labels"],
        "dominant_intent": segment["dominant_intent"],
        "segment_id": segment["segment_id"]
    })
    
    retrieval_result = json.loads(retrieval_tool(query_data))
    
    # Generate backlog
    generation_data = json.dumps({
        "segment_id": segment["segment_id"],
        "segment_text": segment["raw_text"],
        "intent_labels": segment["intent_labels"],
        "dominant_intent": segment["dominant_intent"],
        "retrieved_context": {
            "ado_items": retrieval_result.get("ado_items", []),
            "architecture_constraints": retrieval_result.get("architecture_constraints", [])
        }
    })
    
    generation_result = json.loads(generation_agent(generation_data))
```

### Demo Script

Run the provided demo to see the complete workflow:

```bash
python tests/demo_retrieval_generation.py
```

### Test Suite

Run comprehensive tests:

```bash
# Run all tests
python -m pytest tests/test_retrieval_generation.py -v

# Run specific test
python -m pytest tests/test_retrieval_generation.py::test_retrieval_and_generation_workflow -v -s
```

## Configuration

Configuration is read from `config.poc.yaml`:

```yaml
openai:
  api_key_env_var: OPENAI_API_KEY
  embedding_model: text-embedding-3-small
  chat_model: gpt-4o

pinecone:
  api_key_env_var: PINECONE_API_KEY
  index_name: rde-lab
  environment: us-east-1

retrieval:
  min_similarity_threshold: 0.5  # Filter results below this score
```

## Output Format

### Segments File: `runs/{run_id}/segments.jsonl`

Each line contains:
```json
{
  "segment_id": 1,
  "segment_order": 1,
  "raw_text": "...",
  "intent_labels": ["intent1", "intent2"],
  "dominant_intent": "intent1"
}
```

### Generated Backlog File: `runs/{run_id}/generated_backlog.jsonl`

Each line contains:
```json
{
  "type": "Epic|Feature|Story",
  "title": "Item title",
  "description": "Detailed description",
  "acceptance_criteria": ["AC1", "AC2"],
  "parent_reference": "Parent item reference",
  "rationale": "Why this item is needed",
  "internal_id": "epic_1_1",
  "segment_id": 1,
  "run_id": "..."
}
```

## Key Features

### 1. Intent-Based Embedding

The retrieval tool creates rich query embeddings by combining:
- Dominant intent
- All intent labels
- First ~300 characters of segment text

This ensures semantic similarity matching captures the full context.

### 2. Two-Phase Retrieval

**Phase 1: ADO Items**
- Namespace: `ado_items`
- Top-K: 10 items
- Includes: Epics, Features, Stories from existing backlog

**Phase 2: Architecture Constraints**
- Namespace: `architecture`
- Top-K: 5 items
- Includes: Technical requirements, standards, constraints

### 3. Context-Aware Generation

The generation prompt includes:
- Original segment text and intents
- Retrieved existing ADO items (with similarity scores)
- Retrieved architecture constraints
- Instructions for proper hierarchy and acceptance criteria

### 4. Structured Output

Generated items include:
- Clear type classification (Epic/Feature/Story)
- Proper hierarchy with parent references
- Testable acceptance criteria (for Stories)
- Rationale explaining why the item was created

### 5. Mock Mode Support

Both tools support mock mode for offline testing:
- Set `OPENAI_API_KEY` to empty or missing → mock generation
- Set `PINECONE_API_KEY` to empty or missing → mock retrieval

## Example Output

For a document about MFA implementation, the system generates:

```
Epic: Enhanced Multi-Factor Authentication (MFA) Support
├── Feature: Support for SMS, Email, and Authenticator App MFA Channels
│   ├── Story: Implement MFA via SMS
│   │   └── 5 acceptance criteria
│   ├── Story: Implement MFA via Email
│   │   └── 5 acceptance criteria
│   └── Story: Implement MFA via Authenticator App
│       └── 5 acceptance criteria
└── Feature: Authentication Service Schema Migration for MFA
    └── Story: Modify Database Schema to Store MFA Preferences
        └── 5 acceptance criteria
```

## Next Steps

With Section 6 complete, the next implementations are:

- **Section 7**: Per-Story Tagging (classify stories as new/gap/conflict)
- **Section 8**: ADO Writer Tool (persist items to Azure DevOps)
- **Section 9**: Evaluation Framework (measure quality)

## Troubleshooting

**No items retrieved from Pinecone:**
- Ensure ADO and architecture loaders have been run first
- Check that `PINECONE_API_KEY` is set correctly
- Verify index name in `config.poc.yaml` matches Pinecone
- Check similarity threshold (lower it if too strict)

**Generation produces unexpected results:**
- Review segment intents in `segments.jsonl`
- Check retrieved context quality
- Adjust generation prompt if needed
- Verify `OPENAI_API_KEY` is set

**Files not being created:**
- Check that `runs/` directory exists and is writable
- Verify run_id is being passed correctly
- Check logs for error messages
