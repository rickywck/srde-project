# Plan: Backlog Synthesizer POC (Modular Segmentation & External Loaders)

## 0. Overview
This plan defines a modular backlog synthesis system with the following key features:
1. ADO backlog and architecture constraint ingestion are performed by manual external loaders (not by the Supervisor Agent).
2. A lightweight UI lets users upload meeting notes / transcripts and trigger a synthesis run.
3. A dedicated LLM-based Segmentation Agent produces ordered, intent-aligned segments with full original text preserved (no summarization).
4. A separate Backlog Generation Agent performs per-segment backlog item generation (epics/features/stories) using retrieved context.
5. A dedicated Tagging Agent performs user story-level classification (gap/conflict/extend/new) by retrieving relevant existing user stories and making tagging decisions.
6. Retrieval (ADO + architecture) is orchestrated by the Supervisor for generation; user story retrieval is orchestrated for tagging.
7. Rich metadata & observability (Langfuse) apply only to Supervisor-managed synthesis runs (not ad-hoc loader or evaluation jobs).

Core stack: Strands (single Supervisor for synthesis only), Pinecone, OpenAI `text-embedding-3-small` embeddings, AWS AgentCore Memory, Azure DevOps (REST or MCP), Langfuse.

---
## 1. Architecture Overview

Component Layers:
- External Loaders (manual): ADO Backlog Loader, Architecture Constraint Loader.
- User Interaction: Upload & Run UI.
- Segmentation Agent: LLM semantic segmentation.
- Supervisor Orchestrator: Coordinates segmentation, retrieval, and invokes Generation Agent and Tagging Agent; persists artifacts.
- Backlog Generation Agent: Produces structured backlog items (epics/features/stories) from segment + retrieved context.
- Tagging Agent: Performs user story-level classification by retrieving relevant existing stories and deciding tags (gap/conflict/extend/new).
- Writer Module: Manual trigger to create/modify ADO work items.
- Evaluation Suite: Tagging F1 (with prepared dataset) + LLM-as-judge quality metrics.

Execution Flow:
Manual ingestion (ADO + architecture) → User upload → Segmentation (extract intents) → For each segment: retrieval of ADO + architecture context (Supervisor) → invoke Generation Agent (generate backlogs) → For each generated user story: retrieval of existing user stories (Supervisor) → invoke Tagging Agent (classify tag) → persist → (optional) ADO write → evaluation.

---
## 2. External Loader Modules (Manual)

### 2.1 ADO Backlog Loader (`modules/ado_loader.py`)
Purpose: Populate Pinecone with existing Epics, Features, Stories (and optionally Bugs).
Steps:
1. Fetch items via REST or MCP domains (`work`, `work-items`).
2. Normalize fields (title, description, ACs, tags, hierarchy parent_id, state, updated_at, url, work_item_type).
3. Compute `work_item_hash = sha256(title+description+parent_id+updated_at)`; skip unchanged.
4. Embed (title + description + ACs) using `text-embedding-3-small`.
5. Upsert vectors into `brd-upload` index using per-project namespace.
6. Emit artifact: `ingestion/ado/{project}/ado_vectors.jsonl`.
7. (No dedicated observability instrumentation.)

### 2.2 Architecture Constraints Loader (`modules/arch_loader.py`)
Purpose: Ingest architecture constraint documents for retrieval.
Steps:
1. Convert source formats (.md/.pdf/.docx) to text; store original under `ingestion/architecture/{project}/raw/`.
2. Chunk using `RecursiveTokenChunker` (size=1000, overlap=200).
3. Metadata per chunk: `document_id`, `chunk_id`, `doc_type=architecture`, `source_type=architecture_loader`, `file_name`, `uploaded_at`, `chunk_index`, `total_chunks`, `checksum`, `project`.
4. Embed chunk text; upsert into Pinecone namespace.
5. Artifact: `ingestion/architecture/{project}/arch_vectors.jsonl`.
6. (No dedicated observability instrumentation.)

Rationale: Manual loaders isolate deterministic ingestion from reasoning pipeline; allow independent scheduling & audit.

---
## 3. Upload & Run UI

Minimal web service (FastAPI recommended):
- `POST /upload` (multipart): Accept meeting note / transcript; returns `run_id`.
- `POST /run/{run_id}`: Triggers Supervisor pipeline for that upload.
- `GET /status/{run_id}`: Progress (segment_count, completed_segments, phase).
- `GET /artifacts/{run_id}`: Lists generated artifact paths.
- `POST /write/{run_id}`: Initiates ADO write phase (manual decision).

Client: Simple HTML form + JavaScript polling. Raw file stored at `uploads/{run_id}/raw.{ext}`. Security via PAT / API key.

---
## 4. Segmentation Agent (LLM-Based)

Input: Full raw uploaded document text.
Output: `runs/{run_id}/segments.jsonl` with records:
`run_id`, `segment_id`, `segment_order`, `raw_text`, `token_count`, `intent_labels[]`, `dominant_intent`, `segmentation_version`, `timestamp`.

Algorithm:
1. Detect intents (features, components, user roles, actions, explicit work item references).
2. Determine boundaries ensuring semantic cohesion; target token range ≤ 1200.
3. Merge ultra-short segments (<80 tokens) forward if similarity high (divergence < 0.15).
4. Guarantee full original text preservation (no summarization, no paraphrasing).
5. Retry segmentation once with lowered temperature if first attempt fails.

Trace Span: `segmentation` (metrics: segment_count, avg_tokens, total_tokens, merge_operations).

---
## 5. Backlog Generation Agent (LLM-Based)

Purpose: Convert a single segmented intent + its retrieved ADO backlog items (epics/features/stories) and architecture context into structured backlog items while enforcing terminology consistency and constraint compliance.

Invocation: Called once per segment by Supervisor after intent-based retrieval & prompt assembly (see Section 7.2).

Input Payload:
`run_id`, `segment_id`, `segment_order`, `raw_text`, `intent_labels`, `dominant_intent`, `retrieved_ado_items[]` (epics, features, stories), `retrieved_architecture_chunks[]`, `config_snapshot`, `prompt_template_version`.

Output Records:
- Generated backlog items (without tags) appended to `runs/{run_id}/generated_backlog.jsonl`.

Responsibilities:
1. Analyze original segment text alongside retrieved ADO work items and architecture constraints.
2. Generate hierarchical items referencing existing parents where applicable.
3. Produce acceptance criteria; mark deltas via `ac_delta=true` entries.
4. Enforce architecture constraints: flag violations (`constraint_violation=true`).
5. Return structured JSON payload (no free-form prose outside fields).
6. **Does NOT perform tagging** - tagging is delegated to the Tagging Agent.

Note: Embedding, vector search, and ranking are handled by the Supervisor (see Section 7.2).

Retry & Robustness:
- Single automatic retry on malformed JSON or schema mismatch (temperature ↓ 0.05).
- On second failure: Supervisor logs error; segment marked skipped (record in `errors.jsonl`).

Prompt Strategy (High-Level):
Sections: Original Segment Text → Existing Related ADO Items (structured) → Architecture Constraints Snippets → Generation Instructions & Schema.

Spans:
- `segment_{segment_id}_generation_agent` (metrics: token_usage_prompt, token_usage_completion, item_count, constraint_violations).

Observability:
- Langfuse span `segment_{segment_id}_generation_agent` recorded (Supervisor-managed).

Failure Modes:
- Constraint parsing failure: add warning field in output record.
- Excess token length: Supervisor truncates lowest-ranked architecture chunks before re-invocation.

Security:
- Agent receives only run-scoped context; no credentials for ADO writes.

Versioning:
- `generation_agent_version` added to each generated backlog item.

---
## 6. Tagging Agent (LLM-Based)

Purpose: Classify each generated user story with a tag (gap/conflict/extend/new) by retrieving the most relevant existing user stories and performing similarity analysis and semantic reasoning.

Invocation: Called once per generated user story by Supervisor after Backlog Generation Agent completes.

Input Payload:
`run_id`, `segment_id`, `story_id`, `story_title`, `story_description`, `story_acceptance_criteria`, `parent_feature_id`, `parent_epic_id`, `config_snapshot`, `tagging_template_version`.

Output Records:
- Tagging decisions appended to `runs/{run_id}/tagging_analysis.jsonl`.

Responsibilities:
1. Receive embedded story vector and retrieved existing stories from Supervisor.
2. Compute similarity scores and apply thresholds.
3. Perform semantic reasoning to determine:
   - `conflict`: high similarity (≥0.80) to existing story with contradictory requirements.
   - `gap`: enhancement/extension of existing story (≥0.65) with explicit delta.
   - `extend`: adds related functionality or ACs to existing story (≥0.70) without conflict.
   - `new`: below similarity threshold (<0.55) or no semantic overlap.
4. Return structured tagging record with decision, similarity scores, related story IDs, and reasoning excerpt.

Note: Embedding and retrieval are handled by the Supervisor (see Section 7.3).

Retry & Robustness:
- Single automatic retry on malformed JSON or unclear decision (temperature ↓ 0.05).
- On second failure: Supervisor logs error; story marked as `new` by default with `tagging_failed=true` flag.

Prompt Strategy (High-Level):
Sections: Generated User Story (title, description, ACs) → Retrieved Existing User Stories (structured: id, title, description, similarity score) → Tagging Decision Instructions & Thresholds → Output Schema.

Spans:
- `story_{story_id}_tagging_agent` (metrics: token_usage_prompt, token_usage_completion, retrieved_story_count, decision_tag, max_similarity).

Observability:
- Langfuse span `story_{story_id}_tagging_agent` recorded (Supervisor-managed).

Failure Modes:
- No existing stories retrieved: default to `new` tag.
- Ambiguous decision (multiple tags with similar confidence): log warning; select highest-confidence tag.

Security:
- Agent receives only run-scoped generated story and retrieved story context; no write permissions.

Versioning:
- `tagging_agent_version` added to each tagging analysis record.

---
## 7. Supervisor Orchestration Flow

The Supervisor manages the end-to-end synthesis pipeline for each uploaded document:

### 7.1 Phase 1: Segmentation
1. Supervisor invokes Segmentation Agent with full uploaded document.
2. Segmentation Agent extracts semantic intents and produces ordered segments.
3. Output persisted to `runs/{run_id}/segments.jsonl`.

### 7.2 Phase 2: Per-Segment Generation (Sequential)
For each segment in order:

**Step 2.1: Intent Embedding**
- Embed: `dominant_intent + intent_labels + first_200_chars(raw_text)`.

**Step 2.2: Retrieval (Parallel)**
Supervisor performs dual vector search using the intent embedding:

a) **ADO Backlog Retrieval:**
   - Search Pinecone with intent embedding vector
   - Filter: `source_type=ado_backlog`
   - TopK: 5 (configurable via `retrieval.ado_top_k`)
   - Returns: Existing Epics, Features, User Stories with similarity scores
   - Namespace: per-project

b) **Architecture Constraints Retrieval:**
   - Search Pinecone with intent embedding vector
   - Filter: `doc_type=architecture`
   - TopK: 8 (configurable via `retrieval.architecture_top_k`)
   - Returns: Architecture constraint chunks with similarity scores
   - Namespace: per-project

**Step 2.3: Ranking & Deduplication**
- Composite score = `0.7*similarity + 0.2*recency + 0.1*priority` (priority from architecture metadata if present).
- Deduplicate architecture chunks with token Jaccard > 0.9.
- Trim context if exceeds `segment_max_context_tokens` (drop lowest-ranked architecture chunks first).

**Step 2.4: Prompt Assembly**
```
## Segment Original Text (segment {segment_order})
{raw_text}
## Retrieved ADO Work Items
{ado_items}
## Retrieved Architecture Constraints
{architecture_chunks}
## Task
Generate Epics/Features/Stories that:
- Reuse or extend relevant existing parent Epics/Features when aligned.
- Comply with architecture constraints.
- Maintain established terminology.
- Explicitly mark modifications or deltas.
```

**Step 2.5: Generation Agent Invocation**
- Supervisor invokes Backlog Generation Agent with assembled prompt and context.
- Agent returns hierarchical items (epic→feature→story) with acceptance criteria.
- AC extensions flagged via `ac_delta=true`.

**Step 2.6: Persistence**
- Supervisor persists generated items to `runs/{run_id}/generated_backlog.jsonl`.

### 7.3 Phase 3: Per-Story Tagging (Sequential)
After all segments are generated, for each generated user story:

**Step 3.1: Story Embedding**
- Supervisor embeds story: `title + description + acceptance_criteria`.

**Step 3.2: Retrieval**
- Retrieve TopK=10 (configurable via `retrieval.tagging_top_k`) most relevant existing user stories from Pinecone.
- Filter: `work_item_type=User Story`, `source_type=ado_backlog`.
- Namespace: per-project.
- Returns existing stories with similarity scores.

**Step 3.3: Tagging Agent Invocation**
- Supervisor invokes Tagging Agent with story details and retrieved existing stories.
- Agent returns decision record (tag, similarity scores, related IDs, reasoning).

**Step 3.4: Persistence**
- Supervisor persists tagging analysis to `runs/{run_id}/tagging_analysis.jsonl`.
- Supervisor updates generated story record with assigned tag.

### 7.4 Observability Spans
- Phase 1: `segmentation`
- Phase 2 per segment: `segment_{segment_id}_retrieval`, `segment_{segment_id}_generation_agent`
- Phase 3 per story: `story_{story_id}_tagging_retrieval`, `story_{story_id}_tagging_agent`

---
## 8. ADO Write (Manual Trigger)

Triggered after user review (`POST /write/{run_id}` or CLI). Operations:
1. Create new Epics → Features → Stories maintaining hierarchy.
2. Modify existing items (extend / gap) via PATCH (append ACs, update description, add tags).
3. Idempotency: `item_signature = sha256(type+title+parent_ref+concat_ac)`; skip if signature matches existing.
4. Retry policy: exponential backoff (1s,2s,4s,8s,16s) for 429/5xx (max 5 attempts).
5. Artifact: `runs/{run_id}/ado_write_results.jsonl` (status, created_ids, modified_ids, errors).
6. (No observability spans for writer.)

---
## 9. Evaluation (Optional Post-Run)

### 9.1 Tagging F1 Score
Dataset file: `datasets/{dataset_version}/tagging_test.jsonl` (fields: story_id, story_title, story_description, story_acceptance_criteria, gold_tag, gold_related_ids).

Process:
1. For each test sample, invoke Tagging Agent with story details.
2. Compare predicted tag against gold tag.
3. Compute TP/FP/FN per tag category (conflict, gap, extend, new).
4. Aggregate precision, recall, F1 per category and overall.

Artifacts: `eval/{run_id}/tagging_f1.json` (per-category and overall metrics).

### 9.2 LLM-as-Judge
Checks for each generated story:
- Gap correctness (real enhancement, not contradiction).
- INVEST attributes → `invest_score`.
- AC completeness (coverage, negative cases, measurability) → `ac_completeness`.
- Traceability (source segment reference) → `traceability_score`.

Artifacts: `eval/{run_id}/judge_quality.jsonl`.

---
## 10. Configuration (Extended)

```yaml
external_loaders:
  ado:
    enabled: true
    ingestion_mode: manual
    work_item_types: [Epic, Feature, Story]
  architecture:
    enabled: true
    chunk_size: 1000
    chunk_overlap: 200

segmentation:
  model: gpt-4o
  max_segment_tokens: 1200
  min_merge_threshold: 80
  cohesion_divergence_threshold: 0.15
  temperature: 0.2

retrieval:
  ado_top_k: 5
  architecture_top_k: 8
  tagging_top_k: 10
  similarity_weight: 0.7
  recency_weight: 0.2
  priority_weight: 0.1

thresholds:
  newBelow: 0.55
  conflictAtLeast: 0.80
  extendSimilarity: 0.70
  gapAtLeast: 0.65
  constraint_violation_check: true

embedding:
  model: text-embedding-3-small

ado:
  mode: rest
  organization: your-org
  project: your-project

observability:
  enable_langfuse: true
  per_segment_spans: true
  store_config_snapshot: true

feature_flags:
  modular_mode: true
  streaming_generation: false
```

Config snapshot saved as `runs/{run_id}/config_snapshot.yaml` each run.

---
## 11. Metadata Model (Extended)

### 11.1 Architecture Loader Chunk Metadata
`document_id`, `chunk_id`, `doc_type=architecture`, `source_type=architecture_loader`, `checksum`, `project`, `uploaded_at`, `file_name`, `chunk_index`, `total_chunks`.

### 11.2 Segmentation Segment Metadata
`segment_id`, `segment_order`, `run_id`, `token_count`, `intent_labels`, `dominant_intent`, `segmentation_version`, `full_text_preserved=true`.

### 11.3 ADO Vector Metadata
`work_item_id`, `source_type=ado_backlog`, `project`, `work_item_type`, `state`, `tags`, `parent_id`, `updated_at`, `url`.

### 11.4 Generated Backlog Item
`run_id`, `segment_id`, `segment_order`, `epic_id`, `feature_id`, `story_id`, `type`, `title`, `description`, `acceptance_criteria`, `assigned_tag` (populated by Tagging Agent), `parent_work_item_id`, `extends_work_item_ids`, `constraint_violation`, `ac_delta_entries`, `context_ado_items`, `context_architecture_chunks`, `source_doc`, `source_doc_segment`, `generation_agent_version`, `trace_id`.

### 11.5 Tagging Analysis Record
`run_id`, `segment_id`, `story_id`, `decision_tag` (conflict|gap|extend|new), `similarity_scores[]` (top retrieved stories with scores), `max_similarity`, `related_story_ids[]`, `reasoning_excerpt`, `thresholds_applied`, `tagging_agent_version`, `tagging_failed` (boolean), `trace_id`.

---
## 12. Langfuse Observability (Updated)
Traces:
- Run: `synthesis_run` (metadata: run_id, project, segment_count, story_count, embedding_model) — recorded only for Supervisor pipeline invocations.

Per-Segment Spans (Supervisor-managed):
- `segmentation`, `retrieval`, `generation_agent`.

Per-Story Spans (Supervisor-managed):
- `tagging_retrieval`, `tagging_agent`.

Evaluation:
- No dedicated spans (evaluation jobs run ad-hoc without Langfuse instrumentation).

Metrics Captured (Supervisor only):
- Segment counts, avg token length, story counts, tagging distribution (conflict/gap/extend/new), tagging failure rate.

---
## 13. Further Considerations & Future Extensions

1. Segmentation Reliability: Introduce quality heuristic (mean similarity dispersion) to flag under-segmentation.
2. Adaptive Retrieval: Lower architecture TopK for long segments to maintain token budget.
3. Re-Ranking: Add cross-encoder for ADO items if precision insufficient.
4. Streaming Generation: Optional feature flag for incremental story display.
5. Requirements Loader: Separate from architecture with `doc_type=requirement` for domain separation.
7. Multi-Project Support: Extend namespace strategy with composite key (`project:environment`).
8. Security: Encrypt uploaded raw documents at rest; implement retention policy (e.g., auto-delete after N days).
9. Cost Monitoring: Track token usage per segment and per story (tagging); apply early stop if budget exceeded.
10. Accessibility: Provide diff UI for gap/extend items showing delta vs source work item.
11. Rollback Strategy: Maintain mapping of generated ADO IDs for bulk deletion if run invalidated.
12. Tagging Dataset Management: Version control tagging test datasets; automate periodic dataset refresh from production tagging results.
13. Retagging Support: Enable retagging of existing stories without regeneration; useful for threshold tuning or model upgrades.

---
## 14. Agent Summary

| Agent | Responsibility | Determinism | Trigger | Interfaces |
|-------|----------------|------------|--------|------------|
| Supervisor Agent | Orchestrates segmentation, retrieval, and invokes Generation & Tagging Agents; manages lifecycle & artifacts with Langfuse spans | Mixed (procedural + LLM) | `POST /run/{run_id}` | Pinecone, LLM (prompt assembly), Langfuse, optional ADO writer |
| Backlog Generation Agent | Generates backlog items (epics/features/stories) from segment & retrieved context | Non-deterministic (LLM) | Invoked per segment by Supervisor | LLM, Langfuse |
| Tagging Agent | Classifies user stories (gap/conflict/extend/new) by retrieving relevant existing stories | Non-deterministic (LLM) | Invoked per generated user story by Supervisor | Pinecone, LLM, Langfuse |
| Segmentation Agent | LLM semantic segmentation; outputs ordered segments with intents & raw text | Non-deterministic (LLM) | Invoked by Supervisor start | LLM, Langfuse |
| ADO Writer Module | Create/modify ADO items post-review | Deterministic | `POST /write/{run_id}` | ADO REST/MCP |
| ADO Loader Module | Initial/backfill ingestion of existing backlog | Deterministic | Manual CLI / admin | ADO REST/MCP, Pinecone |
| Architecture Loader Module | Ingest architecture constraints corpus | Deterministic | Manual CLI / admin | FS, Pinecone |
| (Optional) Evaluation Agent | Tagging F1 + judge scoring (can embed in Supervisor) | Mixed | Post-write / manual | Datasets, LLM |
| (Optional) Judge Sub-Agent | Specialized gap correctness & INVEST scoring | Non-deterministic | Batch after generation | LLM |

Minimum viable: Supervisor + Segmentation + Generation + Tagging (others as modules). Optional agents extracted when throughput, latency, or isolation needs grow.

Rationale for Separation:
- Deterministic ingestion & writing kept as modules to minimize agent orchestration overhead.
- Segmentation isolated for targeted retries, metrics, and potential model upgrades independent of generation.
- **Tagging separated from Generation** to enable user story-level classification with dedicated retrieval, independent evaluation (F1 score with prepared dataset), and potential retagging without regenerating backlog items.
- Evaluation decoupling enables asynchronous quality audits without blocking synthesis runs.

Promotion Criteria (convert module → agent):
1. Need autonomous scheduling (e.g., periodic ADO freshness checks).
2. Distinct scaling profile (segmentation becomes bottleneck; scale horizontally).
3. Specialized safety or filtering pipeline for judge evaluations.

Observability Mapping (Supervisor-managed only):
- Segmentation Agent: span `segmentation` per run.
- Backlog Generation Agent: span `generation_agent` per segment.
- Retrieval (procedural step inside Supervisor): span `retrieval` per segment.
- Tagging Agent: span `tagging_agent` per user story; span `tagging_retrieval` for retrieval step.
  (No spans for loaders, ADO writer, or evaluation jobs.)

Failure Handling Summary:
- Segmentation: retry once (lower temperature); abort run on second failure.
- Generation: skip segment on irrecoverable error; log entry in `runs/{run_id}/errors.jsonl`.
- Tagging: default to `new` tag on irrecoverable error; mark `tagging_failed=true` in analysis record.
- Writer: exponential backoff; record failures for manual re-run.

Security & Permissions:
- Loader & Writer modules use elevated PAT/credentials.
- Agents operate with restricted run-scoped tokens; only write under `runs/{run_id}/`.

Versioning:
- Segmentation version tracked in segment metadata (`segmentation_version`).
- Supervisor pipeline version tracked in config snapshot.
- Semantic tags recommended (e.g., `seg-v2.1`, `sup-v1.4`).

Future Agentization Candidates:
- Requirements Loader Agent (if requirement docs gain dynamic updates).
- Re-Ranking Agent (if cross-encoder stage added for precision improvements).

---
End of updated modular plan.
