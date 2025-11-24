# Backlog Generation Workflow Implementation

## Overview

The "Generate Backlog" workflow has been implemented to orchestrate the complete backlog synthesis process from document upload to final output.

## Workflow Steps

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER UPLOADS DOCUMENT                       │
│                   (Meeting notes/transcript)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 1: SEGMENTATION ✅                        │
│                                                                 │
│  • Split document into coherent segments (500-1000 tokens)     │
│  • Identify intents per segment (feature_request, bug, etc.)   │
│  • Save to runs/{run_id}/segments.jsonl                        │
│                                                                 │
│  Implementation: agents/segmentation_agent.py                   │
│  Status: FULLY IMPLEMENTED                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: CONTEXT RETRIEVAL ⚠️                       │
│                                                                 │
│  For each segment:                                              │
│  • Embed intent + text using text-embedding-3-small            │
│  • Query Pinecone for similar ADO backlog items                │
│  • Query Pinecone for relevant architecture constraints        │
│  • Apply similarity threshold (min 0.7)                         │
│                                                                 │
│  Implementation: tools/retrieval_tool.py                        │
│  Status: PLACEHOLDER (not yet implemented)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            STEP 3: BACKLOG GENERATION ⚠️                        │
│                                                                 │
│  For each segment + retrieved context:                          │
│  • Generate epics with high-level scope                         │
│  • Generate features under epics                                │
│  • Generate user stories under features                         │
│  • Create acceptance criteria for stories                       │
│  • Maintain parent-child relationships                          │
│  • Save to runs/{run_id}/generated_backlog.jsonl               │
│                                                                 │
│  Implementation: agents/backlog_generation_agent.py             │
│  Status: PLACEHOLDER (not yet implemented)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                STEP 4: STORY TAGGING (Future)                   │
│                                                                 │
│  For each generated story:                                      │
│  • Compare with existing backlog items                          │
│  • Tag as: new / gap / conflict                                │
│  • Provide reasoning and related items                          │
│  • Save to runs/{run_id}/tagging.jsonl                         │
│                                                                 │
│  Implementation: agents/tagging_agent.py                        │
│  Status: PLACEHOLDER (not yet implemented)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### Backend API Endpoint

**Endpoint:** `POST /generate-backlog/{run_id}`

**Location:** `app.py`

**Functionality:**
- Loads uploaded document from `runs/{run_id}/raw.txt`
- Calls `supervisor.segment_document()` to segment
- Placeholder calls for retrieval and generation
- Returns formatted output with segmentation results
- Saves workflow progress to chat history

**Response Format:**
```json
{
  "run_id": "uuid",
  "status": "partial_success",
  "message": "Segmentation completed, retrieval and generation pending",
  "response": "Formatted text output",
  "workflow_steps": {
    "segmentation": {
      "status": "completed",
      "segments_count": 3,
      "segments_file": "runs/{run_id}/segments.jsonl"
    },
    "retrieval": {
      "status": "not_implemented",
      "message": "Retrieval tool pending implementation"
    },
    "generation": {
      "status": "not_implemented",
      "message": "Backlog generation agent pending implementation"
    }
  },
  "timestamp": "2024-11-24T12:00:00Z"
}
```

### Frontend Integration

**Location:** `static/app.js`

**New Function:** `generateBacklogWorkflow()`

**Trigger:** "Generate Backlog" button click

**Behavior:**
1. Checks if document is uploaded
2. Calls `/generate-backlog/{run_id}` endpoint
3. Displays workflow progress in chat
4. Shows segmentation results
5. Shows status of pending steps

**User Experience:**
- Click "Generate Backlog" button
- See "🚀 Starting backlog generation workflow..." message
- View segmentation results with intents
- See status of pending retrieval and generation steps
- Get next steps guidance

## Current Output Example

```
🎯 Backlog Generation Workflow Complete (Partial)

============================================================
STEP 1: DOCUMENT SEGMENTATION ✅
============================================================
Total Segments: 3

📄 SEGMENT 1
------------------------------------------------------------
Intent: feature_request
All Intents: feature_request, enhancement, user_story

Content Preview:
Topic 1: User Authentication Enhancement
We need to add multi-factor authentication...
------------------------------------------------------------

📄 SEGMENT 2
------------------------------------------------------------
Intent: bug_report
All Intents: bug_report, technical_requirement, decision

Content Preview:
Topic 2: Performance Issues
Several customers reported slow page load times...
------------------------------------------------------------

📄 SEGMENT 3
------------------------------------------------------------
Intent: feature_request
All Intents: user_story, feature_request, discussion

Content Preview:
Topic 3: Mobile App Offline Mode
Product team presented findings...
------------------------------------------------------------

============================================================
STEP 2: CONTEXT RETRIEVAL ⚠️
============================================================
Status: Not yet implemented
TODO: Query Pinecone for relevant ADO items and architecture

============================================================
STEP 3: BACKLOG GENERATION ⚠️
============================================================
Status: Not yet implemented
TODO: Generate epics, features, and user stories

============================================================
NEXT STEPS
============================================================
1. Implement retrieval_tool.py to query Pinecone
2. Implement backlog_generation_agent.py to create items
3. Implement tagging_agent.py to classify stories

📁 Segmentation output saved to:
   runs/{run_id}/segments.jsonl
```

## Testing

### Direct Test
```bash
python tests/test_workflow_direct.py
```
Tests the workflow logic directly without web server.

### API Test
```bash
# Start server
uvicorn app:app --reload

# In another terminal
python tests/test_workflow_api.py
```
Tests the workflow via HTTP API.

### Web UI Test
1. Start server: `uvicorn app:app --reload`
2. Open browser: `http://localhost:8000`
3. Upload a document
4. Click "Generate Backlog" button
5. View results in chat interface

## Files Modified

### Backend
- ✅ `app.py` - Added `/generate-backlog/{run_id}` endpoint
- ✅ `supervisor.py` - No changes (uses existing methods)

### Frontend
- ✅ `static/app.js` - Added `generateBacklogWorkflow()` function
- ✅ Updated button click handler

### Tests
- ✅ `tests/test_workflow_direct.py` - Direct workflow test
- ✅ `tests/test_workflow_api.py` - API endpoint test

## Next Implementation Steps

### Phase 1: Retrieval Tool
**File:** `tools/retrieval_tool.py`

Tasks:
1. Initialize Pinecone client
2. Create embedding function
3. Query ADO backlog items by segment
4. Query architecture constraints
5. Apply similarity thresholds
6. Return structured results

### Phase 2: Backlog Generation Agent
**File:** `agents/backlog_generation_agent.py`

Tasks:
1. Build generation prompts with segment + context
2. Call LLM to generate structured backlog items
3. Parse epics, features, stories with ACs
4. Maintain parent-child relationships
5. Save to `generated_backlog.jsonl`

### Phase 3: Tagging Agent
**File:** `agents/tagging_agent.py`

Tasks:
1. Query similar existing stories
2. Compare and classify (new/gap/conflict)
3. Provide reasoning and related items
4. Save to `tagging.jsonl`

### Phase 4: Integration
Tasks:
1. Update workflow to call retrieval tool
2. Update workflow to call generation agent
3. Update workflow to call tagging agent
4. Update output formatting for all steps
5. Add error handling and retries

## Benefits

✅ **User-Friendly**: Single button click to execute entire workflow

✅ **Observable**: Clear step-by-step progress messages

✅ **Incremental**: Works now with segmentation, ready for future steps

✅ **Testable**: Multiple test scripts for different scenarios

✅ **Maintainable**: Clean separation of workflow orchestration

✅ **Extensible**: Easy to add new steps or modify existing ones

## Usage in Production

Once all agents are implemented:

1. User uploads meeting notes
2. Clicks "Generate Backlog"
3. System automatically:
   - Segments document
   - Retrieves relevant context
   - Generates backlog items
   - Tags stories
4. User reviews results
5. User can optionally write to ADO

Estimated time: 30-60 seconds for typical document.
