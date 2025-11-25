# ADO Export Feature

The ADO Writer Tool exports generated backlog items (Epics, Features, User Stories) to Azure DevOps.

## Configuration

### 1. Environment Variables (.env file)

```bash
ADO_PAT=your_personal_access_token_here
```

### 2. Project Configuration (config.poc.yaml)

```yaml
ado:
  organization: rickywck        # Your ADO organization
  project: rde-lab             # Your ADO project
  pat_env_var: ADO_PAT         # Name of env var containing PAT
```

## Usage

### Via API Endpoint

**Preview Export (Dry Run)**
```bash
POST /ado-export/{run_id}?dry_run=true&filter_tags=new,gap
```

Response shows what would be created:
```json
{
  "status": "ok",
  "mode": "dry_run",
  "summary": {
    "counts": {
      "epics": 1,
      "features": 2,
      "stories": 2
    },
    "filter_tags": ["new", "gap"],
    "ado_config": {
      "organization": "rickywck",
      "project": "rde-lab",
      "pat_present": true
    }
  }
}
```

**Actual Export**
```bash
POST /ado-export/{run_id}?dry_run=false&filter_tags=new,gap
```

Response includes created work item details:
```json
{
  "status": "ok",
  "mode": "write",
  "created_items": [
    {
      "type": "Epic",
      "ado_id": 12345,
      "title": "Voice Chatbot Dispute Submission",
      "url": "https://dev.azure.com/rickywck/rde-lab/_workitems/edit/12345"
    },
    {
      "type": "Feature",
      "ado_id": 12346,
      "title": "Automated Information Capture",
      "parent_ado_id": 12345,
      "url": "https://dev.azure.com/rickywck/rde-lab/_workitems/edit/12346"
    }
  ],
  "counts": {
    "epics_created": 1,
    "features_created": 2,
    "stories_created": 2
  },
  "errors": []
}
```

### Via UI

1. **Complete workflow** to generate backlog items
2. Click **"Preview ADO Export"** to see what will be created (dry run)
3. Review the counts and items
4. Click **"Export to ADO"** to create work items (actual write)

## Filtering

Stories are filtered by their classification tags:
- `new`: Completely new functionality
- `gap`: Fills a gap in existing features
- `conflict`: Conflicts with existing items (excluded by default)

Default filter: `["new", "gap"]`

## Work Item Creation

The tool creates work items in hierarchical order:

1. **Epics** (no parent)
   - Title
   - Description (includes rationale)

2. **Features** (parent: Epic)
   - Title
   - Description (includes rationale)
   - Parent relationship to Epic

3. **User Stories** (parent: Feature)
   - Title
   - Description (includes rationale)
   - Acceptance Criteria
   - Parent relationship to Feature

## Parent-Child Relationships

The tool automatically establishes parent-child links:
- Stories link to their parent Features
- Features link to their parent Epics
- Uses `parent_reference` (title matching) or direct ID fields
- Creates ADO hierarchy relationships via REST API

## Error Handling

- Partial success: Some items created, some failed (status: "partial")
- Complete failure: No items created (status: "error")
- Errors array includes details for each failure
- Successfully created items are reported even if later items fail

## ADO REST API

Uses Azure DevOps REST API 7.0:
- Endpoint: `https://dev.azure.com/{org}/{project}/_apis/wit/workitems`
- Authentication: Basic auth with PAT
- Format: JSON Patch documents
- Work item types: Epic, Feature, User Story

## Required Permissions

Your ADO PAT needs:
- **Work Items**: Read, Write, & Manage
- Scope: Project-level access to target project

## Testing

Run tests with mock ADO API:
```bash
PYTHONPATH=/Users/ricky.c.wong/poc/rde/v2 python tests/test_ado_writer.py
```

## Troubleshooting

**"PAT not present"**
- Ensure `.env` file exists with `ADO_PAT=your_token`
- Restart the application after adding/changing .env

**"No generated backlog found"**
- Run the backlog generation workflow first
- Check `runs/{run_id}/generated_backlog.jsonl` exists

**"0 epics, 0 features, 0 stories"**
- Check tagging results in `runs/{run_id}/tagging.jsonl`
- Verify filter_tags match story classifications
- Try filter_tags=new,gap,conflict to include all stories

**API authentication errors**
- Verify PAT is valid and not expired
- Check PAT has required permissions
- Confirm organization and project names are correct
