"""
ADO Writer Tool - Creates ADO work items from generated backlog

Functionality:
- Reads runs/{run_id}/generated_backlog.jsonl and tagging.jsonl
- Filters stories by tags (default: new, gap)
- Prepares a creation plan (epics, features, stories)
- Dry-run mode (default): returns plan and counts without calling ADO
- Write mode: creates work items in ADO using REST API with proper parent-child relationships

Configuration:
- Loads from config.poc.yaml: organization, project, pat_env_var
- Loads PAT from .env file using the configured pat_env_var name
- Default PAT environment variable: ADO_PAT

ADO REST API:
- Creates Epics first (no parents)
- Creates Features with Epic parents
- Creates User Stories with Feature parents
- Uses JSON Patch format for work item creation
- Establishes parent-child relationships via System.LinkTypes.Hierarchy-Reverse

Work Item Mapping:
- Epic: title, description (includes rationale)
- Feature: title, description (includes rationale), parent epic link
- User Story: title, description (includes rationale), acceptance criteria, parent feature link
"""

import os
import json
import yaml
import base64
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from strands import tool
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _encode_pat(pat: str) -> str:
    """Encode PAT for Basic authentication."""
    token = f":{pat}"
    return base64.b64encode(token.encode()).decode()


def _create_work_item(
    organization: str,
    project: str,
    pat: str,
    work_item_type: str,
    title: str,
    description: str = "",
    acceptance_criteria: str = "",
    parent_id: Optional[int] = None,
    additional_fields: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a work item in Azure DevOps using REST API.
    
    Args:
        organization: ADO organization name
        project: ADO project name
        pat: Personal Access Token
        work_item_type: Type of work item (Epic, Feature, User Story)
        title: Work item title
        description: Work item description
        acceptance_criteria: Acceptance criteria (for stories)
        parent_id: ID of parent work item
        additional_fields: Additional fields to set
    
    Returns:
        Created work item response dict with id, url, etc.
    """
    base_url = f"https://dev.azure.com/{organization}/{project}/_apis"
    headers = {
        "Content-Type": "application/json-patch+json",
        "Authorization": f"Basic {_encode_pat(pat)}"
    }
    
    # Build JSON Patch document for work item creation
    patch_document = [
        {
            "op": "add",
            "path": "/fields/System.Title",
            "value": title
        }
    ]
    
    # Add description if provided
    if description:
        patch_document.append({
            "op": "add",
            "path": "/fields/System.Description",
            "value": description
        })
    
    # Add acceptance criteria if provided (typically for stories)
    if acceptance_criteria:
        patch_document.append({
            "op": "add",
            "path": "/fields/Microsoft.VSTS.Common.AcceptanceCriteria",
            "value": acceptance_criteria
        })
    
    # Add parent relationship if specified
    if parent_id:
        patch_document.append({
            "op": "add",
            "path": "/relations/-",
            "value": {
                "rel": "System.LinkTypes.Hierarchy-Reverse",
                "url": f"https://dev.azure.com/{organization}/{project}/_apis/wit/workitems/{parent_id}",
                "attributes": {
                    "comment": "Parent work item"
                }
            }
        })
    
    # Add any additional fields
    if additional_fields:
        for field_path, field_value in additional_fields.items():
            patch_document.append({
                "op": "add",
                "path": f"/fields/{field_path}",
                "value": field_value
            })
    
    # Create work item
    url = f"{base_url}/wit/workitems/${work_item_type}?api-version=7.0"
    response = requests.post(url, headers=headers, json=patch_document)
    response.raise_for_status()
    
    return response.json()


def _write_to_ado(
    run_id: str,
    plan: Dict[str, List[Dict[str, Any]]],
    summary: Dict[str, Any],
    ado_org: str,
    ado_project: str,
    ado_pat: str
) -> str:
    """
    Execute the ADO write plan: create epics, features, and stories in order.
    
    Args:
        run_id: Run identifier
        plan: Dictionary with epics, features, stories lists
        summary: Summary information for response
        ado_org: ADO organization
        ado_project: ADO project
        ado_pat: Personal Access Token
    
    Returns:
        JSON string with creation results
    """
    created_items = []
    errors = []
    
    # Track created item IDs for parent-child linking
    # Map internal_id or title -> ADO work item ID
    id_map: Dict[str, int] = {}
    
    try:
        # Step 1: Create Epics
        for epic in plan["epics"]:
            try:
                title = epic.get("title", "Untitled Epic")
                description = epic.get("description", "")
                rationale = epic.get("rationale", "")
                
                # Combine description and rationale
                full_description = description
                if rationale:
                    full_description += f"\n\nRationale: {rationale}"
                
                result = _create_work_item(
                    organization=ado_org,
                    project=ado_project,
                    pat=ado_pat,
                    work_item_type="Epic",
                    title=title,
                    description=full_description
                )
                
                ado_id = result["id"]
                internal_id = epic.get("internal_id") or epic.get("id")
                if internal_id:
                    id_map[str(internal_id)] = ado_id
                # Also map by title for parent_reference lookup
                id_map[title] = ado_id
                
                created_items.append({
                    "type": "Epic",
                    "internal_id": internal_id,
                    "ado_id": ado_id,
                    "title": title,
                    "url": result.get("_links", {}).get("html", {}).get("href", "")
                })
                
            except Exception as e:
                errors.append(f"Failed to create Epic '{epic.get('title', 'Unknown')}': {str(e)}")
        
        # Step 2: Create Features
        for feature in plan["features"]:
            try:
                title = feature.get("title", "Untitled Feature")
                description = feature.get("description", "")
                rationale = feature.get("rationale", "")
                
                # Combine description and rationale
                full_description = description
                if rationale:
                    full_description += f"\n\nRationale: {rationale}"
                
                # Find parent epic
                parent_id = None
                parent_ref = feature.get("parent_reference")
                if parent_ref and parent_ref in id_map:
                    parent_id = id_map[parent_ref]
                else:
                    # Try direct ID lookup
                    epic_id = feature.get("epic_id") or feature.get("parent_epic_id")
                    if epic_id and str(epic_id) in id_map:
                        parent_id = id_map[str(epic_id)]
                
                result = _create_work_item(
                    organization=ado_org,
                    project=ado_project,
                    pat=ado_pat,
                    work_item_type="Feature",
                    title=title,
                    description=full_description,
                    parent_id=parent_id
                )
                
                ado_id = result["id"]
                internal_id = feature.get("internal_id") or feature.get("id")
                if internal_id:
                    id_map[str(internal_id)] = ado_id
                # Also map by title
                id_map[title] = ado_id
                
                created_items.append({
                    "type": "Feature",
                    "internal_id": internal_id,
                    "ado_id": ado_id,
                    "title": title,
                    "parent_ado_id": parent_id,
                    "url": result.get("_links", {}).get("html", {}).get("href", "")
                })
                
            except Exception as e:
                errors.append(f"Failed to create Feature '{feature.get('title', 'Unknown')}': {str(e)}")
        
        # Step 3: Create Stories
        for story in plan["stories"]:
            try:
                title = story.get("title", "Untitled Story")
                description = story.get("description", "")
                rationale = story.get("rationale", "")
                acceptance_criteria_list = story.get("acceptance_criteria", [])
                
                # Combine description and rationale
                full_description = description
                if rationale:
                    full_description += f"\n\nRationale: {rationale}"
                
                # Format acceptance criteria
                acceptance_criteria = ""
                if acceptance_criteria_list:
                    if isinstance(acceptance_criteria_list, list):
                        acceptance_criteria = "\n".join([f"- {ac}" for ac in acceptance_criteria_list])
                    else:
                        acceptance_criteria = str(acceptance_criteria_list)
                
                # Find parent feature
                parent_id = None
                parent_ref = story.get("parent_reference")
                if parent_ref and parent_ref in id_map:
                    parent_id = id_map[parent_ref]
                else:
                    # Try direct ID lookup
                    feature_id = story.get("feature_id") or story.get("parent_feature_id")
                    if feature_id and str(feature_id) in id_map:
                        parent_id = id_map[str(feature_id)]
                
                result = _create_work_item(
                    organization=ado_org,
                    project=ado_project,
                    pat=ado_pat,
                    work_item_type="User Story",
                    title=title,
                    description=full_description,
                    acceptance_criteria=acceptance_criteria,
                    parent_id=parent_id
                )
                
                ado_id = result["id"]
                internal_id = story.get("internal_id") or story.get("id")
                
                created_items.append({
                    "type": "User Story",
                    "internal_id": internal_id,
                    "ado_id": ado_id,
                    "title": title,
                    "parent_ado_id": parent_id,
                    "url": result.get("_links", {}).get("html", {}).get("href", "")
                })
                
            except Exception as e:
                errors.append(f"Failed to create Story '{story.get('title', 'Unknown')}': {str(e)}")
        
    except Exception as e:
        errors.append(f"Critical error during ADO write: {str(e)}")
    
    # Return results
    status = "ok" if not errors else "partial" if created_items else "error"
    
    return json.dumps({
        "status": status,
        "mode": "write",
        "run_id": run_id,
        "summary": summary,
        "created_items": created_items,
        "errors": errors,
        "counts": {
            "epics_created": len([i for i in created_items if i["type"] == "Epic"]),
            "features_created": len([i for i in created_items if i["type"] == "Feature"]),
            "stories_created": len([i for i in created_items if i["type"] == "User Story"])
        }
    }, indent=2)


def create_ado_writer_tool(run_id: str):
    """
    Create an ADO writer tool bound to a specific run.

    The tool accepts a JSON string with fields:
    - run_id: optional (defaults to the bound run_id)
    - filter_tags: list[str] (default ["new", "gap"]) applied to Story tags
    - dry_run: bool (default False)
    """

    # Load configuration (reuse retrieval style)
    config_path = "config.poc.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "ado": {
                "organization": os.getenv("ADO_ORG", "unknown-org"),
                "project": os.getenv("ADO_PROJECT", "unknown-project"),
                "pat_env_var": "ADO_PAT"
            }
        }

    ado_cfg = config.get("ado", {})
    ado_pat_env_var = ado_cfg.get("pat_env_var", "ADO_PAT")
    ado_org = ado_cfg.get("organization") or os.getenv("ADO_ORG")
    ado_project = ado_cfg.get("project") or os.getenv("ADO_PROJECT")

    @tool
    def write_to_ado(params_json: str) -> str:
        """
        Writes the generated backlog items to Azure DevOps (ADO).
        
        Args:
            params_json: A JSON string containing:
                - run_id (str, optional): The run ID to export. Defaults to current run.
                - filter_tags (List[str], optional): Tags to filter stories by (e.g., ["new", "gap"]).
                - dry_run (bool, optional): If True, returns a plan without writing to ADO. Defaults to False.
        
        Returns:
            JSON string with the result of the operation.
        """
        try:
            params = json.loads(params_json or "{}")
        except json.JSONDecodeError as e:
            return json.dumps({
                "status": "error",
                "error": f"Invalid JSON for tool params: {e}",
                "run_id": run_id
            }, indent=2)

        effective_run_id: str = params.get("run_id") or run_id
        filter_tags: List[str] = params.get("filter_tags") or ["new", "gap"]
        dry_run: bool = params.get("dry_run", False)

        run_dir = Path("runs") / effective_run_id
        backlog_path = run_dir / "generated_backlog.jsonl"
        tagging_path = run_dir / "tagging.jsonl"

        if not backlog_path.exists():
            return json.dumps({
                "status": "error",
                "error": "No generated backlog found for this run",
                "run_id": effective_run_id
            }, indent=2)

        # Load backlog items
        backlog_items: List[Dict[str, Any]] = []
        with open(backlog_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    backlog_items.append(json.loads(line))
                except Exception:
                    # Skip malformed lines
                    continue

        # Load tagging decisions (map by any available synthetic id or title fallback)
        story_tag_map: Dict[str, str] = {}
        if tagging_path.exists():
            with open(tagging_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    # Use story_internal_id as primary key for matching with backlog items
                    key = rec.get("story_internal_id") or rec.get("story_id") or rec.get("id") or rec.get("title")
                    if key:
                        story_tag_map[str(key)] = rec.get("decision_tag")

        # Index backlog items by possible keys to help parent-child resolution
        by_internal_ids = {}
        for it in backlog_items:
            # Record by known synthetic ids if present (support both internal_id and specific IDs)
            internal_id = it.get("internal_id")
            if internal_id:
                by_internal_ids.setdefault("internal_id", {}).setdefault(str(internal_id), it)
            for k in ("id", "story_id", "feature_id", "epic_id"):
                if k in it:
                    by_internal_ids.setdefault(k, {}).setdefault(str(it[k]), it)

        # Filter stories by tags; include their parent feature/epic if resolvable
        def story_passes(item: Dict[str, Any]) -> bool:
            # Use explicit item-assigned tag if provided, else consult tagging map
            assigned = item.get("assigned_tag")
            if not assigned:
                # Try to derive key for tagging lookup (prioritize internal_id)
                key = item.get("internal_id") or item.get("story_id") or item.get("id") or item.get("title")
                assigned = story_tag_map.get(str(key))
            return (assigned in filter_tags) if assigned else False

        # Build creation plan
        epics: Dict[str, Dict[str, Any]] = {}
        features: Dict[str, Dict[str, Any]] = {}
        stories: List[Dict[str, Any]] = []

        for item in backlog_items:
            item_type = (item.get("type") or item.get("work_item_type") or "").lower()
            if item_type == "story" or item_type == "user story":
                if story_passes(item):
                    stories.append(item)
                    # Attempt to capture parents using parent_reference (title) or IDs
                    parent_ref = item.get("parent_reference")
                    if parent_ref:
                        # Find parent feature by title match
                        for feat_item in backlog_items:
                            if feat_item.get("title") == parent_ref and feat_item.get("type", "").lower() == "feature":
                                feat_id = feat_item.get("internal_id") or feat_item.get("feature_id") or feat_item.get("id")
                                if feat_id:
                                    features[str(feat_id)] = feat_item
                                # Find epic parent of this feature
                                epic_ref = feat_item.get("parent_reference")
                                if epic_ref:
                                    for epic_item in backlog_items:
                                        if epic_item.get("title") == epic_ref and epic_item.get("type", "").lower() == "epic":
                                            epic_id = epic_item.get("internal_id") or epic_item.get("epic_id") or epic_item.get("id")
                                            if epic_id:
                                                epics[str(epic_id)] = epic_item
                                break
                    # Also try direct ID-based parent lookup
                    feat_key = item.get("feature_id") or item.get("parent_feature_id")
                    epic_key = item.get("epic_id") or item.get("parent_epic_id")
                    if feat_key:
                        feat = by_internal_ids.get("internal_id", {}).get(str(feat_key)) or by_internal_ids.get("feature_id", {}).get(str(feat_key)) or by_internal_ids.get("id", {}).get(str(feat_key))
                        if feat:
                            features[str(feat_key)] = feat
                    if epic_key:
                        epi = by_internal_ids.get("internal_id", {}).get(str(epic_key)) or by_internal_ids.get("epic_id", {}).get(str(epic_key)) or by_internal_ids.get("id", {}).get(str(epic_key))
                        if epi:
                            epics[str(epic_key)] = epi
            elif item_type == "feature":
                # Keep as potential parent if any of its children selected later
                key = item.get("internal_id") or item.get("feature_id") or item.get("id")
                if key:
                    features.setdefault(str(key), item)
            elif item_type == "epic":
                key = item.get("internal_id") or item.get("epic_id") or item.get("id")
                if key:
                    epics.setdefault(str(key), item)

        plan = {
            "epics": list(epics.values()),
            "features": list(features.values()),
            "stories": stories,
        }

        # Summaries (include ADO config for transparency)
        summary = {
            "counts": {
                "epics": len(plan["epics"]),
                "features": len(plan["features"]),
                "stories": len(plan["stories"]),
            },
            "filter_tags": filter_tags,
            "dry_run": dry_run,
            "ado_config": {
                "organization": ado_org,
                "project": ado_project,
                "pat_env_var": ado_pat_env_var,
                "pat_present": bool(os.getenv(ado_pat_env_var))
            }
        }

        # If dry-run, return the plan overview only
        if dry_run:
            return json.dumps({
                "status": "ok",
                "mode": "dry_run",
                "run_id": effective_run_id,
                "summary": summary,
                "created_items": [],
                "errors": [],
            }, indent=2)

        # Non-dry run: validate env
        ado_pat = os.getenv(ado_pat_env_var)
        if not (ado_pat and ado_org and ado_project):
            return json.dumps({
                "status": "error",
                "run_id": effective_run_id,
                "summary": summary,
                "created_items": [],
                "errors": [
                    f"Missing ADO configuration. Ensure config.poc.yaml has ado section or set env vars {ado_pat_env_var}, ADO_ORG, ADO_PROJECT."
                ],
            }, indent=2)

        # Perform actual ADO writes using REST API
        return _write_to_ado(
            run_id=effective_run_id,
            plan=plan,
            summary=summary,
            ado_org=ado_org,
            ado_project=ado_project,
            ado_pat=ado_pat
        )

    return write_to_ado
