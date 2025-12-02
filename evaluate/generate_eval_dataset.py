#!/usr/bin/env python3
"""Generate evaluation dataset for tagging (gap / new / duplicate / conflict)
using existing Azure DevOps User Stories as context.

Output format: JSONL (one JSON object per line), each object:
{
    "story_title": str,
    "story_description": str,
    "story_acceptance_criteria": [str, ...],
    "existing_stories": [{"title": str, "description": str, "acceptance_criteria": [str, ...]}],
    "gold_tag": "gap"|"new"|"duplicate"|"conflict",
    "gold_related_ids": [title strings]
}

Tag heuristics (deterministic, no LLM calls):
 - gap: enhancement / extension of ONE existing story
 - duplicate: already covered by the existing story with little to no gap
 - conflict: replacement / incompatible change of ONE existing story
 - new: unrelated new capability (no related IDs)

Evaluation quality requirement: ONLY ONE existing backlog story is referenced
per synthesized example. Post-generation sanitation enforces this invariant and
corrects gold_related_ids if necessary. Sample size is capped at 20.
"""

import os
import sys
import json
import random
import argparse
import re
from typing import List, Dict, Any
import requests
import yaml
from dotenv import load_dotenv

# =============================
# Prompt/Template Constants
# =============================

# Gap templates
GAP_TITLE_TEMPLATE = "Enhance {base_title} with extended capability"
GAP_DESC_TEMPLATE = (
    "Extend '{base_title}' to cover additional scenarios and edge conditions, "
    "improving robustness and observability in subtler ways than currently described."
)
GAP_ADDED_REQUIREMENTS = [
    "Handles edge case input variations",
    "Emits additional audit events for troubleshooting",
    "Meets a modest performance target under load",
]

# Duplicate templates (minimal or no gap)
DUPLICATE_TITLE_TEMPLATE = "Re-state {base_title} requirement"
DUPLICATE_DESC_TEMPLATE = (
    "Restate requirements closely aligned with '{base_title}', using slightly different phrasing "
    "or examples but preserving the original scope and outcomes."
)
DUPLICATE_NOTES = [
    "Clarifies existing acceptance details",
    "Provides an alternate phrasing without added scope",
]

# Conflict templates
CONFLICT_TITLE_TEMPLATE = "Replace {base_title} with revised approach"
CONFLICT_DESC_TEMPLATE = (
    "Propose an alternative approach to '{base_title}' that changes the expected workflow or "
    "behaviour in ways that are not directly compatible with the existing item."
)
CONFLICT_ACCEPTANCE = [
    "Defines alternative workflow and expected outcomes",
    "Specifies behaviour differences compared to current approach",
    "Includes notes on transition considerations",
]

# New templates
NEW_NOUNS = [
    "analytics dashboard",
    "observability hub",
    "bulk export tool",
    "AI assistant",
    "usage heatmap",
]
NEW_ACCEPTANCE = [
    "Accessible via navigation",
    "Data loads within 2s",
    "Basic role-based access enforced",
]


def encode_pat(pat: str) -> str:
    import base64
    token = f":{pat}"
    return base64.b64encode(token.encode()).decode()


def strip_html(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text or '')


def fetch_user_stories(organization: str, project: str, pat: str) -> List[Dict[str, Any]]:
    """Fetch all User Story work items for the project."""
    base_url = f"https://dev.azure.com/{organization}/{project}/_apis"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {encode_pat(pat)}"
    }
    # WIQL for User Stories
    wiql_query = {
        "query": (
            "SELECT [System.Id] FROM WorkItems "
            f"WHERE [System.WorkItemType] = 'User Story' AND [System.TeamProject] = '{project}'"
        )
    }
    wiql_url = f"{base_url}/wit/wiql?api-version=7.0"
    resp = requests.post(wiql_url, headers=headers, json=wiql_query)
    resp.raise_for_status()
    refs = resp.json().get("workItems", [])
    ids = [str(r["id"]) for r in refs]
    if not ids:
        return []
    stories = []
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        details_url = (
            f"{base_url}/wit/workitems?ids={','.join(batch_ids)}&$expand=All&api-version=7.0"
        )
        dresp = requests.get(details_url, headers=headers)
        dresp.raise_for_status()
        stories.extend(dresp.json().get("value", []))
    return stories


def extract_story_fields(work_item: Dict[str, Any]) -> Dict[str, Any]:
    fields = work_item.get("fields", {})
    title = fields.get("System.Title", "")
    description_html = fields.get("System.Description", "")
    acceptance_html = fields.get("Microsoft.VSTS.Common.AcceptanceCriteria", "")
    description = strip_html(description_html).strip()
    acceptance_clean = strip_html(acceptance_html).strip()
    # Split acceptance criteria into lines, filter blanks
    acceptance_list = [c.strip() for c in re.split(r'[\n\r]+', acceptance_clean) if c.strip()]
    return {
        "title": title,
        "description": description,
        "acceptance_criteria": acceptance_list or []
    }


def synthesize_candidate(existing_list: List[Dict[str, Any]], tag: str, rng: random.Random) -> Dict[str, Any]:
    # existing_list MUST contain exactly one story; we still accept >1 and will
    # sanitize later to be defensive.
    base = existing_list[0]
    if tag == "gap":
        story_title = GAP_TITLE_TEMPLATE.format(base_title=base['title'])
        story_description = GAP_DESC_TEMPLATE.format(base_title=base['title'])
        story_acceptance = base["acceptance_criteria"][:2] + rng.sample(GAP_ADDED_REQUIREMENTS, k=min(2, len(GAP_ADDED_REQUIREMENTS)))
        gold_related_ids = [base['title']]
    elif tag == "duplicate":
        story_title = DUPLICATE_TITLE_TEMPLATE.format(base_title=base['title'])
        story_description = DUPLICATE_DESC_TEMPLATE.format(base_title=base['title'])
        ac_base = base.get("acceptance_criteria", [])
        extra_note = rng.choice(DUPLICATE_NOTES) if DUPLICATE_NOTES else "Clarify existing behavior only"
        story_acceptance = (ac_base[:3] if isinstance(ac_base, list) else []) + [extra_note]
        gold_related_ids = [base['title']]
    elif tag == "conflict":
        story_title = CONFLICT_TITLE_TEMPLATE.format(base_title=base['title'])
        story_description = CONFLICT_DESC_TEMPLATE.format(base_title=base['title'])
        story_acceptance = CONFLICT_ACCEPTANCE
        gold_related_ids = [base['title']]
    else:  # new
        chosen = rng.choice(NEW_NOUNS)
        story_title = f"Implement {chosen}"
        story_description = (
            f"Provide a {chosen} offering new capability unrelated to current backlog items."
            " Users can access it from main navigation."
        )
        story_acceptance = NEW_ACCEPTANCE
        gold_related_ids = []
    return {
        "story_title": story_title,
        "story_description": story_description,
        "story_acceptance_criteria": story_acceptance,
        "existing_stories": existing_list[:1],  # enforce single-story here
        "gold_tag": tag,
        "gold_related_ids": gold_related_ids
    }


def build_dataset(stories: List[Dict[str, Any]], sample_size: int, proportion_gap: float, proportion_duplicate: float, proportion_conflict: float, seed: int) -> List[Dict[str, Any]]:
    """Build dataset ensuring exactly one existing story per synthesized entry."""
    rng = random.Random(seed)
    simplified = [extract_story_fields(w) for w in stories]
    if len(simplified) < 1:
        print("Not enough stories fetched to build dataset (need >=1).")
        return []
    rng.shuffle(simplified)
    needed = sample_size
    gap_target = int(needed * proportion_gap)
    duplicate_target = int(needed * proportion_duplicate)
    conflict_target = int(needed * proportion_conflict)
    new_target = needed - gap_target - duplicate_target - conflict_target
    dataset: List[Dict[str, Any]] = []
    idx = 0
    def pick_single(i: int) -> List[Dict[str, Any]]:
        return [simplified[i % len(simplified)]]
    for _ in range(gap_target):
        dataset.append(synthesize_candidate(pick_single(idx), "gap", rng)); idx += 1
    for _ in range(duplicate_target):
        dataset.append(synthesize_candidate(pick_single(idx), "duplicate", rng)); idx += 1
    for _ in range(conflict_target):
        dataset.append(synthesize_candidate(pick_single(idx), "conflict", rng)); idx += 1
    for _ in range(new_target):
        dataset.append(synthesize_candidate(pick_single(idx), "new", rng)); idx += 1
    rng.shuffle(dataset)
    # Post-generation sanitation (defensive): enforce single story and fix related IDs
    for entry in dataset:
        if entry.get("existing_stories"):
            entry["existing_stories"] = entry["existing_stories"][:1]
        if entry.get("gold_tag") in ("gap", "duplicate", "conflict"):
            if entry["existing_stories"]:
                entry["gold_related_ids"] = [entry["existing_stories"][0]["title"]]
            else:
                entry["gold_related_ids"] = []
        else:
            entry["gold_related_ids"] = []
    return dataset


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation tagging dataset from ADO User Stories (single-story context)")
    parser.add_argument("--config", default="config.poc.yaml", help="Path to config file")
    parser.add_argument("--organization", help="Override ADO organization")
    parser.add_argument("--project", help="Override ADO project")
    parser.add_argument("--output", default="eval/datasets/eval_dataset_generated.jsonl", help="Output JSONL file path (one JSON object per line)")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of evaluation entries to generate (max 20)")
    parser.add_argument("--gap", type=float, default=0.25, help="Proportion of gap examples")
    parser.add_argument("--duplicate", type=float, default=0.25, help="Proportion of duplicate examples")
    parser.add_argument("--conflict", type=float, default=0.25, help="Proportion of conflict examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    MAX_SAMPLE = 20
    if args.sample_size > MAX_SAMPLE:
        print(f"Requested sample size {args.sample_size} exceeds max {MAX_SAMPLE}; capping to {MAX_SAMPLE}.")
        args.sample_size = MAX_SAMPLE

    if args.gap + args.duplicate + args.conflict > 0.95:
        print("Invalid proportions: gap + duplicate + conflict must be <= 0.95")
        sys.exit(1)

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    organization = args.organization or config['ado']['organization']
    project = args.project or config['ado']['project']
    pat_env_var = config['ado']['pat_env_var']

    load_dotenv()
    pat = os.getenv(pat_env_var)
    if not pat:
        print(f"Error: ADO PAT environment variable '{pat_env_var}' not set")
        sys.exit(1)

    print(f"Fetching User Stories from ADO {organization}/{project} using config {args.config} ...")
    try:
        user_stories = fetch_user_stories(organization, project, pat)
    except requests.HTTPError as e:
        print(f"HTTP error fetching stories: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print(f"Fetched {len(user_stories)} user stories.")
    dataset = build_dataset(user_stories, args.sample_size, args.gap, args.duplicate, args.conflict, args.seed)
    if not dataset:
        print("No dataset generated (insufficient stories).")
        sys.exit(1)

    # Defensive re-check (in case future changes regress single-story constraint)
    for entry in dataset:
        if entry.get("existing_stories"):
            entry["existing_stories"] = entry["existing_stories"][:1]
        if entry.get("gold_tag") in ("gap", "duplicate", "conflict") and entry["existing_stories"]:
            entry["gold_related_ids"] = [entry["existing_stories"][0]["title"]]
        elif entry.get("gold_tag") == "new":
            entry["gold_related_ids"] = []

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        for obj in dataset:
            f.write(json.dumps(obj) + "\n")
    print(f"Wrote JSONL evaluation dataset with {len(dataset)} entries to {args.output}")


if __name__ == "__main__":
    main()
