"""
Backlog Generation Agent - Specialized agent for generating backlog items from segments
"""

import os
import json
from pathlib import Path
from openai import OpenAI
from strands import tool


def create_backlog_generation_agent(run_id: str):
    """
    Create a backlog generation agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    # Get OpenAI configuration
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    
    openai_client = OpenAI(api_key=api_key) if api_key else None
    
    @tool
    def generate_backlog(segment_data: str) -> str:
        """
        Generate backlog items (epics, features, stories) from a segment with retrieved context.
        
        Args:
            segment_data: JSON string containing:
                - segment_id: The segment identifier
                - segment_text: The original segment text
                - intent_labels: List of intent labels
                - dominant_intent: The dominant intent
                - retrieved_context: Retrieved ADO items and architecture constraints
            
        Returns:
            JSON string containing generated backlog items (epics, features, stories)
        """
        
        try:
            # Parse input
            data = json.loads(segment_data)
            segment_id = data.get("segment_id", 0)
            segment_text = data.get("segment_text", "")
            intent_labels = data.get("intent_labels", [])
            dominant_intent = data.get("dominant_intent", "")
            retrieved_context = data.get("retrieved_context", {})
            
            print(f"Backlog Generation Agent: Processing segment {segment_id} (run_id: {run_id})")
            
            # Build generation prompt
            prompt = _build_generation_prompt(
                segment_text=segment_text,
                intent_labels=intent_labels,
                dominant_intent=dominant_intent,
                ado_items=retrieved_context.get("ado_items", []),
                architecture_constraints=retrieved_context.get("architecture_constraints", [])
            )
            
            # Check if we have OpenAI client
            if not openai_client:
                print("Backlog Generation Agent: Using MOCK mode (missing OPENAI_API_KEY)")
                return _mock_generation(segment_id, segment_text, intent_labels, run_id)
            
            print(f"Backlog Generation Agent: Calling LLM to generate backlog items...")
            
            # Call LLM
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": BACKLOG_GENERATION_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            # Validate structure
            if "backlog_items" not in result:
                raise ValueError("Response missing 'backlog_items' key")
            
            backlog_items = result["backlog_items"]
            
            # Assign internal IDs
            item_counter = {"epic": 1, "feature": 1, "story": 1}
            for item in backlog_items:
                item_type = item.get("type", "story").lower()
                if item_type in item_counter:
                    item["internal_id"] = f"{item_type}_{segment_id}_{item_counter[item_type]}"
                    item_counter[item_type] += 1
                item["segment_id"] = segment_id
                item["run_id"] = run_id
            
            # Ensure output directory exists
            output_dir = Path(f"runs/{run_id}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Append to generated_backlog.jsonl
            backlog_file = output_dir / "generated_backlog.jsonl"
            with open(backlog_file, "a") as f:
                for item in backlog_items:
                    f.write(json.dumps(item) + "\n")
            
            print(f"Backlog Generation Agent: Generated {len(backlog_items)} backlog items")
            
            # Prepare summary
            summary = {
                "status": "success",
                "run_id": run_id,
                "segment_id": segment_id,
                "items_generated": len(backlog_items),
                "backlog_file": str(backlog_file),
                "item_counts": {
                    "epics": sum(1 for item in backlog_items if item.get("type", "").lower() == "epic"),
                    "features": sum(1 for item in backlog_items if item.get("type", "").lower() == "feature"),
                    "stories": sum(1 for item in backlog_items if item.get("type", "").lower() == "story")
                },
                "backlog_items": backlog_items
            }
            
            return json.dumps(summary, indent=2)
            
        except json.JSONDecodeError as e:
            error_msg = {
                "status": "error",
                "error": f"Failed to parse input or LLM response as JSON: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
        
        except Exception as e:
            error_msg = {
                "status": "error",
                "error": f"Backlog generation failed: {str(e)}",
                "run_id": run_id
            }
            return json.dumps(error_msg, indent=2)
    
    return generate_backlog


def _build_generation_prompt(
    segment_text: str,
    intent_labels: list,
    dominant_intent: str,
    ado_items: list,
    architecture_constraints: list
) -> str:
    """Build the generation prompt from segment and context"""
    
    prompt_parts = []
    
    # Section 1: Original segment
    prompt_parts.append("# ORIGINAL SEGMENT TO ANALYZE\n")
    prompt_parts.append(f"**Dominant Intent:** {dominant_intent}\n")
    prompt_parts.append(f"**Intent Labels:** {', '.join(intent_labels)}\n\n")
    prompt_parts.append(f"**Segment Text:**\n{segment_text}\n")
    
    # Section 2: Retrieved ADO items
    prompt_parts.append("\n# RETRIEVED EXISTING ADO BACKLOG ITEMS\n")
    if ado_items:
        for item in ado_items:
            prompt_parts.append(f"\n## {item.get('type', 'Item')} (ID: {item.get('work_item_id', 'N/A')}, Similarity: {item.get('score', 0):.2f})\n")
            prompt_parts.append(f"**Title:** {item.get('title', 'Untitled')}\n")
            prompt_parts.append(f"**Description:** {item.get('description', 'No description')}\n")
    else:
        prompt_parts.append("No relevant existing ADO items found.\n")
    
    # Section 3: Architecture constraints
    prompt_parts.append("\n# RETRIEVED ARCHITECTURE CONSTRAINTS\n")
    if architecture_constraints:
        for constraint in architecture_constraints:
            prompt_parts.append(f"\n## From {constraint.get('source', 'Unknown')} - {constraint.get('section', '')} (Similarity: {constraint.get('score', 0):.2f})\n")
            prompt_parts.append(f"{constraint.get('text', 'No text')}\n")
    else:
        prompt_parts.append("No relevant architecture constraints found.\n")
    
    # Section 4: Instructions
    prompt_parts.append("\n# TASK: GENERATE BACKLOG ITEMS\n")
    prompt_parts.append("""
Based on the segment text above and the retrieved context (existing ADO items and architecture constraints):

1. Generate appropriate backlog items: Epics, Features, and/or User Stories
2. Ensure items are aligned with existing backlog and architecture
3. Create clear, actionable acceptance criteria for each Story
4. Maintain proper hierarchy (Epic → Feature → Story)
5. Reference parent items where appropriate

Return your response as a JSON object with this structure:
```json
{
    "backlog_items": [
        {
            "type": "Epic|Feature|Story",
            "title": "Clear, concise title",
            "description": "Detailed description",
            "acceptance_criteria": ["AC1", "AC2", "AC3"],  // For stories only
            "parent_reference": "Reference to parent epic/feature if applicable",
            "rationale": "Why this item is needed based on segment and context"
        }
    ]
}
```

Guidelines:
- Epic: High-level business objective (use when segment describes major initiative)
- Feature: Specific capability within an epic (use for substantial functionality)
- Story: User-facing deliverable (use for concrete, implementable work)
- Include 3-5 specific, testable acceptance criteria for each Story
- Reference existing ADO items when the new item extends or depends on them
- Respect architecture constraints in your descriptions and acceptance criteria
- Be specific and actionable - avoid vague descriptions
""")
    
    return "".join(prompt_parts)


def _mock_generation(segment_id: int, segment_text: str, intent_labels: list, run_id: str) -> str:
    """Generate mock backlog items for testing"""
    
    # Simple heuristic-based generation
    mock_items = []
    
    # Check intents to decide what to generate
    has_auth = any("auth" in label.lower() for label in intent_labels)
    has_performance = any("performance" in label.lower() or "latency" in label.lower() or "optimize" in label.lower() for label in intent_labels)
    has_offline = any("offline" in label.lower() for label in intent_labels)
    
    if has_auth:
        mock_items.append({
            "type": "Feature",
            "title": "Multi-Factor Authentication Implementation",
            "description": f"Implement multi-factor authentication based on requirements identified in segment analysis. {segment_text[:100]}...",
            "acceptance_criteria": [],
            "parent_reference": "Security & Authentication Improvements Epic",
            "rationale": "Addresses authentication security requirements identified in segment",
            "internal_id": f"feature_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
        mock_items.append({
            "type": "Story",
            "title": "As a user, I want to enable MFA with authenticator app",
            "description": "Allow users to enable multi-factor authentication using TOTP authenticator apps",
            "acceptance_criteria": [
                "User can scan QR code to add account to authenticator app",
                "User must enter verification code to complete MFA setup",
                "User is prompted for MFA code on subsequent logins",
                "User can generate backup codes for account recovery"
            ],
            "parent_reference": "Multi-Factor Authentication Implementation Feature",
            "rationale": "Provides secure MFA option using industry-standard TOTP protocol",
            "internal_id": f"story_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    if has_performance:
        mock_items.append({
            "type": "Story",
            "title": "As a developer, I want optimized database queries with proper indexes",
            "description": "Add database indexes and optimize slow queries identified in performance analysis",
            "acceptance_criteria": [
                "Identify top 10 slowest queries using query analyzer",
                "Add appropriate indexes to relevant tables",
                "Query response time improves by at least 50%",
                "95th percentile API response time is under 200ms"
            ],
            "parent_reference": "API Performance Optimization Feature",
            "rationale": "Addresses performance issues and latency concerns identified in segment",
            "internal_id": f"story_{segment_id}_2",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    if has_offline:
        mock_items.append({
            "type": "Epic",
            "title": "Mobile Offline Mode Support",
            "description": "Enable users to access and work with documents without internet connectivity",
            "acceptance_criteria": [],
            "parent_reference": "",
            "rationale": "Major architectural initiative identified from user research in segment",
            "internal_id": f"epic_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    # Default story if no specific intents matched
    if not mock_items:
        mock_items.append({
            "type": "Story",
            "title": f"Implement requirements from segment {segment_id}",
            "description": f"Address requirements identified in segment: {segment_text[:200]}...",
            "acceptance_criteria": [
                "Requirements are clearly defined",
                "Implementation meets acceptance criteria",
                "Changes are tested and reviewed"
            ],
            "parent_reference": "",
            "rationale": "Generated from segment analysis",
            "internal_id": f"story_{segment_id}_1",
            "segment_id": segment_id,
            "run_id": run_id
        })
    
    # Save to file
    output_dir = Path(f"runs/{run_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    backlog_file = output_dir / "generated_backlog.jsonl"
    
    with open(backlog_file, "a") as f:
        for item in mock_items:
            f.write(json.dumps(item) + "\n")
    
    summary = {
        "status": "success_mock",
        "run_id": run_id,
        "segment_id": segment_id,
        "items_generated": len(mock_items),
        "backlog_file": str(backlog_file),
        "note": "Mock data - set OPENAI_API_KEY for real generation",
        "item_counts": {
            "epics": sum(1 for item in mock_items if item.get("type", "").lower() == "epic"),
            "features": sum(1 for item in mock_items if item.get("type", "").lower() == "feature"),
            "stories": sum(1 for item in mock_items if item.get("type", "").lower() == "story")
        },
        "backlog_items": mock_items
    }
    
    return json.dumps(summary, indent=2)


# System prompt for backlog generation agent
BACKLOG_GENERATION_AGENT_SYSTEM_PROMPT = """You are a backlog synthesis specialist. Your role is to:

1. Analyze segmented document content with retrieved context
2. Generate structured backlog items (Epics, Features, User Stories)
3. Write clear, actionable acceptance criteria
4. Maintain hierarchy and relationships between items

Your inputs include:
- Segmented document text with identified intents
- Retrieved existing ADO backlog items
- Retrieved architecture constraints and requirements

Your output should be:
- Well-structured backlog items with proper hierarchy
- Clear titles and descriptions
- Testable acceptance criteria
- Parent-child relationships (epic → feature → story)

Focus on creating backlog items that are:
- Aligned with existing backlog and architecture
- Properly scoped and actionable
- Complete with all necessary details
- Ready for development team consumption

IMPORTANT: Always return valid JSON with a "backlog_items" array. Each item must have: type, title, description, acceptance_criteria (for stories), parent_reference, and rationale."""
