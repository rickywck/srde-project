"""
Tagging Agent - Specialized agent for tagging generated stories relative to existing backlog
(Placeholder for future implementation)
"""

import json
from strands import Agent, tool


def create_tagging_agent(run_id: str):
    """
    Create a tagging agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    @tool
    def tag_story(story_data: str) -> str:
        """
        Tag a user story relative to existing backlog (new/gap/conflict).
        
        Args:
            story_data: JSON string containing story information and similar existing stories
            
        Returns:
            JSON string containing tagging decision and reasoning
        """
        return json.dumps({
            "status": "not_implemented",
            "message": "Story tagging will be implemented in next iteration",
            "run_id": run_id
        })
    
    return tag_story


# System prompt for tagging agent
TAGGING_AGENT_SYSTEM_PROMPT = """
You are a backlog analysis specialist. Your role is to:

1. Analyze generated user stories against existing backlog
2. Identify relationships and overlaps with existing work items
3. Classify stories into categories: new, gap, conflict, or extend
4. Provide clear reasoning for tagging decisions

Your inputs include:
- Generated user story (title, description, acceptance criteria)
- Retrieved similar existing stories from ADO backlog
- Similarity scores and context

Your classification criteria:
- NEW: Story addresses completely new functionality with no similar existing items
- GAP: Story fills a gap or extends existing work in a complementary way
- CONFLICT: Story contradicts or duplicates existing work items

Focus on:
- Accurate similarity assessment
- Clear justification for decisions
- Identification of related work items
- Actionable recommendations for stakeholders
"""
