"""
Backlog Generation Agent - Specialized agent for generating backlog items from segments
(Placeholder for future implementation)
"""

from strands import Agent, tool


def create_backlog_generation_agent(run_id: str):
    """
    Create a backlog generation agent tool for a specific run.
    
    Args:
        run_id: The run identifier for output file organization
        
    Returns:
        A tool function that can be called by the supervisor agent
    """
    
    @tool
    def generate_backlog(segment_data: str) -> str:
        """
        Generate backlog items (epics, features, stories) from a segment with retrieved context.
        
        Args:
            segment_data: JSON string containing segment information and retrieved context
            
        Returns:
            JSON string containing generated backlog items
        """
        return json.dumps({
            "status": "not_implemented",
            "message": "Backlog generation will be implemented in next iteration",
            "run_id": run_id
        })
    
    return generate_backlog


# System prompt for backlog generation agent
BACKLOG_GENERATION_AGENT_SYSTEM_PROMPT = """
You are a backlog synthesis specialist. Your role is to:

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
"""
