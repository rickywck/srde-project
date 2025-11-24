"""
Passthrough Supervisor Agent
Currently passes all messages directly to LLM and returns response
Will be enhanced with agent orchestration in future iterations
"""

import os
from openai import AsyncOpenAI
from typing import Dict, Any, Optional
import yaml

class SupervisorAgent:
    """
    Supervisor agent that orchestrates the backlog synthesis workflow.
    Currently implements passthrough to LLM for POC.
    """
    
    def __init__(self, config_path: str = "config.poc.yaml"):
        """Initialize supervisor with configuration"""
        self.config = self._load_config(config_path)
        
        # Initialize OpenAI client
        api_key = os.getenv(self.config["openai"]["api_key_env_var"])
        if not api_key:
            raise ValueError(f"OpenAI API key not found in environment variable: {self.config['openai']['api_key_env_var']}")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = self.config["openai"]["chat_model"]
        
        # System prompt for the supervisor
        self.system_prompt = """You are an AI assistant helping to process meeting notes and generate software backlog items.

Your role is to:
- Help users understand their uploaded documents
- Answer questions about the content
- Guide them through the backlog generation process
- Explain what actions are available

Available capabilities (will be implemented in future iterations):
- Segment documents into coherent chunks
- Generate epics, features, and user stories
- Tag stories relative to existing backlog (new/gap/conflict)
- Write items to Azure DevOps

For now, provide helpful guidance and context based on the user's questions and the document content."""
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            # Return default config if file doesn't exist
            return {
                "openai": {
                    "api_key_env_var": "OPENAI_API_KEY",
                    "chat_model": "gpt-4o"
                }
            }
        
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    async def process_message(
        self,
        run_id: str,
        message: str,
        instruction_type: Optional[str] = None,
        document_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and return response.
        Currently passes through to LLM with context.
        
        Args:
            run_id: Unique identifier for this run
            message: User's message
            instruction_type: Optional type hint for the instruction
            document_text: Optional document content for context
        
        Returns:
            Dict with response and status information
        """
        
        # Build context-aware prompt
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add document context if available
        if document_text:
            context_msg = f"""The user has uploaded a document. Here's the content:

--- DOCUMENT START ---
{document_text[:3000]}{"..." if len(document_text) > 3000 else ""}
--- DOCUMENT END ---

Please help the user with their request about this document."""
            messages.append({"role": "system", "content": context_msg})
        
        # Add instruction type hint if provided
        if instruction_type:
            messages.append({
                "role": "system",
                "content": f"The user's request is categorized as: {instruction_type}"
            })
        
        # Add user message
        messages.append({"role": "user", "content": message})
        
        try:
            # Call LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            assistant_message = response.choices[0].message.content
            
            return {
                "response": assistant_message,
                "status": {
                    "run_id": run_id,
                    "model": self.model,
                    "tokens_used": response.usage.total_tokens,
                    "has_document": document_text is not None,
                    "mode": "passthrough"
                }
            }
        
        except Exception as e:
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "status": {
                    "run_id": run_id,
                    "error": str(e),
                    "mode": "passthrough"
                }
            }
    
    async def segment_document(self, run_id: str, document_text: str) -> Dict[str, Any]:
        """
        Segment document into coherent chunks with intent labels.
        To be implemented in future iteration.
        """
        return {
            "status": "not_implemented",
            "message": "Document segmentation will be implemented in next iteration"
        }
    
    async def generate_backlog(self, run_id: str, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate backlog items from a segment with retrieved context.
        To be implemented in future iteration.
        """
        return {
            "status": "not_implemented",
            "message": "Backlog generation will be implemented in next iteration"
        }
    
    async def tag_story(self, run_id: str, story: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tag a user story relative to existing backlog.
        To be implemented in future iteration.
        """
        return {
            "status": "not_implemented",
            "message": "Story tagging will be implemented in next iteration"
        }
