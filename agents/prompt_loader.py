"""
Centralized prompt management for all agents.
Loads prompts from YAML files and provides formatting utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


class PromptLoader:
    """Loads and manages prompts from YAML configuration files."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize the prompt loader.
        
        Args:
            prompts_dir: Directory containing prompt YAML files
        """
        self.prompts_dir = Path(prompts_dir)
        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {prompts_dir}")
    
    @lru_cache(maxsize=32)
    def load_prompt(self, agent_name: str) -> Dict[str, Any]:
        """
        Load prompt configuration for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., 'segmentation_agent')
            
        Returns:
            Dictionary containing prompt configuration
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
            ValueError: If prompt file is invalid
        """
        prompt_file = self.prompts_dir / f"{agent_name}.yaml"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        with open(prompt_file, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ["name", "system_prompt"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Prompt file missing required field '{field}': {prompt_file}")
        
        return config
    
    def get_system_prompt(self, agent_name: str) -> str:
        """Get the system prompt for an agent."""
        config = self.load_prompt(agent_name)
        return config["system_prompt"].strip()
    
    def get_user_prompt_template(self, agent_name: str) -> str:
        """Get the user prompt template for an agent."""
        config = self.load_prompt(agent_name)
        return config.get("user_prompt_template", "").strip()
    
    def get_parameters(self, agent_name: str) -> Dict[str, Any]:
        """Get model parameters for an agent."""
        config = self.load_prompt(agent_name)
        return config.get("parameters", {})
    
    def format_user_prompt(self, agent_name: str, **kwargs) -> str:
        """
        Format the user prompt template with provided variables.
        
        Args:
            agent_name: Name of the agent
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        template = self.get_user_prompt_template(agent_name)
        if not template:
            return ""
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required template variable: {e}")
    
    def get_metadata(self, agent_name: str) -> Dict[str, Any]:
        """Get metadata about the agent prompt."""
        config = self.load_prompt(agent_name)
        return {
            "name": config.get("name"),
            "description": config.get("description"),
            "version": config.get("version", "1.0")
        }


# Global prompt loader instance
_prompt_loader: Optional[PromptLoader] = None


def get_prompt_loader() -> PromptLoader:
    """Get the global prompt loader instance (singleton pattern)."""
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader()
    return _prompt_loader


def load_agent_prompt(agent_name: str) -> Dict[str, Any]:
    """
    Convenience function to load a complete agent prompt configuration.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Dictionary with system_prompt, user_prompt_template, parameters, and metadata
    """
    loader = get_prompt_loader()
    config = loader.load_prompt(agent_name)
    
    return {
        "system_prompt": loader.get_system_prompt(agent_name),
        "user_prompt_template": loader.get_user_prompt_template(agent_name),
        "parameters": loader.get_parameters(agent_name),
        "metadata": loader.get_metadata(agent_name)
    }
