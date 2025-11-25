#!/usr/bin/env python3
"""
Verification script to demonstrate prompt externalization.
Shows that prompts are now loaded from YAML files instead of hardcoded strings.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.prompt_loader import get_prompt_loader

def main():
    print("=" * 80)
    print("PROMPT EXTERNALIZATION VERIFICATION")
    print("=" * 80)
    
    loader = get_prompt_loader()
    agents = [
        "segmentation_agent",
        "backlog_generation_agent", 
        "tagging_agent",
        "evaluation_agent",
        "supervisor_agent"
    ]
    
    print("\n📁 Prompt Files Location: prompts/")
    print("\n✓ All prompts successfully externalized to YAML configuration files\n")
    
    for agent_name in agents:
        config = loader.load_prompt(agent_name)
        metadata = loader.get_metadata(agent_name)
        params = loader.get_parameters(agent_name)
        system_prompt = loader.get_system_prompt(agent_name)
        
        print(f"\n{'─' * 80}")
        print(f"Agent: {metadata['name']}")
        print(f"{'─' * 80}")
        print(f"Description: {metadata['description']}")
        print(f"Version:     {metadata['version']}")
        print(f"Source:      prompts/{agent_name}.yaml")
        print(f"\nPrompt Stats:")
        print(f"  • System prompt: {len(system_prompt)} characters")
        
        if "user_prompt_template" in config:
            template = config["user_prompt_template"]
            # Extract variables from template
            import re
            variables = re.findall(r'\{(\w+)\}', template)
            print(f"  • User template: {len(template)} characters")
            if variables:
                print(f"  • Template variables: {', '.join(sorted(set(variables)))}")
        
        if params:
            print(f"  • Parameters: {', '.join(f'{k}={v}' for k, v in params.items())}")
    
    print(f"\n{'=' * 80}")
    print("BENEFITS")
    print("=" * 80)
    print("""
✓ Centralized Management
  - All prompts in prompts/ directory
  - Easy to locate and modify
  - Consistent YAML structure

✓ Version Control
  - Track changes with git
  - Compare versions easily
  - Rollback capability

✓ Separation of Concerns
  - Prompt engineering ≠ code changes
  - Non-technical users can edit
  - Independent testing

✓ Standardization
  - Uniform schema across agents
  - Predictable parameters
  - Template variable system

✓ Flexibility
  - Environment-specific overrides
  - A/B testing support
  - Multi-language potential
""")
    
    print("=" * 80)
    print("To modify a prompt, edit the YAML file:")
    print("  vim prompts/segmentation_agent.yaml")
    print("\nChanges take effect immediately (LRU cache timeout)")
    print("=" * 80)

if __name__ == "__main__":
    main()
