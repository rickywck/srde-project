# Prompt Externalization and Standardization

## Overview
All agent prompts have been externalized from hardcoded strings to YAML configuration files stored in the `prompts/` directory. This provides centralized, version-controlled, and standardized prompt management.

## Structure

### Directory Layout
```
prompts/
├── segmentation_agent.yaml
├── backlog_generation_agent.yaml
├── tagging_agent.yaml
├── evaluation_agent.yaml
└── supervisor_agent.yaml
```

### YAML Schema
Each prompt file follows a standardized structure:

```yaml
name: agent_name
description: Brief description of the agent's purpose
version: 1.0

system_prompt: |
  The system prompt that defines the agent's role and capabilities.
  Multi-line text supported with YAML pipe (|) notation.

user_prompt_template: |
  Template for user prompts with {variable} placeholders.
  Variables are substituted at runtime using Python's str.format().

parameters:
  temperature: 0.7
  max_tokens: 2000
  response_format: json_object  # Optional
  top_p: 0.9  # Optional

# Optional: agent-specific configuration
evaluation_schema:  # Only in evaluation_agent.yaml
  completeness:
    score: "int (1-5)"
    reasoning: "string"
  ...
```

## Components

### 1. Prompt Loader (`agents/prompt_loader.py`)

Centralized utility for loading and managing prompts:

```python
from agents.prompt_loader import get_prompt_loader

loader = get_prompt_loader()

# Load complete configuration
config = loader.load_prompt("segmentation_agent")

# Get specific components
system_prompt = loader.get_system_prompt("segmentation_agent")
user_template = loader.get_user_prompt_template("segmentation_agent")
parameters = loader.get_parameters("segmentation_agent")

# Format user prompt with variables
user_prompt = loader.format_user_prompt(
    "segmentation_agent",
    document_text="..."
)
```

**Features:**
- LRU caching for performance
- Validation of required fields
- Template variable substitution
- Metadata extraction

### 2. Updated Agents

All agents now use the prompt loader:

#### Segmentation Agent
```python
def create_segmentation_agent(run_id: str):
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("segmentation_agent")
    params = prompt_loader.get_parameters("segmentation_agent")
    
    # Build user prompt from template
    segmentation_prompt = prompt_loader.format_user_prompt(
        "segmentation_agent",
        document_text=document_text
    )
```

#### Backlog Generation Agent
```python
def create_backlog_generation_agent(run_id: str):
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("backlog_generation_agent")
    params = prompt_loader.get_parameters("backlog_generation_agent")
    
    # Format context for prompt
    prompt = prompt_loader.format_user_prompt(
        "backlog_generation_agent",
        segment_text=segment_text,
        intent_labels=", ".join(intent_labels),
        dominant_intent=dominant_intent,
        ado_items_formatted=ado_formatted,
        architecture_constraints_formatted=arch_formatted
    )
```

#### Tagging Agent
```python
def create_tagging_agent(run_id: str):
    prompt_loader = get_prompt_loader()
    system_prompt = prompt_loader.get_system_prompt("tagging_agent")
    params = prompt_loader.get_parameters("tagging_agent")
    
    # Format similar stories context
    user_prompt = prompt_loader.format_user_prompt(
        "tagging_agent",
        story_title=story.get("title"),
        story_description=story.get("description"),
        story_acceptance_criteria=ac_text,
        similarity_threshold=threshold,
        similar_stories_formatted=similar_formatted
    )
```

#### Evaluation Agent
```python
def create_evaluation_agent(run_id: str):
    prompt_loader = get_prompt_loader()
    evaluation_system_prompt = prompt_loader.get_system_prompt("evaluation_agent")
    params = prompt_loader.get_parameters("evaluation_agent")
    eval_config = prompt_loader.load_prompt("evaluation_agent")
    evaluation_schema = eval_config.get("evaluation_schema", EVALUATION_SCHEMA)
    
    user_prompt = prompt_loader.format_user_prompt(
        "evaluation_agent",
        segment_text=segment_text[:4000],
        ado_items_formatted=ado_formatted,
        architecture_constraints_formatted=arch_formatted,
        backlog_items_formatted=backlog_formatted,
        evaluation_schema=json.dumps(evaluation_schema, indent=2)
    )
```

#### Supervisor Agent
```python
class SupervisorAgent:
    def __init__(self, config_path: str = "config.poc.yaml"):
        prompt_loader = get_prompt_loader()
        self.system_prompt = prompt_loader.get_system_prompt("supervisor_agent")
        model_params = prompt_loader.get_parameters("supervisor_agent")
        
        self.model = OpenAIModel(
            model_id=self.config["openai"]["chat_model"],
            params=model_params
        )
```

## Benefits

### 1. Centralized Management
- All prompts in one location
- Easy to find and update
- Consistent structure across agents

### 2. Version Control
- Track prompt changes over time
- Compare versions with git diff
- Roll back if needed
- Document prompt evolution

### 3. Separation of Concerns
- Prompt engineering separate from code
- Non-technical users can modify prompts
- Code changes don't require prompt reviews

### 4. Testability
- Easy A/B testing of prompts
- Swap prompts without code changes
- Test different parameter combinations

### 5. Consistency
- Standardized YAML schema
- Uniform parameter names
- Predictable structure

### 6. Flexibility
- Environment-specific prompts possible
- Easy to add new template variables
- Support for multi-language prompts

## Migration Notes

### Removed Components
- Hardcoded `SEGMENTATION_AGENT_SYSTEM_PROMPT` constant
- Hardcoded `BACKLOG_GENERATION_AGENT_SYSTEM_PROMPT` constant
- Hardcoded `TAGGING_AGENT_SYSTEM_PROMPT` constant
- Helper functions like `_build_generation_prompt()` and `_build_llm_prompt()`
- Inline prompt strings in agent code

### Preserved Functionality
- All agents produce identical outputs
- Same template variables supported
- Parameters match original hardcoded values
- Mock modes still functional

## Usage Examples

### Modifying a Prompt
Edit the YAML file directly:

```bash
vim prompts/segmentation_agent.yaml
```

Changes take effect immediately (with LRU cache timeout).

### Adding a New Variable
1. Add placeholder to template in YAML:
   ```yaml
   user_prompt_template: |
     New variable: {my_new_variable}
   ```

2. Pass variable when formatting:
   ```python
   prompt = loader.format_user_prompt(
       "agent_name",
       my_new_variable="value"
   )
   ```

### Creating a New Agent
1. Create `prompts/my_agent.yaml` with required fields
2. Load in agent code:
   ```python
   def create_my_agent(run_id: str):
       prompt_loader = get_prompt_loader()
       system_prompt = prompt_loader.get_system_prompt("my_agent")
       params = prompt_loader.get_parameters("my_agent")
       # ... use prompts
   ```

## Testing

Verify prompt loading:
```bash
python -c "from agents.prompt_loader import get_prompt_loader; \
           loader = get_prompt_loader(); \
           print(loader.get_system_prompt('segmentation_agent'))"
```

Test agent creation:
```bash
python -c "from agents.segmentation_agent import create_segmentation_agent; \
           agent = create_segmentation_agent('test'); \
           print(f'Created: {agent.__name__}')"
```

Run integration test:
```bash
python tests/test_supervisor_integration.py
```

## Best Practices

1. **Prompt Versioning**: Increment version number when making significant changes
2. **Documentation**: Update description field when changing prompt logic
3. **Variables**: Use clear, descriptive variable names in templates
4. **Parameters**: Keep temperature/max_tokens consistent per agent type
5. **Testing**: Test prompt changes with representative inputs
6. **Backup**: Keep backup of working prompts before major changes

## Future Enhancements

Potential improvements:
- Prompt variant support (A/B testing)
- Environment-specific overrides (dev/staging/prod)
- Prompt performance tracking
- Multi-language support
- Prompt templates library
- Hot-reloading for development
- Prompt validation on startup
