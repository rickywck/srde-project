# Appendix
## Part 2: Hands-On Project
### Project Theme: Accelerating Engineering Through AI-First Agentic Solutions
Candidate Goal
Design, build, and demo a working agentic solution that solves a real engineering problem using AI in every phase of the SDLC.
### Sample Project: Backlog Synthesizer
#### Objective: Build a multi-agent system that ingests: - Customer meeting transcripts (raw text or PDF) - A confluence/wiki export containing architecture constraints - Current backlog tickets from JIRA or GitHub (real integration preferred, but mocked JSON acceptable)
#### Output: - Structured user stories, epics, and tasks - Acceptance criteria for each story - System or feature tags - Gaps or conflicts across existing backlog vs. new requests
#### Constraints: - Agents must handle task decomposition, planning, and tool invocation - Memory must persist across document parsing, gap detection, and story writing - Audit logs must show how conclusions were reached - AI usage must be documented throughout SDLC stages
## Required Steps
1. Problem Framing (AI-Enhanced)
Use AI tools to explore alternative framings or uncover edge cases
Submit AI prompts used and the iterations that influenced scope
2. Design
Architecture diagram of agent roles, tool interfaces, and memory structure
Include planned interactions, tool outputs, and trace paths
Submit prompts or AI design assistance (e.g., Claude helping sketch flow)
3. Evaluation Plan
Define how to measure success of the system
Create a small “golden dataset” (3–5 sample transcripts with manually written ideal outputs)
Define metrics (e.g., completeness of acceptance criteria, accuracy of feature tagging, F1 score for conflict detection)
Implement one automated evaluation (e.g., keyword match or LLM-as-judge scoring)
4. Implementation
Multi-agent framework
Context and memory engine (Redis, Chroma, Weaviate, etc.)
Modular tool abstractions (JIRA, Notion, JSON parsers, etc.)
Evidence of error handling and retry logic