"""MCP ADO integration tool to process Azure DevOps related queries.

Exposes `mcp_ado_tool(query: str)` as a Strands tool to route queries to the
MCP server and return helpful Azure DevOps responses.
"""

import os
from pathlib import Path

from dotenv import load_dotenv, dotenv_values
from mcp import StdioServerParameters, stdio_client
from strands import Agent, tool
from strands.tools.mcp import MCPClient
from strands_tools import file_write

from agents.model_factory import ModelFactory


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    # Load .env so model factory can access credentials like OPENAI_API_KEY
    load_dotenv(ENV_PATH)

@tool
def mcp_ado_tool(query: str) -> str:
    """
    Process Azure DevOps related queries using the MCP server.

    Args:
        query: The user's question

    Returns:
        A helpful response addressing user query
    """

    model = ModelFactory.create_openai_model_for_agent()

    response = str()

    try:
        env_overrides = {}
        if ENV_PATH.exists():
            env_overrides = {
                key: value
                for key, value in dotenv_values(ENV_PATH).items()
                if value is not None
            }

        env = {**os.environ, **env_overrides}
        ado_mcp_server = MCPClient(
            lambda: stdio_client(
                StdioServerParameters(
                    command="npx",
                    args=["@tiberriver256/mcp-server-azure-devops"],
                    env=env,
                )
            )
        )

        with ado_mcp_server:

            tools = ado_mcp_server.list_tools_sync() + [file_write]
            # Create the research agent with specific capabilities
            ado_agent = Agent(
                model=model,
                system_prompt=(
                    "You are an Azure DevOps research assistant. "
                    "Use the available Azure DevOps MCP server tools to gather project, boards, work items, "
                    "pipelines, repositories, and other relevant information. "
                    "Prefer retrieving data via the provided tools instead of making assumptions. "
                    "When referencing results, include enough context so the user can locate the artifacts."
                ),
                tools=tools,
            )
            response = str(ado_agent(query))
            print("\n\n")

        if len(response) > 0:
            return response

        return "I apologize, but I couldn't properly analyze your question. Could you please rephrase or provide more context?"

    except Exception as e:
        return f"Error processing your query: {str(e)}"


if __name__ == "__main__":
    mcp_ado_tool("List all epics")