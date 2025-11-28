"""
Top-level Supervisor module shim for tests and external imports.

Exports SupervisorAgent from agents.supervisor_agent for backward compatibility.
"""
from agents.supervisor_agent import SupervisorAgent  # noqa: F401
