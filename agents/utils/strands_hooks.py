import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import httpx


class StrandsHookClient:
    """
    Minimal client to emit Strands-style progress hooks to the server so the UI
    can receive updates via SSE.

    Usage:
        hook = StrandsHookClient(run_id)
        await hook.emit(event="started", agent="workflow", message="...")
    """

    def __init__(self, run_id: str, base_url: Optional[str] = None):
        self.run_id = run_id
        self.base_url = base_url or os.getenv("RDE_BASE_URL", "http://localhost:8000")
        self._endpoint = f"{self.base_url}/strands-hook"

    async def emit(self, *, event: str, agent: Optional[str] = None, message: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "event": event,
            "message": message,
            "meta": meta or {},
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(self._endpoint, json=payload)
        except Exception:
            # Best-effort: ignore network errors; workflow continues
            return
