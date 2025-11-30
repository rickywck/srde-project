import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

from tools.token_utils import estimate_tokens
from agents.tagging_agent import create_tagging_agent


class SupervisorRunHelper:
    """
    Helper for SupervisorAgent to:
    - Log input token estimates
    - Snapshot file states before agent run
    - Detect side-effects after run and decide response_type
    - Optionally perform auto-tagging when backlog is generated
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_input_tokens(self, full_query: str) -> Optional[int]:
        try:
            approx = estimate_tokens(full_query)
            self.logger.info("Supervisor: input tokens approx=%s", approx)
            return approx
        except Exception as e:
            self.logger.debug("Supervisor: token estimate failed: %s", e)
            return None

    def _file_stats(self, p: Path) -> Dict[str, Any]:
        if not p.exists():
            return {"exists": False, "mtime_ns": None, "size": None}
        st = p.stat()
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))
        return {"exists": True, "mtime_ns": mtime_ns, "size": st.st_size}

    def snapshot_before(self, run_id: str) -> Dict[str, Any]:
        run_dir = Path("runs") / run_id
        return {
            "run_dir": str(run_dir),
            "ado": self._file_stats(run_dir / "ado_export_last.json"),
            "tag": self._file_stats(run_dir / "tagging.jsonl"),
            "backlog": self._file_stats(run_dir / "generated_backlog.jsonl"),
        }

    def detect_response_type(self, run_id: str, before: Dict[str, Any], enable_auto_tagging: bool) -> Dict[str, Any]:
        try:
            run_dir = Path(before.get("run_dir", str(Path("runs") / run_id)))
            ado_file = run_dir / "ado_export_last.json"
            tag_file = run_dir / "tagging.jsonl"
            backlog_file = run_dir / "generated_backlog.jsonl"

            after = {
                "ado": self._file_stats(ado_file),
                "tag": self._file_stats(tag_file),
                "backlog": self._file_stats(backlog_file),
            }

            result_type: Optional[str] = None
            status_updates: Dict[str, Any] = {}

            # Priority 1: ADO
            b, a = before["ado"], after["ado"]
            if a["exists"] and (not b["exists"] or a["mtime_ns"] != b["mtime_ns"] or a["size"] != b["size"]):
                result_type = "ado_export"
                self.logger.info("Supervisor: Detected ADO export update")

            # Priority 2: Tagging
            if result_type is None:
                b, a = before["tag"], after["tag"]
                if a["exists"] and (not b["exists"] or a["mtime_ns"] != b["mtime_ns"] or a["size"] != b["size"]):
                    result_type = "tagging"
                    self.logger.info("Supervisor: Detected tagging update")

            # Priority 3: Backlog
            if result_type is None:
                b, a = before["backlog"], after["backlog"]
                if a["exists"] and (not b["exists"] or a["mtime_ns"] != b["mtime_ns"] or a["size"] != b["size"]):
                    result_type = "backlog_generated"
                    self.logger.info("Supervisor: Detected backlog generation/update")
                    if enable_auto_tagging:
                        try:
                            tagging_agent = create_tagging_agent(run_id)
                            tag_out_str = tagging_agent()
                            status_updates["auto_tagging"] = True
                            status_updates["auto_tagging_result"] = tag_out_str[:500]
                            # If tagging file now exists, prefer tagging view
                            ta = self._file_stats(tag_file)
                            if ta["exists"]:
                                result_type = "tagging"
                                self.logger.info("Supervisor: Auto-tagging completed")
                        except Exception as e:
                            status_updates["auto_tagging"] = False
                            status_updates["auto_tagging_error"] = str(e)
                            self.logger.warning("Supervisor: Auto-tagging error: %s", e)

            return {"response_type": result_type, "status_updates": status_updates}
        except Exception:
            # Non-fatal; ignore detection errors
            self.logger.debug("Supervisor: Side-effect detection encountered an error; continuing")
            return {"response_type": None, "status_updates": {}}
