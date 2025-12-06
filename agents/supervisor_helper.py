import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

from tools.utils.token_utils import estimate_tokens
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
            "evaluation": self._file_stats(run_dir / "evaluation.jsonl"),
        }

    def detect_response_type(self, run_id: str, before: Dict[str, Any], enable_auto_tagging: bool) -> Dict[str, Any]:
        """Detect which artifacts were modified and determine the appropriate response type.
        
        Compares file state before/after agent execution to identify side-effects.
        Uses priority ordering: ADO export > Evaluation > Tagging > Backlog generation.
        Optionally triggers auto-tagging when backlog is first generated.
        
        Args:
            run_id: Run identifier
            before: File state snapshot from before agent execution
            enable_auto_tagging: If True, automatically tag stories when backlog is generated
            
        Returns:
            Dict with:
            - response_type: "ado_export" | "evaluation" | "tagging" | "backlog_generated" | None
            - status_updates: Additional metadata (e.g., auto-tagging results)
        """
        try:
            run_dir = Path(before.get("run_dir", str(Path("runs") / run_id)))
            ado_file = run_dir / "ado_export_last.json"
            tag_file = run_dir / "tagging.jsonl"
            backlog_file = run_dir / "generated_backlog.jsonl"
            evaluation_file = run_dir / "evaluation.jsonl"

            # Capture current file state
            after = {
                "ado": self._file_stats(ado_file),
                "tag": self._file_stats(tag_file),
                "backlog": self._file_stats(backlog_file),
                "evaluation": self._file_stats(evaluation_file),
            }

            result_type: Optional[str] = None
            status_updates: Dict[str, Any] = {}

            # Priority 1: ADO export (highest priority - indicates successful export)
            b, a = before["ado"], after["ado"]
            if a["exists"] and (not b["exists"] or a["mtime_ns"] != b["mtime_ns"] or a["size"] != b["size"]):
                result_type = "ado_export"
                self.logger.info("Supervisor: Detected ADO export update")

            # Priority 2: Evaluation (high priority - backlog quality assessment completed)
            if result_type is None:
                b, a = before["evaluation"], after["evaluation"]
                if a["exists"] and (not b["exists"] or a["mtime_ns"] != b["mtime_ns"] or a["size"] != b["size"]):
                    result_type = "evaluation"
                    self.logger.info("Supervisor: Detected evaluation update")

            # Priority 3: Tagging (medium priority - stories have been classified)
            if result_type is None:
                b, a = before["tag"], after["tag"]
                if a["exists"] and (not b["exists"] or a["mtime_ns"] != b["mtime_ns"] or a["size"] != b["size"]):
                    result_type = "tagging"
                    self.logger.info("Supervisor: Detected tagging update")

            # Priority 4: Backlog generation (lowest priority - backlog items created)
            # Optionally trigger auto-tagging to transition from "backlog_generated" to "tagging" response
            if result_type is None:
                b, a = before["backlog"], after["backlog"]
                if a["exists"] and (not b["exists"] or a["mtime_ns"] != b["mtime_ns"] or a["size"] != b["size"]):
                    result_type = "backlog_generated"
                    self.logger.info("Supervisor: Detected backlog generation/update")
                    
                    # If auto-tagging enabled, attempt to tag the newly generated backlog
                    if enable_auto_tagging:
                        try:
                            tagging_agent = create_tagging_agent(run_id)
                            tag_out_str = tagging_agent()
                            status_updates["auto_tagging"] = True
                            status_updates["auto_tagging_result"] = tag_out_str[:500]
                            
                            # After auto-tagging, check if tagging file now exists
                            # If so, prefer tagging view over backlog_generated
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
            # Non-fatal; ignore detection errors and continue with no detected response type
            self.logger.debug("Supervisor: Side-effect detection encountered an error; continuing")
            return {"response_type": None, "status_updates": {}}

    # ------------------------------------------------------------------
    # Dashboard helpers

    def build_dashboard(
        self,
        run_id: str,
        sessions_dir: str,
        model_id: Optional[str],
        agent: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Aggregate session + artifact stats for the mini-dashboard."""

        repo_data = self._read_session_repository(run_id, sessions_dir)
        conversation = None
        sources: List[str] = []

        if agent:
            conversation = self._summarize_in_memory_messages(agent)
            if conversation.get("total_messages"):
                sources.append("agent_cache")

        if not conversation or not conversation.get("total_messages"):
            conversation = repo_data.get("conversation", {})
        elif repo_data.get("conversation", {}).get("total_messages") and "session_repository" not in sources:
            sources.append("session_repository")

        if not sources:
            sources = [repo_data.get("source") or "unknown"]

        artifacts = self._collect_artifacts(run_id)

        return {
            "run_id": run_id,
            "model_id": model_id,
            "session_state": repo_data.get("session_state", {}),
            "conversation": conversation or {},
            "sources": {
                "conversation": sources
            },
            "artifacts": artifacts
        }

    def _summarize_in_memory_messages(self, agent: Any) -> Dict[str, Any]:
        messages = getattr(agent, "messages", None)
        if not messages:
            return {}
        normalized: List[Dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)
            created = getattr(msg, "created_at", None)
            if isinstance(msg, dict):
                role = msg.get("role", role)
                content = msg.get("content", content)
                created = msg.get("created_at", created)
            normalized.append({
                "role": role,
                "content": content,
                "created_at": created
            })
        return self._summarize_messages(normalized)

    def _read_session_repository(self, run_id: str, sessions_dir: str) -> Dict[str, Any]:
        session_dir = Path(sessions_dir) / f"session_{run_id}"
        session_state = {
            "available": False,
            "session_created_at": None,
            "session_updated_at": None,
            "conversation_manager": None,
            "removed_message_count": 0
        }
        conversation = {
            "total_messages": 0,
            "by_role": {},
            "last_message_role": None,
            "last_message_at": None,
            "last_message_preview": None,
            "tool_usage": [],
            "token_estimate": 0
        }
        source = "session_repository"

        if not session_dir.exists():
            return {
                "session_state": session_state,
                "conversation": conversation,
                "source": source
            }

        session_state["available"] = True

        meta = self._load_json(session_dir / "session.json")
        if meta:
            session_state["session_created_at"] = meta.get("created_at")
            session_state["session_updated_at"] = meta.get("updated_at")

        agent_dir = session_dir / "agents" / "agent_default"
        agent_meta = self._load_json(agent_dir / "agent.json")
        if agent_meta:
            cm_state = agent_meta.get("conversation_manager_state", {}) or {}
            session_state["conversation_manager"] = cm_state.get("__name__")
            session_state["removed_message_count"] = cm_state.get("removed_message_count", 0)
            session_state.setdefault("session_updated_at", agent_meta.get("updated_at"))

        messages_dir = agent_dir / "messages"
        messages = self._collect_messages_from_repo(messages_dir)
        if messages:
            conversation.update(self._summarize_messages(messages))

        return {
            "session_state": session_state,
            "conversation": conversation,
            "source": source
        }

    def _collect_messages_from_repo(self, messages_dir: Path) -> List[Dict[str, Any]]:
        if not messages_dir.exists():
            return []
        entries: List[Dict[str, Any]] = []
        for path in sorted(messages_dir.glob("message_*.json"), key=self._message_sort_key):
            data = self._load_json(path)
            if not data:
                continue
            message = data.get("message", {}) or {}
            entries.append({
                "role": message.get("role"),
                "content": message.get("content"),
                "created_at": data.get("created_at")
            })
        return entries

    def _message_sort_key(self, path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except Exception:
            return 0

    def _summarize_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not messages:
            return {
                "total_messages": 0,
                "by_role": {},
                "last_message_role": None,
                "last_message_at": None,
                "last_message_preview": None,
                "tool_usage": [],
                "token_estimate": 0
            }

        counts = Counter()
        tool_counts = Counter()
        combined_text: List[str] = []
        last_message = messages[-1]

        for msg in messages:
            role = msg.get("role") or "unknown"
            counts[role] += 1
            text = self._extract_text(msg.get("content"))
            if text:
                combined_text.append(text)
                if role == "assistant":
                    for tool_name in self._extract_tool_calls(text):
                        tool_counts[tool_name] += 1

        token_estimate = 0
        if combined_text:
            try:
                token_estimate = estimate_tokens("\n".join(combined_text)) or 0
            except Exception as e:
                self.logger.debug("Supervisor: token estimate (dashboard) failed: %s", e)

        preview = self._extract_text(last_message.get("content")) if last_message else None
        if preview and len(preview) > 160:
            preview = preview[:160] + "…"

        return {
            "total_messages": sum(counts.values()),
            "by_role": dict(counts),
            "last_message_role": last_message.get("role") if last_message else None,
            "last_message_at": last_message.get("created_at") if last_message else None,
            "last_message_preview": preview,
            "tool_usage": [
                {"name": name, "count": count}
                for name, count in tool_counts.most_common()
            ],
            "token_estimate": token_estimate
        }

    def _extract_text(self, content: Any) -> Optional[str]:
        if not content:
            return None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    texts.append(str(item["text"]))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts).strip() if texts else None
        if isinstance(content, dict) and content.get("text"):
            return str(content["text"]).strip()
        return None

    def _extract_tool_calls(self, text: str) -> List[str]:
        if not text or "tool_call" not in text:
            return []
        normalized = text.strip()
        try:
            payload = normalized
            if not payload.startswith("["):
                payload = f"[{payload}]"
            payload = payload.replace("}\n{", "},{")
            data = json.loads(payload)
            if isinstance(data, dict):
                data = [data]
            names: List[str] = []
            for entry in data:
                call = entry.get("tool_call") if isinstance(entry, dict) else None
                if isinstance(call, dict) and call.get("name"):
                    names.append(call["name"])
            if names:
                return names
        except Exception:
            pass
        regex = re.compile(r'"tool_call"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"', re.MULTILINE)
        return regex.findall(normalized)

    def _collect_artifacts(self, run_id: str) -> Dict[str, Any]:
        run_dir = Path("runs") / run_id
        backlog_file = run_dir / "generated_backlog.jsonl"
        tagging_file = run_dir / "tagging.jsonl"
        evaluation_file = run_dir / "evaluation.jsonl"
        ado_file = run_dir / "ado_export_last.json"

        return {
            "backlog_items": self._count_jsonl(backlog_file) or 0,
            "tagging_items": self._count_jsonl(tagging_file) or 0,
            "evaluation_runs": self._count_jsonl(evaluation_file) or 0,
            "ado_export": self._read_ado_export(ado_file)
        }

    def _read_ado_export(self, path: Path) -> Dict[str, Any]:
        ado_info = {
            "has_export": False,
            "status": None,
            "mode": None,
            "counts": None,
            "updated_at": None
        }
        data = self._load_json(path)
        if not data:
            return ado_info
        ado_info.update({
            "has_export": True,
            "status": data.get("status"),
            "mode": data.get("mode"),
            "counts": data.get("counts") or data.get("summary", {}).get("counts"),
            "updated_at": data.get("timestamp") or data.get("exported_at") or data.get("updated_at")
        })
        return ado_info

    def _count_jsonl(self, path: Path) -> Optional[int]:
        if not path.exists():
            return 0
        try:
            count = 0
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        count += 1
            return count
        except Exception as e:
            self.logger.debug("Supervisor: Failed to count %s: %s", path, e)
            return 0

    def _load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.logger.debug("Supervisor: Failed to parse %s: %s", path, e)
        return None
