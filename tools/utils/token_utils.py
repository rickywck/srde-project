def estimate_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        # Fallback heuristic: ~4 chars per token
        s = text or ""
        return max(1, (len(s) + 3) // 4)
