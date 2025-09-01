import os
from typing import Dict, Any

_SUPABASE_READY = False
_client = None

def _init_supabase():
    global _client, _SUPABASE_READY
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return
    try:
        # Lazy import so the app still runs if supabase isn't installed
        from supabase import create_client, Client  # type: ignore
        _client = create_client(url, key)
        _SUPABASE_READY = True
    except Exception:
        _SUPABASE_READY = False

def log_event(table: str, payload: Dict[str, Any]) -> None:
    """
    Log usage analytics (non-blocking). If Supabase isn't configured, this no-ops.
    To enable, set env vars SUPABASE_URL and SUPABASE_KEY and install `supabase` package.
    """
    global _SUPABASE_READY, _client
    if not _SUPABASE_READY:
        _init_supabase()
    if not _SUPABASE_READY:
        return
    try:
        _client.table(table).insert(payload).execute()
    except Exception:
        # Never crash the app for analytics
        pass
