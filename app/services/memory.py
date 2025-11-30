from collections import defaultdict, deque
from typing import Deque, Dict, List
import uuid

MAX_MESSAGES = 20

# session_id -> deque of {"role": ..., "content":...}
_sessions: Dict[str, Deque[dict]] = defaultdict(
    lambda: deque(maxlen=MAX_MESSAGES)
)

def ensure_session_id(session_id: str | None) -> str:
    """If no session_id, create a new one."""
    return session_id or str(uuid.uuid4())

def get_history(session_id: str) -> List[dict]:
    """Return a list of message dicts for this session."""
    return list(_sessions[session_id])

def append_message(session_id: str, role: str, content: str) -> None:
    """Add one new message to session history."""
    _sessions[session_id].append({"role": role, "content": content})

def reset_session(session_id: str) -> None:
    """Clear a session completely (not used yet, but handy)."""
    _sessions.pop(session_id, None)
