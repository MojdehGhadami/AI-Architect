"""Entry points: `run_agent` for an orchestrator/Chainlit app, and a small CLI.

    python -m ai_architect.runner
"""

from __future__ import annotations

import asyncio
import json
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .state import role_and_content


@lru_cache(maxsize=1)
def _agent():
    from .graph import build_graph

    return build_graph()


def run_turn(user_text: str, thread_id: str) -> Dict[str, Any]:
    """Run one user turn and return the assistant's reply for this thread."""
    config = {"configurable": {"thread_id": thread_id}}
    state = _agent().invoke({"messages": [{"role": "user", "content": user_text}]}, config)

    reply = ""
    for message in reversed(state.get("messages", [])):
        role, content = role_and_content(message)
        if role == "assistant" and content:
            reply = content
            break

    try:
        debug = json.loads(reply)
    except Exception:
        debug = None

    return {
        "final_answer": reply,
        "debug": debug,
        "awaiting_user": bool(state.get("pending_questions")),
        "requirements": state.get("requirements"),
        "retrieval_plan": state.get("retrieval_plan"),
    }


async def run_agent(user_text: str, history: Optional[List[dict]] = None, thread_id: str = "default") -> Dict[str, Any]:
    """Async wrapper; runs the sync graph in a worker thread."""
    return await asyncio.to_thread(run_turn, user_text, thread_id)


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    thread_id = str(uuid.uuid4())
    print("AI Architect. Describe your workload (Ctrl+C to quit).")
    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        result = run_turn(user_text, thread_id)
        print("\narchitect>\n" + result["final_answer"])


if __name__ == "__main__":
    main()
