# src/agents/history.py

import json
import logging
from typing import cast

from src.llm.bedrock_client import call_llm
from src.states import TravelChatBotState

logger = logging.getLogger(__name__)


def update_history_summary(state: TravelChatBotState) -> TravelChatBotState:
    """
    Compress the last exchange (user_input + master response)
    into a short 30–50 word history_summary and store it on state.
    """
    previous_summary = (state.get("history_summary") or "").strip()
    user_input = (state.get("user_input") or "").strip()
    master_plan = cast(dict, state.get("master_plan") or {})
    master_message = (
        master_plan.get("assistant_message", "")
        or state.get("master_message", "")
        or ""
    )

    summary_payload = {
        "previous_summary": previous_summary,
        "last_user_message": user_input,
        "last_assistant_message": master_message,
    }

    summary_prompt = """
You are a summarizer.

Given:
- previous_summary (may be empty)
- last_user_message
- last_assistant_message

Produce a single updated summary of the conversation so far
in 30–50 words, focusing on trip details and user preferences.
Respond with plain text only.
    """.strip()

    logger.debug("update_history_summary: calling LLM summarizer")
    new_summary = call_llm(
        system_prompt=summary_prompt,
        user_prompt=json.dumps(summary_payload),
        temperature=0.0,
        max_tokens=150,
    ).strip()

    state["history_summary"] = new_summary
    logger.debug("update_history_summary: new summary=%s", new_summary)
    return state
