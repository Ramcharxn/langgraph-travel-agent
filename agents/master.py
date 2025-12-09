# src/agents/master.py

import json
import logging
from typing import cast

from src.llm.bedrock_client import call_llm
from src.prompts import MASTER_SYSTEM_PROMPT, MASTER_RESPONSE_SYSTEM_PROMPT
from src.states import TravelChatBotState

logger = logging.getLogger(__name__)


def master_agent(state: TravelChatBotState) -> TravelChatBotState:
    """
    Top-level MASTER agent.

    - Classifies the user's intent.
    - Updates trip_info / preferences.
    - Decides whether to call specialists (activities/logistics) or answer directly.
    - Sets master_route: "activities", "logistics", or "master_response".
    """
    logger.debug("master_agent: entered")

    user_input = (state.get("user_input") or "").strip()
    history_summary = (state.get("history_summary") or "").strip()

    # No user message: simple greeting + direct route to master_response
    if not user_input:
        msg = "Hi! Tell me what kind of trip or question you have in mind."
        state["master_message"] = msg
        state["master_plan"] = {
            "intent": "generic_chat",
            "needs_specialist": False,
            "assistant_message": msg,
            "trip_info_updates": {},
            "preferences_updates": {},
        }
        state["master_route"] = "master_response"
        logger.info("master_agent: no user_input, routing -> master_response")
        return state

    # Previous knowledge
    trip_info = cast(dict, state.get("trip_info") or {})
    preferences = cast(dict, state.get("preferences") or {})

    # Build compact JSON input for LLM
    llm_input = {
        "user_query": user_input,
        "trip_info": trip_info,
        "preferences": preferences,
        "history_summary": history_summary,
    }

    llm_user_prompt = json.dumps(llm_input)
    logger.debug(
        "master_agent: calling LLM with trip_info=%s, preferences=%s",
        trip_info,
        preferences,
    )

    raw = call_llm(
        system_prompt=MASTER_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        max_tokens=400,
        temperature=0.3,
    )

    logger.debug("master_agent: raw LLM output: %s", raw)

    # Default fallback if parsing fails
    fallback = {
        "intent": "generic_chat",
        "needs_specialist": False,
        "assistant_message": "Here is a simple high-level answer based on your question.",
        "trip_info_updates": {},
        "preferences_updates": {},
    }

    try:
        parsed = json.loads(raw) if raw else fallback
    except Exception as e:
        logger.exception("master_agent: failed to parse JSON: %s", e)
        parsed = fallback

    intent = parsed.get("intent", "generic_chat")
    needs_specialist = bool(parsed.get("needs_specialist", False))
    trip_updates = parsed.get("trip_info_updates") or {}
    pref_updates = parsed.get("preferences_updates") or {}
    assistant_message = parsed.get("assistant_message") or fallback["assistant_message"]

    # Merge updates into state
    trip_info.update(trip_updates)
    preferences.update(pref_updates)
    state["trip_info"] = trip_info
    state["preferences"] = preferences
    state["intent"] = intent

    # Save full master plan
    state["master_plan"] = parsed

    # SAFETY OVERRIDE:
    # For structured trip intents, we always want to go through specialists,
    # even if the LLM mistakenly set needs_specialist = false.
    if intent in ("activities_q", "logistics_q", "plan_full"):
        needs_specialist = True
        parsed["needs_specialist"] = True

    metadata = state.get("metadata") or {}

    if needs_specialist:
        # Decide which specialists we want THIS TURN
        if intent == "logistics_q":
            targets = ["logistics"]
        elif intent == "plan_full":
            targets = ["activities", "logistics"]
        else:
            # default to activities when we need a specialist but it's
            # not explicitly logistics-only or full-plan
            targets = ["activities"]

        metadata["specialist_targets"] = targets
        metadata["specialist_index"] = 0
        state["metadata"] = metadata

        # First specialist to call this turn
        state["master_route"] = targets[0]
    else:
        # No specialists needed: master already has the final message
        state["master_message"] = assistant_message
        state["master_route"] = "master_response"

    logger.info(
        "master_agent: intent=%s, needs_specialist=%s, targets=%s, route=%s",
        intent,
        needs_specialist,
        metadata.get("specialist_targets") if needs_specialist else None,
        state["master_route"],
    )
    return state


def master_response_agent(state: TravelChatBotState) -> TravelChatBotState:
    """
    MASTER_RESPONSE agent.

    - Combines master_plan + activities_plan + logistics_plan into ONE
      final user-facing message.
    - Respects the “chat-first, itinerary-only-when-asked” logic in
      MASTER_RESPONSE_SYSTEM_PROMPT.
    """
    logger.debug("master_response_agent: entered")

    master_plan = cast(dict, state.get("master_plan") or {})
    activities_plan = state.get("activities_plan")
    logistics_plan = state.get("logistics_plan")
    trip_info = cast(dict, state.get("trip_info") or {})
    preferences = cast(dict, state.get("preferences") or {})
    intent = state.get("intent")
    user_query = (state.get("user_input") or "").strip()

    has_activities = activities_plan is not None
    has_logistics = logistics_plan is not None

    logger.debug(
        "master_response_agent: sources => master_plan=%s, activities_plan=%s, logistics_plan=%s",
        "yes" if master_plan else "no",
        "yes" if has_activities else "no",
        "yes" if has_logistics else "no",
    )

    # If there are no specialist outputs, just reuse master_message/assistant_message
    if not has_activities and not has_logistics:
        msg = (
            state.get("master_message")
            or master_plan.get("assistant_message")
            or "Here is a simple high-level answer based on your question."
        )
        state["master_message"] = msg
        return state

    # We DO have activities and/or logistics: synthesize everything
    llm_input = {
        "user_query": user_query,
        "intent": intent,
        "trip_info": trip_info,
        "preferences": preferences,
        "master_plan": master_plan,
        "activities_plan": activities_plan,
        "logistics_plan": logistics_plan,
    }

    llm_user_prompt = json.dumps(llm_input)

    reply = call_llm(
        system_prompt=MASTER_RESPONSE_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        max_tokens=700,
        temperature=0.5,
    )

    if not reply:
        logger.warning(
            "master_response_agent: empty reply from LLM, falling back to master_message/master_plan"
        )
        reply = (
            state.get("master_message")
            or master_plan.get("assistant_message")
            or "Here is a simple summary of your trip based on the information I have."
        )

    state["master_message"] = reply.strip()
    logger.debug("master_response_agent: finished, reply ready for user")
    return state
