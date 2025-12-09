# src/agents/specialists.py

import json
import logging
from typing import cast

from src.llm.bedrock_client import call_llm
from src.prompts import ACTIVITIES_SYSTEM_PROMPT, LOGISTICS_SYSTEM_PROMPT
from src.states import TravelChatBotState
from src.tools import ACTIVITIES_TOOLS, LOGISTICS_TOOLS

logger = logging.getLogger(__name__)


def activities_agent(state: TravelChatBotState) -> TravelChatBotState:
    """
    ACTIVITIES specialist.

    Uses Bedrock LLM (with ACTIVITIES_TOOLS bound) to generate or update
    a structured activities_plan.

    Reads:
      - user_input
      - trip_info
      - preferences
      - existing activities_plan
      - activities_tools_results

    Writes:
      - activities_plan
      - metadata['activities_needs_tools'] (bool)
    """
    logger.debug("activities_agent: entered")

    user_query = (state.get("user_input") or "").strip()
    trip_info = cast(dict, state.get("trip_info") or {})
    preferences = cast(dict, state.get("preferences") or {})
    existing_plan = state.get("activities_plan") or {}
    tool_results = state.get("activities_tools_results") or []

    llm_input = {
        "user_query": user_query,
        "trip_info": trip_info,
        "preferences": preferences,
        "existing_plan": existing_plan,
        "tool_results": tool_results,
    }

    llm_user_prompt = json.dumps(llm_input)

    raw = call_llm(
        system_prompt=ACTIVITIES_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        max_tokens=500,
        temperature=0.4,
        tools=ACTIVITIES_TOOLS,
    )

    logger.debug("activities_agent: raw LLM output: %s", raw)

    # Fallback if parsing fails
    fallback = {
        "activities_plan": existing_plan,
        "needs_tools": False,
    }

    try:
        parsed = json.loads(raw) if raw else fallback
    except Exception as e:
        logger.exception("activities_agent: failed to parse JSON: %s", e)
        parsed = fallback

    activities_plan = parsed.get("activities_plan") or {}
    needs_tools = bool(parsed.get("needs_tools", False))

    state["activities_plan"] = activities_plan

    metadata = state.get("metadata") or {}
    metadata["activities_needs_tools"] = needs_tools
    state["metadata"] = metadata

    items = (activities_plan or {}).get("items", []) or []
    logger.info(
        "activities_agent: tools_bound=ACTIVITIES_TOOLS, needs_tools=%s, items_count=%s",
        needs_tools,
        len(items),
    )
    return state


def logistics_agent(state: TravelChatBotState) -> TravelChatBotState:
    """
    LOGISTICS specialist.

    Uses Bedrock LLM (with LOGISTICS_TOOLS bound) to generate or update
    a structured logistics_plan.

    Reads:
      - user_input
      - trip_info
      - preferences
      - existing logistics_plan
      - logistics_tools_results

    Writes:
      - logistics_plan
      - metadata['logistics_needs_tools'] (bool)
    """
    logger.debug("logistics_agent: entered")

    user_query = (state.get("user_input") or "").strip()
    trip_info = cast(dict, state.get("trip_info") or {})
    preferences = cast(dict, state.get("preferences") or {})
    existing_plan = state.get("logistics_plan") or {}
    tool_results = state.get("logistics_tools_results") or []

    llm_input = {
        "user_query": user_query,
        "trip_info": trip_info,
        "preferences": preferences,
        "existing_plan": existing_plan,
        "tool_results": tool_results,
    }

    llm_user_prompt = json.dumps(llm_input)

    raw = call_llm(
        system_prompt=LOGISTICS_SYSTEM_PROMPT,
        user_prompt=llm_user_prompt,
        max_tokens=500,
        temperature=0.4,
        tools=LOGISTICS_TOOLS,
    )

    logger.debug("logistics_agent: raw LLM output: %s", raw)

    fallback = {
        "logistics_plan": existing_plan,
        "needs_tools": False,
    }

    try:
        parsed = json.loads(raw) if raw else fallback
    except Exception as e:
        logger.exception("logistics_agent: failed to parse JSON: %s", e)
        parsed = fallback

    logistics_plan = parsed.get("logistics_plan") or {}
    needs_tools = bool(parsed.get("needs_tools", False))

    state["logistics_plan"] = logistics_plan

    metadata = state.get("metadata") or {}
    metadata["logistics_needs_tools"] = needs_tools
    state["metadata"] = metadata

    legs = (logistics_plan or {}).get("legs", []) or []
    logger.info(
        "logistics_agent: tools_bound=LOGISTICS_TOOLS, needs_tools=%s, legs_count=%s",
        needs_tools,
        len(legs),
    )
    return state
