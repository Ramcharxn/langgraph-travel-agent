# src/states.py
"""
Shared state definitions for the travel assistant.

These TypedDicts describe exactly what each agent should read/write,
so that they pass compact JSON between them instead of long text.
"""

from typing import TypedDict, Literal, Optional, List, Dict, Any


class UserPreferences(TypedDict, total=False):
    budget_level: Optional[Literal["low", "medium", "high"]]
    accommodation_type: Optional[str]          # "hostel", "hotel", "airbnb", etc.
    diet: Optional[str]                        # "vegetarian", "vegan", "halal", etc.
    transport_preference: Optional[str]        # "public", "taxi", "walk", etc.
    pace: Optional[Literal["slow", "medium", "fast"]]
    interests: Optional[List[str]]             # ["food", "museums", "hiking"]
    with_kids: Optional[bool]
    language: Optional[str]


class TripInfo(TypedDict, total=False):
    origin: Optional[str]
    destination: Optional[str]
    start_date: Optional[str]      # "YYYY-MM-DD"
    end_date: Optional[str]        # "YYYY-MM-DD"
    travel_month: Optional[str]    # e.g. "March"
    num_days: Optional[int]
    country: Optional[str]
    timezone: Optional[str]


class ActivityItem(TypedDict, total=False):
    day: Optional[int]                 # relative ordering index (1, 2, 3, ...)
    time_of_day: Optional[str]         # "morning", "afternoon", "evening", etc.
    title: Optional[str]
    category: Optional[str]            # "event", "sightseeing", "food", etc.
    address: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    date: Optional[str]                # "2025-03-10" or None
    start_time: Optional[str]          # "19:30" or "19:30:00" or None
    duration_hours: Optional[float]
    approx_cost: Optional[float]
    currency: Optional[str]
    notes: Optional[str]


class ActivitiesPlan(TypedDict, total=False):
    items: List[ActivityItem]


class LogisticsLeg(TypedDict, total=False):
    day: Optional[int]                 # relative ordering / day index
    mode: Optional[str]                # "flight", "train", "taxi", "metro", etc.
    from_place: Optional[str]
    to_place: Optional[str]
    duration_hours: Optional[float]
    distance_km: Optional[float]
    departure_time: Optional[str]      # "2025-03-10T09:30" or local time
    arrival_time: Optional[str]
    carrier: Optional[str]             # airline, train operator, etc.
    price: Optional[float]
    currency: Optional[str]
    booking_link: Optional[str]
    notes: Optional[str]


class LogisticsPlan(TypedDict, total=False):
    legs: List[LogisticsLeg]


class ToolResult(TypedDict, total=False):
    tool_name: Optional[str]
    raw_output: Optional[str]
    parsed: Optional[Dict[str, Any]]


class TravelChatBotState(TypedDict, total=False):
    # Latest user message
    user_input: Optional[str]

    # High-level intent label
    intent: Optional[
        Literal["generic_chat", "activities_q", "logistics_q", "plan_full"]
    ]

    # Master agent's natural-language answer for this turn
    master_message: Optional[str]

    # Structured trip information + user preferences
    trip_info: Optional[TripInfo]
    preferences: Optional[UserPreferences]

    # Full JSON object returned by the MASTER agent
    master_plan: Optional[Dict[str, Any]]

    # Plans produced by specialists
    activities_plan: Optional[ActivitiesPlan]
    logistics_plan: Optional[LogisticsPlan]

    # Raw tool outputs if you ever want to store them
    activities_tools_results: Optional[List[ToolResult]]
    logistics_tools_results: Optional[List[ToolResult]]

    # Short running summary of the conversation
    history_summary: Optional[str]

    # Legacy / optional list form if you ever need it
    history_summaries: Optional[List[str]]

    # Any extra scratchpad data for routing, flags, etc.
    metadata: Optional[Dict[str, Any]]

    # Where the master decided to route this turn
    master_route: Optional[Literal["activities", "logistics", "master_response"]]
