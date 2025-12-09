# src/tools/events.py

import json
import logging
import os
from typing import Any, Dict, List

import requests
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Read Ticketmaster API key from environment.
# DO NOT commit your real key to GitHub.
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY")

TICKETMASTER_BASE_URL = "https://app.ticketmaster.com/discovery/v2/events.json"


def _simplify_ticketmaster_events(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert the Ticketmaster response into a simple list of dicts:
    [
      {
        "name": str | None,
        "venue": str | None,
        "city": str | None,
        "country": str | None,
        "date": str | None,    # ISO local date
        "time": str | None,    # local time (HH:MM:SS) or None
        "lat": float | None,
        "lon": float | None,
      },
      ...
    ]
    """
    results: List[Dict[str, Any]] = []

    events = (
        data.get("_embedded", {})
        .get("events", [])
    )

    for ev in events:
        name = ev.get("name")

        dates = ev.get("dates", {}).get("start", {}) or {}
        local_date = dates.get("localDate")
        local_time = dates.get("localTime")

        venues = (
            ev.get("_embedded", {})
            .get("venues", [])
        ) or []

        venue_name = None
        city = None
        country = None
        lat = None
        lon = None

        if venues:
            v = venues[0]
            venue_name = v.get("name")
            city = (v.get("city") or {}).get("name")
            country = (v.get("country") or {}).get("name")

            loc = v.get("location") or {}
            try:
                lat = float(loc.get("latitude")) if loc.get("latitude") else None
                lon = float(loc.get("longitude")) if loc.get("longitude") else None
            except (TypeError, ValueError):
                lat, lon = None, None

        results.append(
            {
                "name": name,
                "venue": venue_name,
                "city": city,
                "country": country,
                "date": local_date,
                "time": local_time,
                "lat": lat,
                "lon": lon,
            }
        )

    return results


@tool
def activities_events_tool(query: str) -> str:
    """
    Search upcoming events (concerts, festivals, sports, etc.) via Ticketmaster.

    Parameters
    ----------
    query : str
        A short free-text description of what the user wants.
        Examples:
          - "big events in Mumbai next few months"
          - "concerts in March near Mumbai"
          - "sports events in India in February"

    Behavior
    --------
    - Uses the Ticketmaster Discovery API to search for matching events.
    - Currently biases to India (countryCode='IN') because most examples
      are Mumbai-focused; you can adjust as needed.
    - Returns a JSON string with:
        {
          "tool": "activities_events_tool",
          "params_used": { ... },
          "results": [
            {
              "name": str | null,
              "venue": str | null,
              "city": str | null,
              "country": str | null,
              "date": str | null,   // local date, e.g. "2025-03-10"
              "time": str | null,   // local time, e.g. "19:30:00"
              "lat": float | null,
              "lon": float | null
            },
            ...
          ]
        }

    The activities_agent should parse this JSON and map fields into its
    activities_plan schema (title, address, date, start_time, lat/lon, etc.).
    """

    logger.info("activities_events_tool: query=%r", query)

    if not TICKETMASTER_API_KEY:
        logger.warning(
            "activities_events_tool: TICKETMASTER_API_KEY not set, returning empty results."
        )
        payload = {
            "tool": "activities_events_tool",
            "params_used": {},
            "results": [],
            "error": "TICKETMASTER_API_KEY not configured",
        }
        return json.dumps(payload)

    # Basic params; adjust as needed (city, radius, etc.).
    params: Dict[str, Any] = {
        "apikey": TICKETMASTER_API_KEY,
        "keyword": query,
        "size": 10,
        "sort": "date,asc",
        "countryCode": "IN",  # bias events to India; change if you want global
        "locale": "*",
    }

    try:
        resp = requests.get(TICKETMASTER_BASE_URL, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.exception("activities_events_tool: error calling Ticketmaster: %s", e)
        payload = {
            "tool": "activities_events_tool",
            "params_used": params,
            "results": [],
            "error": f"Ticketmaster API error: {e}",
        }
        return json.dumps(payload)

    simplified = _simplify_ticketmaster_events(data)

    payload = {
        "tool": "activities_events_tool",
        "params_used": {k: v for k, v in params.items() if k != "apikey"},
        "results": simplified,
    }

    return json.dumps(payload)


# The list of tools the activities_agent binds to its LLM.
ACTIVITIES_TOOLS = [activities_events_tool]
