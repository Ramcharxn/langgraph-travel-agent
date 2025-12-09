# src/tools/logistics_rag.py

import logging
from typing import List

from langchain_core.tools import tool
from langchain_core.documents import Document

from rag import flight_retriever

logger = logging.getLogger(__name__)


def _format_flight_docs(docs: List[Document]) -> str:
    """
    Turn retrieved flight documents into a human-readable string.

    Each document's metadata is expected to have:
      - carrier_name, carrier
      - airport_name, airport
      - year, month
    and the page_content already contains a short description of
    on-time performance metrics.
    """
    lines: List[str] = []

    for d in docs:
        meta = d.metadata or {}
        carrier_name = meta.get("carrier_name") or meta.get("carrier") or "Unknown carrier"
        carrier = meta.get("carrier") or ""
        airport_name = meta.get("airport_name") or meta.get("airport") or "Unknown airport"
        airport = meta.get("airport") or ""
        year = meta.get("year")
        month = meta.get("month")

        prefix = f"{carrier_name}"
        if carrier:
            prefix += f" ({carrier})"
        prefix += " at "

        prefix += f"{airport_name}"
        if airport:
            prefix += f" ({airport})"

        if year and month:
            prefix += f" in {year}-{int(month):02d}"

        lines.append(f"{prefix}: {d.page_content}")

    return "\n\n".join(lines)


@tool
def logistics_rag_tool(query: str) -> str:
    """
    Retrieve historical flight on-time / delay statistics using the RAG index.

    Parameters
    ----------
    query : str
        Free-text question describing the route or airline you care about.
        Examples:
          - "on-time performance of Indigo flights from BOM to DEL"
          - "which airline is most reliable from Chennai to Mumbai?"
          - "delays for Air India flights in January"

    Behavior
    --------
    - Uses a FAISS-based retriever (built from your flights CSV) to fetch
      the top-k relevant chunks.
    - Returns a plain text description of the matched records, formatted
      for the LLM to read and use in its reasoning.

    Returns
    -------
    str
        Human-readable summary of matching stats, or a message indicating
        nothing was found.
    """
    logger.info("logistics_rag_tool: query=%r", query)

    try:
        docs = flight_retriever.get_relevant_documents(query)
    except Exception as e:
        logger.exception("logistics_rag_tool: error retrieving docs: %s", e)
        return "There was an error retrieving flight statistics for your query."

    if not docs:
        return "No matching flight statistics were found for your query."

    return _format_flight_docs(docs)


# The list of tools the logistics_agent binds to its LLM.
LOGISTICS_TOOLS = [logistics_rag_tool]
