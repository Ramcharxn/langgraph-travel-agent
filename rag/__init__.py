# src/rag/__init__.py
"""
RAG utilities for flight on-time performance data.

Exports:
- CSV_PATH, INDEX_DIR
- embeddings
- load_flight_documents
- build_or_load_flight_index
- flight_vectorstore
- flight_retriever
"""

from .flights_index import (
    CSV_PATH,
    INDEX_DIR,
    embeddings,
    load_flight_documents,
    build_or_load_flight_index,
    flight_vectorstore,
    flight_retriever,
)

__all__ = [
    "CSV_PATH",
    "INDEX_DIR",
    "embeddings",
    "load_flight_documents",
    "build_or_load_flight_index",
    "flight_vectorstore",
    "flight_retriever",
]
