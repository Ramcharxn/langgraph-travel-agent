# src/__init__.py
"""
Travel assistant package.

This package wires together:
- agents (master, specialists, history)
- tools (events, logistics RAG)
- rag (flight stats index)
- llm (Bedrock Chat wrapper)
- graph (LangGraph state machine)
"""

from .graph import build_graph

__all__ = ["build_graph"]
