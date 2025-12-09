# src/llm/__init__.py

"""
LLM utilities for the travel assistant.

Currently exposes:
- base_llm: the shared ChatBedrockConverse instance
- call_llm: thin convenience wrapper for invoking the LLM
"""

from .bedrock_client import base_llm, call_llm

__all__ = ["base_llm", "call_llm"]
