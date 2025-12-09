# src/main.py

import logging

from src.graph import build_graph
from src.states import TravelChatBotState


def run_cli() -> None:
    """
    Simple command-line interface for the travel assistant.

    Uses a single LangGraph app instance with a fixed thread_id so that
    conversation state (history_summary, trip_info, etc.) is preserved
    across turns via the MemorySaver checkpointer.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = build_graph()
    thread_id = "cli-session"

    print("Travel Assistant")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Assistant: Goodbye! Safe travels ✈️")
            break

        # Minimal initial state each turn; the rest is loaded by the checkpointer.
        state: TravelChatBotState = {
            "user_input": user_input,
        }

        result_state = app.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        reply = result_state.get("master_message", "") or ""
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    run_cli()
