# Langgraph-Travel-Multi-Agent-Assistant

A **modular, production-style, multi-agent LLM system** built using **Amazon Bedrock** and **LangGraph**, designed for conversational travel planning, real-time event discovery, and logistics reasoning using **tool-calling**, **RAG**, and **persistent conversational memory**.

This project focuses on **agent orchestration and system design**, not just prompt engineering.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Amazon Bedrock](https://img.shields.io/badge/Amazon%20Bedrock-FF9900?style=flat&logo=amazonaws&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-6366F1?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-0EA5E9?style=flat)
![HuggingFace](https://img.shields.io/badge/HuggingFace-F9A826?style=flat&logo=huggingface&logoColor=black)
![RAG](https://img.shields.io/badge/RAG-EC4899?style=flat)
![Multi-Agent](https://img.shields.io/badge/Multi--Agent-10B981?style=flat)
![Vector Search](https://img.shields.io/badge/Vector%20Search-14B8A6?style=flat)

---

## 🔹 Technical Highlights

- **LangGraph-based agent orchestration**
- **Amazon Bedrock as the LLM runtime**
- **Multi-agent architecture with specialists**
- **Tool-augmented LLMs for real-time data**
- **RAG (Retrieval-Augmented Generation)**
- **Context-aware, multi-turn conversation**
- **Persistent conversational memory**

---

## 🎯 Problem Statement

Most travel chatbots either:
- Over-generate rigid itineraries, or
- Fail to reason across tools, memory, and follow-up questions.

This project addresses that gap by building a **stateful, multi-agent LLM system**
that separates intent detection, planning, tool usage, and response synthesis,
resulting in more controllable and extensible travel planning behavior.

---

## 🧠 Agent Flow

- **MASTER Agent**  
  Classifies intent, updates structured trip state, and routes execution via LangGraph.

- **ACTIVITIES Agent**  
  Retrieves and structures events/activities using real-time API tool calls.

- **LOGISTICS Agent**  
  Uses RAG to analyze transport options and historical flight reliability.

- **MASTER_RESPONSE Agent**  
  Synthesizes structured plans into a chat-friendly response, generating itineraries only when requested.

- **HISTORY Agent**  
  Compresses past interactions into a compact memory for multi-turn awareness.

---

## 🧩 Project Structure

```text
.
├── agents/                         # LLM agents (master, specialists, history)
│   ├── master.py
│   ├── specialists.py
│   ├── history.py
│   └── __init__.py
│
├── tools/                          # Tool-calling layer
│   ├── events.py                   # Ticketmaster API integration
│   ├── logistics_rag.py            # RAG-based flight insights
│   └── __init__.py
│
├── rag/                            # Retrieval-Augmented Generation
│   ├── flights_index.py            # FAISS + embeddings for flight data
│   └── __init__.py
│
├── llm/                            # LLM runtime abstraction
│   ├── bedrock_client.py           # Amazon Bedrock wrapper
│   └── __init__.py
│
├── data/
│   └── vectorstores/
│       └── flight_faiss/            # Persisted FAISS index for RAG
│
├── graph.py                        # LangGraph agent orchestration
├── prompts.py                      # All system & agent prompts
├── states.py                       # Typed shared state definitions
├── main.py                         # CLI entrypoint
├── __init__.py
│
├── README.md
└── .gitignore
```

## 🛠️ Technologies Used

- **Python 3.11+**
- **Amazon Bedrock** (`ChatBedrockConverse`)
- **LangGraph** (agent orchestration & state machine)
- **LangChain Tools** (tool calling & integration)
- **FAISS** (vector storage for semantic retrieval)
- **HuggingFace Embeddings**
- **Ticketmaster Discovery API** (real-time event data)
- **Retrieval-Augmented Generation (RAG)**

## 🚀 Future Work

- Web & API Interface
- Persistent User Profiles
- Tool & RAG Caching Layers
- Streaming & Partial Responses
- Advanced Routing & Cost-Aware Planning
- Observability & Tracing
- Evaluation & Guardrails
- Multi-Model Support
