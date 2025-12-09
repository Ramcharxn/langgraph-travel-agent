# src/rag/flights_index.py

import logging
import os
from typing import List

import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

# ---- CONFIG ----

# Path to your CSV with aggregated flight on-time stats.
# You can override via environment if needed.
CSV_PATH = os.getenv("FLIGHT_CSV_PATH", "data/flights_ontime.csv")

# Folder where the FAISS index will be stored.
INDEX_DIR = os.getenv("FLIGHT_INDEX_DIR", "data/vectorstores/flight_faiss")

# Single global embedding model (reuse across app)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def load_flight_documents(csv_path: str = CSV_PATH) -> List[Document]:
    """
    Load the flights CSV and convert each row into a LangChain Document.

    Expected columns (loosely, names are handled in a forgiving way):
    - YEAR or year
    - MONTH or month
    - airport / AIRPORT / ORIGIN
    - airport_name / AIRPORT_NAME / ORIGIN_CITY_NAME
    - carrier / CARRIER / OP_UNIQUE_CARRIER
    - carrier_name / CARRIER_NAME / OP_CARRIER_NAME
    - arr_flights / ARR_FLIGHTS (total arriving flights)
    - arr_del15 / ARR_DEL15 (arrivals delayed 15+ minutes)

    The exact dataset schema may vary; this function tries common variants and
    skips rows that do not have enough information.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Flights CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    docs: List[Document] = []

    for _, row in df.iterrows():
        year = int(row.get("YEAR") or row.get("year") or 0)
        month = int(row.get("MONTH") or row.get("month") or 0)

        airport = (
            row.get("airport")
            or row.get("AIRPORT")
            or row.get("ORIGIN")
            or ""
        )
        airport_name = (
            row.get("airport_name")
            or row.get("AIRPORT_NAME")
            or row.get("ORIGIN_CITY_NAME")
            or ""
        )
        carrier = (
            row.get("carrier")
            or row.get("CARRIER")
            or row.get("OP_UNIQUE_CARRIER")
            or ""
        )
        carrier_name = (
            row.get("carrier_name")
            or row.get("CARRIER_NAME")
            or row.get("OP_CARRIER_NAME")
            or ""
        )

        # Basic metrics (best-effort, skip if nothing meaningful)
        arr_flights = (
            row.get("arr_flights")
            or row.get("ARR_FLIGHTS")
            or row.get("arrivals")
        )
        arr_del15 = (
            row.get("arr_del15")
            or row.get("ARR_DEL15")
            or row.get("delayed_15")
        )

        if not year or not month or not airport or not carrier:
            # Not enough context to be useful for QA
            continue

        text_parts: List[str] = []
        if pd.notna(arr_flights):
            try:
                text_parts.append(f"{int(arr_flights)} arriving flights")
            except Exception:
                text_parts.append(f"{arr_flights} arriving flights")

        if pd.notna(arr_del15):
            try:
                text_parts.append(
                    f"{int(arr_del15)} arrivals were delayed 15+ minutes"
                )
            except Exception:
                text_parts.append(
                    f"{arr_del15} arrivals were delayed 15+ minutes"
                )

        if not text_parts:
            # No metrics, nothing interesting to embed
            continue

        content = (
            f"In {year}-{month:02d}, "
            f"{carrier_name or carrier} "
            f"at {airport_name or airport} had "
            + ", ".join(text_parts)
            + "."
        )

        metadata = {
            "year": year,
            "month": month,
            "airport": str(airport),
            "airport_name": str(airport_name),
            "carrier": str(carrier),
            "carrier_name": str(carrier_name),
        }

        docs.append(Document(page_content=content, metadata=metadata))

    logger.info(
        "load_flight_documents: loaded %d CSV rows into %d documents",
        len(df),
        len(docs),
    )
    return docs


def build_or_load_flight_index(
    index_dir: str = INDEX_DIR,
    csv_path: str = CSV_PATH,
) -> FAISS:
    """
    Build or load the FAISS vectorstore for flight on-time performance.

    - If an index already exists at `index_dir`, it will be loaded.
    - Otherwise, the CSV at `csv_path` will be loaded and a new index built
      and saved to disk.
    """
    if os.path.isdir(index_dir) and os.listdir(index_dir):
        logger.info(
            "build_or_load_flight_index: loading existing index from %s",
            index_dir,
        )
        vectorstore = FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore

    logger.info(
        "build_or_load_flight_index: no index found, building new FAISS index "
        "from %s into %s",
        csv_path,
        index_dir,
    )
    os.makedirs(index_dir, exist_ok=True)

    docs = load_flight_documents(csv_path)
    vectorstore = FAISS.from_documents(docs, embeddings)
    logger.info("build_or_load_flight_index: FAISS index built with %d docs", len(docs))

    # Persist to disk
    vectorstore.save_local(index_dir)
    return vectorstore


# Build or load at import time so tools can use flight_retriever immediately
flight_vectorstore = build_or_load_flight_index()
flight_retriever = flight_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
