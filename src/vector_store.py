import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.ingestion import ingest_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

VECTOR_STORE_DIR = PROJECT_ROOT / "chroma_db"


def get_embeddings():
    """Create the embedding model."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from .env")

    model_name = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )

    return OpenAIEmbeddings(model=model_name)


def create_vector_store():
    """Ingest documents and store their embeddings in Chroma."""
    chunks = ingest_documents()

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )

    # Give every chunk a stable ID to avoid duplicates on reruns.
    ids = []

    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        page = chunk.metadata.get("page", 0)
        start_index = chunk.metadata.get("start_index", 0)

        ids.append(f"{source}:{page}:{start_index}")

    # Re-running this with unchanged files updates existing IDs.
    vector_store.add_documents(documents=chunks, ids=ids)

    print(f"Chunks processed: {len(chunks)}")
    print(f"Vector database location: {VECTOR_STORE_DIR}")

    return vector_store


def load_vector_store():
    """Open an existing vector database."""
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            "Vector database not found. Run create_vector_store() first."
        )

    return Chroma(
        collection_name="knowledge_base",
        embedding_function=get_embeddings(),
        persist_directory=str(VECTOR_STORE_DIR),
    )


if __name__ == "__main__":
    create_vector_store()