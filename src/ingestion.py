from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def load_documents():
    """Load all PDF files from the documents directory."""
    documents = []

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_path.name

        documents.extend(pages)

    return documents


def split_documents(documents):
    """Split documents into smaller chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    return splitter.split_documents(documents)


def ingest_documents():
    """Load PDFs and split their contents into chunks."""
    documents = load_documents()
    chunks = split_documents(documents)

    print(f"Documents loaded: {len(documents)} pages")
    print(f"Chunks created: {len(chunks)}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0].page_content[:500])
        print("\nMetadata:")
        print(chunks[0].metadata)

    return chunks


if __name__ == "__main__":
    ingest_documents()