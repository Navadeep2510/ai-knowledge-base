from src.vector_store import load_vector_store


def search_documents(question: str):
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        question,
        k=3,
    )

    for document, score in results:
        print(f"\nSource: {document.metadata.get('source')}")
        print(f"Page: {document.metadata.get('page')}")
        print(f"Distance: {score}")
        print(f"Content: {document.page_content[:300]}")


if __name__ == "__main__":
    question = input("Ask a question about your PDFs: ")
    search_documents(question)