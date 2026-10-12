from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.vector_store import load_vector_store


def create_rag_chain():
    """Create a RAG chain using the existing vector database."""

    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful question-answering assistant.
Answer using only the provided context.
If the context does not contain the answer, say you
could not find the answer in the provided documents.
Do not invent facts.

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    def format_documents(documents):
        return "\n\n".join(
            f"Source: {doc.metadata.get('source', 'Unknown')}, "
            f"Page: {doc.metadata.get('page', 'Unknown')}\n"
            f"{doc.page_content}"
            for doc in documents
        )

    chain = (
        {
            "context": retriever | format_documents,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    chain = create_rag_chain()

    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.strip().lower() == "exit":
            break

        if not question.strip():
            continue

        answer = chain.invoke(question)
        print("\nAnswer:\n", answer)