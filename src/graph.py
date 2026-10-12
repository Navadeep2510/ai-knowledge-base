from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.vector_store import load_vector_store


class RAGState(TypedDict):
    question: str
    documents: list
    context_found: bool
    answer: str
    sources: list


def retrieve(state: RAGState):
    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        state["question"],
        k=3,
    )

    return {"documents": documents}


def check_context(state: RAGState):
    documents = state["documents"]

    # Basic check: determine whether any text was retrieved.
    context_found = any(
        doc.page_content.strip() for doc in documents
    )

    return {"context_found": context_found}


def generate_answer(state: RAGState):
    documents = state["documents"]

    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}, "
        f"Page: {doc.metadata.get('page', 'Unknown')}\n"
        f"{doc.page_content}"
        for doc in documents
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Answer the question using only the context below.
If the context does not contain the answer, say so.
Do not invent facts.

Context:
{context}"""
        ),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": state["question"],
    })

    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
        }
        for doc in documents
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }


def insufficient_context(state: RAGState):
    return {
        "answer": "I couldn't find relevant document content to answer that question.",
        "sources": [],
    }


def route_after_check(state: RAGState):
    if state["context_found"]:
        return "generate_answer"

    return "insufficient_context"


builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve)
builder.add_node("check_context", check_context)
builder.add_node("generate_answer", generate_answer)
builder.add_node("insufficient_context", insufficient_context)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "check_context")

builder.add_conditional_edges(
    "check_context",
    route_after_check,
    {
        "generate_answer": "generate_answer",
        "insufficient_context": "insufficient_context",
    },
)

builder.add_edge("generate_answer", END)
builder.add_edge("insufficient_context", END)

rag_graph = builder.compile()


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.strip().lower() == "exit":
            break

        if not question.strip():
            continue

        result = rag_graph.invoke({
            "question": question,
            "documents": [],
            "context_found": False,
            "answer": "",
            "sources": [],
        })

        print("\nAnswer:\n", result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(source)