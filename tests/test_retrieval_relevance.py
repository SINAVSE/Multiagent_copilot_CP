
import pytest

from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.reranker import CrossEncoderReranker

EVALUATION_CASES = [
    (
        "GITLAB-0001",
        "What are the rules for booking domestic flights "
        "and selecting airfare class of service?",
    ),
    (
        "GITLAB-0002",
        "What guidelines should team members follow when "
        "spending company money?",
    ),
    (
        "GITLAB-0003",
        "What lodging and meals and incidental expense "
        "per diem rates apply to sponsored travel?",
    ),
    (
        "GITLAB-0004",
        "What are the red flags when engaging third parties "
        "that could indicate bribery or corruption?",
    ),
    (
        "GITLAB-0005",
        "How should employees report suspected fraud, "
        "and what confidentiality protections apply?",
    ),
    (
        "GITLAB-0006",
        "What are the rules for accepting gifts and "
        "entertainment from external parties?",
    ),
    (
        "GITLAB-0007",
        "What are the eligibility requirements and pay "
        "provisions for GitLab parental leave?",
    ),
    (
        "GITLAB-0008",
        "How much advance notice should employees provide "
        "for planned time away, and how should they "
        "prepare work handovers?",
    ),
    (
        "GITLAB-0009",
        "What are the rules for flexible paid time off, "
        "including planning, coordination and entering "
        "PTO in Workday?",
    ),
]

@pytest.fixture(scope="module")
def retrieval_pipeline():
    return (
        HybridRetriever(),
        CrossEncoderReranker(),
    )

@pytest.mark.parametrize(
    "expected_document_id, query",
    EVALUATION_CASES,
)
def test_expected_document_appears_in_top_five(
    retrieval_pipeline,
    expected_document_id,
    query,
):
    retriever, reranker = retrieval_pipeline

    hybrid_results = retriever.retrieve(
        query=query,
        top_k=10,
        candidate_k=10,
    )

    reranked_results = reranker.rerank(
        query=query,
        retrieval_result=hybrid_results,
        top_k=5,
    )

    retrieved_documents = [
        item.chunk.document_id
        for item in reranked_results.retrieved_chunks
    ]

    print(f"\nQuery: {query}")
    print(f"Expected: {expected_document_id}")

    for rank, item in enumerate(
        reranked_results.retrieved_chunks,
        start=1,
    ):
        print(
            f"\nRank: {rank}"
            f"\nDocument: {item.chunk.document_id}"
            f"\nSection: {item.chunk.section_path}"
            f"\nChunk ID: {item.chunk.chunk_id}"
            f"\nText: {item.chunk.text[:300]}"
        )

    assert expected_document_id in retrieved_documents, (
        f"Expected {expected_document_id} was not found "
        f"in the top five results for: {query}"
    )


def test_planned_time_away_handover_section_is_retrieved(
    retrieval_pipeline,
):
    retriever, reranker = retrieval_pipeline

    query = (
        "Before planned time away, how should employees "
        "document their work, reassign in-progress tasks, "
        "and arrange a backup or handover?"
    )

    hybrid_results = retriever.retrieve(
        query=query,
        top_k=10,
        candidate_k=10,
    )

    reranked_results = reranker.rerank(
        query=query,
        retrieval_result=hybrid_results,
        top_k=5,
    )

    sections = [
        item.chunk.section_path or ""
        for item in reranked_results.retrieved_chunks
    ]

    print("\nHandover query:", query)
    print("Retrieved sections:")
    for section in sections:
        print(section)

    assert any(
        "Work Transition" in section
        or "Documentation Tips" in section
        or "Documentation and Iteration" in section
        for section in sections
    ), "Relevant planned-time-away handover section was not retrieved."