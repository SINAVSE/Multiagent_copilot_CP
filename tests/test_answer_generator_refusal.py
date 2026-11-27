
from src.rag.schemas import RetrievalResult

from src.rag.generation.answer_generator import AnswerGenerator


class EmptyRetriever:
    def retrieve(self, **kwargs):
        return RetrievalResult(
            query=kwargs["query"],
            retrieved_chunks=[],
            retrieval_method="hybrid",
            metadata={},
        )


class EmptyReranker:
    def rerank(self, query, retrieval_result, top_k=None):
        return retrieval_result


class ModelMustNotBeCalled:
    def invoke(self, prompt):
        raise AssertionError(
            "LLM should not be called without retrieved evidence."
        )


def test_answer_generator_refuses_without_evidence():
    generator = AnswerGenerator(
        retriever=EmptyRetriever(),
        reranker=EmptyReranker(),
        model=ModelMustNotBeCalled(),
    )

    result = generator.generate(
        "What is GitLab's current stock price?"
    )

    assert result.insufficient_evidence is True
    assert result.citations == []
