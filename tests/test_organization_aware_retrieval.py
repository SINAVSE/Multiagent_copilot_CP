
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.rag.generation.answer_generator import (
    AnswerGenerator,
)


def make_result(*organizations):

    return SimpleNamespace(
        retrieved_chunks=[
            SimpleNamespace(
                chunk=SimpleNamespace(
                    organization=organization
                )
            )
            for organization in organizations
        ],
        retrieval_method="hybrid",
    )


def make_generator(result):

    generator = AnswerGenerator.__new__(
        AnswerGenerator
    )

    generator.top_k = 5
    generator.candidate_k = 10

    generator.retriever = Mock()
    generator.reranker = Mock()
    generator.model = Mock()

    generator.retriever.retrieve.return_value = result
    generator.reranker.rerank.return_value = result
    generator.prepare_evidence = Mock(return_value=result)

    return generator


@pytest.mark.parametrize(
    "organization",
    [
        "GitLab",
        "Rice University",
        "University of Arizona",
        "University of Utah",
        "University of Connecticut",
        "University of Colorado System",
        "University of Washington",
    ],
)
def test_scope_filter_forwarded(organization):

    result = make_result(organization)

    generator = make_generator(result)

    scope = PolicyScope(organization)

    returned = generator.retrieve(
        "What is the travel policy?",
        policy_scope=scope,
    )

    assert returned is result

    generator.retriever.retrieve.assert_called_once_with(
        query="What is the travel policy?",
        top_k=10,
        candidate_k=10,
        where={"organization": organization},
    )


def test_legacy_retrieval_arguments():

    generator = make_generator(
        make_result("GitLab")
    )

    generator.retrieve("Travel policy")

    generator.retriever.retrieve.assert_called_once_with(
        query="Travel policy",
        top_k=10,
        candidate_k=10,
    )


def test_cross_organization_candidate_rejected():

    generator = make_generator(
        make_result("GitLab", "Rice University")
    )

    with pytest.raises(PolicyScopeError):

        generator.retrieve(
            "Travel policy",
            policy_scope=PolicyScope("GitLab"),
        )

    generator.reranker.rerank.assert_not_called()

    generator.model.invoke.assert_not_called()


def test_reranker_contamination_rejected():

    generator = make_generator(
        make_result("GitLab")
    )

    generator.reranker.rerank.return_value = (
        make_result("Rice University")
    )

    with pytest.raises(PolicyScopeError):

        generator.retrieve(
            "Travel policy",
            policy_scope=PolicyScope("GitLab"),
        )

    generator.prepare_evidence.assert_not_called()


def test_prepared_evidence_contamination_rejected():

    generator = make_generator(
        make_result("GitLab")
    )

    generator.prepare_evidence.return_value = (
        make_result("Rice University")
    )

    with pytest.raises(PolicyScopeError):

        generator.retrieve(
            "Travel policy",
            policy_scope=PolicyScope("GitLab"),
        )


def test_direct_evidence_cannot_bypass_scope():

    generator = make_generator(
        make_result("GitLab")
    )

    with pytest.raises(PolicyScopeError):

        generator.generate(
            query="Travel policy",
            retrieval_result=make_result(
                "Rice University"
            ),
            policy_scope=PolicyScope("GitLab"),
        )

    generator.prepare_evidence.assert_not_called()

    generator.model.invoke.assert_not_called()


def test_missing_metadata_rejected():

    generator = make_generator(
        make_result(None)
    )

    with pytest.raises(PolicyScopeError):

        generator.retrieve(
            "Travel policy",
            policy_scope=PolicyScope("GitLab"),
        )


def test_invalid_scope_rejected_before_retrieval():

    generator = make_generator(
        make_result("GitLab")
    )

    with pytest.raises(PolicyScopeError):

        generator.retrieve(
            "Travel policy",
            policy_scope={"organization": "GitLab"},
        )

    generator.retriever.retrieve.assert_not_called()


def test_empty_scoped_evidence_returns_safe_answer():

    generator = make_generator(make_result())

    answer = generator.generate(
        query="Travel policy",
        policy_scope=PolicyScope("GitLab"),
    )

    assert answer.insufficient_evidence is True

    assert answer.citations == []

    assert answer.retrieved_evidence == []

    generator.model.invoke.assert_not_called()


def test_direct_evidence_checked_after_preparation():

    generator = make_generator(
        make_result("GitLab")
    )

    generator.prepare_evidence.return_value = (
        make_result("Rice University")
    )

    with pytest.raises(PolicyScopeError):

        generator.generate(
            query="Travel policy",
            retrieval_result=make_result("GitLab"),
            policy_scope=PolicyScope("GitLab"),
        )

    generator.model.invoke.assert_not_called()
