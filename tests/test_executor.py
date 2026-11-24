
from types import SimpleNamespace
from unittest.mock import patch

from src.agents.executor import Executor


def test_executor_retrieves_policy_before_calculation():
    """Executor must retrieve policy even when only calculation is selected."""

    fake_evidence = [
        {
            "chunk_id": "POLICY-001",
            "title": "Travel Expense Policy",
            "section": "Meal Allowance",
            "text": "The domestic meal allowance is INR 1800 per day.",
        }
    ]

    # Executor expects an object with attributes, not a dictionary.
    fake_rag_result = SimpleNamespace(
        answer="The domestic meal allowance is INR 1800 per day.",
        insufficient_evidence=False,
        missing_information=[],
        citations=[],
        retrieved_evidence=fake_evidence,
        retrieval_method="hybrid",
    )

    # Policy extraction also returns an object with attributes.
    fake_policy = SimpleNamespace(
        limit_type="fixed",
        policy_limit=1800,
        currency="INR",
        limit_unit="per_day",
        approval_required=False,
        evidence_quotes=[],
        missing_information=[],
    )

    with (
        patch(
            "src.agents.executor.AnswerGenerator"
        ) as mock_answer_generator,
        patch(
            "src.agents.executor.extract_expense_policy",
            return_value=fake_policy,
        ) as mock_extract_policy,
    ):
        mock_answer_generator.return_value.generate.return_value = (
            fake_rag_result
        )

        # Initialize Executor after mocking AnswerGenerator.
        executor = Executor()

        result = executor.execute(
            selected_tools=["calculate_expense_eligibility"],
            query="I spent INR 2500 on meals. Is it within policy?",
            expense_details={
                "claimed_amount": 2500,
            },
        )

    # Verify policy retrieval happened.
    mock_answer_generator.return_value.generate.assert_called_once()

    # Verify extraction received retrieved evidence.
    mock_extract_policy.assert_called_once()
    extraction_args = mock_extract_policy.call_args
    assert fake_evidence == extraction_args.kwargs["retrieved_evidence"]

    # Verify calculation.
    calculation = result["calculate_expense_eligibility"]

    assert calculation["compliance_status"] == "exceeds_policy"
    assert calculation["claimed_amount"] == 2500
    assert calculation["policy_limit"] == 1800
    assert calculation["eligible_amount"] == 1800
    assert calculation["excess_amount"] == 700
    assert calculation["currency"] == "INR"