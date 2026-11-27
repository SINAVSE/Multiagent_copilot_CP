from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.agents.executor import Executor
from src.agents.errors import (
    ToolExecutionError,
    ErrorCode,
    classify_error,
)


QUERY = "Can I claim INR 2500 for meals?"

DETAILS = {
    "claimed_amount": 2500,
    "currency": "INR",
    "headcount": 3,
}

EXPENSE_TOOLS = [
    "search_hr_policy",
    "calculate_expense_eligibility",
]


@pytest.fixture
def generator():

    instance = Mock()

    chunk = SimpleNamespace(
        chunk_id="POLICY-001",
        title="Travel Expense Policy",
        section_path="Meal Allowance",
        section_heading="Meal Allowance",
        text="The meal allowance is INR 1800.",
    )

    instance.retrieve.return_value = SimpleNamespace(
        retrieved_chunks=[
            SimpleNamespace(chunk=chunk)
        ],
        retrieval_method="hybrid",
    )

    instance.generate.return_value = SimpleNamespace(
        answer="The meal allowance is INR 1800.",
        insufficient_evidence=False,
        missing_information=[],
        retrieval_method="hybrid",
        citations=[],
        retrieved_evidence=[
            {
                "chunk_id": "POLICY-001",
                "title": "Travel Expense Policy",
                "section": "Meal Allowance",
                "text": "The meal allowance is INR 1800.",
            }
        ],
    )

    return instance


def make_policy(limit_type="fixed"):

    return SimpleNamespace(
        limit_type=limit_type,
        policy_limit=1800,
        currency="INR",
        approval_required=False,
        evidence_quotes=[],
        missing_information=[],
    )


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("Retrieval timed out"),
        ConnectionError("Service unavailable"),
        RuntimeError("Database unavailable"),
    ],
)
def test_retrieval_failure_stops_execution(
    generator,
    failure,
):

    generator.retrieve.side_effect = failure

    executor = Executor(
        answer_generator=generator
    )

    with pytest.raises(ToolExecutionError) as error:

        executor.execute(
            selected_tools=EXPENSE_TOOLS,
            query=QUERY,
            expense_details=DETAILS,
        )

    assert error.value.tool_name == "search_hr_policy"
    assert error.value.__cause__ is failure

    generator.retrieve.assert_called_once()
    generator.generate.assert_not_called()


def test_unexpected_policy_extraction_failure(generator):

    executor = Executor(
        answer_generator=generator
    )

    with patch(
        "src.agents.executor.extract_expense_policy",
        side_effect=TimeoutError("Extraction timeout"),
    ):

        with pytest.raises(ToolExecutionError) as error:

            executor.execute(
                selected_tools=EXPENSE_TOOLS,
                query=QUERY,
                expense_details=DETAILS,
            )

    assert error.value.tool_name == (
        "extract_expense_policy"
    )

    generator.generate.assert_not_called()


@pytest.mark.parametrize(
    "failure_type",
    [ValueError, TypeError],
)
def test_policy_validation_fallback_is_preserved(
    generator,
    failure_type,
):

    executor = Executor(
        answer_generator=generator
    )

    with patch(
        "src.agents.executor.extract_expense_policy",
        side_effect=failure_type("Invalid policy evidence"),
    ):

        result = executor.execute(
            selected_tools=EXPENSE_TOOLS,
            query=QUERY,
            expense_details=DETAILS,
        )

    expense = result[
        "calculate_expense_eligibility"
    ]

    assert expense["compliance_status"] == (
        "unable_to_determine"
    )

    assert expense["policy_limit"] is None
    assert expense["approval_required"] is None

    generator.generate.assert_called_once()


def test_fixed_calculation_failure(generator):

    executor = Executor(
        answer_generator=generator
    )

    with (
        patch(
            "src.agents.executor.extract_expense_policy",
            return_value=make_policy("fixed"),
        ),
        patch(
            "src.agents.executor.calculate_expense",
            side_effect=RuntimeError("Calculation failed"),
        ),
    ):

        with pytest.raises(ToolExecutionError) as error:

            executor.execute(
                selected_tools=EXPENSE_TOOLS,
                query=QUERY,
                expense_details=DETAILS,
            )

    assert error.value.tool_name == (
        "calculate_expense_eligibility"
    )

    generator.generate.assert_not_called()


def test_per_person_calculation_failure(generator):

    executor = Executor(
        answer_generator=generator
    )

    with (
        patch(
            "src.agents.executor.extract_expense_policy",
            return_value=make_policy("per_person"),
        ),
        patch(
            "src.agents.executor.calculate_per_person_expense",
            side_effect=RuntimeError("Calculation failed"),
        ),
    ):

        with pytest.raises(ToolExecutionError) as error:

            executor.execute(
                selected_tools=EXPENSE_TOOLS,
                query=QUERY,
                expense_details=DETAILS,
            )

    assert error.value.tool_name == (
        "calculate_expense_eligibility"
    )

    generator.generate.assert_not_called()


def test_answer_generation_failure(generator):

    generator.generate.side_effect = (
        TimeoutError("Answer generation timeout")
    )

    executor = Executor(
        answer_generator=generator
    )

    with pytest.raises(ToolExecutionError) as error:

        executor.execute(
            selected_tools=["search_hr_policy"],
            query=QUERY,
        )

    assert error.value.tool_name == (
        "generate_policy_answer"
    )

    generator.retrieve.assert_called_once()
    generator.generate.assert_called_once()


@pytest.mark.parametrize(
    "has_file,file_path",
    [
        (False, None),
        (True, "mock_receipt.pdf"),
    ],
)
def test_unavailable_receipt_tool_fails_safely(
    generator,
    has_file,
    file_path,
):

    executor = Executor(
        answer_generator=generator
    )

    with pytest.raises(ToolExecutionError) as error:

        executor.execute(
            selected_tools=[
                "extract_bill_details",
                "search_hr_policy",
            ],
            query="Analyze my receipt",
            has_file=has_file,
            file_path=file_path,
        )

    assert error.value.tool_name == (
        "extract_bill_details"
    )

    generator.retrieve.assert_not_called()
    generator.generate.assert_not_called()


def test_unknown_tool_never_executes(generator):

    executor = Executor(
        answer_generator=generator
    )

    with pytest.raises(ValueError, match="Unknown tools"):

        executor.execute(
            selected_tools=["unsupported_tool"],
            query=QUERY,
        )

    generator.retrieve.assert_not_called()
    generator.generate.assert_not_called()


def test_tool_exception_does_not_expose_secret(generator):

    secret = "SENSITIVE_TEST_CREDENTIAL"

    generator.retrieve.side_effect = RuntimeError(secret)

    executor = Executor(
        answer_generator=generator
    )

    with pytest.raises(ToolExecutionError) as error:

        executor.execute(
            selected_tools=["search_hr_policy"],
            query=QUERY,
        )

    assert secret not in str(error.value)

    classification = classify_error(error.value)

    assert classification.code == (
        ErrorCode.TOOL_EXECUTION_FAILED
    )

    assert classification.retryable is False
    assert secret not in classification.safe_message


def test_existing_tool_error_is_not_double_wrapped(generator):

    original = ToolExecutionError("search_hr_policy")

    generator.retrieve.side_effect = original

    executor = Executor(
        answer_generator=generator
    )

    with pytest.raises(ToolExecutionError) as error:

        executor.execute(
            selected_tools=["search_hr_policy"],
            query=QUERY,
        )

    assert error.value is original


def test_insufficient_evidence_is_not_technical_failure(generator):

    generator.retrieve.return_value.retrieved_chunks = []

    generator.generate.return_value = SimpleNamespace(
        answer="Insufficient policy evidence.",
        insufficient_evidence=True,
        missing_information=["Applicable policy"],
        retrieval_method="hybrid",
        citations=[],
        retrieved_evidence=[],
    )

    executor = Executor(
        answer_generator=generator
    )

    with patch(
        "src.agents.executor.extract_expense_policy"
    ) as extract_policy:

        result = executor.execute(
            selected_tools=EXPENSE_TOOLS,
            query=QUERY,
            expense_details=DETAILS,
        )

    assert result["calculate_expense_eligibility"][
        "compliance_status"
    ] == "unable_to_determine"

    assert result["search_hr_policy"][
        "insufficient_evidence"
    ] is True

    extract_policy.assert_not_called()
