
"""
CAP-04: Automated functional regression tests.

Tests the actual Employee Operations Copilot workflow
against the frozen regression dataset.

Live tests require hosted LLM and embedding API access.
"""

import json
import os
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]

BASELINE_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "baseline_regression_v1.json"
)

RESULTS_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "regression_execution_results.json"
)


def load_baseline():
    with BASELINE_PATH.open(
        encoding="utf-8-sig"
    ) as file:
        return json.load(file)


BASELINE = load_baseline()
TEST_CASES = BASELINE["test_cases"]


# --------------------------------------------------
# Dataset validation - no API calls
# --------------------------------------------------

def test_baseline_metadata():
    assert BASELINE["metadata"]["frozen"] is True
    assert BASELINE["metadata"]["version"] == "1.0"


def test_baseline_test_count():
    assert len(TEST_CASES) == 9


def test_baseline_unique_test_ids():
    test_ids = [
        case["test_id"]
        for case in TEST_CASES
    ]

    assert len(test_ids) == len(set(test_ids))


@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=[case["test_id"] for case in TEST_CASES],
)
def test_baseline_case_structure(case):
    required_fields = {
        "test_id",
        "category",
        "query",
        "conversation_history",
        "expected",
        "baseline",
    }

    assert required_fields.issubset(case.keys())
    assert case["query"].strip()
    assert isinstance(case["expected"], dict)
    assert isinstance(case["conversation_history"], list)

    assert case["baseline"]["status"] in {
        "PASS",
        "FAIL",
        "PARTIAL",
    }


# --------------------------------------------------
# Live workflow initialization
# --------------------------------------------------

@pytest.fixture(scope="module")
def workflow():
    if os.getenv("RUN_LIVE_REGRESSION") != "1":
        pytest.skip(
            "Set RUN_LIVE_REGRESSION=1 "
            "to enable hosted API tests."
        )

    from src.agents.workflow import EmployeeOpsWorkflow

    return EmployeeOpsWorkflow()


@pytest.fixture(scope="module")
def execution_results():
    results = []

    yield results

    if results:
        RESULTS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with RESULTS_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )


# --------------------------------------------------
# Assertion helpers
# --------------------------------------------------

def contains_amount(answer, amount):
    pattern = (
        rf"(?<![\d.]){amount}"
        rf"(?:\.0+)?(?![\d.])"
    )

    return bool(re.search(pattern, answer))


def evaluate_result(case, result):
    test_id = case["test_id"]
    expected = case["expected"]

    answer = str(
        result.get("final_answer", "")
    ).lower()

    tools = result.get("tool_results", {}) or {}

    search_result = (
        tools.get("search_hr_policy", {}) or {}
    )

    expense = (
        tools.get("calculate_expense_eligibility", {})
        or {}
    )

    evidence = (
        result.get("retrieved_evidence", []) or []
    )

    citations = (
        search_result.get("citations", []) or []
    )

    selected_tools = (
        result.get("selected_tools", []) or []
    )

    checks = {}

    # REG-001 and REG-009:
    # Individual travel meal allowance.
    if test_id in {"REG-001", "REG-009"}:

        checks["correct_amount"] = contains_amount(
            answer,
            expected["policy_amount"],
        )

        checks["correct_limit_unit"] = (
            "per day" in answer
            or "daily" in answer
        )

        checks["citation_present"] = bool(citations)

        checks["answer_approved"] = (
            result.get("is_approved") is True
        )

    # REG-002: Travel-to-leave topic switching.
    elif test_id == "REG-002":

        leave_terms = (
            "emergency leave",
            "parental leave",
            "sick leave",
            "sick time",
            "military leave",
            "time off",
        )

        checks["relevant_leave_answer"] = any(
            term in answer
            for term in leave_terms
        )

        checks["leave_evidence_present"] = any(
            any(
                term in str(item).lower()
                for term in (
                    "leave",
                    "time off",
                    "time away",
                )
            )
            for item in evidence
        )

        checks["answer_approved"] = (
            result.get("is_approved") is True
        )

    # REG-003: Traveling team meal follow-up.
    elif test_id == "REG-003":

        checks["correct_amount"] = contains_amount(
            answer,
            80,
        )

        checks["correct_limit_unit"] = (
            "per person" in answer
            and "per meal" in answer
        )

        checks["no_misleading_daily_limit"] = (
            "daily meal allowance" not in answer
        )

        checks["citation_present"] = bool(citations)

    # REG-004: Out-of-scope refusal.
    elif test_id == "REG-004":

        refusal_terms = (
            "out of scope",
            "outside my scope",
            "outside the scope",
            "only assist",
            "only help",
        )

        checks["explicit_refusal"] = any(
            term in answer
            for term in refusal_terms
        )

        checks["no_policy_retrieval"] = (
            "search_hr_policy" not in tools
            and "search_hr_policy" not in selected_tools
            and not evidence
        )

        checks["no_expense_calculation"] = (
            "calculate_expense_eligibility"
            not in tools
        )

    # REG-005: Ambiguous expense handling.
    elif test_id == "REG-005":

        checks["clarification_present"] = (
            bool(expense.get("missing_information"))
            or "please provide" in answer
            or "please clarify" in answer
            or "could you" in answer
        )

        checks["no_definitive_violation"] = (
            expense.get("compliance_status")
            in {None, "unable_to_determine"}
            and "policy violation" not in answer
            and "exceeds this limit" not in answer
        )

    # REG-006: Claimed amount integrity.
    elif test_id == "REG-006":

        checks["claimed_amount_preserved"] = (
            expense.get("claimed_amount")
            == expected["claimed_amount"]
        )

    # REG-007: Critic consistency.
    elif test_id == "REG-007":

        actual_amount = expense.get(
            "claimed_amount"
        )

        correct_amount = (
            actual_amount == 100
        )

        critic_approved = (
            result.get("is_approved") is True
        )

        unsupported_violation = (
            "policy violation" in answer
            or "exceeds this limit" in answer
        )

        checks["no_approved_inconsistent_amount"] = (
            correct_amount
            or not critic_approved
        )

        checks["no_approved_unsupported_violation"] = (
            not unsupported_violation
            or not critic_approved
        )

    # REG-008: Retrieval ranking.
    elif test_id == "REG-008":

        chunk_ids = [
            item.get("chunk_id")
            for item in evidence
        ]

        relevant_chunk = expected[
            "relevant_chunk_id"
        ]

        max_rank = expected["target_max_rank"]

        checks["relevant_chunk_in_top_k"] = (
            relevant_chunk in chunk_ids[:max_rank]
        )

        checks["irrelevant_equipment_excluded"] = (
            "GITLAB-0001-S0045"
            not in chunk_ids
        )

    else:
        raise ValueError(
            f"Unsupported regression case: {test_id}"
        )

    return checks


# --------------------------------------------------
# Live regression test execution
# --------------------------------------------------

@pytest.mark.parametrize(
    "case",
    TEST_CASES,
    ids=[case["test_id"] for case in TEST_CASES],
)
def test_live_regression(
    case,
    workflow,
    execution_results,
):

    start_time = time.perf_counter()

    record = {
        "test_id": case["test_id"],
        "category": case["category"],
        "query": case["query"],
        "recorded_manual_status": (
            case["baseline"]["status"]
        ),
    }

    try:
        result = workflow.run(
            query=case["query"],
            expense_details={},
            thread_id=str(uuid.uuid4()),
            conversation_history=deepcopy(
                case["conversation_history"]
            ),
        )

        checks = evaluate_result(
            case,
            result,
        )

        passed = bool(checks) and all(
            checks.values()
        )

        record.update({
            "status": (
                "PASS" if passed else "FAIL"
            ),
            "checks": checks,
            "actual_response": result.get(
                "final_answer", ""
            ),
            "workflow_status": result.get(
                "status"
            ),
            "critic_approved": result.get(
                "is_approved"
            ),
            "retry_count": result.get(
                "retry_count", 0
            ),
            "selected_tools": result.get(
                "selected_tools", []
            ),
            "claimed_amount": (
                result.get("tool_results", {})
                .get(
                    "calculate_expense_eligibility",
                    {},
                )
                .get("claimed_amount")
            ),
            "retrieved_chunk_ids": [
                item.get("chunk_id")
                for item in result.get(
                    "retrieved_evidence", []
                )
            ],
        })

    except Exception as exc:

        record.update({
            "status": "ERROR",
            "error_type": type(exc).__name__,
        })

        raise

    finally:

        record["latency_seconds"] = round(
            time.perf_counter() - start_time,
            3,
        )

        execution_results.append(record)

    failed_checks = [
        name
        for name, status in checks.items()
        if not status
    ]

    assert passed, (
        f"{case['test_id']} failed: "
        f"{failed_checks}"
    )
