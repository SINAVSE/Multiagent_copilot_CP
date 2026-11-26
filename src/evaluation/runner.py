from uuid import uuid4
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.workflow import EmployeeOpsWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "real_corpus"
    / "end_to_end_eval_v1.0.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results"
    / "real_policy_evaluation_results.json"
)

CALCULATOR_STATUSES = {
    "within_policy",
    "requires_approval",
    "exceeds_policy",
    "unable_to_determine",
}


def load_dataset(dataset_path: Path) -> dict:
    """Load and validate the evaluation dataset."""

    with dataset_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    cases = dataset.get("evaluation_cases")

    if not isinstance(cases, list):
        raise ValueError(
            "Dataset must contain an evaluation_cases list."
        )

    return dataset


def get_missing_information(result: dict) -> list:
    """Collect missing-information details from workflow outputs."""

    missing = list(result.get("missing_information") or [])

    tool_results = result.get("tool_results") or {}

    for tool_result in tool_results.values():
        if isinstance(tool_result, dict):
            missing.extend(
                tool_result.get("missing_information") or []
            )

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(missing))


def evaluate_case(workflow: EmployeeOpsWorkflow, case: dict) -> dict:
    """Run one evaluation case and capture its results."""

    started = time.perf_counter()

    
    case_result = {
        "test_id": case.get("test_id"),
        "category": case.get("category"),
        "query": case.get("query"),

        # CAP-06: Gold answer and evidence references
        "gold_answer": case.get("gold_answer"),
        "expected_document_ids": case.get(
            "expected_document_ids", []
        ),
        "expected_chunk_ids": case.get(
            "expected_chunk_ids", []
        ),
        "citation_required": case.get(
            "citation_required", False
        ),

        "expected_status": case.get("expected_status"),
        "expected_amount": case.get("expected_amount"),
        "expected_currency": case.get("expected_currency"),
        "requires_clarification": case.get(
            "requires_clarification"
        ),
    }


    try:
        result = workflow.run(
        query=case["query"],
        expense_details=case.get("expense_details") or {},
        thread_id=str(uuid4()),
        conversation_history=case.get("conversation_history") or [],
        )

        latency = round(time.perf_counter() - started, 3)

        tool_results = result.get("tool_results") or {}
        selected_tools = result.get("selected_tools") or []

        expected_tools = case.get("expected_tool_calls") or []
        actual_tools = list(tool_results.keys())

        missing_tools = [
            tool for tool in expected_tools
            if tool not in actual_tools
        ]

        tool_execution = {}

        for tool in expected_tools:
            tool_result = tool_results.get(tool)

            if tool_result is None:
                tool_execution[tool] = "not_executed"

            elif not isinstance(tool_result, dict):
                tool_execution[tool] = "returned_non_dict_result"

            else:
                tool_status = tool_result.get("status")

                if tool_status == "not_implemented":
                    tool_execution[tool] = "not_implemented"
                elif tool_status == "missing_file":
                    tool_execution[tool] = "missing_file"
                else:
                    tool_execution[tool] = "result_returned"

        calculation = tool_results.get(
            "calculate_expense_eligibility"
        )

        if not isinstance(calculation, dict):
            calculation = {}

        actual_calculator_status = calculation.get(
            "compliance_status"
        )

        expected_status = case.get("expected_status")

        # Compare statuses only when both use the calculator's
        # recognised status vocabulary.
        status_match = None

        if (
            actual_calculator_status in CALCULATOR_STATUSES
            and expected_status in CALCULATOR_STATUSES
        ):
            status_match = (
                actual_calculator_status == expected_status
            )

        missing_information = get_missing_information(result)

        clarification_observed = (
            len(missing_information) > 0
        )

        retrieved_evidence = result.get(
            "retrieved_evidence"
        ) or []

        case_result.update({
            "workflow_completed": True,
            "latency_seconds": latency,
            "selected_tools": selected_tools,
            "actual_tools": actual_tools,
            "expected_tools": expected_tools,
            "missing_expected_tools": missing_tools,
            "tool_execution": tool_execution,
            "critic_approved": result.get("is_approved"),
            "retry_count": result.get("retry_count", 0),
            "workflow_status": result.get("status"),
            "final_answer": result.get("final_answer", ""),
            "actual_calculator_status": actual_calculator_status,
            "status_match": status_match,
            "actual_calculated_amount": calculation.get(
                "eligible_amount"
            ),
            "actual_claimed_amount": calculation.get(
                "claimed_amount"
            ),
            "actual_excess_amount": calculation.get(
                "excess_amount"
            ),
            "missing_information": missing_information,
            "clarification_observed": clarification_observed,
            "retrieved_evidence_count": len(retrieved_evidence),
            "execution_log": result.get("execution_log", []),
            "tool_results": tool_results,
            "error": None,
        })

    except Exception as exc:
        case_result.update({
            "workflow_completed": False,
            "latency_seconds": round(
                time.perf_counter() - started, 3
            ),
            "error": f"{type(exc).__name__}: {exc}",
        })

    return case_result


def build_summary(results: list[dict]) -> dict:
    """Build high-level evaluation statistics."""

    total = len(results)

    completed = sum(
        item.get("workflow_completed", False)
        for item in results
    )

    approved = sum(
        item.get("critic_approved") is True
        for item in results
    )

    expected_tool_count = 0
    executed_tool_count = 0
    not_implemented_count = 0

    for item in results:
        for tool in item.get("expected_tools", []):
            expected_tool_count += 1

            status = item.get(
                "tool_execution", {}
            ).get(tool)

            if status in {
                "result_returned",
                "not_implemented",
                "missing_file",
            }:
                executed_tool_count += 1

            if status == "not_implemented":
                not_implemented_count += 1

    comparable_statuses = [
        item["status_match"]
        for item in results
        if item.get("status_match") is not None
    ]

    status_matches = sum(comparable_statuses)

    return {
        "total_cases": total,
        "workflow_completed": completed,
        "workflow_errors": total - completed,
        "critic_approved": approved,
        "expected_tool_calls": expected_tool_count,
        "expected_tool_calls_found": executed_tool_count,
        "not_implemented_tool_calls": not_implemented_count,
        "tool_call_coverage": (
            round(executed_tool_count / expected_tool_count, 3)
            if expected_tool_count else None
        ),
        "comparable_status_cases": len(comparable_statuses),
        "status_matches": status_matches,
        "status_match_rate": (
            round(status_matches / len(comparable_statuses), 3)
            if comparable_statuses else None
        ),
    }


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    limit: int | None = None,
    selected_case: str | None = None,
) -> None:
    """Run the evaluation dataset and save results."""

    dataset = load_dataset(dataset_path)
    cases = dataset["evaluation_cases"]

    if selected_case:
        cases = [
            case for case in cases
            if case.get("test_id") == selected_case
        ]

        if not cases:
            raise ValueError(
                f"Test case not found: {selected_case}"
            )

    if limit is not None:
        cases = cases[:limit]

    print(f"Evaluation cases: {len(cases)}")
    print("Initialising Employee Operations Copilot...")

    workflow = EmployeeOpsWorkflow()

    results = []

    for index, case in enumerate(cases, start=1):
        test_id = case.get("test_id", f"case_{index}")

        print(
            f"[{index}/{len(cases)}] Running {test_id}...",
            end=" ",
            flush=True,
        )

        case_result = evaluate_case(workflow, case)
        results.append(case_result)

        if case_result.get("workflow_completed"):
            print("Completed")
        else:
            print("Error")

    summary = build_summary(results)

    output = {
        "evaluation_run": {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "dataset_version": dataset.get("dataset_version"),
            "total_cases": len(results),
        },
        "summary": summary,
        "results": results,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    print("\nEvaluation Summary")
    print("------------------")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nDetailed results saved to:\n{output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Employee Operations Copilot."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases.",
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Run one specific test case, e.g. A01.",
    )

    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        limit=args.limit,
        selected_case=args.case,
    )


if __name__ == "__main__":
    main()