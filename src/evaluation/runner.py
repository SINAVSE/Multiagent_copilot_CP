
import argparse
import json
import re
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
    / "evaluation_gitlab_v1.1.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "results"
    / "evaluation_gitlab_v1.1_results.json"
)

FAILED_TOOL_STATUSES = {
    "error",
    "failed",
    "not_implemented",
    "missing_file",
}


def normalize_text(value: Any) -> str:
    """Normalize text for case-insensitive matching."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def as_list(value: Any) -> list:
    """Convert optional values to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_dataset(dataset_path: Path) -> dict:
    """Load and validate the GitLab evaluation dataset."""

    with dataset_path.open("r", encoding="utf-8-sig") as file:
        dataset = json.load(file)

    cases = dataset.get("evaluation_cases")

    if not isinstance(cases, list) or not cases:
        raise ValueError(
            "Dataset must contain a non-empty evaluation_cases list."
        )

    required_fields = {"test_id", "query", "gold_answer"}

    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"Case {index} must be a JSON object.")

        missing = required_fields - case.keys()

        if missing:
            raise ValueError(
                f"Case {index} is missing required fields: {sorted(missing)}"
            )

    return dataset


def get_missing_information(result: dict) -> list:
    """Collect missing-information details from workflow outputs."""

    missing = list(result.get("missing_information") or [])

    for tool_result in (result.get("tool_results") or {}).values():
        if isinstance(tool_result, dict):
            missing.extend(
                tool_result.get("missing_information") or []
            )

    return list(dict.fromkeys(missing))


def get_evidence(result: dict) -> list[dict]:
    """Collect retrieved evidence from the workflow result."""

    evidence = result.get("retrieved_evidence") or []

    return [
        item for item in evidence
        if isinstance(item, dict)
    ]


def evaluate_sources(
    expected_sources: list,
    evidence: list[dict],
) -> dict:
    """Compare expected document IDs with retrieved document IDs."""

    expected = set(str(item) for item in expected_sources)

    actual = {
        str(item.get("document_id"))
        for item in evidence
        if item.get("document_id")
    }

    matched = expected & actual

    return {
        "expected": sorted(expected),
        "retrieved": sorted(actual),
        "matched": sorted(matched),
        "missing": sorted(expected - actual),
        "recall": (
            round(len(matched) / len(expected), 3)
            if expected else None
        ),
        "all_expected_found": (
            len(matched) == len(expected)
            if expected else None
        ),
    }

def evaluate_sections(
    expected_sections: list,
    evidence: list[dict],
) -> dict:
    """Compare expected section names with retrieved section paths.

    A match requires an exact section-name match against one
    segment of the retrieved section path.
    """

    actual_sections = [
        str(item.get("section") or "")
        for item in evidence
        if item.get("section")
    ]

    # Split section paths into individual heading names.
    actual_section_segments = []

    for section_path in actual_sections:
        segments = [
            normalize_text(segment)
            for segment in section_path.split(">")
            if normalize_text(segment)
        ]
        actual_section_segments.extend(segments)

    matched = []
    missing = []

    for expected in expected_sections:
        expected_text = normalize_text(expected)

        found = (
            expected_text in actual_section_segments
        )

        if found:
            matched.append(expected)
        else:
            missing.append(expected)

    return {
        "expected": expected_sections,
        "retrieved": actual_sections,
        "matched": matched,
        "missing": missing,
        "recall": (
            round(len(matched) / len(expected_sections), 3)
            if expected_sections else None
        ),
        "all_expected_found": (
            len(matched) == len(expected_sections)
            if expected_sections else None
        ),
    }

def evaluate_tool_execution(
    expected_tools: list,
    tool_results: dict,
) -> dict:
    """Check whether expected tools returned usable results."""

    execution = {}

    for tool in expected_tools:
        tool_result = tool_results.get(tool)

        if tool_result is None:
            execution[tool] = "not_executed"
            continue

        if not isinstance(tool_result, dict):
            execution[tool] = "returned_non_dict_result"
            continue

        status = normalize_text(tool_result.get("status"))

        if status in FAILED_TOOL_STATUSES:
            execution[tool] = status
        else:
            execution[tool] = "result_returned"

    successful = [
        tool for tool, status in execution.items()
        if status == "result_returned"
    ]

    return {
        "execution": execution,
        "successful_tools": successful,
        "missing_tools": [
            tool for tool, status in execution.items()
            if status == "not_executed"
        ],
        "successful_count": len(successful),
        "expected_count": len(expected_tools),
    }


def evaluate_case(
    workflow: EmployeeOpsWorkflow,
    case: dict,
) -> dict:
    """Run one GitLab evaluation case."""

    started = time.perf_counter()

    expected_tools = as_list(case.get("expected_tool_calls"))
    expected_sources = as_list(case.get("expected_sources"))
    expected_sections = as_list(case.get("expected_sections"))

    case_result = {
        "test_id": case.get("test_id"),
        "category": case.get("category"),
        "query": case.get("query"),
        "gold_answer": case.get("gold_answer"),
        "expected_status": case.get("expected_status"),
        "requires_clarification": case.get(
            "requires_clarification"
        ),
        "source_verification_required": case.get(
            "source_verification_required"
        ),
        "expected_tool_calls": expected_tools,
        "expected_sources": expected_sources,
        "expected_sections": expected_sections,
    }

    try:
        result = workflow.run(
            query=case["query"],
            has_file=False,
            expense_details=None,
        )

        if not isinstance(result, dict):
            raise TypeError("Workflow did not return a dictionary.")

        latency = round(time.perf_counter() - started, 3)

        tool_results = result.get("tool_results") or {}
        selected_tools = result.get("selected_tools") or []
        evidence = get_evidence(result)

        tool_metrics = evaluate_tool_execution(
            expected_tools,
            tool_results,
        )

        source_metrics = evaluate_sources(
            expected_sources,
            evidence,
        )

        section_metrics = evaluate_sections(
            expected_sections,
            evidence,
        )

        # Preserve all missing-information details for diagnostics.
        missing_information = get_missing_information(result)

        # Planner missing_information represents details missing
        # from the employee request before policy retrieval.
        employee_missing_information = (
            result.get("missing_information") or []
        )

        insufficient_evidence = any(
            isinstance(tool_result, dict)
            and tool_result.get("insufficient_evidence") is True
            for tool_result in tool_results.values()
        )

        # Policy evidence gaps alone do not mean the employee
        # needs to provide additional information.
        clarification_observed = bool(employee_missing_information)

        expected_clarification = case.get(
            "requires_clarification"
        )

        clarification_match = None

        if isinstance(expected_clarification, bool):
            clarification_match = (
                expected_clarification == clarification_observed
            )

        # Capture calculator status only if a calculator result exists.
        calculation = tool_results.get(
            "calculate_expense_eligibility"
        )

        actual_status = (
            calculation.get("compliance_status")
            if isinstance(calculation, dict)
            else None
        )

        expected_status = case.get("expected_status")

        status_match = None

        if actual_status and expected_status:
            status_match = (
                normalize_text(actual_status)
                == normalize_text(expected_status)
            )

        case_result.update({
            "workflow_completed": True,
            "latency_seconds": latency,
            "workflow_status": result.get("status"),
            "critic_approved": result.get("is_approved"),
            "retry_count": result.get("retry_count", 0),
            "selected_tools": selected_tools,
            "actual_tools": list(tool_results.keys()),
            "tool_execution": tool_metrics,
            "source_evaluation": source_metrics,
            "section_evaluation": section_metrics,
            "retrieved_evidence_count": len(evidence),
            "retrieved_evidence": evidence,
            "missing_information": missing_information,
            "insufficient_evidence": insufficient_evidence,
            "clarification_observed": clarification_observed,
            "clarification_match": clarification_match,
            "actual_calculator_status": actual_status,
            "status_match": status_match,
            "generated_answer": result.get("final_answer", ""),
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
            "tool_execution": {
                "execution": {
                    tool: "not_executed"
                    for tool in expected_tools
                },
                "successful_tools": [],
                "missing_tools": expected_tools,
                "successful_count": 0,
                "expected_count": len(expected_tools),
            },
            "source_evaluation": None,
            "section_evaluation": None,
            "clarification_match": None,
        })

    return case_result


def build_summary(results: list[dict]) -> dict:
    """Calculate aggregate evaluation metrics."""

    total = len(results)

    completed = sum(
        item.get("workflow_completed", False)
        for item in results
    )

    expected_tool_count = sum(
        item.get("tool_execution", {}).get("expected_count", 0)
        for item in results
    )

    successful_tool_count = sum(
        item.get("tool_execution", {}).get("successful_count", 0)
        for item in results
    )

    source_cases = [
        item for item in results
        if item.get("source_evaluation")
        and item["source_evaluation"]["recall"] is not None
    ]

    section_cases = [
        item for item in results
        if item.get("section_evaluation")
        and item["section_evaluation"]["recall"] is not None
    ]

    source_expected = sum(
        len(item["source_evaluation"]["expected"])
        for item in source_cases
    )

    source_matched = sum(
        len(item["source_evaluation"]["matched"])
        for item in source_cases
    )

    section_expected = sum(
        len(item["section_evaluation"]["expected"])
        for item in section_cases
    )

    section_matched = sum(
        len(item["section_evaluation"]["matched"])
        for item in section_cases
    )

    clarification_cases = [
        item for item in results
        if item.get("clarification_match") is not None
    ]

    clarification_matches = sum(
        item["clarification_match"]
        for item in clarification_cases
    )

    comparable_statuses = [
        item["status_match"]
        for item in results
        if item.get("status_match") is not None
    ]

    completed_latencies = [
        item["latency_seconds"]
        for item in results
        if item.get("workflow_completed")
    ]

    return {
        "total_cases": total,
        "workflow_completed": completed,
        "workflow_errors": total - completed,
        "workflow_completion_rate": (
            round(completed / total, 3) if total else None
        ),
        "expected_tool_calls": expected_tool_count,
        "successful_tool_calls": successful_tool_count,
        "tool_execution_rate": (
            round(
                successful_tool_count / expected_tool_count, 3
            )
            if expected_tool_count else None
        ),
        "source_cases_evaluated": len(source_cases),
        "expected_source_references": source_expected,
        "matched_source_references": source_matched,
        "source_recall": (
            round(source_matched / source_expected, 3)
            if source_expected else None
        ),
        "section_cases_evaluated": len(section_cases),
        "expected_section_references": section_expected,
        "matched_section_references": section_matched,
        "section_recall": (
            round(section_matched / section_expected, 3)
            if section_expected else None
        ),
        "clarification_cases_evaluated": len(clarification_cases),
        "clarification_matches": clarification_matches,
        "clarification_accuracy": (
            round(
                clarification_matches / len(clarification_cases), 3
            )
            if clarification_cases else None
        ),
        "comparable_status_cases": len(comparable_statuses),
        "status_matches": sum(comparable_statuses),
        "status_match_rate": (
            round(
                sum(comparable_statuses) / len(comparable_statuses), 3
            )
            if comparable_statuses else None
        ),
        "critic_approved": sum(
            item.get("critic_approved") is True
            for item in results
        ),
        "average_latency_seconds": (
            round(
                sum(completed_latencies) / len(completed_latencies), 3
            )
            if completed_latencies else None
        ),
    }


def run_evaluation(
    dataset_path: Path,
    output_path: Path,
    limit: int | None = None,
    selected_case: str | None = None,
) -> None:
    """Run evaluation and save the detailed results."""

    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1.")

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

    print(f"Dataset: {dataset.get('dataset_name', dataset_path.name)}")
    print(f"Dataset version: {dataset.get('dataset_version')}")
    print(f"Evaluation cases: {len(cases)}")
    print("Initialising Employee Operations Copilot...")

    workflow = None
    results = []

    try:
        workflow = EmployeeOpsWorkflow()

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

    finally:
        if workflow is not None:
            workflow.close()

    summary = build_summary(results)

    output = {
        "evaluation_run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset.get("dataset_name"),
            "dataset_version": dataset.get("dataset_version"),
            "dataset_status": dataset.get("status"),
            "dataset_path": str(dataset_path),
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
        description="Evaluate the Employee Operations Copilot "
                    "using the GitLab policy dataset."
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
        help="Run one specific test case.",
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