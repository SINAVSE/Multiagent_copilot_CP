
from typing import Any

from src.rag.generation.answer_generator import AnswerGenerator
from src.tools.expense_calculator import (
    calculate_expense,
    calculate_per_person_expense,
    unable_to_determine,
)
from src.tools.policy_extractor import extract_expense_policy


class Executor:
    """Executes tools selected by the Planner."""

    def __init__(self, answer_generator: Any = None):
        self.answer_generator = (
            answer_generator or AnswerGenerator()
        )

    def execute(
        self,
        selected_tools: list[str],
        query: str,
        has_file: bool = False,
        file_path: str | None = None,
        critic_feedback: str | None = None,
        expense_details: dict | None = None,
    ) -> dict:

        results = {}
        details = expense_details or {}

        # Enforce retrieval before expense calculation.
        tool_order = [
            "search_hr_policy",
            "calculate_expense_eligibility",
            "extract_bill_details",
        ]

        tools_to_run = list(selected_tools)

        if "calculate_expense_eligibility" in tools_to_run:
            if "search_hr_policy" not in tools_to_run:
                tools_to_run.insert(0, "search_hr_policy")

        ordered_tools = [
            tool for tool in tool_order
            if tool in tools_to_run
        ]

        for tool_name in ordered_tools:

            if tool_name == "search_hr_policy":
                result = self.answer_generator.generate(
                    query,
                    critic_feedback=critic_feedback,
                )

                results[tool_name] = {
                    "answer": result.answer,
                    "insufficient_evidence": result.insufficient_evidence,
                    "missing_information": result.missing_information,
                    "retrieval_method": result.retrieval_method,
                    "citations": [
                        citation.model_dump()
                        for citation in result.citations
                    ],
                    "retrieved_evidence": result.retrieved_evidence,
                }

            elif tool_name == "calculate_expense_eligibility":

                search_result = results.get("search_hr_policy", {})
                evidence = search_result.get("retrieved_evidence", [])

                if (
                    search_result.get("insufficient_evidence", True)
                    or not evidence
                ):
                    result = unable_to_determine(
                        reason="Insufficient policy evidence was retrieved.",
                        currency=details.get("currency", "INR"),
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=[
                            "Applicable policy limit",
                            "Supporting policy evidence",
                        ],
                    )
                    results[tool_name] = result.model_dump()
                    continue

                try:
                    policy = extract_expense_policy(
                        query=query,
                        retrieved_evidence=evidence,
                    )
                except (ValueError, TypeError):
                    result = unable_to_determine(
                        reason="Policy extraction or evidence validation failed.",
                        currency=details.get("currency", "INR"),
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=[
                            "Verified policy extraction",
                        ],
                    )
                    results[tool_name] = result.model_dump()
                    continue

                if (
                    policy.limit_type == "unknown"
                    or policy.policy_limit is None
                    or policy.approval_required is None
                ):
                    result = unable_to_determine(
                        reason="The policy limit or approval requirement could not be established.",
                        currency=policy.currency or details.get("currency", "INR"),
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=policy.missing_information
                        or ["Verified policy limit and approval requirement"],
                    )

                elif policy.limit_type == "fixed":
                    claimed_amount = details.get("claimed_amount")

                    if claimed_amount is None:
                        result = unable_to_determine(
                            reason="Claimed amount is missing.",
                            currency=policy.currency,
                            missing_information=["Claimed amount"],
                        )
                    else:
                        result = calculate_expense(
                            claimed_amount=claimed_amount,
                            policy_limit=policy.policy_limit,
                            currency=policy.currency or "INR",
                            approval_required=policy.approval_required,
                        )

                elif policy.limit_type == "per_person":
                    claimed_amount = details.get("claimed_amount")
                    headcount = details.get("headcount")

                    if claimed_amount is None or headcount is None:
                        result = unable_to_determine(
                            reason="Claimed amount or headcount is missing.",
                            currency=policy.currency,
                            claimed_amount=claimed_amount,
                            missing_information=[
                                "Claimed amount",
                                "Headcount",
                            ],
                        )
                    else:
                        result = calculate_per_person_expense(
                            total_claimed_amount=claimed_amount,
                            headcount=headcount,
                            per_person_policy_limit=policy.policy_limit,
                            currency=policy.currency or "INR",
                            approval_required=policy.approval_required,
                        )

                else:
                    result = unable_to_determine(
                        reason="Unsupported policy limit type.",
                        currency=policy.currency,
                    )

                output = result.model_dump()
                output["policy_evidence"] = [
                    quote.quote
                    for quote in policy.evidence_quotes
                ]
                results[tool_name] = output

            elif tool_name == "extract_bill_details":
                if not has_file or not file_path:
                    results[tool_name] = {
                        "status": "missing_file",
                        "message": "No bill file was provided.",
                    }
                else:
                    results[tool_name] = {
                        "status": "not_implemented",
                        "message": "Bill extraction tool is not implemented yet.",
                    }

        unknown_tools = set(selected_tools) - set(tool_order)

        if unknown_tools:
            raise ValueError(
                f"Unknown tools selected: {unknown_tools}"
            )

        return results