
from typing import Any

from src.agents.errors import ToolExecutionError

from src.rag.generation.answer_generator import AnswerGenerator
from src.tools.expense_calculator import (
    calculate_expense,
    calculate_per_person_expense,
    unable_to_determine,
)
from src.tools.policy_extractor import extract_expense_policy

from src.agents.runtime_budget import BudgetExceededError


class Executor:
    """Executes tools selected by the Planner."""

    def __init__(self, answer_generator: Any = None):
        self.answer_generator = (
            answer_generator or AnswerGenerator()
        )

    @staticmethod
    def _execute_tool(tool_name, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)

        except (ToolExecutionError, BudgetExceededError):
            raise

        except Exception as exc:
            raise ToolExecutionError(tool_name) from exc

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

        tool_order = [
            "search_hr_policy",
            "calculate_expense_eligibility",
            "extract_bill_details",
        ]

        tools_to_run = list(selected_tools)

        # Expense calculations must always have policy evidence.
        if "calculate_expense_eligibility" in tools_to_run:
            if "search_hr_policy" not in tools_to_run:
                tools_to_run.insert(0, "search_hr_policy")

        unknown_tools = set(selected_tools) - set(tool_order)

        if unknown_tools:
            raise ValueError(
                f"Unknown tools selected: {unknown_tools}"
            )

        ordered_tools = [
            tool for tool in tool_order
            if tool in tools_to_run
        ]

        if "extract_bill_details" in ordered_tools:
            raise ToolExecutionError(
                "extract_bill_details"
            )

        retrieval_result = None

        # --------------------------------------------------
        # Policy retrieval
        # --------------------------------------------------

        if "search_hr_policy" in ordered_tools:

            retrieval_result = self._execute_tool(
                "search_hr_policy",
                self.answer_generator.retrieve,
                query,
            )

            evidence = [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "title": item.chunk.title,
                    "section": (
                        item.chunk.section_path
                        or item.chunk.section_heading
                        or "Not specified"
                    ),
                    "text": item.chunk.text,
                }
                for item in retrieval_result.retrieved_chunks
            ]

            results["search_hr_policy"] = {
                "answer": "",
                "insufficient_evidence": not bool(evidence),
                "missing_information": [],
                "retrieval_method": retrieval_result.retrieval_method,
                "citations": [],
                "retrieved_evidence": evidence,
            }

        # --------------------------------------------------
        # Tool execution
        # --------------------------------------------------

        for tool_name in ordered_tools:

            if tool_name == "search_hr_policy":
                continue

            elif tool_name == "calculate_expense_eligibility":

                search_result = results.get(
                    "search_hr_policy",
                    {},
                )

                evidence = search_result.get(
                    "retrieved_evidence",
                    [],
                )

                # Do not calculate without supporting policy evidence.
                if (
                    search_result.get("insufficient_evidence", True)
                    or not evidence
                ):
                    result = unable_to_determine(
                        reason=(
                            "Insufficient policy evidence "
                            "was retrieved."
                        ),
                        currency=details.get("currency"),
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=[
                            "Applicable policy limit",
                            "Supporting policy evidence",
                        ],
                    )

                    results[tool_name] = result.model_dump()
                    continue

                # Extract policy rules from retrieved evidence.
                try:
                    policy = extract_expense_policy(
                        query=query,
                        retrieved_evidence=evidence,
                    )

                except (ValueError, TypeError):
                    result = unable_to_determine(
                        reason=(
                            "Policy extraction or evidence "
                            "validation failed."
                        ),
                        currency=details.get("currency"),
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=[
                            "Verified policy extraction",
                        ],
                    )

                    results[tool_name] = result.model_dump()
                    continue

                except BudgetExceededError:
                    raise

                except Exception as exc:
                    raise ToolExecutionError(
                        "extract_expense_policy"
                    ) from exc

                # --------------------------------------------------
                # Currency validation
                # --------------------------------------------------

                claimed_currency = details.get("currency")
                policy_currency = policy.currency

                if (
                    claimed_currency
                    and policy_currency
                    and claimed_currency.strip().upper()
                    != policy_currency.strip().upper()
                ):
                    result = unable_to_determine(
                        reason=(
                            "The claimed expense currency does not match "
                            "the currency specified in the policy evidence. "
                            "Currency conversion has not been performed."
                        ),
                        currency=claimed_currency,
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=[
                            "A policy limit expressed in the claimed "
                            "currency or an explicitly authorized "
                            "currency conversion rule",
                        ],
                    )

                    results[tool_name] = result.model_dump()
                    continue

                if not claimed_currency or not policy_currency:
                    missing = []

                    if not claimed_currency:
                        missing.append("Claimed expense currency")

                    if not policy_currency:
                        missing.append("Policy currency")

                    result = unable_to_determine(
                        reason=(
                            "The claimed expense currency or policy "
                            "currency could not be established."
                        ),
                        currency=claimed_currency or policy_currency,
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=missing,
                    )

                    results[tool_name] = result.model_dump()
                    continue

                # --------------------------------------------------
                # Unknown policy limit
                # --------------------------------------------------

                if (
                    policy.limit_type == "unknown"
                    or policy.policy_limit is None
                ):
                    result = unable_to_determine(
                        reason=(
                            "The applicable policy limit "
                            "could not be established."
                        ),
                        currency=policy_currency,
                        claimed_amount=details.get("claimed_amount"),
                        missing_information=(
                            policy.missing_information
                            or [
                                "Verified applicable policy limit"
                            ]
                        ),
                    )

                # --------------------------------------------------
                # Fixed monetary limit
                # --------------------------------------------------

                elif policy.limit_type == "fixed":

                    claimed_amount = details.get("claimed_amount")

                    if claimed_amount is None:
                        result = unable_to_determine(
                            reason="Claimed amount is missing.",
                            currency=policy_currency,
                            missing_information=["Claimed amount"],
                        )

                    else:
                        result = self._execute_tool(
                            "calculate_expense_eligibility",
                            calculate_expense,
                            claimed_amount=claimed_amount,
                            policy_limit=policy.policy_limit,
                            currency=policy_currency,
                            approval_required=(
                                policy.approval_required
                            ),
                        )

                # --------------------------------------------------
                # Per-person monetary limit
                # --------------------------------------------------

                elif policy.limit_type == "per_person":

                    claimed_amount = details.get("claimed_amount")
                    headcount = details.get("headcount")

                    if (
                        claimed_amount is None
                        or headcount is None
                    ):
                        missing = []

                        if claimed_amount is None:
                            missing.append("Claimed amount")

                        if headcount is None:
                            missing.append("Headcount")

                        result = unable_to_determine(
                            reason=(
                                "Required expense details "
                                "are missing."
                            ),
                            currency=policy_currency,
                            claimed_amount=claimed_amount,
                            missing_information=missing,
                        )

                    else:
                        result = self._execute_tool(
                            "calculate_expense_eligibility",
                            calculate_per_person_expense,
                            total_claimed_amount=claimed_amount,
                            headcount=headcount,
                            per_person_policy_limit=(
                                policy.policy_limit
                            ),
                            currency=policy_currency,
                            approval_required=(
                                policy.approval_required
                            ),
                        )

                else:
                    result = unable_to_determine(
                        reason="Unsupported policy limit type.",
                        currency=policy_currency,
                    )

                # Attach exact supporting policy quotations.
                output = result.model_dump()

                output["policy_evidence"] = [
                    quote.quote
                    for quote in policy.evidence_quotes
                ]

                results[tool_name] = output

            # --------------------------------------------------
            # Bill extraction
            # --------------------------------------------------

            elif tool_name == "extract_bill_details":

                if not has_file or not file_path:
                    results[tool_name] = {
                        "status": "missing_file",
                        "message": "No bill file was provided.",
                    }

                else:
                    results[tool_name] = {
                        "status": "not_implemented",
                        "message": (
                            "Bill extraction tool is "
                            "not implemented yet."
                        ),
                    }

        # --------------------------------------------------
        # Final answer generation
        # --------------------------------------------------

        if retrieval_result is not None:

            expense_result = results.get(
                "calculate_expense_eligibility"
            )

            answer = self._execute_tool(
                "generate_policy_answer",
                self.answer_generator.generate,
                query=query,
                critic_feedback=critic_feedback,
                expense_result=expense_result,
                retrieval_result=retrieval_result,
            )

            results["search_hr_policy"].update({
                "answer": answer.answer,
                "insufficient_evidence": (
                    answer.insufficient_evidence
                ),
                "missing_information": (
                    answer.missing_information
                ),
                "retrieval_method": (
                    answer.retrieval_method
                ),
                "citations": [
                    citation.model_dump()
                    for citation in answer.citations
                ],
                "retrieved_evidence": (
                    answer.retrieved_evidence
                ),
            })

        return results