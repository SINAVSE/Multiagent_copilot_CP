from typing import Any
from src.rag.generation.answer_generator import AnswerGenerator


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
    ) -> dict:
        results = {}

        for tool_name in selected_tools:

            if tool_name == "search_hr_policy":
                result = self.answer_generator.generate(query,critic_feedback=critic_feedback,)

                results[tool_name] = {
                    "answer": result.answer,
                    "insufficient_evidence": (
                        result.insufficient_evidence
                    ),
                    "missing_information": (
                        result.missing_information
                    ),
                    "retrieval_method": result.retrieval_method,
                    "citations": [
                        citation.model_dump()
                        for citation in result.citations
                    ],
                    "retrieved_evidence": result.retrieved_evidence,
                }

            elif tool_name == "calculate_expense_eligibility":
                results[tool_name] = {
                    "status": "not_implemented",
                    "message": (
                        "Expense calculation tool is not "
                        "implemented yet."
                    ),
                }

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
                            "Bill extraction tool is not "
                            "implemented yet."
                        ),
                    }

            else:
                raise ValueError(
                    f"Unknown tool selected: {tool_name}"
                )

        return results