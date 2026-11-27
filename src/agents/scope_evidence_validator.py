
from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)


def validate_scoped_tool_results(scope, tool_results):

    if not isinstance(scope, PolicyScope):

        raise PolicyScopeError(
            "A validated policy scope is required."
        )

    if not isinstance(tool_results, dict):

        raise PolicyScopeError(
            "Invalid scoped tool results."
        )

    policy_result = tool_results.get("search_hr_policy")

    if policy_result is None:

        if "calculate_expense_eligibility" in tool_results:

            raise PolicyScopeError(
                "Expense assessment requires policy evidence."
            )

        return

    if not isinstance(policy_result, dict):

        raise PolicyScopeError(
            "Invalid policy search result."
        )

    if policy_result.get("policy_scope") != scope.to_dict():

        raise PolicyScopeError(
            "Policy result scope does not match."
        )

    evidence = policy_result.get("retrieved_evidence")
    citations = policy_result.get("citations")

    if not isinstance(evidence, list):
        raise PolicyScopeError("Invalid evidence list.")

    if not isinstance(citations, list):
        raise PolicyScopeError("Invalid citation list.")

    verified_ids = set()

    for item in evidence:

        if not isinstance(item, dict):
            raise PolicyScopeError("Invalid evidence item.")

        if item.get("organization") != scope.organization:

            raise PolicyScopeError(
                "Cross-organization evidence detected."
            )

        chunk_id = item.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():

            raise PolicyScopeError(
                "Evidence identifier is missing."
            )

        verified_ids.add(chunk_id)

    for citation in citations:

        if not isinstance(citation, dict):
            raise PolicyScopeError("Invalid citation item.")

        if citation.get("organization") != scope.organization:

            raise PolicyScopeError(
                "Cross-organization citation detected."
            )

        if citation.get("chunk_id") not in verified_ids:

            raise PolicyScopeError(
                "Citation does not match verified evidence."
            )

    if policy_result.get("insufficient_evidence") is False:

        if not evidence or not citations:

            raise PolicyScopeError(
                "Supporting policy evidence is missing."
            )

    expense = tool_results.get(
        "calculate_expense_eligibility"
    )

    if expense is not None:

        if not isinstance(expense, dict):

            raise PolicyScopeError(
                "Invalid expense calculation result."
            )

        status = expense.get("compliance_status")

        if status != "unable_to_determine":

            if (
                policy_result.get("insufficient_evidence")
                is not False
                or not evidence
                or not citations
            ):

                raise PolicyScopeError(
                    "Expense assessment lacks sufficient "
                    "applicable policy evidence."
                )
