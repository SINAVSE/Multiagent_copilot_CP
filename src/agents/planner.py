import json
from typing import Any

from config.clients import llm


AVAILABLE_TOOLS = {
    "search_hr_policy": (
        "Search HR and expense policy documents for relevant rules."
    ),
    "calculate_expense_eligibility": (
        "Calculate expense eligibility using supplied policy limits "
        "and expense details. Never assume approval."
    ),
    "extract_bill_details": (
        "Extract merchant, date, amount, currency and tax details "
        "from an uploaded bill."
    ),
}


class Planner:
    """Plans the tools and steps needed to handle an employee request."""

    def __init__(self, model: Any = None):
        self.model = model or llm

    def plan(self, query: str, has_file: bool = False) -> dict:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        prompt = f"""
You are the Planner for an Employee Operations and
Expense Compliance Copilot.

Analyze the employee request and select the required tools.

AVAILABLE TOOLS:
{json.dumps(AVAILABLE_TOOLS, indent=2)}

RULES:
1. Select only tools necessary for the request.
2. Policy questions normally require search_hr_policy.
3. Expense eligibility questions require policy search and
   calculation when sufficient expense details are available.
4. Bill or receipt extraction requires extract_bill_details.
5. Do not calculate using invented policy limits.
6. Do not claim that an expense is approved.
7. If information is missing, mention it in the plan.
8. Never invent a tool name.

Return ONLY valid JSON with this structure:

{{
    "plan": ["Step 1", "Step 2"],
    "selected_tools": ["search_hr_policy"],
    "missing_information": []
}}

EMPLOYEE REQUEST:
{query}

UPLOADED FILE AVAILABLE:
{has_file}
"""

        response = self.model.invoke(prompt)
        content = response.content

        if not isinstance(content, str):
            raise ValueError("Unexpected LLM response format.")

        try:
            result = json.loads(content.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Planner returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError("Planner response must be a JSON object.")

        if not isinstance(result.get("plan"), list):
            raise ValueError("Planner must return a plan list.")

        if not isinstance(result.get("selected_tools"), list):
            raise ValueError("Planner must return selected_tools.")

        if not isinstance(result.get("missing_information"), list):
            raise ValueError(
                "Planner must return missing_information."
            )

        invalid_tools = (
            set(result["selected_tools"]) - set(AVAILABLE_TOOLS)
        )

        if invalid_tools:
            raise ValueError(
                f"Planner selected invalid tools: {invalid_tools}"
            )

        return result