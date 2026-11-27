import json
from typing import Any
import re
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


def extract_expense_details(query: str) -> dict:
    """Extract explicitly stated expense amount and currency."""

    patterns = [
        # USD 250, INR 1,500, EUR 80, GBP 100
        (
            r"\b(USD|INR|EUR|GBP)\s*"
            r"([0-9][0-9,]*(?:\.\d{1,2})?)\b",
            "prefix_code",
        ),

        # $250, ₹1,500, €80, £100
        (
            r"([$₹€£])\s*"
            r"([0-9][0-9,]*(?:\.\d{1,2})?)\b",
            "symbol",
        ),

        # 250 USD, 1500 INR
        (
            r"\b([0-9][0-9,]*(?:\.\d{1,2})?)\s*"
            r"(USD|INR|EUR|GBP)\b",
            "suffix_code",
        ),

        # Rs. 1500, Rs 1,500
        (
            r"\bRs\.?\s*"
            r"([0-9][0-9,]*(?:\.\d{1,2})?)\b",
            "rupees",
        ),
    ]

    currency_symbols = {
        "$": "USD",
        "₹": "INR",
        "€": "EUR",
        "£": "GBP",
    }

    for pattern, pattern_type in patterns:
        match = re.search(pattern, query, re.IGNORECASE)

        if not match:
            continue

        if pattern_type == "prefix_code":
            currency = match.group(1).upper()
            amount = match.group(2)

        elif pattern_type == "symbol":
            currency = currency_symbols[match.group(1)]
            amount = match.group(2)

        elif pattern_type == "suffix_code":
            amount = match.group(1)
            currency = match.group(2).upper()

        elif pattern_type == "rupees":
            amount = match.group(1)
            currency = "INR"

        else:
            continue

        try:
            amount = float(amount.replace(",", ""))
        except ValueError:
            continue

        return {
            "claimed_amount": amount,
            "currency": currency,
        }

    return {}

class Planner:
    """Plans the tools and steps needed to handle an employee request."""

    def __init__(self, model: Any = None):
        self.model = model or llm

    def plan(self, query: str, has_file: bool = False,expense_details: dict | None = None,) -> dict:
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

STRUCTURED EXPENSE DETAILS:
{json.dumps(expense_details or {}, indent=2)}

Use these structured details as user-provided facts.
Do not treat them as policy limits or approval decisions.

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

        # Extract details explicitly stated in the employee's request.
        extracted_details = extract_expense_details(query)

        # Preserve any structured details already supplied by the UI.
        provided_details = expense_details or {}

        # Explicitly supplied values take precedence over extracted values.
        final_expense_details = {
            **extracted_details,
            **provided_details,
        }

        result["expense_details"] = final_expense_details

        return result