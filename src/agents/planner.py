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

def extract_headcount(query: str) -> int | None:
    """Extract an explicitly stated attendee/participant count."""

    number_words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    pattern = (
        r"\b(\d+|one|two|three|four|five|six|seven|eight|"
        r"nine|ten)\s+"
        r"(?:attendees?|participants?|people|persons?|guests?)\b"
    )

    match = re.search(pattern, query, re.IGNORECASE)

    if not match:
        return None

    value = match.group(1).lower()

    try:
        count = int(value)
    except ValueError:
        count = number_words.get(value)

    return count if count and count > 0 else None


def extract_expense_details(query: str) -> dict:
    """Extract explicitly stated expense amount, currency and headcount."""

    details = {}

    patterns = [
        (
            r"\b(USD|INR|EUR|GBP)\s*"
            r"([0-9][0-9,]*(?:\.\d{1,2})?)\b",
            "prefix_code",
        ),
        (
            r"([$₹€£])\s*"
            r"([0-9][0-9,]*(?:\.\d{1,2})?)\b",
            "symbol",
        ),
        (
            r"\b([0-9][0-9,]*(?:\.\d{1,2})?)\s*"
            r"(USD|INR|EUR|GBP)\b",
            "suffix_code",
        ),
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

        else:
            amount = match.group(1)
            currency = "INR"

        try:
            amount = float(amount.replace(",", ""))
        except ValueError:
            continue

        details["claimed_amount"] = amount
        details["currency"] = currency
        break

    headcount = extract_headcount(query)

    if headcount is not None:
        details["headcount"] = headcount

    return details

def has_expense_calculation_intent(query: str) -> bool:
    """Check whether the employee wants an actual expense assessed."""

    patterns = [
        r"\b(spent|spend|paid|pay|cost|purchased|purchase|bought|buy)\b",
        r"\b(claim|claimed|reimburse|reimbursement)\b",
        r"\b(expense|bill|invoice|receipt)\b",
        r"\b(gift|dinner|meal|entertainment|team outing)\b",
        r"\b(hotel|flight|per diem|transport)\b",
        r"\b(within the limit|exceed the limit|over the limit|"
        r"excess amount|eligible amount)\b",
    ]

    return any(
        re.search(pattern, query, re.IGNORECASE)
        for pattern in patterns
    )

def extract_expense_category(query: str) -> str | None:
    """Identify an explicitly described expense category."""

    patterns = [
        (r"\b(customer\s+)?(meal|dinner|lunch)\b", "customer_meal"),
        (r"\b(gift|gifts)\b", "gift"),
        (r"\b(team outing|team event)\b", "team_outing"),
        (r"\b(wfh|work from home|home office|equipment)\b", "wfh_equipment"),
        (r"\b(hotel|accommodation|lodging)\b", "hotel"),
        (r"\b(flight|airfare)\b", "flight"),
        (r"\b(per diem)\b", "per_diem"),
        (r"\b(local transport|taxi|cab|transport)\b", "transport"),
    ]

    for pattern, category in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return category

    return None

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
2. Policy interpretation, eligibility, policy limits and
   approval-condition questions normally require only
   search_hr_policy.
3. Select calculate_expense_eligibility only when the employee
   wants an actual or proposed expense assessed against a policy
   limit and an expense amount is available.
3a. Do not treat a monetary amount mentioned as a policy allowance
    or policy limit as an employee's claimed expense.
3b. An eligibility question does not automatically require
    expense calculation.
3c. When the user asks whether a specific gift, dinner, purchase
    or other expense is within policy, use policy search and
    calculation when the expense amount is available.
4. Bill or receipt extraction requires extract_bill_details.
5. Do not calculate using invented policy limits.
6. Do not claim that an expense is approved.
7. If information is missing, mention it in the plan.
7a. Distinguish between a general policy explanation and a
    case-specific compliance assessment.
7b. For general policy questions, retrieve and explain the
    applicable policy. Do not request employee details merely
    because additional details could make the answer more specific.
7c. For case-specific assessments, identify only the employee-
    provided facts necessary to assess the particular situation.
7d. For gifts to government officials, a case-specific assessment
    may require:
    - Whether the recipient is a U.S. or foreign government official.
    - The proposed gift or entertainment value and currency.
    - Whether there are pending licensing, investigation or
      regulatory matters.
    - Whether additional state, tribal or local restrictions apply.
7e. Do not classify missing policy documents, missing policy
    thresholds or missing retrieval evidence as information
    that the employee must provide.
7f. When a general policy answer is possible but a definitive
    case-specific assessment is not, provide the policy explanation
    and identify the additional details needed for that assessment.
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
        # Fallback: every substantive employee-operations request
        # must have at least one tool selected.
        if not result["selected_tools"]:
            result["selected_tools"] = ["search_hr_policy"]
            result["plan"].append(
                "Search HR and expense policy documents for relevant evidence."
            )

        # Preserve structured expense details supplied by the UI.
        provided_details = expense_details or {}

        # Only extract an amount when the request concerns an actual
        # or proposed expense, rather than merely mentioning a policy limit.
        expense_intent = has_expense_calculation_intent(query)

        extracted_details = (
            extract_expense_details(query)
            if expense_intent
            else {}
        )

        # Explicitly supplied values take precedence.
        final_expense_details = {
            **extracted_details,
            **provided_details,
        }

        result["expense_details"] = final_expense_details

        category = extract_expense_category(query)

        if category:
            final_expense_details["expense_category"] = category

        # Deterministic safeguard against unnecessary calculator calls.
        # Only calculate when amount, currency and expense category
        # are established. Policy-only questions remain retrieval-only.

        has_calculation_inputs = (
            final_expense_details.get("claimed_amount") is not None
            and final_expense_details.get("currency") is not None
            and final_expense_details.get("expense_category") is not None
        )

        if not has_calculation_inputs:
            result["selected_tools"] = [
                tool for tool in result["selected_tools"]
                if tool != "calculate_expense_eligibility"
            ]

        elif "search_hr_policy" in result["selected_tools"]:
            if "calculate_expense_eligibility" not in result["selected_tools"]:
                result["selected_tools"].append(
                    "calculate_expense_eligibility"
                )

        return result