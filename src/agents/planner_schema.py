from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)


class PlannerIntent(str, Enum):
    POLICY_QUERY = "POLICY_QUERY"
    EXPENSE_ELIGIBILITY = "EXPENSE_ELIGIBILITY"
    RECEIPT_ANALYSIS = "RECEIPT_ANALYSIS"
    GREETING = "GREETING"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"


class PlannerTool(str, Enum):
    SEARCH_POLICY = "search_hr_policy"
    CALCULATE_EXPENSE = "calculate_expense_eligibility"
    EXTRACT_RECEIPT = "extract_bill_details"


INTENT_TOOL_COMPATIBILITY = {
    PlannerIntent.POLICY_QUERY: {
        PlannerTool.SEARCH_POLICY,
    },
    PlannerIntent.EXPENSE_ELIGIBILITY: {
        PlannerTool.SEARCH_POLICY,
        PlannerTool.CALCULATE_EXPENSE,
    },
    PlannerIntent.RECEIPT_ANALYSIS: {
        PlannerTool.EXTRACT_RECEIPT,
        PlannerTool.SEARCH_POLICY,
        PlannerTool.CALCULATE_EXPENSE,
    },
    PlannerIntent.GREETING: set(),
    PlannerIntent.OUT_OF_SCOPE: set(),
    PlannerIntent.CLARIFICATION_REQUIRED: set(),
}


class PlannerOutput(BaseModel):
    # Reject unexpected fields from the language model.
    model_config = ConfigDict(extra="forbid")

    intent: PlannerIntent

    plan: list[StrictStr] = Field(
        min_length=1,
        max_length=10,
    )

    selected_tools: list[PlannerTool] = Field(
        max_length=3,
    )

    missing_information: list[StrictStr] = Field(
        max_length=10,
    )

    @field_validator("plan", "missing_information")
    @classmethod
    def validate_text_items(cls, values):
        if any(not item.strip() for item in values):
            raise ValueError(
                "Empty text items are not allowed."
            )

        return [item.strip() for item in values]

    @field_validator("selected_tools")
    @classmethod
    def validate_unique_tools(cls, tools):
        if len(tools) != len(set(tools)):
            raise ValueError(
                "Duplicate tools are not allowed."
            )

        return tools

    @model_validator(mode="after")
    def validate_intent_and_tools(self):
        selected = set(self.selected_tools)

        allowed = INTENT_TOOL_COMPATIBILITY[
            self.intent
        ]

        if not selected.issubset(allowed):
            raise ValueError(
                f"Tools incompatible with intent: "
                f"{self.intent.value}"
            )

        if self.intent in {
            PlannerIntent.POLICY_QUERY,
            PlannerIntent.EXPENSE_ELIGIBILITY,
        }:
            if PlannerTool.SEARCH_POLICY not in selected:
                raise ValueError(
                    "Policy search is required."
                )

        if self.intent == PlannerIntent.RECEIPT_ANALYSIS:
            if PlannerTool.EXTRACT_RECEIPT not in selected:
                raise ValueError(
                    "Receipt extraction is required."
                )

        if PlannerTool.CALCULATE_EXPENSE in selected:
            if PlannerTool.SEARCH_POLICY not in selected:
                raise ValueError(
                    "Expense calculation requires policy search."
                )

            if (
                self.selected_tools.index(
                    PlannerTool.SEARCH_POLICY
                )
                >
                self.selected_tools.index(
                    PlannerTool.CALCULATE_EXPENSE
                )
            ):
                raise ValueError(
                    "Policy search must precede calculation."
                )

        if self.intent == PlannerIntent.CLARIFICATION_REQUIRED:
            if not self.missing_information:
                raise ValueError(
                    "Clarification requires missing information."
                )

        return self
