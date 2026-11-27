import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from contextlib import contextmanager


MAX_AGENT_STEPS = 10

DEFAULT_TOKEN_BUDGET = 32_000

DEFAULT_MAX_COMPLETION_TOKENS = 2_048


class BudgetExceededError(RuntimeError):
    """The request cannot continue within its runtime budget."""

    def __init__(self, budget_type: str):

        self.budget_type = budget_type

        super().__init__(
            "The permitted runtime budget has been exhausted."
        )


@dataclass
class RuntimeBudget:

    max_agent_steps: int = MAX_AGENT_STEPS

    max_tokens: int = DEFAULT_TOKEN_BUDGET

    max_completion_tokens: int = (
        DEFAULT_MAX_COMPLETION_TOKENS
    )

    agent_steps: int = 0

    llm_calls: int = 0

    embedding_calls: int = 0

    input_tokens: int = 0

    output_tokens: int = 0

    total_tokens: int = 0

    estimated_input_tokens: int = 0

    estimated_tokens: int = 0

    estimated_usage_calls: int = 0

    unreported_usage_calls: int = 0

    step_history: list = field(
        default_factory=list
    )

    def __post_init__(self):

        limits = {
            "max_agent_steps": self.max_agent_steps,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
        }

        for name, value in limits.items():

            if (
                type(value) is not int
                or value <= 0
            ):
                raise ValueError(
                    f"{name} must be a positive integer."
                )

        if self.max_agent_steps > MAX_AGENT_STEPS:

            raise ValueError(
                "Agent-step limit cannot exceed 10."
            )

        if self.max_completion_tokens > self.max_tokens:

            raise ValueError(
                "Completion limit cannot exceed token budget."
            )

    @classmethod
    def from_environment(cls):
        """Create a request budget from validated settings."""

        def read_integer(name, default):

            value = os.getenv(name)

            if value is None:
                return default

            try:
                return int(value)

            except ValueError as exc:
                raise ValueError(
                    f"{name} must be a valid integer."
                ) from exc

        return cls(
            max_agent_steps=read_integer(
                "COPILOT_MAX_AGENT_STEPS",
                MAX_AGENT_STEPS,
            ),
            max_tokens=read_integer(
                "COPILOT_TOKEN_BUDGET",
                DEFAULT_TOKEN_BUDGET,
            ),
            max_completion_tokens=read_integer(
                "COPILOT_MAX_COMPLETION_TOKENS",
                DEFAULT_MAX_COMPLETION_TOKENS,
            ),
        )

    @property
    def remaining_steps(self):

        return max(
            0,
            self.max_agent_steps - self.agent_steps,
        )

    @property
    def accounted_tokens(self):

        return (
            self.total_tokens
            + self.estimated_tokens
        )

    @property
    def remaining_tokens(self):

        return max(
            0,
            self.max_tokens - self.accounted_tokens,
        )

    def consume_step(self, agent_name: str):

        if self.remaining_steps <= 0:

            raise BudgetExceededError("agent_steps")

        self.agent_steps += 1

        self.step_history.append({
            "step": self.agent_steps,
            "agent": agent_name,
        })

    @staticmethod
    def estimate_input_tokens(prompt: str):

        if not isinstance(prompt, str):

            raise TypeError(
                "Token estimation requires a string prompt."
            )

        # Conservative byte-based approximation.
        # Includes an allowance for message framing.
        # This is not an exact model tokenizer.
        return len(prompt.encode("utf-8")) + 128

    def begin_llm_call(self, prompt: str):

        estimated_input = self.estimate_input_tokens(
            prompt
        )

        available_output = (
            self.remaining_tokens - estimated_input
        )

        if available_output <= 0:

            raise BudgetExceededError("tokens")

        output_limit = min(
            self.max_completion_tokens,
            available_output,
        )

        self.llm_calls += 1

        self.estimated_input_tokens += estimated_input

        return estimated_input, output_limit

    @staticmethod
    def valid_token_count(value):

        return (
            type(value) is int
            and value >= 0
        )

    def record_llm_response(
        self,
        response,
        estimated_input: int,
        reserved_output: int,
    ):

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if not isinstance(usage, dict):

            metadata = getattr(
                response,
                "response_metadata",
                {},
            )

            if isinstance(metadata, dict):

                usage = (
                    metadata.get("token_usage")
                    or metadata.get("usage")
                )

        if isinstance(usage, dict):

            input_count = usage.get(
                "input_tokens",
                usage.get("prompt_tokens"),
            )

            output_count = usage.get(
                "output_tokens",
                usage.get("completion_tokens"),
            )

            total_count = usage.get("total_tokens")

            valid_components = (
                self.valid_token_count(input_count)
                and self.valid_token_count(output_count)
            )

            if valid_components:

                minimum_total = (
                    input_count + output_count
                )

                if total_count is None:

                    total_count = minimum_total

                if (
                    self.valid_token_count(total_count)
                    and total_count >= minimum_total
                ):

                    self.input_tokens += input_count

                    self.output_tokens += output_count

                    self.total_tokens += total_count

                    self.check_token_budget()

                    return

        # Missing or inconsistent provider usage metadata.
        # Account conservatively using the pre-call allowance.
        self.estimated_usage_calls += 1

        self.estimated_tokens += (
            estimated_input + reserved_output
        )

        self.check_token_budget()

    def record_failed_llm_call(
        self,
        estimated_input: int,
        reserved_output: int,
    ):

        # A failed request may still have been processed
        # by the inference provider.
        self.unreported_usage_calls += 1

        self.estimated_tokens += (
            estimated_input + reserved_output
        )

    def record_embedding_call(self):

        self.embedding_calls += 1

    def check_token_budget(self):

        if self.accounted_tokens > self.max_tokens:

            raise BudgetExceededError("tokens")

    def snapshot(self):

        return {
            "max_agent_steps": self.max_agent_steps,
            "agent_steps": self.agent_steps,
            "remaining_steps": self.remaining_steps,
            "max_tokens": self.max_tokens,
            "max_completion_tokens": self.max_completion_tokens,
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_tokens": self.estimated_tokens,
            "accounted_tokens": self.accounted_tokens,
            "remaining_tokens": self.remaining_tokens,
            "estimated_usage_calls": self.estimated_usage_calls,
            "unreported_usage_calls": self.unreported_usage_calls,
            "step_history": [
                dict(item)
                for item in self.step_history
            ],
        }


_active_budget = ContextVar(
    "active_runtime_budget",
    default=None,
)


def get_active_budget():

    return _active_budget.get()


@contextmanager
def budget_scope(budget: RuntimeBudget):

    token = _active_budget.set(budget)

    try:

        yield budget

    finally:

        _active_budget.reset(token)
