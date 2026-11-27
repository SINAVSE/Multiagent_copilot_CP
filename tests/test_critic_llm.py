# CAP-11: Explicit opt-in for hosted integration tests.
import os
import pytest

if os.getenv("RUN_LIVE_TESTS") != "1":
    pytest.skip(
        "Set RUN_LIVE_TESTS=1 to enable hosted integration tests.",
        allow_module_level=True,
    )


from src.agents.critic import Critic



# Controlled test evidence (not an actual company policy)
evidence_text = (
    "TEST POLICY: The meal allowance is INR 1800 "
    "per employee per travel day."
)

def create_tool_results(answer):
    return {
        "search_hr_policy": {
            "answer": answer,
            "insufficient_evidence": False,
            "citations": [
                {"chunk_id": "TEST-001"}
            ],
            "retrieved_evidence": [
                {
                    "chunk_id": "TEST-001",
                    "title": "Test Expense Policy",
                    "section": "Meal Allowance",
                    "text": evidence_text
                }
            ]
        }
    }


@pytest.mark.parametrize(
    "amount,expected_approval",
    [
        (1800, True),
        (2500, False),
    ],
)
def test_live_critic_evidence_verification(
    amount,
    expected_approval,
):

    critic = Critic()

    answer = (
        f"The meal allowance is INR {amount} "
        "per employee per travel day."
    )

    result = critic.review_with_recovery(
        query="What is the meal allowance?",
        tool_results=create_tool_results(answer),
        final_answer=answer,
        retry_allowed=lambda: False,
    )

    assert result["approved"] is expected_approval
