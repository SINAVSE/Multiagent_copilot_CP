
from src.agents.critic import Critic

critic = Critic()

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


# Test 1: Correct answer
print("\nTEST 1: Correct answer")

correct_result = critic.review(
    query="What is the meal allowance?",
    tool_results=create_tool_results(
        "The meal allowance is INR 1800 per employee per travel day."
    )
)

print(correct_result)


# Test 2: Incorrect answer
print("\nTEST 2: Incorrect answer")

incorrect_result = critic.review(
    query="What is the meal allowance?",
    tool_results=create_tool_results(
        "The meal allowance is INR 2500 per employee per travel day."
    )
)

print(incorrect_result)