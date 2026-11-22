
from src.agents.critic import Critic


critic = Critic()

# Simulated Executor output
tool_results = {
    "search_hr_policy": {
        "answer": "The policy provides meal-related travel guidance.",
        "insufficient_evidence": False,
        "citations": [
            {"chunk_id": "GITLAB-0001-S0027"},
            #{"chunk_id": "GITLAB-0001-S0028"},
            {"chunk_id": "GITLAB-9999-S9999"},
        ],
        "retrieved_evidence": [
            {"chunk_id": "GITLAB-0001-S0027"},
            {"chunk_id": "GITLAB-0001-S0028"}
        ]
    }
}

result = critic.validate_executor_output(
    query="What are the meal rules while traveling?",
    tool_results=tool_results
)

print("CRITIC VALIDATION RESULT:")
print(result)