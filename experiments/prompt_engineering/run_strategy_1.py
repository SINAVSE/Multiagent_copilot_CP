import json
import time
from pathlib import Path
from config.clients import llm
from experiments.prompt_engineering.schemas import ComplianceResponse


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "policies"
    / "synthetic"
    / "synthetic_policy_v1.0.md"
)

PROMPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "prompts"
    / "strategy_1_few_shot.txt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "results"
)

RESULTS_FILE = RESULTS_DIR / "strategy_1_results.jsonl"


def main():
    # 1. Load policy
    policy_context = POLICY_PATH.read_text(encoding="utf-8")

    # 2. Load prompt
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    # 3. Define first test query
    user_query = (
        # "I have spent INR 1500 on meals during one domestic "
        # "travel day. Is it within the policy limit?"
        # "I have spent INR 500 on meals during one domestic "
        # "travel day and purchased a watch worth INR 2000 for myself. Is it within the policy limit?"
        "Total bill for team outing is INR 9000. Is it within the policy limit?"
    )

    # 4. Insert inputs into the prompt
    # Use replace() because the prompt contains JSON braces.
    prompt = prompt_template.replace(
        "{policy_context}", policy_context
    ).replace(
        "{user_query}", user_query
    )

    # 5. Include the Pydantic schema in the prompt
    schema = ComplianceResponse.model_json_schema()

    prompt += (
        "\n\nRequired JSON response schema:\n"
        + json.dumps(schema, indent=2)
        + "\nReturn only valid JSON. Do not use Markdown fences."
    )

    # 6. Call the LLM
    print("Sending request to Qwen3-8B...")

    start_time = time.time()
    response = llm.invoke(prompt)
    latency = time.time() - start_time

    raw_response = response.content

    if not isinstance(raw_response, str):
        raise TypeError("LLM response content is not a text string.")

    print(f"LLM response received in {latency:.2f} seconds.")

    # 7. Extract JSON from the response
    json_start = raw_response.find("{")

    if json_start == -1:
        raise ValueError("No JSON object found in the LLM response.")

    decoder = json.JSONDecoder()
    response_data, _ = decoder.raw_decode(
        raw_response[json_start:]
    )

    # 8. Validate using Pydantic
    validated_response = ComplianceResponse.model_validate(
        response_data
    )

    print("\nPydantic validation successful.")
    print(validated_response.model_dump_json(indent=2))

    # 9. Save result
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_record = {
        "strategy": "strategy_1_few_shot",
        "query": user_query,
        "raw_response": raw_response,
        "validated_response": validated_response.model_dump(
            mode="json"
        ),
        "latency_seconds": round(latency, 2),
    }

    with RESULTS_FILE.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(result_record, ensure_ascii=False) + "\n"
        )

    print(f"\nResult saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
