import json
import time
from pathlib import Path
from config.clients import llm_self_consistency
from experiments.prompt_engineering.schemas import ComplianceResponse

# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

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
    / "strategy_2_self_consistency.txt"
)

CHALLENGE_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "results"
    / "strategy_1_challenge_results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "results"
    / "strategy_2_challenge_results.json"
)

GENERATIONS_PER_QUERY = 3


# --------------------------------------------------
# 2. Load policy, prompt and challenge queries
# --------------------------------------------------

policy_context = POLICY_PATH.read_text(encoding="utf-8")
prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

with CHALLENGE_PATH.open("r", encoding="utf-8") as file:
    challenge_data = json.load(file)

# The Strategy 1 result file stores each query and its
# expected behaviour. We reuse the same query text here.
challenge_queries = [
    {
        "test_id": item["test_id"],
        "query": item["query"],
        "expected_behavior": item.get("expected_behavior"),
    }
    for item in challenge_data
]


# --------------------------------------------------
# 3. Prepare output schema
# --------------------------------------------------

schema_instruction = json.dumps(
    ComplianceResponse.model_json_schema(),
    indent=2,
)


# --------------------------------------------------
# 4. Run three independent generations per query
# --------------------------------------------------

all_results = []

for index, test_case in enumerate(challenge_queries, start=1):

    test_id = test_case["test_id"]
    user_query = test_case["query"]

    print(
        f"\nProcessing {index}/{len(challenge_queries)}: "
        f"{test_id}"
    )

    prompt = (
        prompt_template
        .replace("{policy_context}", policy_context)
        .replace("{user_query}", user_query)
    )

    prompt += (
        "\n\nRequired Pydantic JSON schema:\n"
        + schema_instruction
    )

    generations = []

    for generation_number in range(
        1, GENERATIONS_PER_QUERY + 1
    ):

        print(
            f"  Generation "
            f"{generation_number}/{GENERATIONS_PER_QUERY}"
        )

        raw_response = None
        start_time = time.perf_counter()

        generation_result = {
            "generation": generation_number,
            "execution_status": "failed",
            "latency_seconds": None,
            "raw_response": None,
            "validated_response": None,
            "error": None,
        }

        try:
            response = llm_self_consistency.invoke(prompt)

            latency = time.perf_counter() - start_time
            raw_response = response.content

            # Extract the JSON object from the response.
            decoder = json.JSONDecoder()
            json_start = raw_response.find("{")

            if json_start == -1:
                raise ValueError("No JSON object found.")

            parsed_json, _ = decoder.raw_decode(
                raw_response[json_start:]
            )

            # Validate the response using Pydantic.
            validated = ComplianceResponse.model_validate(
                parsed_json
            )

            generation_result.update({
                "execution_status": "success",
                "latency_seconds": round(latency, 3),
                "raw_response": raw_response,
                "validated_response": validated.model_dump(),
            })

            print(
                "    Validation: PASSED | "
                f"Status: {validated.compliance_status}"
            )

        except Exception as error:
            generation_result.update({
                "latency_seconds": round(
                    time.perf_counter() - start_time, 3
                ),
                "raw_response": raw_response,
                "error": str(error),
            })

            print(f"    FAILED: {error}")

        generations.append(generation_result)

    # Compare statuses from successfully validated responses.
    successful_statuses = [
        item["validated_response"]["compliance_status"]
        for item in generations
        if item["execution_status"] == "success"
    ]

    status_counts = {
        status: successful_statuses.count(status)
        for status in set(successful_statuses)
    }

    if len(successful_statuses) == 0:
        status_agreement = "no_valid_responses"
    elif len(status_counts) == 1 and len(successful_statuses) == 3:
        status_agreement = "unanimous"
    elif len(successful_statuses) < 3:
        status_agreement = "incomplete_generations"
    elif max(status_counts.values()) > 1:
        status_agreement = "majority_agreement"
    else:
        status_agreement = "no_consensus"

    all_results.append({
        "test_id": test_id,
        "query": user_query,
        "expected_behavior": test_case["expected_behavior"],
        "generations": generations,
        "status_counts": status_counts,
        "status_agreement": status_agreement,
    })


# --------------------------------------------------
# 5. Save all results
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(
        all_results,
        file,
        indent=2,
        ensure_ascii=False,
    )


# --------------------------------------------------
# 6. Print execution summary
# --------------------------------------------------

total_generations = (
    len(challenge_queries) * GENERATIONS_PER_QUERY
)

successful_generations = sum(
    generation["execution_status"] == "success"
    for result in all_results
    for generation in result["generations"]
)

print("\n" + "=" * 50)
print("STRATEGY 2 EXECUTION SUMMARY")
print("=" * 50)

print(f"Queries processed: {len(challenge_queries)}")
print(f"Total generations: {total_generations}")
print(f"Successful generations: {successful_generations}")
print(
    f"Failed generations: "
    f"{total_generations - successful_generations}"
)
print(f"Results saved to: {OUTPUT_PATH}")