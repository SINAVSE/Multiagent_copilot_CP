
import json
import time
from pathlib import Path

from config.clients import llm
from experiments.prompt_engineering.schemas import ComplianceResponse


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_FILE = (
    PROJECT_ROOT
    / "data"
    / "policies"
    / "synthetic"
    / "synthetic_policy_v1.0.md"
)

PROMPT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "prompts"
    / "strategy_3_decomposition.txt"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "results"
    / "strategy_1_challenge_results.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
    / "results"
    / "strategy_3_challenge_results.json"
)


# --------------------------------------------------
# 2. Load input files
# --------------------------------------------------

policy_context = POLICY_FILE.read_text(encoding="utf-8")
prompt_template = PROMPT_FILE.read_text(encoding="utf-8")

with INPUT_FILE.open("r", encoding="utf-8") as file:
    challenge_cases = json.load(file)


# --------------------------------------------------
# 3. Prepare output schema
# --------------------------------------------------

schema = ComplianceResponse.model_json_schema()

schema_instruction = (
    "\n\nReturn valid JSON matching this JSON schema:\n"
    + json.dumps(schema, indent=2)
)


# --------------------------------------------------
# 4. Execute Strategy 3
# --------------------------------------------------

results = []

for case in challenge_cases:

    test_id = case["test_id"]
    user_query = case["query"]

    print(f"Running {test_id}: {user_query}")

    prompt = (
        prompt_template
        .replace("{policy_context}", policy_context)
        .replace("{user_query}", user_query)
        + schema_instruction
    )

    raw_response = None
    start_time = time.perf_counter()

    try:
        response = llm.invoke(prompt)

        latency = time.perf_counter() - start_time
        raw_response = response.content

        if not isinstance(raw_response, str):
            raise ValueError("LLM response content is not a string.")

        # Extract the first JSON object from the response.
        decoder = json.JSONDecoder()
        json_start = raw_response.find("{")

        if json_start == -1:
            raise ValueError("No JSON object found in model response.")

        parsed_json, _ = decoder.raw_decode(
            raw_response[json_start:]
        )

        # Validate against the Pydantic schema.
        validated_response = ComplianceResponse.model_validate(
            parsed_json
        )

        result = {
            "test_id": test_id,
            "query": user_query,
            "expected_behavior": case.get("expected_behavior"),
            "execution_status": "success",
            "validation_status": "passed",
            "latency_seconds": round(latency, 3),
            "raw_response": raw_response,
            "validated_response": validated_response.model_dump(),
            "error": None,
        }

        print(
            f"  SUCCESS | "
            f"{validated_response.compliance_status} | "
            f"{latency:.2f}s"
        )

    except Exception as error:

        latency = time.perf_counter() - start_time

        result = {
            "test_id": test_id,
            "query": user_query,
            "expected_behavior": case.get("expected_behavior"),
            "execution_status": "failed",
            "validation_status": "not_passed",
            "latency_seconds": round(latency, 3),
            "raw_response": raw_response,
            "validated_response": None,
            "error": str(error),
        }

        print(f"  FAILED | {error}")

    results.append(result)


# --------------------------------------------------
# 5. Save results
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)


# --------------------------------------------------
# 6. Print execution summary
# --------------------------------------------------

successful = sum(
    item["execution_status"] == "success"
    for item in results
)

failed = len(results) - successful

print("\nStrategy 3 execution completed.")
print(f"Total cases: {len(results)}")
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Results saved to: {OUTPUT_FILE}")