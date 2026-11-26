
"""
CAP-07: Unified Prompt Engineering Benchmark.

Compares:
S0 - Minimal structured-output baseline
S1 - Structured output with few-shot examples
S2 - Self-consistency with majority selection
S3 - Decomposition prompting

Uses the frozen synthetic evaluation dataset.
No RAG, tools or multi-agent orchestration.
"""

import argparse
import hashlib
import json
import time

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from config.clients import llm, llm_self_consistency
from config.settings import settings

from experiments.prompt_engineering.schemas import (
    ComplianceResponse,
)


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "prompt_engineering"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "evaluation_v1.0.json"
)

POLICY_PATH = (
    PROJECT_ROOT
    / "data"
    / "policies"
    / "synthetic"
    / "synthetic_policy_v1.0.md"
)

PROMPTS_DIR = EXPERIMENT_DIR / "prompts"

RESULTS_DIR = (
    EXPERIMENT_DIR
    / "results"
    / "benchmark_v1.0"
)

STRATEGIES = {
    "S0": {
        "name": "Minimal Baseline",
        "prompt_file": "strategy_0_baseline.txt",
        "generations": 1,
        "temperature": 0,
        "client": llm,
    },
    "S1": {
        "name": "Few-Shot",
        "prompt_file": "strategy_1_few_shot.txt",
        "generations": 1,
        "temperature": 0,
        "client": llm,
    },
    "S2": {
        "name": "Self-Consistency",
        "prompt_file": "strategy_2_self_consistency.txt",
        "generations": 3,
        "temperature": 0.7,
        "client": llm_self_consistency,
    },
    "S3": {
        "name": "Decomposition",
        "prompt_file": "strategy_3_decomposition.txt",
        "generations": 1,
        "temperature": 0,
        "client": llm,
    },
}


# --------------------------------------------------
# 2. Utility methods
# --------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def fingerprint(config):
    content = json.dumps(
        config,
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def load_existing_results(path):

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def append_result(path, result):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                result,
                ensure_ascii=False,
                default=str,
            ) + "\n"
        )

        file.flush()


# --------------------------------------------------
# 3. Prompt construction
# --------------------------------------------------

def build_prompt(
    strategy,
    policy_context,
    user_query,
):

    prompt_path = (
        PROMPTS_DIR
        / strategy["prompt_file"]
    )

    template = prompt_path.read_text(
        encoding="utf-8-sig"
    )

    if (
        "{policy_context}" not in template
        or "{user_query}" not in template
    ):
        raise ValueError(
            "Required prompt placeholders are missing."
        )

    prompt = template.replace(
        "{policy_context}",
        policy_context,
    ).replace(
        "{user_query}",
        user_query,
    )

    schema = ComplianceResponse.model_json_schema()

    prompt += (
        "\n\nRequired JSON response schema:\n"
        + json.dumps(schema, indent=2)
        + "\nReturn only valid JSON. "
          "Do not use Markdown fences."
    )

    return prompt


# --------------------------------------------------
# 4. JSON extraction and validation
# --------------------------------------------------

def extract_json(raw_response):

    if not isinstance(raw_response, str):
        raise TypeError(
            "Model response must be a string."
        )

    try:
        parsed = json.loads(raw_response)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    json_start = raw_response.find("{")

    if json_start == -1:
        raise ValueError(
            "No JSON object found."
        )

    parsed, _ = json.JSONDecoder().raw_decode(
        raw_response[json_start:]
    )

    if not isinstance(parsed, dict):
        raise ValueError(
            "Expected a JSON object."
        )

    return parsed


# --------------------------------------------------
# 5. Token usage extraction
# --------------------------------------------------

def extract_token_usage(response):

    usage = response.usage_metadata or {}

    metadata = response.response_metadata or {}

    fallback = metadata.get("token_usage") or {}

    def first_available(*values):

        return next(
            (
                value
                for value in values
                if value is not None
            ),
            None,
        )

    input_tokens = first_available(
        usage.get("input_tokens"),
        fallback.get("prompt_tokens"),
    )

    output_tokens = first_available(
        usage.get("output_tokens"),
        fallback.get("completion_tokens"),
    )

    total_tokens = first_available(
        usage.get("total_tokens"),
        fallback.get("total_tokens"),
    )

    if (
        total_tokens is None
        and input_tokens is not None
        and output_tokens is not None
    ):
        total_tokens = (
            input_tokens + output_tokens
        )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


# --------------------------------------------------
# 6. Execute one LLM generation
# --------------------------------------------------

def execute_generation(
    strategy,
    prompt,
    generation_number,
):

    started = time.perf_counter()

    result = {
        "generation": generation_number,
        "execution_status": "failed",
        "strict_json_valid": False,
        "json_extracted": False,
        "schema_valid": False,
        "raw_response": None,
        "validated_response": None,
        "token_usage": {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        },
        "latency_seconds": None,
        "error_type": None,
    }

    try:

        response = strategy["client"].invoke(
            prompt
        )

        result["token_usage"] = (
            extract_token_usage(response)
        )

        raw_response = response.content

        result["raw_response"] = raw_response

        if isinstance(raw_response, str):
            try:
                result["strict_json_valid"] = (
                    isinstance(
                        json.loads(raw_response),
                        dict,
                    )
                )
            except json.JSONDecodeError:
                pass

        parsed = extract_json(raw_response)

        result["json_extracted"] = True

        validated = ComplianceResponse.model_validate(
            parsed
        )

        result.update({
            "execution_status": "success",
            "schema_valid": True,
            "validated_response": validated.model_dump(
                mode="json"
            ),
        })

    except Exception as error:

        result["error_type"] = (
            type(error).__name__
        )

    finally:

        result["latency_seconds"] = round(
            time.perf_counter() - started,
            3,
        )

    return result


# --------------------------------------------------
# 7. Self-consistency selection
# --------------------------------------------------

def decision_signature(response):

    extracted = (
        response.get("extracted_information")
        or {}
    )

    return (
        response.get("compliance_status"),
        extracted.get("amount"),
        str(
            extracted.get("currency") or ""
        ).upper(),
        extracted.get("employees"),
        extracted.get("travel_days"),
        bool(
            response.get("missing_information")
        ),
    )


def select_consensus(generations):

    valid = [
        generation
        for generation in generations
        if generation["schema_valid"]
    ]

    if not valid:

        return {
            "selection_status": "no_valid_response",
            "selected_generation": None,
            "selected_response": None,
            "status_counts": {},
            "structured_disagreement": None,
        }

    statuses = Counter(
        item["validated_response"][
            "compliance_status"
        ]
        for item in valid
    )

    signatures = Counter(
        decision_signature(
            item["validated_response"]
        )
        for item in valid
    )

    majority_signature, count = (
        signatures.most_common(1)[0]
    )

    # Require at least two matching responses
    # from the three planned generations.
    if count < 2:

        return {
            "selection_status": "no_consensus",
            "selected_generation": None,
            "selected_response": None,
            "status_counts": dict(statuses),
            "structured_disagreement": (
                len(signatures) > 1
            ),
        }

    selected = next(
        item
        for item in valid
        if decision_signature(
            item["validated_response"]
        ) == majority_signature
    )

    selection_status = (
        "unanimous"
        if count == len(generations)
        else "majority_consensus"
    )

    return {
        "selection_status": selection_status,
        "selected_generation": selected[
            "generation"
        ],
        "selected_response": selected[
            "validated_response"
        ],
        "status_counts": dict(statuses),
        "structured_disagreement": (
            len(signatures) > 1
        ),
    }


# --------------------------------------------------
# 8. Execute one evaluation case
# --------------------------------------------------

def evaluate_case(
    strategy_id,
    case,
    policy_context,
    experiment_fingerprint,
):

    strategy = STRATEGIES[strategy_id]

    started = time.perf_counter()

    prompt = build_prompt(
        strategy,
        policy_context,
        case["query"],
    )

    generations = []

    for number in range(
        1,
        strategy["generations"] + 1,
    ):

        print(
            f"    Generation {number}/"
            f"{strategy['generations']}",
            flush=True,
        )

        result = execute_generation(
            strategy,
            prompt,
            number,
        )

        generations.append(result)

    if strategy_id == "S2":

        selection = select_consensus(
            generations
        )

    else:

        generation = generations[0]

        selection = {
            "selection_status": (
                "single_response"
                if generation["schema_valid"]
                else "no_valid_response"
            ),
            "selected_generation": (
                1
                if generation["schema_valid"]
                else None
            ),
            "selected_response": (
                generation["validated_response"]
            ),
            "status_counts": {},
            "structured_disagreement": None,
        }

    selected = selection[
        "selected_response"
    ]

    actual_status = (
        selected.get("compliance_status")
        if selected is not None
        else None
    )

    expected_status = case.get(
        "expected_status"
    )

    # Preserve gold answers in the result only.
    # Gold answers are not included in the LLM prompt.
    return {
        "timestamp": utc_now(),
        "benchmark_version": "1.0",
        "dataset_id": "synthetic_evaluation_v1.0",
        "experiment_fingerprint": (
            experiment_fingerprint
        ),
        "strategy_id": strategy_id,
        "strategy_name": strategy["name"],
        "model": settings.llm_model,
        "temperature": strategy["temperature"],
        "test_id": case["test_id"],
        "category": case.get("category"),
        "query": case["query"],
        "gold_answer": case.get("gold_answer"),
        "expected_status": expected_status,
        "expected_amount": case.get(
            "expected_amount"
        ),
        "expected_currency": case.get(
            "expected_currency"
        ),
        "requires_clarification": case.get(
            "requires_clarification"
        ),
        "expected_extraction": case.get(
            "expected_extraction"
        ),
        "evaluation_mode": (
            "document_text_interpretation"
            if case["test_id"].startswith("D")
            else "text"
        ),
        "ocr_evaluated": False,
        "generations": generations,
        **selection,
        "actual_status": actual_status,
        "status_match": (
            actual_status == expected_status
            if expected_status is not None
            else None
        ),
        "schema_valid": selected is not None,
        "generation_count": len(generations),
        "task_latency_seconds": round(
            time.perf_counter() - started,
            3,
        ),
    }


# --------------------------------------------------
# 9. Main benchmark execution
# --------------------------------------------------

def run_benchmark(args):

    dataset = load_json(DATASET_PATH)

    cases = dataset["evaluation_cases"]

    if args.case:

        cases = [
            case
            for case in cases
            if case["test_id"] == args.case
        ]

        if not cases:
            raise ValueError(
                f"Unknown test case: {args.case}"
            )

    if args.limit is not None:

        if args.limit < 1:
            raise ValueError(
                "Limit must be greater than zero."
            )

        cases = cases[:args.limit]

    strategy_ids = (
        list(STRATEGIES)
        if args.strategy == "all"
        else [args.strategy]
    )

    policy_context = POLICY_PATH.read_text(
        encoding="utf-8-sig"
    )

    output_dir = (
        RESULTS_DIR / args.run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nCAP-07 PROMPT BENCHMARK")
    print("=" * 50)
    print(f"Model: {settings.llm_model}")
    print(f"Cases selected: {len(cases)}")
    print(f"Strategies: {strategy_ids}")
    print(f"Output: {output_dir}")

    for strategy_id in strategy_ids:

        strategy = STRATEGIES[strategy_id]

        output_path = (
            output_dir
            / f"{strategy_id}_results.jsonl"
        )

        experiment_config = {
            "model": settings.llm_model,
            "strategy_id": strategy_id,
            "temperature": strategy["temperature"],
            "generations": strategy["generations"],
            "dataset_sha256": sha256_file(
                DATASET_PATH
            ),
            "policy_sha256": sha256_file(
                POLICY_PATH
            ),
            "prompt_sha256": sha256_file(
                PROMPTS_DIR
                / strategy["prompt_file"]
            ),
            "runner_sha256": sha256_file(
                Path(__file__)
            ),
            "schema": (
                ComplianceResponse.model_json_schema()
            ),
        }

        experiment_fingerprint = fingerprint(
            experiment_config
        )

        existing = load_existing_results(
            output_path
        )

        if existing and not args.resume:

            raise ValueError(
                f"Results already exist: {output_path}. "
                "Use --resume or another --run-name."
            )

        for record in existing:

            if (
                record.get(
                    "experiment_fingerprint"
                )
                != experiment_fingerprint
            ):

                raise ValueError(
                    "Experiment configuration changed. "
                    "Use a new --run-name."
                )

        completed_ids = {
            record["test_id"]
            for record in existing
        }

        print(
            f"\nRunning {strategy_id}: "
            f"{strategy['name']}"
        )

        print(
            f"Previously recorded: "
            f"{len(completed_ids)}"
        )

        for index, case in enumerate(
            cases,
            start=1,
        ):

            test_id = case["test_id"]

            if test_id in completed_ids:

                print(
                    f"[{index}/{len(cases)}] "
                    f"{test_id}: Skipped"
                )

                continue

            print(
                f"[{index}/{len(cases)}] "
                f"Running {test_id}",
                flush=True,
            )

            result = evaluate_case(
                strategy_id,
                case,
                policy_context,
                experiment_fingerprint,
            )

            append_result(
                output_path,
                result,
            )

            completed_ids.add(test_id)

            print(
                f"    Selection: "
                f"{result['selection_status']}"
            )

            print(
                f"    Status match: "
                f"{result['status_match']}"
            )

            print(
                f"    Latency: "
                f"{result['task_latency_seconds']}s"
            )

        print(
            f"Results saved: {output_path}"
        )

    print("\nBenchmark execution completed.")


# --------------------------------------------------
# 10. Command-line interface
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "CAP-07 Prompt Engineering Benchmark"
        )
    )

    parser.add_argument(
        "--strategy",
        choices=["all", "S0", "S1", "S2", "S3"],
        default="all",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--case",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="full",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        not args.run_name
        or Path(args.run_name).name
        != args.run_name
        or args.run_name in {".", ".."}
    ):
        parser.error(
            "Invalid run name."
        )

    run_benchmark(args)


if __name__ == "__main__":
    main()
