from src.agents.runtime_budget import (
    get_active_budget,
)


def invoke_model(model, prompt: str):
    """Invoke a model with optional runtime budget enforcement."""

    budget = get_active_budget()

    # Preserve compatibility for standalone components,
    # experiments and existing offline unit tests.
    if budget is None:

        return model.invoke(prompt)

    estimated_input, output_limit = (
        budget.begin_llm_call(prompt)
    )

    try:

        response = model.invoke(
            prompt,
            max_tokens=output_limit,
        )

    except Exception:

        # Failed calls may still consume provider tokens.
        budget.record_failed_llm_call(
            estimated_input=estimated_input,
            reserved_output=output_limit,
        )

        raise

    # Account for returned usage before downstream
    # JSON parsing or schema validation.
    budget.record_llm_response(
        response=response,
        estimated_input=estimated_input,
        reserved_output=output_limit,
    )

    return response


def invoke_embedding(operation, *args, **kwargs):
    """Track application-level embedding invocations."""

    budget = get_active_budget()

    if budget is not None:

        budget.record_embedding_call()

    return operation(*args, **kwargs)
