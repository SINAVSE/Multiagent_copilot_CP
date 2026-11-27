import json
import re

from src.agents.errors import (
    AgentResponseError,
    ErrorCode,
)


MAX_RESPONSE_CHARS = 100_000


def _reject_duplicate_keys(pairs):

    result = {}

    for key, value in pairs:

        if key in result:
            raise ValueError(
                "Duplicate JSON keys are not permitted."
            )

        result[key] = value

    return result


def _reject_invalid_constant(value):

    raise ValueError(
        "Non-standard JSON constants are not permitted."
    )


def _extract_content(
    response,
    allow_content_blocks: bool,
) -> str:

    content = getattr(response, "content", response)

    if isinstance(content, list) and allow_content_blocks:

        text_parts = []

        for item in content:

            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

        content = "".join(text_parts)

    if not isinstance(content, str):

        raise AgentResponseError(
            ErrorCode.INVALID_RESPONSE,
            "Unexpected LLM response format.",
        )

    if not content.strip():

        raise AgentResponseError(
            ErrorCode.INVALID_RESPONSE,
            "LLM returned empty response content.",
        )

    if len(content) > MAX_RESPONSE_CHARS:

        raise AgentResponseError(
            ErrorCode.INVALID_RESPONSE,
            "LLM response exceeds the permitted size.",
        )

    return content.strip()


def parse_json_response(
    response,
    agent_name: str = "Agent",
    allow_content_blocks: bool = False,
) -> dict:

    content = _extract_content(
        response,
        allow_content_blocks,
    )

    # Accept only a complete JSON code fence.
    if content.startswith("```"):

        match = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if not match:

            raise AgentResponseError(
                ErrorCode.MALFORMED_JSON,
                f"{agent_name} returned malformed JSON fencing.",
            )

        content = match.group(1).strip()

    try:

        result = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_invalid_constant,
        )

    except (ValueError, RecursionError) as exc:

        raise AgentResponseError(
            ErrorCode.MALFORMED_JSON,
            f"{agent_name} returned invalid JSON.",
        ) from exc

    if not isinstance(result, dict):

        raise AgentResponseError(
            ErrorCode.SCHEMA_VALIDATION,
            f"{agent_name} returned an invalid JSON object.",
        )

    return result
