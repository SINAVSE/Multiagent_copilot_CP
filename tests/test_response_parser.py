from types import SimpleNamespace

import pytest

from src.agents.errors import (
    AgentResponseError,
    ErrorCode,
    ToolExecutionError,
    classify_error,
)

from src.agents.response_parser import (
    MAX_RESPONSE_CHARS,
    parse_json_response,
)


FENCE = chr(96) * 3
VALID_JSON = '{"intent": "GREETING"}'


@pytest.mark.parametrize(
    "content",
    [
        VALID_JSON,
        "  " + VALID_JSON + "  ",
        FENCE + "json\n" + VALID_JSON + "\n" + FENCE,
        FENCE + "\n" + VALID_JSON + "\n" + FENCE,
        FENCE + "JSON\n" + VALID_JSON + "\n" + FENCE,
    ],
)
def test_valid_json_responses(content):

    response = SimpleNamespace(content=content)

    assert parse_json_response(response) == {
        "intent": "GREETING"
    }


@pytest.mark.parametrize(
    "content",
    [
        '{"intent": "GREETING"',
        '{"approved": true,}',
        "{'approved': True}",
        'Here is the JSON: {"approved": true}',
        FENCE + 'json\n{"approved": true}',
        FENCE + 'python\n{"approved": true}\n' + FENCE,
        '{"approved": true, "approved": false}',
        '{"amount": NaN}',
        '{"amount": Infinity}',
        '{"amount": -Infinity}',
    ],
)
def test_malformed_json_is_rejected(content):

    with pytest.raises(AgentResponseError) as error:
        parse_json_response(content)

    assert error.value.code == ErrorCode.MALFORMED_JSON


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "[1, 2]",
        "null",
        "true",
        "100",
        '"hello"',
    ],
)
def test_non_object_json_is_rejected(content):

    with pytest.raises(AgentResponseError) as error:
        parse_json_response(content)

    assert error.value.code == ErrorCode.SCHEMA_VALIDATION


@pytest.mark.parametrize(
    "content",
    ["", " ", None, 123, [], {}],
)
def test_invalid_content_is_rejected(content):

    with pytest.raises(AgentResponseError) as error:
        parse_json_response(content)

    assert error.value.code == ErrorCode.INVALID_RESPONSE


def test_structured_content_blocks():

    response = SimpleNamespace(
        content=[
            {"type": "text", "text": '{"approved": '},
            {"type": "text", "text": "true}"},
        ]
    )

    result = parse_json_response(
        response,
        allow_content_blocks=True,
    )

    assert result == {"approved": True}


def test_content_blocks_require_explicit_permission():

    response = SimpleNamespace(
        content=['{"approved": true}']
    )

    with pytest.raises(AgentResponseError):
        parse_json_response(response)


def test_response_size_limit():

    with pytest.raises(AgentResponseError) as error:
        parse_json_response(
            "x" * (MAX_RESPONSE_CHARS + 1)
        )

    assert error.value.code == ErrorCode.INVALID_RESPONSE


@pytest.mark.parametrize(
    "code",
    [
        ErrorCode.INVALID_RESPONSE,
        ErrorCode.MALFORMED_JSON,
        ErrorCode.SCHEMA_VALIDATION,
    ],
)
def test_model_response_errors_are_retryable(code):

    result = classify_error(
        AgentResponseError(code, "Test response error")
    )

    assert result.code == code
    assert result.retryable is True


def test_timeout_is_retryable():

    result = classify_error(
        TimeoutError("Test timeout")
    )

    assert result.code == ErrorCode.API_TIMEOUT
    assert result.retryable is True


def test_connection_failure_is_retryable():

    result = classify_error(
        ConnectionError("Test connection failure")
    )

    assert result.code == ErrorCode.API_CONNECTION
    assert result.retryable is True


def test_tool_execution_failure_is_not_retryable():

    result = classify_error(
        ToolExecutionError("search_hr_policy")
    )

    assert result.code == ErrorCode.TOOL_EXECUTION_FAILED
    assert result.retryable is False


def test_unexpected_error_is_not_retryable():

    result = classify_error(
        RuntimeError("internal failure")
    )

    assert result.code == ErrorCode.INTERNAL_ERROR
    assert result.retryable is False


def test_generic_value_error_is_not_model_error():

    result = classify_error(
        ValueError("Invalid application input")
    )

    assert result.code == ErrorCode.INTERNAL_ERROR
    assert result.retryable is False


def test_safe_message_does_not_expose_secret():

    secret = "SENSITIVE_TEST_CREDENTIAL"

    result = classify_error(RuntimeError(secret))

    assert secret not in result.safe_message


def test_response_error_preserves_value_error_contract():

    exc = AgentResponseError(
        ErrorCode.MALFORMED_JSON,
        "Invalid JSON",
    )

    assert isinstance(exc, ValueError)
