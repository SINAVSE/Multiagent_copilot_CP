from dataclasses import dataclass
from enum import Enum

import httpx
import openai


class ErrorCode(str, Enum):

    INVALID_RESPONSE = "invalid_response"
    MALFORMED_JSON = "malformed_json"
    SCHEMA_VALIDATION = "schema_validation"
    API_TIMEOUT = "api_timeout"
    API_CONNECTION = "api_connection"
    AUTHENTICATION_FAILED = "authentication_failed"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    API_SERVER_ERROR = "api_server_error"
    API_REQUEST_FAILED = "api_request_failed"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ErrorClassification:

    code: ErrorCode
    retryable: bool
    safe_message: str


class AgentResponseError(ValueError):
    """Invalid or unusable language model response."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
    ):
        super().__init__(message)
        self.code = code


class ToolExecutionError(RuntimeError):
    """A required business tool could not complete."""

    def __init__(self, tool_name: str):
        super().__init__(
            "A required business tool could not complete."
        )
        self.tool_name = tool_name


def classify_error(exc: Exception) -> ErrorClassification:

    if isinstance(exc, AgentResponseError):
        return ErrorClassification(
            code=exc.code,
            retryable=exc.code in {
                ErrorCode.INVALID_RESPONSE,
                ErrorCode.MALFORMED_JSON,
                ErrorCode.SCHEMA_VALIDATION,
            },
            safe_message=(
                "The model returned an unusable response."
            ),
        )

    if isinstance(
        exc,
        (
            openai.AuthenticationError,
            openai.PermissionDeniedError,
        ),
    ):
        return ErrorClassification(
            code=ErrorCode.AUTHENTICATION_FAILED,
            retryable=False,
            safe_message=(
                "The AI service is currently unavailable. "
                "Please contact the application administrator."
            ),
        )

    if isinstance(exc, openai.RateLimitError):
        return ErrorClassification(
            code=ErrorCode.RATE_LIMIT_EXCEEDED,
            retryable=False,
            safe_message=(
                "The AI service request limit has been reached. "
                "Please try again later."
            ),
        )

    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            httpx.TimeoutException,
            TimeoutError,
        ),
    ):
        return ErrorClassification(
            code=ErrorCode.API_TIMEOUT,
            retryable=True,
            safe_message=(
                "The AI service did not respond in time."
            ),
        )

    if isinstance(
        exc,
        (
            openai.APIConnectionError,
            httpx.NetworkError,
            ConnectionError,
        ),
    ):
        return ErrorClassification(
            code=ErrorCode.API_CONNECTION,
            retryable=True,
            safe_message=(
                "Unable to connect to the required service."
            ),
        )

    if isinstance(exc, openai.APIStatusError):

        status = exc.status_code

        if status in {401, 403}:
            return ErrorClassification(
                ErrorCode.AUTHENTICATION_FAILED,
                False,
                "AI service access is unavailable.",
            )

        if status == 429:
            return ErrorClassification(
                ErrorCode.RATE_LIMIT_EXCEEDED,
                False,
                "The AI service request limit has been reached.",
            )

        if status >= 500:
            return ErrorClassification(
                ErrorCode.API_SERVER_ERROR,
                status in {500, 502, 503, 504},
                "The AI service is temporarily unavailable.",
            )

        return ErrorClassification(
            ErrorCode.API_REQUEST_FAILED,
            False,
            "The AI service could not process the request.",
        )

    if isinstance(exc, ToolExecutionError):
        return ErrorClassification(
            code=ErrorCode.TOOL_EXECUTION_FAILED,
            retryable=False,
            safe_message=(
                "A required policy or expense tool could "
                "not complete. No compliance decision "
                "has been established."
            ),
        )

    return ErrorClassification(
        code=ErrorCode.INTERNAL_ERROR,
        retryable=False,
        safe_message=(
            "The request could not be completed safely. "
            "Please try again later."
        ),
    )
