from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "streamlit_app.py"
)


def make_result(intent, status, answer):
    return {
        "intent": intent,
        "status": status,
        "final_answer": answer,
        "is_approved": False,
        "plan": ["Process employee request"],
        "selected_tools": [],
        "missing_information": [],
        "tool_results": {},
        "retrieved_evidence": [],
        "critic_feedback": None,
        "retry_count": 0,
        "execution_log": [],
    }


def render_result(result):

    app = AppTest.from_file(
        str(APP_PATH),
        default_timeout=30,
    )

    app.session_state["conversation_id"] = (
        "cap10-ui-test"
    )

    app.session_state["messages"] = [
        {
            "role": "user",
            "content": "Example request",
        },
        {
            "role": "assistant",
            "content": result["final_answer"],
        },
    ]

    app.session_state["last_result"] = result

    app.run()

    return app


@pytest.mark.parametrize(
    "intent,status,answer,expected_message",
    [
        (
            "GREETING",
            "greeting",
            "Hello! How can I assist you?",
            "Greeting handled successfully.",
        ),
        (
            "OUT_OF_SCOPE",
            "out_of_scope",
            "This request is outside my scope.",
            "This request is outside the supported business scope.",
        ),
        (
            "CLARIFICATION_REQUIRED",
            "clarification_required",
            "Please provide additional information.",
            "Additional information is required before proceeding.",
        ),
    ],
)
def test_direct_response_status_rendering(
    intent,
    status,
    answer,
    expected_message,
):

    app = render_result(
        make_result(intent, status, answer)
    )

    assert len(app.exception) == 0

    info_messages = [
        element.value for element in app.info
    ]

    assert expected_message in info_messages

    assert len(app.error) == 0

    captions = [
        element.value for element in app.caption
    ]

    expected_intent = (
        "Request intent: "
        + intent.replace("_", " ").title()
    )

    assert expected_intent in captions


def test_direct_response_does_not_require_citations():

    app = render_result(
        make_result(
            "GREETING",
            "greeting",
            "Hello!",
        )
    )

    assert len(app.exception) == 0

    captions = [
        element.value for element in app.caption
    ]

    assert (
        "Policy retrieval was not required for this response."
        in captions
    )

    assert (
        "Evidence retrieval was not required."
        in captions
    )


def test_direct_response_is_not_critic_failure():

    app = render_result(
        make_result(
            "OUT_OF_SCOPE",
            "out_of_scope",
            "This request is outside my scope.",
        )
    )

    assert len(app.exception) == 0
    assert len(app.error) == 0

    markdown_messages = [
        element.value for element in app.markdown
    ]

    assert any(
        "Critic review:" in message
        and "Not required" in message
        for message in markdown_messages
    )


@pytest.mark.parametrize(
    "failed_node,error_code",
    [
        ("planner", "malformed_json"),
        ("executor", "tool_execution_failed"),
        ("critic", "authentication_failed"),
    ],
)
def test_technical_failure_status_rendering(
    failed_node,
    error_code,
):

    result = make_result(
        "POLICY_QUERY",
        "technical_failure",
        "No verified policy or expense decision has been established.",
    )

    result.update({
        "failed_node": failed_node,
        "error_code": error_code,
        "technical_retry_count": 1,
        "technical_errors": [
            {
                "agent": failed_node,
                "error_code": error_code,
                "will_retry": False,
            }
        ],
    })

    app = render_result(result)

    assert len(app.exception) == 0

    errors = [
        element.value for element in app.error
    ]

    assert (
        "The request could not be completed safely."
        in errors
    )

    captions = [
        element.value for element in app.caption
    ]

    assert (
        f"Failure category: {error_code}" in captions
    )

    assert (
        f"Failed stage: {failed_node}" in captions
    )

    markdown = [
        element.value for element in app.markdown
    ]

    assert any(
        "No verified approval" in value
        for value in markdown
    )

    assert any(
        "Technical retries:" in value
        for value in markdown
    )


def test_recovered_response_displays_retry_count():

    result = make_result(
        "POLICY_QUERY",
        "approved",
        "Mock supported policy response.",
    )

    result.update({
        "is_approved": True,
        "technical_retry_count": 1,
        "technical_errors": [
            {
                "agent": "planner",
                "error_code": "malformed_json",
                "will_retry": True,
            }
        ],
    })

    app = render_result(result)

    assert len(app.exception) == 0
    assert len(app.error) == 0

    success = [
        element.value for element in app.success
    ]

    assert "Critic review passed" in success

    markdown = [
        element.value for element in app.markdown
    ]

    assert any(
        "Technical retries:" in value
        and "1" in value
        for value in markdown
    )


# --------------------------------------------------
# CAP-12: Runtime budget UI validation
# --------------------------------------------------


def make_runtime_budget(
    agent_steps=3,
    accounted_tokens=450,
    estimated_tokens=0,
):

    return {
        "max_agent_steps": 10,
        "agent_steps": agent_steps,
        "remaining_steps": max(
            0, 10 - agent_steps
        ),
        "max_tokens": 32000,
        "max_completion_tokens": 2048,
        "llm_calls": 2,
        "embedding_calls": 1,
        "input_tokens": 300,
        "output_tokens": 150,
        "total_tokens": 450,
        "estimated_tokens": estimated_tokens,
        "accounted_tokens": accounted_tokens,
        "remaining_tokens": max(
            0, 32000 - accounted_tokens
        ),
        "estimated_usage_calls": (
            1 if estimated_tokens else 0
        ),
        "unreported_usage_calls": 0,
        "step_history": [
            {"step": 1, "agent": "planner"},
            {"step": 2, "agent": "executor"},
            {"step": 3, "agent": "critic"},
        ],
    }


def test_runtime_budget_metrics_render():

    result = make_result(
        "POLICY_QUERY",
        "approved",
        "Mock approved policy response.",
    )

    result.update({
        "is_approved": True,
        "runtime_budget": make_runtime_budget(),
        "budget_exhausted": False,
    })

    app = render_result(result)

    assert len(app.exception) == 0

    metrics = {
        item.label: str(item.value)
        for item in app.metric
    }

    assert metrics["Agent steps"] == "3 / 10"

    assert metrics["Accounted chat tokens"] == "450"

    assert metrics["LLM calls"] == "2"

    assert metrics["Embedding invocations"] == "1"


@pytest.mark.parametrize(
    "reason,expected_caption",
    [
        (
            "agent_steps",
            "Agent-step limit reached.",
        ),
        (
            "tokens",
            "Chat-token budget exhausted.",
        ),
    ],
)
def test_budget_exhaustion_status_rendering(
    reason,
    expected_caption,
):

    result = make_result(
        "POLICY_QUERY",
        "technical_failure",
        "No verified policy or expense decision.",
    )

    result.update({
        "error_code": "budget_exhausted",
        "budget_exhausted": True,
        "budget_exhaustion_reason": reason,
        "failed_node": "critic",
        "runtime_budget": make_runtime_budget(
            agent_steps=10,
            accounted_tokens=32000,
        ),
    })

    app = render_result(result)

    assert len(app.exception) == 0

    errors = [
        item.value for item in app.error
    ]

    captions = [
        item.value for item in app.caption
    ]

    assert "Runtime budget exhausted." in errors

    assert expected_caption in captions

    assert len(app.success) == 0


def test_estimated_token_usage_is_identified():

    result = make_result(
        "POLICY_QUERY",
        "approved",
        "Mock approved policy response.",
    )

    result.update({
        "is_approved": True,
        "runtime_budget": make_runtime_budget(
            accounted_tokens=1450,
            estimated_tokens=1000,
        ),
    })

    app = render_result(result)

    assert len(app.exception) == 0

    captions = [
        item.value for item in app.caption
    ]

    assert any(
        "Estimated usage has been included"
        in item
        for item in captions
    )


def test_token_overrun_does_not_break_ui():

    result = make_result(
        "POLICY_QUERY",
        "technical_failure",
        "Token budget exceeded.",
    )

    result.update({
        "budget_exhausted": True,
        "budget_exhaustion_reason": "tokens",
        "runtime_budget": make_runtime_budget(
            accounted_tokens=40000,
        ),
    })

    app = render_result(result)

    assert len(app.exception) == 0

    metrics = {
        item.label: str(item.value)
        for item in app.metric
    }

    assert metrics["Accounted chat tokens"] == "40,000"


def test_legacy_result_without_budget_still_renders():

    result = make_result(
        "GREETING",
        "greeting",
        "Hello!",
    )

    app = render_result(result)

    assert len(app.exception) == 0

    metric_labels = [
        item.label for item in app.metric
    ]

    assert "Agent steps" not in metric_labels

