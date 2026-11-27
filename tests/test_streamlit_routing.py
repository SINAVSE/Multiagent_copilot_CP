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
