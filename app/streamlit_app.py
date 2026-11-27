import sys
import sqlite3
import uuid
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.workflow import EmployeeOpsWorkflow


CHECKPOINT_DB = (
    ROOT_DIR
    / "data"
    / "checkpoints"
    / "employee_ops.sqlite"
)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Employee Operations Copilot",
    page_icon="💼",
    layout="centered",
)


# ---------------------------------------------------------
# Workflow initialization
# ---------------------------------------------------------

@st.cache_resource
def get_workflow():
    return EmployeeOpsWorkflow()


workflow = get_workflow()


# ---------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

if "clear_question_on_next_run" not in st.session_state:
    st.session_state.clear_question_on_next_run = False


# Clear input only after successful workflow execution.
# This must happen before the text_area is created.

if st.session_state.clear_question_on_next_run:
    st.session_state.question_input = ""
    st.session_state.clear_question_on_next_run = False


# ---------------------------------------------------------
# Previous conversation helpers
# ---------------------------------------------------------

def get_saved_thread_ids():
    """
    Return saved LangGraph conversation IDs,
    ordered from newest to oldest.
    """

    if not CHECKPOINT_DB.exists():
        return []

    connection = None

    try:
        connection = sqlite3.connect(
            f"{CHECKPOINT_DB.as_uri()}?mode=ro",
            uri=True,
        )

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                thread_id,
                MAX(rowid) AS latest_checkpoint
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY latest_checkpoint DESC
            """
        )

        rows = cursor.fetchall()

        return [row[0] for row in rows]

    except sqlite3.Error as error:
        st.sidebar.warning(
            f"Could not retrieve saved conversations: {error}"
        )
        return []

    finally:
        if connection:
            connection.close()


def load_selected_conversation(selected_thread_id):
    """
    Load the latest saved question and answer
    from the selected LangGraph conversation.
    """

    if not selected_thread_id:
        st.sidebar.warning("Please select a conversation.")
        return

    try:
        saved_state = workflow.get_thread_state(
            selected_thread_id
        )

        if hasattr(saved_state, "values"):
            state_values = saved_state.values
        else:
            state_values = saved_state

        if not isinstance(state_values, dict):
            st.sidebar.error(
                "Could not read the saved conversation."
            )
            return

        previous_query = (
            state_values.get("user_query")
            or state_values.get("query")
            or ""
        )

        previous_answer = (
            state_values.get("final_answer", "")
        )

        if not previous_query and not previous_answer:
            st.sidebar.info(
                "No conversation messages found."
            )
            return

        # Restore the selected conversation ID.
        st.session_state.conversation_id = selected_thread_id

        # Restore the latest saved exchange.
        st.session_state.messages = []

        if previous_query:
            st.session_state.messages.append({
                "role": "user",
                "content": previous_query,
            })

        if previous_answer:
            st.session_state.messages.append({
                "role": "assistant",
                "content": previous_answer,
            })

        # Clear any unsent question.
        st.session_state.question_input = ""

        st.rerun()

    except Exception as error:
        st.sidebar.error(
            f"Unable to load conversation: {error}"
        )


# ---------------------------------------------------------
# Sidebar: conversation controls
# ---------------------------------------------------------

st.sidebar.title("Conversations")


# New conversation button

if st.sidebar.button(
    "＋ New conversation",
    use_container_width=True,
):

    st.session_state.conversation_id = str(uuid.uuid4())

    st.session_state.messages = []

    st.session_state.question_input = ""

    st.rerun()


# Retrieve saved conversations

saved_thread_ids = get_saved_thread_ids()


# Previous conversations dropdown

if saved_thread_ids:

    selected_thread_id = st.sidebar.selectbox(
        "Previous conversations",
        options=saved_thread_ids,
        format_func=lambda thread_id: (
            f"Conversation {thread_id[:8]}"
        ),
        index=0,
        key="selected_conversation",
    )

    # Load selected conversation button

    if st.sidebar.button(
        "Load selected conversation",
        use_container_width=True,
    ):

        load_selected_conversation(
            selected_thread_id
        )

else:

    st.sidebar.info(
        "No previous conversations available."
    )


# ---------------------------------------------------------
# Main conversation interface
# ---------------------------------------------------------

st.title("Employee Operations Copilot")

st.caption(
    "Ask questions about HR policies, business travel, "
    "per-diem, gifts, team outings and expense compliance."
)


# ---------------------------------------------------------
# Display conversation history
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ---------------------------------------------------------
# User input
# ---------------------------------------------------------

with st.container():

    query = st.text_area(
        "Your question",
        key="question_input",
        height=90,
        placeholder="Type your question here...",
        label_visibility="collapsed",
    )

    send_clicked = st.button(
        "Send",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Process user request
# ---------------------------------------------------------

if send_clicked:

    execution_query = query.strip()

    if not execution_query:

        st.warning(
            "Please enter a question before sending."
        )

    else:

        # Preserve existing conversation history.
        conversation_history = list(
            st.session_state.messages
        )

        try:

            with st.spinner(
                "Preparing your response..."
            ):

                result = workflow.run(
                    query=execution_query,
                    expense_details={},
                    thread_id=st.session_state.conversation_id,
                    conversation_history=conversation_history,
                )

            final_answer = (
                result.get("final_answer")
                if isinstance(result, dict)
                else None
            )

            if not final_answer:

                raise RuntimeError(
                    "The workflow did not return a final answer."
                )

            # Save user message.
            st.session_state.messages.append({
                "role": "user",
                "content": execution_query,
            })

            # Save assistant response.
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
            })

            # Clear input on the next rerun.
            # This happens only after successful execution.
            st.session_state.clear_question_on_next_run = True

            st.rerun()

        except Exception as error:

            import logging
            logging.exception("Agent request failed")

            st.error(
                "I could not complete your request. "
                "Please try again."
            )

            st.caption(f"Error type: {type(error).__name__}")