
import sys
from pathlib import Path
import uuid
import streamlit as st

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.workflow import EmployeeOpsWorkflow


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Employee Operations Copilot",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Employee Operations & Expense Compliance Copilot")

st.caption(
    "AI-assisted employee policy guidance and expense compliance "
    "using GitLab policy documents."
)

st.info(
    "This assistant provides policy guidance, not final HR decisions "
    "or reimbursement approvals."
)

# --------------------------------------------------
# Conversation session management
# --------------------------------------------------

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# --------------------------------------------------
# Workflow initialization
# --------------------------------------------------

@st.cache_resource
def get_workflow():
    return EmployeeOpsWorkflow()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("About the Copilot")

    st.write(
        "The system uses a Planner, Executor and Critic workflow "
        "with retrieval-augmented policy answers."
    )

    st.markdown("**Policy knowledge base:** 9 GitLab documents")
    st.markdown("**Workflow:** Planner - Intent Router - Direct Response or Executor - Critic")

    st.divider()

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.caption(
        f"Conversation ID: "
        f"{st.session_state.conversation_id[:8]}"
    )

    st.divider()
    st.caption("Bill/receipt extraction is not yet implemented.")


# --------------------------------------------------
# Previous conversation
# --------------------------------------------------

# Conversation display container
conversation_container = st.container()

with conversation_container:
    st.subheader("Conversation")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# --------------------------------------------------
# User input
# --------------------------------------------------

st.subheader("Ask an employee policy question")

query = st.text_area(
    "Your question",
    placeholder=(
        "Example: What is the notice requirement for planned time off?"
    ),
    height=110,
)

include_expense = st.checkbox(
    "Include expense details for an eligibility check"
)


# --------------------------------------------------
# Expense details
# --------------------------------------------------

claimed_amount = None
currency = "INR"
headcount = None

if include_expense:

    st.markdown("#### Expense details")

    col1, col2, col3 = st.columns(3)

    with col1:
        claimed_amount = st.number_input(
            "Claimed amount",
            min_value=0.01,
            value=None,
            step=100.0,
            placeholder="Enter amount",
        )

    with col2:
        currency = st.selectbox(
            "Currency",
            ["INR", "USD", "EUR", "GBP"],
        )

    with col3:
        headcount = st.number_input(
            "Number of people",
            min_value=1,
            value=None,
            step=1,
            placeholder="Enter number",
        )


# --------------------------------------------------
# Submit
# --------------------------------------------------

submitted = st.button(
    "Run Employee Operations Copilot",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# Execute workflow
# --------------------------------------------------

if submitted:

    if not query.strip():
        st.warning("Please enter your question.")
        st.stop()

    execution_query = query.strip()
    expense_details = {}

    if include_expense:

        if claimed_amount is None or headcount is None:
            st.warning(
                "Please enter the claimed amount and number of people."
            )
            st.stop()

        expense_details = {
            "claimed_amount": float(claimed_amount),
            "currency": currency,
            "headcount": int(headcount),
        }

    try:

        with st.spinner(
            "Employee Operations Copilot is processing your request..."
        ):

            workflow = get_workflow()

            result = workflow.run(
                query=execution_query,
                expense_details=expense_details,
                thread_id=st.session_state.conversation_id,
                conversation_history=st.session_state.messages,
            )

        # Save the user's message
        st.session_state.messages.append({
            "role": "user",
            "content": query.strip(),
        })

        # Display the user's message immediately
        # with st.chat_message("user"):
        #     st.markdown(query.strip())

        # Save the assistant's response
        assistant_response = result.get(
            "final_answer",
            "The workflow did not produce a final answer."
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_response,
        })

        # Persist the latest workflow result before rerunning.
        st.session_state.last_result = result
        st.rerun()

    except Exception as exc:

        st.error(
            "The application encountered an error while processing "
            "your request."
        )

        st.caption(f"Error type: {type(exc).__name__}")


# --------------------------------------------------
# Render persisted result independently of submission
# --------------------------------------------------

result = st.session_state.last_result

if result is not None:
    st.divider()
    st.subheader("Latest Copilot Response Details")

    status = result.get("status", "unknown")
    approved = result.get("is_approved", False)
    intent = result.get("intent", "")

    direct_response = status in {
        "greeting",
        "out_of_scope",
        "clarification_required",
    }

    if intent:
        st.caption(
            f"Request intent: {intent.replace('_', ' ').title()}"
        )

    if status == "approved" and approved:
        st.success("Critic review passed")

    elif status == "greeting":
        st.info("Greeting handled successfully.")

    elif status == "out_of_scope":
        st.info(
            "This request is outside the supported business scope."
        )

    elif status == "clarification_required":
        st.info(
            "Additional information is required before proceeding."
        )

    elif status == "critic_rejected":
        st.error("The answer did not pass the review process.")

    else:
        st.warning(f"Workflow status: {status}")

    # --------------------------------------------------
    # Planner output
    # --------------------------------------------------

    with st.expander("View agent plan", expanded=True):

        st.write("**Planned steps**")

        for step in result.get("plan", []):
            st.markdown(f"- {step}")

        st.write("**Selected tools**")
        st.write(result.get("selected_tools", []))

        missing = result.get("missing_information", [])

        if missing:
            st.warning("Missing information identified by Planner")

            for item in missing:
                st.markdown(f"- {item}")

    # --------------------------------------------------
    # Tool results
    # --------------------------------------------------

    tool_results = result.get("tool_results", {})

    # --------------------------------------------------
    # Policy citations
    # --------------------------------------------------

    search_result = tool_results.get(
        "search_hr_policy", {}
    )

    citations = search_result.get("citations", [])

    st.markdown("### Policy Sources")

    if not citations:
        if direct_response:
            st.caption(
                "Policy retrieval was not required for this response."
            )
        else:
            st.info("No policy citations were returned.")

    if citations:
        if not approved:
            st.warning("Sources considered during review; the answer was not approved.")

        for citation in citations:

            title = citation.get(
                "title", "Policy document"
            )

            document_id = citation.get(
                "document_id", ""
            )

            section = citation.get(
                "section_path", "Section not specified"
            )

            source_url = citation.get(
                "source_url", ""
            )

            st.markdown(
                f"**{title}** ({document_id})"
            )

            st.caption(f"Section: {section}")

            if source_url:
                st.markdown(
                    f"[Open source document]({source_url})"
                )

    # --------------------------------------------------
    # Expense assessment
    # --------------------------------------------------

    expense_result = tool_results.get(
        "calculate_expense_eligibility"
    )

    if expense_result:

        st.markdown("### Expense Eligibility Assessment")
        if not approved:
            st.warning("This assessment has not passed Critic review.")

        compliance_status = expense_result.get(
            "compliance_status", "unknown"
        )

        st.write(
            f"**Compliance status:** {compliance_status}"
        )

        # Display calculation metrics
        metric_columns = st.columns(4)

        metrics = [
            ("Claimed amount", "claimed_amount"),
            ("Policy limit", "policy_limit"),
            ("Amount within limit", "eligible_amount"),
            ("Excess amount", "excess_amount"),
        ]

        for column, (label, field) in zip(
            metric_columns, metrics
        ):

            value = expense_result.get(field)

            if value is not None:
                column.metric(
                    label,
                    f"{expense_result.get('currency', '')} {value}",
                )
            else:
                column.metric(label, "Not determined")

        # Calculation details
        calculation = expense_result.get("calculation")

        if calculation:
            st.write("**Calculation details**")
            st.write(calculation)

        # Approval requirement
        approval_required = expense_result.get(
            "approval_required"
        )

        if approval_required is True:
            st.warning("Additional approval is required.")

        elif approval_required is False:
            st.write(
                "**Approval requirement:** No additional approval "
                "requirement was identified in the extracted evidence."
            )

        else:
            st.info(
                "**Approval requirement:** Not established by "
                "the available policy evidence."
            )

        # Recommendation
        recommendation = expense_result.get(
            "recommendation"
        )

        if recommendation:
            st.write("**Recommendation**")
            st.write(recommendation)

        # Supporting policy evidence
        st.markdown("#### Supporting Policy Evidence")

        policy_evidence = expense_result.get(
            "policy_evidence", []
        )

        if policy_evidence:

            for quote in policy_evidence:
                st.markdown(f"> {quote}")

        else:
            st.warning(
                "No extracted policy quotations are available "
                "for this expense calculation."
            )

        # Missing information
        expense_missing = expense_result.get(
            "missing_information", []
        )

        if expense_missing:

            st.warning("Additional information required")

            for item in expense_missing:
                st.markdown(f"- {item}")

    # --------------------------------------------------
    # Retrieved evidence
    # --------------------------------------------------

    evidence = result.get("retrieved_evidence", [])

    with st.expander("View retrieved policy evidence"):
        st.write(f"Retrieved chunks: {len(evidence)}")
        if not evidence:
            if direct_response:
                st.caption(
                    "Evidence retrieval was not required."
                )
            else:
                st.info("No policy evidence was retrieved.")
        for item in evidence:
            st.markdown(f"**{item.get('title', 'Policy document')}**")
            st.caption(
                f"Chunk: {item.get('chunk_id', '')} | "
                f"Section: {item.get('section_path') or item.get('section', '')}"
            )
            st.write(item.get("text", ""))
            st.divider()

    # --------------------------------------------------
    # Critic execution trace
    # --------------------------------------------------

    with st.expander(
        "View agent execution and Critic review"
    ):

        st.write(f"**Final workflow status:** {status}")
        if direct_response:
            st.write("**Critic review:** Not required")
        else:
            st.write(f"**Critic approved:** {approved}")

        st.write(
            f"**Retry count:** {result.get('retry_count', 0)}"
        )

        for entry in result.get("execution_log", []):

            st.markdown(
                f"**Agent:** {entry.get('agent', 'Unknown')}"
            )

            st.write(
                f"Approved: {entry.get('approved', 'N/A')}"
            )

            issues = entry.get("issues", [])

            if issues:
                st.write("Issues:")

                for issue in issues:
                    st.markdown(f"- {issue}")

            feedback = entry.get("feedback")

            if feedback:
                st.write(f"Feedback: {feedback}")

    # --------------------------------------------------
    # Complete workflow output
    # --------------------------------------------------

    if st.checkbox("Enable detailed debugging (local only)"):
        with st.expander("View complete workflow output"):
            st.json(result)
