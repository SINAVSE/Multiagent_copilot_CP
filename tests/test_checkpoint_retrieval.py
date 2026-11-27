
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing workflow
from src.agents.workflow import EmployeeOpsWorkflow


def main():

    # Initialize the existing workflow and SQLite checkpointer
    workflow = EmployeeOpsWorkflow()

    try:
        graph = workflow.build()

        # Original conversation ID / LangGraph thread ID
        config = {
            "configurable": {
                "thread_id": "758fdc88"
            }
        }

        # Retrieve previously saved checkpoint
        saved_state = graph.get_state(config)

        print("\n========== CHECKPOINT RETRIEVAL ==========")
        print("Conversation ID:", "758fdc88")
        print("Database:", workflow.database_path)

        if not saved_state or not saved_state.values:
            print("\nNo saved checkpoint state found for this ID.")
        else:
            print("\nCheckpoint successfully retrieved!")
            print("\nSaved workflow state:")
            print(saved_state.values)

            print("\nCheckpoint metadata:")
            print(saved_state.metadata)

            print("\nNext workflow nodes:")
            print(saved_state.next)

        print("\n==========================================")

    finally:
        workflow.connection.close()


if __name__ == "__main__":
    main()