from src.agents.workflow import EmployeeOpsWorkflow


workflow = EmployeeOpsWorkflow()

result = workflow.run(
    query="What are the meal per diem rules for sponsored business travel?"
)

print("\n===== DEBUG: TOOL RESULTS =====")

for tool_name, tool_result in result.get("tool_results", {}).items():
    print(f"\nTool: {tool_name}")
    print("Result keys:", tool_result.keys())
    print(
        "Retrieved evidence count:",
        len(tool_result.get("retrieved_evidence", []))
    )
    for item in tool_result.get("retrieved_evidence", []):
        print("\nChunk ID:", item.get("chunk_id"))
        print("Title:", item.get("title"))
        print("Section:", item.get("section"))
        print("Text:", item.get("text"))
    print("Citations:", tool_result.get("citations"))

print("\n===== WORKFLOW RESULT =====")

print("\nStatus:")
print(result.get("status"))

print("\nSelected tools:")
print(result.get("selected_tools"))

print("\nGenerated answer:")
print(result.get("final_answer"))

print("\nCritic approved:")
print(result.get("is_approved"))

print("\nCritic feedback:")
print(result.get("critic_feedback"))

print("\nExecution log:")
print(result.get("execution_log"))