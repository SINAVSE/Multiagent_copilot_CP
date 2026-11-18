from pydantic import ValidationError
from schemas import ComplianceResponse

#print("TEST SCRIPT STARTED")
# Test 1: Valid response
valid_response = {
    "request_classification": "Domestic travel meal expense",
    "extracted_information": {
        "amount": 1500,
        "currency": "INR",
        "employees": 1,
        "travel_days": 1
    },
    "applicable_policy": [
        "Domestic meal per-diem: INR 1800 per employee per travel day."
    ],
    "calculation": "INR 1800 - INR 1500 = INR 300 remaining",
    "compliance_status": "within_policy",
    "missing_information": [],
    "reason": "The expense is below the applicable daily limit.",
    "recommendation": "Review supporting documentation."
}

result = ComplianceResponse.model_validate(valid_response)

print("Test 1 PASSED: Valid response accepted.")
print(result.model_dump_json(indent=2))


# Test 2: Invalid compliance status
invalid_response = valid_response.copy()
invalid_response["compliance_status"] = "approved"

try:
    ComplianceResponse.model_validate(invalid_response)
    print("Test 2 FAILED: Invalid status was accepted.")
except ValidationError:
    print("Test 2 PASSED: Invalid status rejected.")
