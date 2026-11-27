from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

ComplianceStatus = Literal[
    "within_policy",
    "requires_approval",
    "exceeds_policy",
    "unable_to_determine",
]

class ExtractedInformation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount: Optional[float] = None
    currency: Optional[str] = None
    employees: Optional[int] = None
    travel_days: Optional[int] = None

class ComplianceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_classification: str
    extracted_information: ExtractedInformation
    applicable_policy: list[str] = Field(default_factory=list)
    calculation: Optional[str] = None
    compliance_status: ComplianceStatus
    missing_information: list[str] = Field(default_factory=list)
    reason: str
    recommendation: str