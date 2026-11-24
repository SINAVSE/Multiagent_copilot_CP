from src.tools.policy_extractor import (
    ExtractedExpensePolicy,
    PolicyEvidenceQuote,
    validate_policy_extraction,
)


def test_valid_policy_evidence():
    evidence = [
        {
            "chunk_id": "CHUNK-01",
            "text": "Hotel accommodation limit is INR 6000 per night.",
        }
    ]

    extracted = ExtractedExpensePolicy(
        limit_type="fixed",
        policy_limit=6000,
        currency="INR",
        limit_unit="per night",
        evidence_quotes=[
            PolicyEvidenceQuote(
                chunk_id="CHUNK-01",
                quote="Hotel accommodation limit is INR 6000 per night.",
            )
        ],
    )

    is_valid, issues = validate_policy_extraction(
        extracted, evidence
    )

    assert is_valid is True
    assert issues == []


def test_invalid_chunk_id():
    extracted = ExtractedExpensePolicy(
        limit_type="fixed",
        policy_limit=6000,
        evidence_quotes=[
            PolicyEvidenceQuote(
                chunk_id="UNKNOWN",
                quote="Hotel limit is INR 6000.",
            )
        ],
    )

    is_valid, issues = validate_policy_extraction(
        extracted, []
    )

    assert is_valid is False
    assert any("Unknown evidence chunk" in issue for issue in issues)


def test_mismatched_quotation():
    evidence = [
        {
            "chunk_id": "CHUNK-01",
            "text": "Hotel accommodation limit is INR 6000 per night.",
        }
    ]

    extracted = ExtractedExpensePolicy(
        limit_type="fixed",
        policy_limit=6000,
        evidence_quotes=[
            PolicyEvidenceQuote(
                chunk_id="CHUNK-01",
                quote="Hotel accommodation limit is INR 8000 per night.",
            )
        ],
    )

    is_valid, issues = validate_policy_extraction(
        extracted, evidence
    )

    assert is_valid is False
    assert any("Quotation does not match" in issue for issue in issues)

def test_extracted_limit_not_in_quote():
    evidence = [
        {
            "chunk_id": "CHUNK-01",
            "text": "Hotel accommodation limit is INR 6000 per night.",
        }
    ]

    extracted = ExtractedExpensePolicy(
        limit_type="fixed",
        policy_limit=8000,
        currency="INR",
        evidence_quotes=[
            PolicyEvidenceQuote(
                chunk_id="CHUNK-01",
                quote="Hotel accommodation limit is INR 6000 per night.",
            )
        ],
    )

    is_valid, issues = validate_policy_extraction(
        extracted, evidence
    )

    assert is_valid is False
    assert any("policy limit does not appear" in issue for issue in issues)