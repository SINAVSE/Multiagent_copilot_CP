
import hashlib
import json

import pytest

from src.agents.applicability_registry import (
    ApplicabilityRegistry,
)
from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)


@pytest.fixture
def registry_files(tmp_path):

    manifest_path = tmp_path / "manifest.jsonl"
    registry_path = tmp_path / "registry.json"

    document = {
        "document_id": "TEST-001",
        "title": "India Employee Benefits",
        "organization": "GitLab",
        "source_url": "https://example.org/policy",
        "processed_file_hash": "a" * 64,
    }

    manifest_path.write_text(
        json.dumps(document) + "\n",
        encoding="utf-8",
    )

    record = {
        **document,
        "candidate_applicability": "COUNTRY_SPECIFIC",
        "candidate_countries": ["IN"],
        "verified_applicability": "UNKNOWN",
        "verified_countries": [],
        "review_status": "pending_content_review",
        "eligible_for_definitive_decision": False,
    }

    payload = {
        "version": "2.0-draft",
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "total_documents": 1,
        "documents": [record],
    }

    registry_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return manifest_path, registry_path, payload


def load_registry(files):

    manifest_path, registry_path, payload = files

    registry_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return ApplicabilityRegistry(
        manifest_path,
        registry_path,
    )


def test_valid_registry(registry_files):

    registry = load_registry(registry_files)

    assert registry.summary()["total_documents"] == 1
    assert registry.summary()["verified_documents"] == 0


def test_manifest_hash_mismatch(registry_files):

    registry_files[2]["manifest_sha256"] = "invalid"

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_duplicate_document_id(registry_files):

    payload = registry_files[2]

    payload["documents"].append(
        dict(payload["documents"][0])
    )

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_missing_document(registry_files):

    registry_files[2]["documents"] = []

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_wrong_document_count(registry_files):

    registry_files[2]["total_documents"] = 2

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


@pytest.mark.parametrize(
    "field",
    [
        "organization",
        "source_url",
        "processed_file_hash",
        "title",
    ],
)
def test_metadata_mismatch(registry_files, field):

    registry_files[2]["documents"][0][field] = (
        "modified value"
    )

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_draft_cannot_claim_verification(registry_files):

    registry_files[2]["documents"][0][
        "review_status"
    ] = "verified"

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_draft_cannot_allow_definitive_decision(
    registry_files,
):

    registry_files[2]["documents"][0][
        "eligible_for_definitive_decision"
    ] = True

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_invalid_candidate_country(registry_files):

    registry_files[2]["documents"][0][
        "candidate_countries"
    ] = ["INVALID"]

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


@pytest.mark.parametrize(
    "country, expected",
    [
        ("India", True),
        ("Germany", False),
        ("United States", False),
        (None, False),
    ],
)
def test_research_country_filter(
    registry_files, country, expected
):

    registry = load_registry(registry_files)

    result = registry.research_access(
        PolicyScope("GitLab", country),
        "TEST-001",
    )

    assert result["allowed"] is expected


def test_cross_organization_research(registry_files):

    registry = load_registry(registry_files)

    result = registry.research_access(
        PolicyScope("Rice University", "US"),
        "TEST-001",
    )

    assert result["allowed"] is False


def test_matching_country_not_definitive(
    registry_files,
):

    registry = load_registry(registry_files)

    scope = PolicyScope("GitLab", "India")

    research = registry.research_access(
        scope, "TEST-001"
    )

    assessment = registry.assess(
        scope, "TEST-001"
    )

    assert research["allowed"] is True
    assert research["definitive_decision_allowed"] is False
    assert assessment["allowed"] is False


def test_missing_registry_record(registry_files):

    registry = load_registry(registry_files)

    result = registry.research_access(
        PolicyScope("GitLab", "India"),
        "UNKNOWN",
    )

    assert result["allowed"] is False


def test_registry_returns_defensive_copy(registry_files):

    registry = load_registry(registry_files)

    record = registry.get("TEST-001")

    record["candidate_countries"].append("DE")

    assert registry.get("TEST-001")[
        "candidate_countries"
    ] == ["IN"]


def test_unknown_applicability_research_only(
    registry_files,
):

    record = registry_files[2]["documents"][0]

    record["candidate_applicability"] = "UNKNOWN"
    record["candidate_countries"] = []

    registry = load_registry(registry_files)

    scope = PolicyScope("GitLab", "India")

    assert registry.research_access(
        scope, "TEST-001"
    )["allowed"] is True

    assert registry.assess(
        scope, "TEST-001"
    )["allowed"] is False


@pytest.mark.parametrize(
    "version",
    [None, "2.0", "verified", "3.0-draft"],
)
def test_unsupported_registry_version(
    registry_files, version
):

    registry_files[2]["version"] = version

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_unsupported_candidate_type(registry_files):

    registry_files[2]["documents"][0][
        "candidate_applicability"
    ] = "GLOBAL"

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_unknown_scope_with_country_rejected(
    registry_files,
):

    registry_files[2]["documents"][0][
        "candidate_applicability"
    ] = "UNKNOWN"

    with pytest.raises(PolicyScopeError):
        load_registry(registry_files)


def test_document_id_inventory(registry_files):

    registry = load_registry(registry_files)

    assert registry.document_ids() == (
        "TEST-001",
    )
