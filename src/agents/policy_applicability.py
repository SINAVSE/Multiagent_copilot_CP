
from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
    canonical_country,
)


def assess_applicability(scope, metadata):

    if not isinstance(scope, PolicyScope):

        raise PolicyScopeError(
            "A validated policy scope is required."
        )

    if not isinstance(metadata, dict):

        return {
            "allowed": False,
            "reason": "missing_policy_metadata",
        }

    if metadata.get("organization") != scope.organization:

        return {
            "allowed": False,
            "reason": "organization_mismatch",
        }

    if metadata.get("review_status") != "verified":

        return {
            "allowed": False,
            "reason": "unverified_applicability",
        }

    applicability = metadata.get(
        "verified_applicability", "UNKNOWN"
    )

    countries = metadata.get(
        "verified_countries", []
    )

    if not isinstance(countries, list):

        return {
            "allowed": False,
            "reason": "invalid_country_metadata",
        }

    try:

        countries = [
            canonical_country(country)
            for country in countries
        ]

    except PolicyScopeError:

        return {
            "allowed": False,
            "reason": "invalid_country_metadata",
        }

    if len(countries) != len(set(countries)):

        return {
            "allowed": False,
            "reason": "duplicate_country_metadata",
        }

    if applicability == "GLOBAL":

        if countries:

            return {
                "allowed": False,
                "reason": "conflicting_global_metadata",
            }

        return {
            "allowed": True,
            "reason": "global_scope_match",
            "eligibility_established": False,
        }

    if applicability not in (
        "COUNTRY_SPECIFIC",
        "MULTI_COUNTRY",
    ):

        return {
            "allowed": False,
            "reason": "unknown_applicability",
        }

    if (
        applicability == "COUNTRY_SPECIFIC"
        and len(countries) != 1
    ):

        return {
            "allowed": False,
            "reason": "invalid_country_metadata",
        }

    if (
        applicability == "MULTI_COUNTRY"
        and len(countries) < 2
    ):

        return {
            "allowed": False,
            "reason": "invalid_country_metadata",
        }

    if scope.employment_country is None:

        return {
            "allowed": False,
            "reason": "employment_country_required",
        }

    if scope.employment_country not in countries:

        return {
            "allowed": False,
            "reason": "country_mismatch",
        }

    return {
        "allowed": True,
        "reason": "country_scope_match",
        "eligibility_established": False,
    }


def validate_applicable_evidence(
    scope,
    retrieval_result,
    applicability_registry,
):

    if not isinstance(scope, PolicyScope):

        raise PolicyScopeError(
            "A validated policy scope is required."
        )

    if not isinstance(applicability_registry, dict):

        raise PolicyScopeError(
            "Invalid applicability registry."
        )

    for candidate in retrieval_result.retrieved_chunks:

        chunk = candidate.chunk

        if chunk.organization != scope.organization:

            raise PolicyScopeError(
                "Cross-organization evidence detected."
            )

        metadata = applicability_registry.get(
            chunk.document_id
        )

        if (
            not isinstance(metadata, dict)
            or metadata.get("document_id")
            != chunk.document_id
        ):

            raise PolicyScopeError(
                "Missing or mismatched applicability record."
            )

        decision = assess_applicability(
            scope,
            metadata,
        )

        if not decision["allowed"]:

            raise PolicyScopeError(
                "Inapplicable policy evidence: "
                + decision["reason"]
            )

    return retrieval_result
