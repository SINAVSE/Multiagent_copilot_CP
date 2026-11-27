
from dataclasses import dataclass
import re
import unicodedata


ORGANIZATION_ALIASES = {
    "GitLab": ("gitlab",),
    "Rice University": ("rice university",),
    "University of Arizona": ("university of arizona",),
    "University of Utah": ("university of utah",),
    "University of Connecticut": (
        "university of connecticut",
        "uconn",
    ),
    "University of Colorado System": (
        "university of colorado system",
    ),
    "University of Washington": (
        "university of washington",
    ),
}

COUNTRY_ALIASES = {
    "US": ("us", "usa", "united states"),
    "IN": ("in", "india"),
    "GB": ("gb", "uk", "united kingdom"),
    "AU": ("au", "australia"),
    "NZ": ("nz", "new zealand"),
    "DE": ("de", "germany"),
    "BE": ("be", "belgium"),
    "NL": ("nl", "netherlands"),
    "ES": ("es", "spain"),
    "JP": ("jp", "japan"),
    "KR": ("kr", "south korea"),
    "IL": ("il", "israel"),
    "CA": ("ca", "canada"),
    "FR": ("fr", "france"),
    "IE": ("ie", "ireland"),
    "SG": ("sg", "singapore"),

}


class PolicyScopeError(ValueError):

    # Missing, unsupported or conflicting policy scope.
    pass


def normalize(value):

    if not isinstance(value, str):
        raise PolicyScopeError(
            "Expected a text value."
        )

    return " ".join(
        unicodedata.normalize(
            "NFKC", value
        ).casefold().split()
    )


def canonical_organization(value):

    key = normalize(value)

    for organization, aliases in ORGANIZATION_ALIASES.items():

        if key in aliases:
            return organization

    raise PolicyScopeError(
        "Please select a supported organization."
    )


def canonical_country(value):

    key = normalize(value)

    for country, aliases in COUNTRY_ALIASES.items():

        if key in aliases:
            return country

    raise PolicyScopeError(
        "Unsupported employment country."
    )


@dataclass(frozen=True)
class PolicyScope:

    organization: str
    employment_country: str | None = None

    def __post_init__(self):

        organization = canonical_organization(
            self.organization
        )

        country = self.employment_country

        if country is not None:
            country = canonical_country(country)

        object.__setattr__(
            self, "organization", organization
        )

        object.__setattr__(
            self, "employment_country", country
        )

    def to_dict(self):

        return {
            "organization": self.organization,
            "employment_country": self.employment_country,
        }

    def organization_filter(self):

        return {
            "organization": self.organization
        }


def organizations_in_query(query):

    text = normalize(query)

    matches = set()

    for organization, aliases in ORGANIZATION_ALIASES.items():

        for alias in aliases:

            pattern = (
                r"(?<!\w)"
                + re.escape(alias)
                + r"(?!\w)"
            )

            if re.search(pattern, text):
                matches.add(organization)

    return matches


def resolve_policy_scope(
    query,
    organization=None,
    employment_country=None,
    previous_scope=None,
):

    previous = (
        PolicyScope(**previous_scope)
        if previous_scope is not None
        else None
    )

    explicit = (
        canonical_organization(organization)
        if organization is not None
        else None
    )

    mentioned = organizations_in_query(query)

    if len(mentioned) > 1:

        raise PolicyScopeError(
            "Please ask about one organization at a time."
        )

    mentioned_org = next(iter(mentioned), None)

    if (
        explicit is not None
        and mentioned_org is not None
        and explicit != mentioned_org
    ):

        raise PolicyScopeError(
            "The selected organization conflicts "
            "with the organization in your question."
        )

    selected = (
        explicit
        or mentioned_org
        or (previous.organization if previous else None)
    )

    if selected is None:

        raise PolicyScopeError(
            "Please select your organization."
        )

    country = employment_country

    if (
        country is None
        and previous is not None
        and previous.organization == selected
    ):

        country = previous.employment_country

    scope = PolicyScope(
        organization=selected,
        employment_country=country,
    )

    return {
        "scope": scope,
        "scope_changed": (
            previous is not None
            and previous != scope
        ),
    }


def validate_organization_evidence(scope, result):

    if not isinstance(scope, PolicyScope):

        raise PolicyScopeError(
            "A validated policy scope is required."
        )

    for candidate in result.retrieved_chunks:

        if candidate.chunk.organization != scope.organization:

            raise PolicyScopeError(
                "Retrieved evidence belongs to "
                "a different organization."
            )

    return result
