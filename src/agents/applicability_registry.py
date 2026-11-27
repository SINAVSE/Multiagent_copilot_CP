
from pathlib import Path
from collections import Counter
from copy import deepcopy
import hashlib
import json

from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
    canonical_country,
)
from src.agents.policy_applicability import (
    assess_applicability,
)


class ApplicabilityRegistry:

    def __init__(self, manifest_path, registry_path):

        manifest_path = Path(manifest_path)
        registry_path = Path(registry_path)

        manifest_bytes = manifest_path.read_bytes()

        manifest_hash = hashlib.sha256(
            manifest_bytes
        ).hexdigest()

        documents = [
            json.loads(line)
            for line in manifest_bytes.decode(
                "utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        manifest = self._index_documents(documents)

        registry = json.loads(
            registry_path.read_text(
                encoding="utf-8-sig"
            )
        )

        if registry.get("version") != "2.0-draft":

            raise PolicyScopeError(
                "Unsupported applicability registry version."
            )

        if registry.get("manifest_sha256") != manifest_hash:

            raise PolicyScopeError(
                "Applicability registry manifest hash mismatch."
            )

        records = self._index_documents(
            registry["documents"]
        )

        if set(records) != set(manifest):

            raise PolicyScopeError(
                "Registry document inventory mismatch."
            )

        if registry.get("total_documents") != len(records):

            raise PolicyScopeError(
                "Registry document count mismatch."
            )

        for document_id, record in records.items():

            original = manifest[document_id]

            for field in (
                "organization",
                "source_url",
                "processed_file_hash",
                "title",
            ):

                if (
                    field not in record
                    or field not in original
                    or record[field] != original[field]
                ):

                    raise PolicyScopeError(
                        "Registry metadata mismatch: "
                        + document_id + " / " + field
                    )

            candidate_countries = record.get(
                "candidate_countries"
            )

            if not isinstance(candidate_countries, list):

                raise PolicyScopeError(
                    "Invalid candidate country metadata."
                )

            candidate_type = record.get(
                "candidate_applicability"
            )

            if candidate_type not in (
                "UNKNOWN", "COUNTRY_SPECIFIC"
            ):

                raise PolicyScopeError(
                    "Unsupported candidate applicability."
                )

            if candidate_type == "UNKNOWN" and candidate_countries:

                raise PolicyScopeError(
                    "Conflicting unknown applicability metadata."
                )

            for country in candidate_countries:

                if not isinstance(country, str):

                    raise PolicyScopeError(
                        "Invalid candidate country."
                    )

                canonical_country(country)

            if (
                record.get("candidate_applicability")
                == "COUNTRY_SPECIFIC"
                and len(candidate_countries) != 1
            ):

                raise PolicyScopeError(
                    "Invalid country-specific candidate."
                )

            if registry.get("version") == "2.0-draft":

                if (
                    record.get("review_status") == "verified"
                    or record.get("verified_applicability")
                    != "UNKNOWN"
                    or record.get("verified_countries") != []
                    or record.get(
                        "eligible_for_definitive_decision"
                    ) is not False
                ):

                    raise PolicyScopeError(
                        "Draft registry contains an "
                        "unauthorized verification."
                    )

        self._records = deepcopy(records)
        self.manifest_sha256 = manifest_hash

    @staticmethod
    def _index_documents(documents):

        if not isinstance(documents, list):

            raise PolicyScopeError(
                "Invalid document inventory."
            )

        indexed = {}

        for item in documents:

            if not isinstance(item, dict):

                raise PolicyScopeError(
                    "Invalid document record."
                )

            document_id = item.get("document_id")

            if (
                not isinstance(document_id, str)
                or not document_id.strip()
                or document_id in indexed
            ):

                raise PolicyScopeError(
                    "Missing or duplicate document identifier."
                )

            indexed[document_id] = item

        return indexed

    def document_ids(self):

        return tuple(sorted(self._records))

    def get(self, document_id):

        return deepcopy(
            self._records.get(document_id)
        )

    def summary(self):

        return {
            "total_documents": len(self._records),
            "candidate_distribution": dict(
                Counter(
                    item["candidate_applicability"]
                    for item in self._records.values()
                )
            ),
            "verified_documents": sum(
                item.get("review_status") == "verified"
                for item in self._records.values()
            ),
        }

    def assess(self, scope, document_id):

        return assess_applicability(
            scope,
            self.get(document_id),
        )

    def research_access(self, scope, document_id):

        if not isinstance(scope, PolicyScope):

            raise PolicyScopeError(
                "A validated policy scope is required."
            )

        record = self.get(document_id)

        if record is None:

            return {
                "allowed": False,
                "reason": "missing_policy_metadata",
            }

        if record["organization"] != scope.organization:

            return {
                "allowed": False,
                "reason": "organization_mismatch",
            }

        if record.get("review_status") == "verified":

            decision = assess_applicability(
                scope, record
            )

            if not decision["allowed"]:
                return decision

        elif record.get("candidate_applicability") == (
            "COUNTRY_SPECIFIC"
        ):

            countries = [
                canonical_country(country)
                for country in record["candidate_countries"]
            ]

            if scope.employment_country is None:

                return {
                    "allowed": False,
                    "reason": "employment_country_required",
                }

            if scope.employment_country not in countries:

                return {
                    "allowed": False,
                    "reason": "candidate_country_mismatch",
                }

        return {
            "allowed": True,
            "reason": "informational_research_only",
            "eligibility_established": False,
            "definitive_decision_allowed": False,
        }
