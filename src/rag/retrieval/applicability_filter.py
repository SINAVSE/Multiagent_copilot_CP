
from collections import Counter

from src.agents.applicability_registry import (
    ApplicabilityRegistry,
)

from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.rag.schemas import RetrievalResult


class ApplicabilityFilter:

    def __init__(self, registry):

        if not isinstance(registry, ApplicabilityRegistry):

            raise PolicyScopeError(
                "A validated applicability registry is required."
            )

        self.registry = registry

    def allowed_document_ids(self, scope):

        if not isinstance(scope, PolicyScope):

            raise PolicyScopeError(
                "A validated policy scope is required."
            )

        return tuple(
            document_id
            for document_id in self.registry.document_ids()
            if self.registry.research_access(
                scope, document_id
            )["allowed"]
        )

    def select(self, scope, result):

        if not isinstance(scope, PolicyScope):

            raise PolicyScopeError(
                "A validated policy scope is required."
            )

        if not isinstance(result, RetrievalResult):

            raise PolicyScopeError(
                "A valid RetrievalResult is required."
            )

        selected = []
        rejected = Counter()

        for candidate in result.retrieved_chunks:

            chunk = candidate.chunk

            if chunk.organization != scope.organization:

                raise PolicyScopeError(
                    "Cross-organization retrieval detected."
                )

            record = self.registry.get(
                chunk.document_id
            )

            if record is None:

                raise PolicyScopeError(
                    "Retrieved document is absent from registry."
                )

            for field in (
                "organization",
                "title",
                "source_url",
            ):

                if getattr(chunk, field) != record[field]:

                    raise PolicyScopeError(
                        "Retrieved metadata mismatch: " + field
                    )

            access = self.registry.research_access(
                scope,
                chunk.document_id,
            )

            if not access["allowed"]:

                rejected[access["reason"]] += 1
                continue

            selected.append(candidate)

        filtered = result.model_copy(
            update={
                "retrieved_chunks": selected,
            }
        )

        diagnostics = {
            "access_mode": "informational_research_only",
            "input_candidates": len(
                result.retrieved_chunks
            ),
            "accepted_candidates": len(selected),
            "rejected_candidates": sum(
                rejected.values()
            ),
            "rejection_reasons": dict(rejected),
            "eligibility_established": False,
            "definitive_decision_allowed": False,
        }

        return filtered, diagnostics
