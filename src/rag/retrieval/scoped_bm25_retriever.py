
from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)

from src.rag.retrieval.applicability_filter import (
    ApplicabilityFilter,
)


class ScopedBM25Retriever:

    def __init__(self, retriever, applicability_filter):

        if not isinstance(retriever, BM25Retriever):

            raise TypeError(
                "A BM25Retriever is required."
            )

        if not isinstance(
            applicability_filter, ApplicabilityFilter
        ):

            raise TypeError(
                "An ApplicabilityFilter is required."
            )

        self.retriever = retriever
        self.applicability_filter = applicability_filter

    def retrieve(self, query, scope, top_k=5):

        if not isinstance(scope, PolicyScope):

            raise PolicyScopeError(
                "A validated policy scope is required."
            )

        if type(top_k) is not int or top_k < 1:

            raise ValueError("Invalid top_k.")

        allowed_ids = set(
            self.applicability_filter.allowed_document_ids(
                scope
            )
        )

        result = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            where=scope.organization_filter(),
            allowed_document_ids=allowed_ids,
        )

        filtered, diagnostics = (
            self.applicability_filter.select(
                scope, result
            )
        )

        if any(
            item.chunk.document_id not in allowed_ids
            for item in filtered.retrieved_chunks
        ):

            raise PolicyScopeError(
                "Document allowlist violation."
            )

        filtered.metadata = {
            **filtered.metadata,
            "policy_scope": scope.to_dict(),
            "allowed_document_count": len(allowed_ids),
            "applicability": diagnostics,
            "returned_chunks": len(
                filtered.retrieved_chunks
            ),
            "access_mode": "informational_research_only",
            "definitive_decision_allowed": False,
        }

        return filtered
