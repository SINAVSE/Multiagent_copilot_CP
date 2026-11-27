
from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.rag.schemas import RetrievalResult

from src.rag.retrieval.dense_retriever import (
    DenseRetriever,
)

from src.rag.retrieval.applicability_filter import (
    ApplicabilityFilter,
)

from src.rag.retrieval.chunk_integrity import (
    ChunkIntegrityValidator,
)


class ScopedDenseRetriever:

    def __init__(
        self,
        retriever,
        applicability_filter,
        canonical_chunks,
    ):

        if not isinstance(retriever, DenseRetriever):
            raise TypeError("DenseRetriever required.")

        if retriever.embedding_function is None:
            raise ValueError(
                "Explicit embedding function required."
            )

        if not isinstance(
            applicability_filter, ApplicabilityFilter
        ):
            raise TypeError("ApplicabilityFilter required.")

        chunks = list(canonical_chunks)

        self.canonical_chunks = {
            chunk.chunk_id: chunk for chunk in chunks
        }

        if len(self.canonical_chunks) != len(chunks):
            raise ValueError("Duplicate canonical chunk IDs.")

        self.retriever = retriever
        self.applicability_filter = applicability_filter

    def retrieve(self, query, scope, top_k=5):

        if not isinstance(scope, PolicyScope):
            raise PolicyScopeError(
                "Validated policy scope required."
            )

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if type(top_k) is not int or top_k < 1:
            raise ValueError("Invalid top_k.")

        allowed_ids = set(
            self.applicability_filter.allowed_document_ids(
                scope
            )
        )

        if not allowed_ids:
            result = RetrievalResult(
                query=query,
                retrieved_chunks=[],
                retrieval_method="dense",
                elapsed_ms=0.0,
            )

        else:

            where = {
                "$and": [
                    {"organization": scope.organization},
                    {
                        "document_id": {
                            "$in": sorted(allowed_ids)
                        }
                    },
                ]
            }

            result = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                where=where,
            )

        verified = []

        for item in result.retrieved_chunks:

            chunk = item.chunk

            if chunk.organization != scope.organization:
                raise PolicyScopeError(
                    "Cross-organization dense result."
                )

            if chunk.document_id not in allowed_ids:
                raise PolicyScopeError(
                    "Dense document allowlist violation."
                )

            canonical = self.canonical_chunks.get(
                chunk.chunk_id
            )

            if canonical is None:
                raise ValueError(
                    "Unknown dense chunk identifier."
                )

            for field in (
                ChunkIntegrityValidator.REQUIRED_FIELDS
            ):

                if getattr(chunk, field) != getattr(
                    canonical, field
                ):
                    raise ValueError(
                        "Dense chunk integrity mismatch: "
                        + field
                    )

            verified.append(
                item.model_copy(
                    update={"chunk": canonical}
                )
            )

        result = result.model_copy(
            update={"retrieved_chunks": verified}
        )

        filtered, diagnostics = (
            self.applicability_filter.select(
                scope, result
            )
        )

        if diagnostics["rejected_candidates"] != 0:
            raise PolicyScopeError(
                "Unexpected dense applicability rejection."
            )

        filtered.metadata = {
            **filtered.metadata,
            "policy_scope": scope.to_dict(),
            "allowed_document_count": len(allowed_ids),
            "applicability": diagnostics,
            "access_mode": "informational_research_only",
            "definitive_decision_allowed": False,
        }

        return filtered
