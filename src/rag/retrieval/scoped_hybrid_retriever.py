
from src.agents.policy_scope import (
    PolicyScope,
    PolicyScopeError,
)

from src.rag.retrieval.hybrid_retriever import (
    HybridRetriever,
)

from src.rag.retrieval.scoped_dense_retriever import (
    ScopedDenseRetriever,
)

from src.rag.retrieval.scoped_bm25_retriever import (
    ScopedBM25Retriever,
)


class ScopedAdapter:

    def __init__(self, retriever, scope):

        self.retriever = retriever
        self.scope = scope

    def retrieve(self, query, top_k=5, where=None):

        if where is not None:
            raise ValueError(
                "External filters are not supported."
            )

        return self.retriever.retrieve(
            query=query,
            scope=self.scope,
            top_k=top_k,
        )


class ScopedHybridRetriever:

    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        rrf_k=60,
    ):

        if not isinstance(
            dense_retriever, ScopedDenseRetriever
        ):
            raise TypeError(
                "ScopedDenseRetriever required."
            )

        if not isinstance(
            bm25_retriever, ScopedBM25Retriever
        ):
            raise TypeError(
                "ScopedBM25Retriever required."
            )

        if (
            dense_retriever.applicability_filter
            is not bm25_retriever.applicability_filter
        ):
            raise ValueError(
                "Retrievers must share an applicability filter."
            )

        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("Invalid rrf_k.")

        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf_k = rrf_k

        self.selector = (
            dense_retriever.applicability_filter
        )

    def retrieve(
        self,
        query,
        scope,
        top_k=5,
        candidate_k=None,
    ):

        if not isinstance(scope, PolicyScope):
            raise PolicyScopeError(
                "Validated policy scope required."
            )

        hybrid = HybridRetriever(
            dense_retriever=ScopedAdapter(
                self.dense, scope
            ),
            bm25_retriever=ScopedAdapter(
                self.bm25, scope
            ),
            rrf_k=self.rrf_k,
        )

        result = hybrid.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        filtered, diagnostics = self.selector.select(
            scope, result
        )

        if diagnostics["rejected_candidates"] != 0:
            raise PolicyScopeError(
                "Unexpected hybrid applicability rejection."
            )

        allowed_ids = set(
            self.selector.allowed_document_ids(scope)
        )

        if any(
            item.chunk.document_id not in allowed_ids
            for item in filtered.retrieved_chunks
        ):
            raise PolicyScopeError(
                "Hybrid document allowlist violation."
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
