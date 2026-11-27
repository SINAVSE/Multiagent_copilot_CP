from unittest.mock import patch

import pytest

from src.rag.retrieval.bm25_retriever import (
    BM25Retriever,
)

from tests.test_evidence_selector import (
    make_item,
)


def make_chunk(
    chunk_id="DOC-001-S1",
    title="Sponsored Business Travel Policy",
    heading="Meal Reimbursement",
    section="Travel > Expenses > Meals",
    text="Employees receive the applicable allowance.",
):

    item = make_item(
        chunk_id=chunk_id,
        document_id=chunk_id.split("-S")[0],
        text=text,
        section=section,
        heading=heading,
    )

    return item.chunk.model_copy(
        update={"title": title}
    )


def make_corpus():

    return [
        make_chunk(),

        make_chunk(
            chunk_id="DOC-002-S1",
            title="Equipment Policy",
            heading="Hardware",
            section="Equipment > Hardware",
            text="Employees receive an equipment allowance.",
        ),

        make_chunk(
            chunk_id="DOC-003-S1",
            title="Annual Leave Policy",
            heading="Holiday Entitlement",
            section="Leave > Annual Leave",
            text="Employees receive paid annual leave.",
        ),

        make_chunk(
            chunk_id="DOC-004-S1",
            title="Anti-Fraud Policy",
            heading="Reporting",
            section="Compliance > Fraud",
            text="Employees must report suspected fraud.",
        ),

        make_chunk(
            chunk_id="DOC-005-S1",
            title="Remote Work Policy",
            heading="Work Locations",
            section="Employment > Remote Work",
            text="Employees may work from approved locations.",
        ),
    ]


def create_retriever(
    include_metadata=True,
    chunks=None,
):

    corpus = (
        make_corpus()
        if chunks is None
        else chunks
    )

    with patch(
        "src.rag.retrieval.bm25_retriever.load_chunks",
        return_value=corpus,
    ), patch(
        "src.rag.retrieval.bm25_retriever.process_chunks",
        return_value=corpus,
    ):

        return BM25Retriever(
            include_metadata=include_metadata
        )


def test_searchable_text_contains_document_title():

    chunk = make_chunk()

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert chunk.title in text


def test_searchable_text_contains_section_heading():

    chunk = make_chunk()

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert chunk.section_heading in text


def test_searchable_text_contains_section_path():

    chunk = make_chunk()

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert chunk.section_path in text


def test_searchable_text_preserves_chunk_body():

    chunk = make_chunk()

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert chunk.text in text


def test_text_only_configuration():

    chunk = make_chunk()

    text = BM25Retriever.build_searchable_text(
        chunk,
        include_metadata=False,
    )

    assert text == chunk.text

    assert chunk.title not in text


def test_missing_optional_metadata():

    chunk = make_chunk(
        heading=None,
        section=None,
    )

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert chunk.title in text

    assert chunk.text in text

    assert "None" not in text


def test_duplicate_metadata_fields_removed():

    chunk = make_chunk(
        title="Travel Policy",
        heading="Travel Policy",
        section="Travel Policy",
    )

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert text.count("Travel Policy") == 1


def test_duplicate_metadata_case_insensitive():

    chunk = make_chunk(
        title="Travel Policy",
        heading="TRAVEL POLICY",
        section="Travel  Policy",
    )

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert (
        text.casefold().count("travel policy")
        == 1
    )


def test_empty_metadata_fields_ignored():

    chunk = make_chunk(
        heading=" ",
        section=" ",
    )

    text = BM25Retriever.build_searchable_text(
        chunk
    )

    assert len(text.splitlines()) == 2


def test_case_insensitive_tokenization():

    assert BM25Retriever._tokenize(
        "SPONSORED Travel"
    ) == [
        "sponsored",
        "travel",
    ]


def test_hyphenated_policy_tokenization():

    assert BM25Retriever._tokenize(
        "Anti-Fraud Policy"
    ) == [
        "anti",
        "fraud",
        "policy",
    ]


def test_ampersand_normalization():

    assert BM25Retriever._tokenize(
        "Gifts & Entertainment"
    ) == [
        "gifts",
        "and",
        "entertainment",
    ]


def test_unicode_normalization():

    composed = "Caf\u00e9"

    decomposed = "Cafe\u0301"

    assert (
        BM25Retriever._tokenize(composed)
        == BM25Retriever._tokenize(decomposed)
    )


def test_full_width_unicode_normalization():

    assert BM25Retriever._tokenize(
        "\uff34\uff32\uff21\uff36\uff25\uff2c"
    ) == [
        "travel"
    ]


def test_empty_tokenization():

    assert BM25Retriever._tokenize(
        "   "
    ) == []


@pytest.mark.parametrize(
    "value",
    [None, 1, "true", []],
)
def test_invalid_metadata_configuration(value):

    with pytest.raises(ValueError):

        create_retriever(
            include_metadata=value
        )


def test_default_metadata_indexing_enabled():

    retriever = create_retriever()

    assert retriever.include_metadata is True


def test_metadata_terms_are_indexed():

    retriever = create_retriever()

    tokens = retriever.tokenized_corpus[0]

    assert "sponsored" in tokens

    assert "reimbursement" in tokens


def test_text_only_terms_exclude_metadata():

    retriever = create_retriever(
        include_metadata=False
    )

    tokens = retriever.tokenized_corpus[0]

    assert "sponsored" not in tokens

    assert "reimbursement" not in tokens

    assert "allowance" in tokens


def test_metadata_query_ranks_correct_policy_first():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Sponsored travel meal reimbursement",
        top_k=3,
    )

    assert result.retrieved_chunks[0].chunk.chunk_id == (
        "DOC-001-S1"
    )


def test_anti_fraud_policy_ranked_first():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Anti-fraud reporting",
        top_k=3,
    )

    assert result.retrieved_chunks[0].chunk.chunk_id == (
        "DOC-004-S1"
    )


def test_section_metadata_improves_matching():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Holiday entitlement",
        top_k=3,
    )

    assert result.retrieved_chunks[0].chunk.chunk_id == (
        "DOC-003-S1"
    )


def test_metadata_index_improves_lexical_score():

    enhanced = create_retriever(
        include_metadata=True
    )

    text_only = create_retriever(
        include_metadata=False
    )

    query = "Sponsored reimbursement"

    enhanced_scores = enhanced.bm25.get_scores(
        enhanced._tokenize(query)
    )

    baseline_scores = text_only.bm25.get_scores(
        text_only._tokenize(query)
    )

    assert enhanced_scores[0] > baseline_scores[0]


def test_retrieval_metadata_identifies_configuration():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Sponsored travel",
        top_k=2,
    )

    assert result.metadata[
        "include_metadata"
    ] is True

    assert result.metadata[
        "scoring_algorithm"
    ] == "BM25Okapi"

    assert result.metadata[
        "tokenization"
    ] == "unicode_nfkc_casefold"


def test_text_only_metadata_identifies_configuration():

    retriever = create_retriever(
        include_metadata=False
    )

    result = retriever.retrieve(
        query="Employee allowance",
        top_k=2,
    )

    assert result.metadata[
        "include_metadata"
    ] is False


def test_top_k_is_preserved():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Employee allowance",
        top_k=2,
    )

    assert len(result.retrieved_chunks) == 2


def test_ranks_are_sequential():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Sponsored travel",
        top_k=3,
    )

    assert [
        item.rank
        for item in result.retrieved_chunks
    ] == [1, 2, 3]


def test_document_filter_is_preserved():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Sponsored travel",
        top_k=5,
        where={
            "document_id": "DOC-002"
        },
    )

    assert len(result.retrieved_chunks) == 1

    assert result.retrieved_chunks[0].chunk.document_id == (
        "DOC-002"
    )


def test_original_chunk_content_is_preserved():

    retriever = create_retriever()

    result = retriever.retrieve(
        query="Sponsored travel",
        top_k=1,
    )

    chunk = result.retrieved_chunks[0].chunk

    assert chunk.text == (
        "Employees receive the applicable allowance."
    )


def test_empty_corpus_rejected():

    with pytest.raises(
        ValueError,
        match="No chunks available",
    ):

        create_retriever(
            chunks=[]
        )
