import pytest

from src.rag.retrieval.chunk_integrity import (
    ChunkIntegrityValidator,
)

from tests.test_evidence_selector import make_item


def make_chunk(**updates):

    chunk = make_item(
        chunk_id="TEST-001",
        document_id="DOC-001",
        text="Meal allowance is USD 100.",
    ).chunk

    data = chunk.model_dump()

    data.update(updates)

    return type(chunk).model_validate(data)


def test_identical_chunks_are_compatible():

    first = make_chunk()

    second = make_chunk()

    result = ChunkIntegrityValidator.merge(
        first, second
    )

    assert result.model_dump() == first.model_dump()


@pytest.mark.parametrize(
    "field,value",
    [
        ("chunk_id", "TEST-002"),
        ("document_id", "DOC-002"),
        ("title", "Another Policy"),
        ("organization", "Another Organization"),
        ("source_url", "https://example.invalid/new"),
        ("text", "Meal allowance is USD 200."),
    ],
)
def test_required_field_conflicts_rejected(field, value):

    first = make_chunk()

    second = make_chunk(
        **{field: value}
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        ChunkIntegrityValidator.merge(
            first, second
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("section_heading", "Meals"),
        ("section_path", "Travel > Meals"),
        ("chunk_index", 5),
        ("chunk_size", 500),
        ("overlap", 100),
        ("token_count", 200),
        ("page_number", 7),
    ],
)
def test_missing_optional_metadata_is_enriched(
    field,
    value,
):

    first = make_chunk(
        **{field: None}
    )

    second = make_chunk(
        **{field: value}
    )

    result = ChunkIntegrityValidator.merge(
        first, second
    )

    assert getattr(result, field) == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("section_heading", "Meals"),
        ("section_path", "Travel > Meals"),
        ("chunk_index", 5),
        ("chunk_size", 500),
        ("overlap", 100),
        ("token_count", 200),
        ("page_number", 7),
    ],
)
def test_existing_optional_metadata_preserved(
    field,
    value,
):

    first = make_chunk(
        **{field: value}
    )

    second = make_chunk(
        **{field: None}
    )

    result = ChunkIntegrityValidator.merge(
        first, second
    )

    assert getattr(result, field) == value


@pytest.mark.parametrize(
    "field,first_value,second_value",
    [
        ("section_heading", "Meals", "Hotels"),
        ("section_path", "Travel > Meals", "Travel > Hotels"),
        ("chunk_index", 1, 2),
        ("chunk_size", 500, 800),
        ("overlap", 50, 100),
        ("token_count", 200, 300),
        ("page_number", 3, 5),
    ],
)
def test_optional_metadata_conflicts_rejected(
    field,
    first_value,
    second_value,
):

    first = make_chunk(
        **{field: first_value}
    )

    second = make_chunk(
        **{field: second_value}
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        ChunkIntegrityValidator.merge(
            first, second
        )


def test_additional_metadata_preserved():

    first = make_chunk()

    second = make_chunk(
        category="Travel",
        version="1.0",
    )

    result = ChunkIntegrityValidator.merge(
        first, second
    )

    assert result.category == "Travel"

    assert result.version == "1.0"


def test_additional_metadata_conflicts_rejected():

    first = make_chunk(
        version="1.0"
    )

    second = make_chunk(
        version="2.0"
    )

    with pytest.raises(
        ValueError,
        match="Conflicting evidence",
    ):

        ChunkIntegrityValidator.merge(
            first, second
        )


def test_null_additional_metadata_is_enriched():

    first = make_chunk(
        category=None
    )

    second = make_chunk(
        category="Travel"
    )

    result = ChunkIntegrityValidator.merge(
        first, second
    )

    assert result.category == "Travel"


def test_original_objects_are_not_modified():

    first = make_chunk(
        page_number=None
    )

    second = make_chunk(
        page_number=7
    )

    first_before = first.model_dump()

    second_before = second.model_dump()

    result = ChunkIntegrityValidator.merge(
        first, second
    )

    assert first.model_dump() == first_before

    assert second.model_dump() == second_before

    assert result.page_number == 7


def test_merge_order_does_not_lose_metadata():

    first = make_chunk(
        page_number=None,
        chunk_size=500,
    )

    second = make_chunk(
        page_number=7,
        chunk_size=None,
    )

    forward = ChunkIntegrityValidator.merge(
        first, second
    )

    reverse = ChunkIntegrityValidator.merge(
        second, first
    )

    assert forward.model_dump() == reverse.model_dump()

    assert forward.page_number == 7

    assert forward.chunk_size == 500


def test_zero_optional_metadata_preserved():

    first = make_chunk(
        chunk_index=0
    )

    second = make_chunk(
        chunk_index=None
    )

    result = ChunkIntegrityValidator.merge(
        first, second
    )

    assert result.chunk_index == 0
