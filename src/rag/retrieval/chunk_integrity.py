from src.rag.schemas import DocumentChunk


class ChunkIntegrityValidator:
    """Validate and merge compatible policy evidence."""

    REQUIRED_FIELDS = (
        "chunk_id",
        "document_id",
        "title",
        "organization",
        "source_url",
        "text",
    )

    @classmethod
    def merge(
        cls,
        existing: DocumentChunk,
        incoming: DocumentChunk,
    ) -> DocumentChunk:

        existing_data = existing.model_dump()

        incoming_data = incoming.model_dump()

        # Policy identity and content must agree.

        for field in cls.REQUIRED_FIELDS:

            if existing_data.get(field) != incoming_data.get(field):

                raise ValueError(
                    "Conflicting evidence uses the same "
                    f"chunk identifier: {field}."
                )

        merged_data = dict(existing_data)

        # Preserve optional metadata when available.
        # Conflicting non-null values remain invalid.

        for field, incoming_value in incoming_data.items():

            if field in cls.REQUIRED_FIELDS:
                continue

            existing_value = merged_data.get(field)

            if incoming_value is None:
                continue

            if existing_value is None:

                merged_data[field] = incoming_value

                continue

            if existing_value != incoming_value:

                raise ValueError(
                    "Conflicting evidence uses the same "
                    f"chunk identifier: {field}."
                )

        return DocumentChunk.model_validate(
            merged_data
        )
