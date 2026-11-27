
from pathlib import Path
from contextlib import contextmanager
import json
import os


@contextmanager
def ingestion_lock(directory):

    directory = Path(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock_path = directory / "ingestion.lock"

    try:

        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )

    except FileExistsError:

        raise RuntimeError(
            "Ingestion lock already exists. "
            "Check whether another process is running."
        )

    try:

        with os.fdopen(
            descriptor, "w", encoding="utf-8"
        ) as file:

            json.dump(
                {"process_id": os.getpid()},
                file,
            )

        yield

    finally:

        lock_path.unlink(missing_ok=True)


def validate_or_create_provenance(
    directory,
    expected,
    existing_count,
):

    directory = Path(directory)

    if type(existing_count) is not int or existing_count < 0:

        raise ValueError("Invalid existing record count.")

    required_fields = (
        "embedding_model",
        "embedding_dimensions",
        "embedding_backend",
        "manifest_sha256",
        "chunk_sha256",
        "collection_name",
    )

    if not isinstance(expected, dict):

        raise ValueError("Invalid provenance metadata.")

    if any(
        field not in expected
        for field in required_fields
    ):

        raise ValueError("Incomplete provenance metadata.")

    if expected["embedding_dimensions"] != 384:

        raise ValueError("Invalid embedding dimensions.")

    for field in required_fields:

        if field == "embedding_dimensions":
            continue

        if (
            not isinstance(expected[field], str)
            or not expected[field].strip()
        ):

            raise ValueError(
                "Invalid provenance field: " + field
            )

    for field in ("manifest_sha256", "chunk_sha256"):

        value = expected[field]

        if (
            len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):

            raise ValueError(
                "Invalid provenance fingerprint: " + field
            )

    provenance_path = directory / "provenance.json"

    if provenance_path.exists():

        actual = json.loads(
            provenance_path.read_text(
                encoding="utf-8"
            )
        )

        if actual != expected:

            raise ValueError(
                "Embedding provenance mismatch."
            )

        return "verified"

    if existing_count > 0:

        raise ValueError(
            "Existing vectors have no provenance record."
        )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = directory / "provenance.tmp"

    temporary_path.write_text(
        json.dumps(expected, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(provenance_path)

    return "created"
