
import re

from src.data_ingestion.chunking_experiments_v2 import (
    split_text,
    token_count,
)

NAVIGATION_LABELS = {
    "ARIZONA": {
        "university policies",
        "finance and budget policies",
        "graduate assistant/associate policies",
    },
    "RICE": {
        "helpful links",
        "< back to rice policies",
    },
}

HEADING_PATTERN = re.compile(
    r"^(#{1,6})[ \t]+(.+?)\s*$"
)


def normalize(value):
    return " ".join(value.split()).casefold()


def remove_navigation(text, source_id):

    labels = NAVIGATION_LABELS.get(
        source_id, set()
    )

    retained = []
    removed = []

    for line in text.splitlines(keepends=True):

        heading = HEADING_PATTERN.match(
            line.strip()
        )

        label = (
            heading.group(2)
            if heading
            else line.strip()
        )

        if normalize(label) in labels:
            removed.append(line)
        else:
            retained.append(line)

    return "".join(retained), removed


def split_sections(text):

    sections = []
    stack = []
    lines = []
    current_path = []

    def save():

        body = "".join(lines)

        if body.strip():

            sections.append({
                "text": body,
                "section_path": (
                    " > ".join(current_path)
                    or "Document introduction"
                ),
                "section_heading": (
                    current_path[-1]
                    if current_path
                    else "Document introduction"
                ),
            })

        lines.clear()

    fence = None

    for line in text.splitlines(keepends=True):

        stripped = line.lstrip()

        marker = re.match(
            r"(`{3,}|~{3,})",
            stripped
        )

        if marker:

            symbol = marker.group(1)[0]

            if fence is None:
                fence = symbol

            elif fence == symbol:
                fence = None

            lines.append(line)
            continue

        match = (
            HEADING_PATTERN.match(
                line.rstrip("\r\n")
            )
            if fence is None
            else None
        )

        if match:

            save()

            level = len(match.group(1))

            heading = match.group(2).strip()

            stack = [
                item for item in stack
                if item[0] < level
            ]

            stack.append((level, heading))

            current_path = [
                item[1] for item in stack
            ]

        lines.append(line)

    save()

    reconstructed = "".join(
        section["text"]
        for section in sections
    )

    assert reconstructed.split() == text.split(), (
        "Section splitting lost content."
    )

    return sections


def merge_short_sections(
    sections,
    tokenizer,
    size=480,
    preferred_minimum=64,
):

    merged = []

    for section in sections:

        current = dict(section)

        if not merged:

            merged.append(current)
            continue

        previous = merged[-1]

        previous_count = token_count(
            previous["text"], tokenizer
        )

        current_count = token_count(
            current["text"], tokenizer
        )

        combined = (
            previous["text"]
            + current["text"]
        )

        combined_count = token_count(
            combined, tokenizer
        )

        should_merge = (
            previous_count < preferred_minimum
            or current_count < preferred_minimum
        )

        if should_merge and combined_count <= size:

            previous["text"] = combined

            previous["section_path"] = (
                previous["section_path"]
                + " | "
                + current["section_path"]
            )

        else:

            merged.append(current)

    original = "".join(
        section["text"]
        for section in sections
    )

    reconstructed = "".join(
        section["text"]
        for section in merged
    )

    assert original.split() == reconstructed.split(), (
        "Section merging lost content."
    )

    return merged


def create_enhanced_chunks(
    text,
    tokenizer,
    source_id,
    size=480,
    overlap=80,
):

    cleaned, removed = remove_navigation(
        text, source_id
    )

    sections = split_sections(cleaned)

    merged = merge_short_sections(
        sections,
        tokenizer,
        size=size,
    )

    chunks = []

    for index, section in enumerate(merged):

        windows = split_text(
            section["text"],
            tokenizer,
            size,
            overlap,
        )

        assert windows

        assert windows[0]["char_start"] == 0

        assert (
            windows[-1]["char_end"]
            == len(section["text"])
        )

        for previous, current in zip(
            windows,
            windows[1:]
        ):

            assert (
                current["char_start"]
                <= previous["char_end"]
            )

        for chunk in windows:

            chunk["section_index"] = index

            chunk["section_heading"] = (
                section["section_heading"]
            )

            chunk["section_path"] = (
                section["section_path"]
            )

            chunks.append(chunk)

    assert chunks

    return chunks, {
        "removed_navigation_lines": len(removed),
        "original_sections": len(sections),
        "merged_sections": len(merged),
        "content_preservation_passed": True,
    }
