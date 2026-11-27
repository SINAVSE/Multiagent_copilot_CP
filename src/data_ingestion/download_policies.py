import argparse
import hashlib
import json
import re

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import fitz
import requests
from lxml import html

from src.data_ingestion.discover_policy_links import (
    SESSION,
    ROBOTS,
    USER_AGENT,
    robots_allowed,
    wait_for_host,
)

ROOT = Path("data/real_corpus/expanded_v2")

RAW_DIR = ROOT / "raw"

PROCESSED_DIR = ROOT / "processed"

MANIFEST = ROOT / "pilot_manifest.jsonl"

CANDIDATES = Path(
    "data/real_corpus/metadata/policy_candidates_v2.json"
)

BASELINE_MANIFEST = Path(
    "data/real_corpus/metadata/corpus_manifest.jsonl"
)

MAX_FILE_BYTES = 10 * 1024 * 1024

MIN_WORDS = 150

TIMEOUT = 30

ALLOWED_HOSTS = {
    "GITLAB": {"handbook.gitlab.com"},
    "UW": {"policy.uw.edu"},
    "UTAH": {"regulations.utah.edu"},
    "UCONN": {"policy.uconn.edu"},
    "RICE": {"policy.rice.edu"},
    "ARIZONA": {"policy.arizona.edu"},
    "COLORADO": {"www.cu.edu"},
}


def sha256(data):

    return hashlib.sha256(data).hexdigest()


def normalized_hash(text):

    normalized = " ".join(text.casefold().split())

    return sha256(normalized.encode("utf-8"))


def load_jsonl(path):

    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]


def download(url):

    if not robots_allowed(url):

        raise ValueError(
            "Automated access not permitted or unavailable."
        )

    parts = urlsplit(url)

    origin = f"{parts.scheme}://{parts.netloc}"

    parser = ROBOTS[origin]

    delay = max(
        2,
        parser.crawl_delay(USER_AGENT) or 0,
    )

    rate = parser.request_rate(USER_AGENT)

    if rate and rate.requests > 0:

        delay = max(
            delay,
            rate.seconds / rate.requests,
        )

    wait_for_host(url, delay)

    with SESSION.get(
        url,
        timeout=TIMEOUT,
        stream=True,
        allow_redirects=False,
    ) as response:

        if response.status_code in (429, 503):

            raise RuntimeError(
                f"Server requested pause: {response.status_code}"
            )

        if 300 <= response.status_code < 400:

            raise ValueError(
                "Redirect requires source review."
            )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        data = bytearray()

        for block in response.iter_content(
            chunk_size=65536
        ):

            data.extend(block)

            if len(data) > MAX_FILE_BYTES:

                raise ValueError(
                    "Document exceeds maximum file size."
                )

    return bytes(data), content_type


def extract_html(data):

    page = html.fromstring(data)

    headings = (
        page.xpath("//main//h1 | //article//h1")
        + page.xpath("//h1")
    )

    title = ""

    for heading in headings:

        value = " ".join(
            heading.text_content().split()
        )

        if (
            value
            and "cookie" not in value.casefold()
            and value.casefold() not in (
                "university policies",
                "university of connecticut",
                "uconn"
            )
        ):
            title = value
            break

    if not title:
        title = page.xpath("string(//title)").strip()

    containers = (
        page.xpath("//main")
        or page.xpath("//article")
        or page.xpath("//body")
    )

    if not containers:
        raise ValueError("No readable HTML content.")

    body = max(
        containers,
        key=lambda node: len(
            node.text_content().split()
        )
    )

    for node in body.xpath(
        ".//script | .//style | .//nav | "
        ".//footer | .//aside | .//noscript | "
        ".//form | .//*[@hidden] | "
        ".//*[@aria-hidden='true']"
    ):
        if node.getparent() is not None:
            node.drop_tree()

    blocks = []

    def normalize(value):
        return " ".join((value or "").split())

    def emit(value):
        value = value.strip()
        if value:
            blocks.append(value)

    def walk(node, depth=0):

        if not isinstance(node.tag, str):
            return

        tag = node.tag.lower()

        if tag in (
            "h1", "h2", "h3",
            "h4", "h5", "h6"
        ):
            value = normalize(node.text_content())

            if value:
                emit("#" * int(tag[1]) + " " + value)

            return

        if tag == "table":

            for row in node.xpath(".//tr"):

                cells = [
                    normalize(cell.text_content())
                    for cell in row.xpath("./th | ./td")
                ]

                if cells:
                    emit(" | ".join(cells))

            return

        if tag == "pre":

            emit(node.text_content())

            return

        block_tags = {
            "div", "section", "article",
            "main", "p", "ul", "ol", "li",
            "dl", "dt", "dd", "table",
            "blockquote", "pre",
            "h1", "h2", "h3",
            "h4", "h5", "h6"
        }

        pending = [node.text or ""]

        prefix = ""

        if tag == "li":

            parent = node.getparent()

            if parent is not None and parent.tag == "ol":

                siblings = [
                    item for item in parent
                    if item.tag == "li"
                ]

                try:
                    first = int(parent.get("start", "1"))
                except ValueError:
                    first = 1

                number = first + siblings.index(node)

                prefix = str(number) + ". "

            else:

                prefix = "- "

            prefix = "  " * depth + prefix

        def flush():

            nonlocal prefix

            value = normalize("".join(pending))

            if value:

                emit(prefix + value)

                prefix = ""

            pending.clear()

        for child in node:

            child_tag = (
                child.tag.lower()
                if isinstance(child.tag, str)
                else ""
            )

            if child_tag in block_tags:

                flush()

                walk(
                    child,
                    depth + (
                        1 if child_tag in ("ol", "ul")
                        else 0
                    )
                )

            elif child_tag == "br":

                pending.append(" ")

            elif child_tag:

                pending.append(child.text_content())

            pending.append(child.tail or "")

        flush()

    walk(body)

    content = "\n\n".join(blocks)

    return title, content


def extract_pdf(data):

    with fitz.open(
        stream=data,
        filetype="pdf"
    ) as document:

        if document.needs_pass:

            raise ValueError(
                "Encrypted PDF."
            )

        title = (
            document.metadata or {}
        ).get("title", "")

        text = "\n\n".join(
            page.get_text()
            for page in document
        )

    return title, text


def extract(data, content_type):

    if data.startswith(b"%PDF-"):

        title, text = extract_pdf(data)

        extension = ".pdf"

    elif "html" in content_type:

        title, text = extract_html(data)

        extension = ".html"

    else:

        raise ValueError(
            "Unsupported content type."
        )

    word_count = len(text.split())

    if word_count < MIN_WORDS:

        raise ValueError(
            f"Insufficient content: {word_count} words."
        )

    return title, text, extension, word_count


def stable_id(candidate):

    fingerprint = sha256(
        candidate["source_url"].encode("utf-8")
    )[:16]

    return (
        candidate["source_id"]
        + "-V2-"
        + fingerprint
    )


def eligible(candidate, existing_urls):

    source_id = candidate["source_id"]

    url = candidate["source_url"]

    parts = urlsplit(url)

    return (
        source_id in ALLOWED_HOSTS
        and parts.scheme == "https"
        and parts.hostname in ALLOWED_HOSTS[source_id]
        and url.rstrip("/") not in existing_urls
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--source",
        choices=sorted(ALLOWED_HOSTS),
        default=None,
    )

    args = parser.parse_args()

    if not 1 <= args.limit <= 100:

        raise ValueError(
            "Batch limit must be between 1 and 100."
        )

    if not 1 <= args.max_attempts <= 150:

        raise ValueError(
            "Invalid maximum attempts."
        )

    report = json.loads(
        CANDIDATES.read_text(
            encoding="utf-8-sig"
        )
    )

    existing = load_jsonl(MANIFEST)

    original = load_jsonl(BASELINE_MANIFEST)

    known_urls = {
        row["source_url"].rstrip("/")
        for row in existing + original
    }

    known_hashes = {
        row["normalized_content_hash"]
        for row in existing
    }

    known_raw_hashes = {
        row["raw_file_hash"]
        for row in existing
    }

    for row in original:

        known_raw_hashes.add(
            row["file_hash"]
        )

        original_path = Path(row["local_path"])

        if original_path.is_file():

            known_raw_hashes.add(
                sha256(original_path.read_bytes())
            )

    for original_file in Path(
        "data/real_corpus/processed/gitlab"
    ).glob("*.md"):

        known_hashes.add(
            normalized_hash(
                original_file.read_text(
                    encoding="utf-8-sig"
                )
            )
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    collected = 0

    attempted = 0

    skipped = 0

    blocked_sources = set()

    candidates = sorted(
        [
            row for row in report["candidates"]
            if args.source is None
            or row["source_id"] == args.source
        ],
        key=lambda row: (
            "3-030" not in row["source_url"],
            row["source_id"] != "UW",
            row["source_url"],
        ),
    )

    for candidate in candidates:

        if collected >= args.limit:

            break

        if attempted >= args.max_attempts:

            break

        if candidate["source_id"] in blocked_sources:

            continue

        if not eligible(
            candidate,
            known_urls,
        ):

            skipped += 1

            continue

        url = candidate["source_url"]

        attempted += 1

        print("\nATTEMPT:", attempted)

        print("URL:", url)

        try:

            data, content_type = download(url)

            raw_hash = sha256(data)

            if raw_hash in known_raw_hashes:

                print("SKIPPED: Duplicate raw file")

                skipped += 1

                continue

            title, text, extension, words = extract(
                data,
                content_type,
            )

            content_hash = normalized_hash(text)

            if content_hash in known_hashes:

                print("SKIPPED: Duplicate content")

                skipped += 1

                continue

            document_id = stable_id(candidate)

            raw_path = RAW_DIR / (
                document_id + extension
            )

            processed_path = PROCESSED_DIR / (
                document_id + ".md"
            )

            if raw_path.exists() or processed_path.exists():

                raise ValueError(
                    "Existing file requires integrity review."
                )

            raw_path.write_bytes(data)

            processed_path.write_text(
                text.rstrip() + "\n",
                encoding="utf-8",
            )

            row = {
                "document_id": document_id,
                "title": title or candidate["title"],
                "organization": candidate["organization"],
                "source_id": candidate["source_id"],
                "source_url": url,
                "categories": candidate["categories"],
                "country": (
                    "US"
                    if candidate["source_id"] in ("UW", "UTAH", "UCONN", "RICE", "ARIZONA", "COLORADO")
                    else None
                ),
                "retrieved_date": datetime.now(
                    timezone.utc
                ).isoformat(),
                "document_type": extension.lstrip("."),
                "raw_file_hash": raw_hash,
                "normalized_content_hash": content_hash,
                "processed_file_hash": sha256(
                    processed_path.read_bytes()
                ),
                "local_path": raw_path.as_posix(),
                "processed_path": processed_path.as_posix(),
                "word_count": words,
                "processing_status": "pending_quality_review",
                "redistribution_status": "requires_review",
            }

            with MANIFEST.open(
                "a",
                encoding="utf-8",
            ) as stream:

                stream.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    ) + "\n"
                )

            known_urls.add(url.rstrip("/"))

            known_hashes.add(content_hash)

            known_raw_hashes.add(raw_hash)

            collected += 1

            print("COLLECTED:", document_id)

            print("TITLE:", row["title"])

            print("WORDS:", words)

        except RuntimeError as error:

            print("SOURCE PAUSED:", str(error))

            blocked_sources.add(
                candidate["source_id"]
            )

        except (
            requests.RequestException,
            ValueError,
            OSError,
        ) as error:

            skipped += 1

            print("SKIPPED:", str(error))

    print("\n========== PILOT SUMMARY ==========")

    print("Attempted downloads:", attempted)

    print("New documents collected:", collected)

    print("Skipped:", skipped)

    print("Total stored documents:", len(
        load_jsonl(MANIFEST)
    ))

    print("Validated documents: pending review")

    print("Hosted AI API calls: 0")

    print("Manifest:", MANIFEST)


if __name__ == "__main__":

    main()
