import hashlib
import json
import os
import re
import subprocess

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = (
    Path(os.environ["TEMP"])
    / "gitlab-handbook-cap14"
)

HANDBOOK = REPOSITORY / "content" / "handbook"

OUTPUT = Path(
    "data/real_corpus/metadata/"
    "gitlab_repository_inventory_v2.json"
)

CATEGORIES = {
    "travel": [
        "travel", "airfare", "lodging",
        "mileage", "per diem"
    ],
    "expense": [
        "expense", "reimbursement",
        "entertainment", "allowance"
    ],
    "procurement": [
        "procurement", "purchasing",
        "spending", "vendor", "contract"
    ],
    "hr": [
        "employment", "employee", "leave",
        "absence", "time off", "hiring",
        "termination", "onboarding"
    ],
    "benefits": [
        "benefits", "insurance",
        "retirement", "healthcare"
    ],
    "compensation": [
        "compensation", "payroll",
        "salary", "bonus", "equity"
    ],
    "ethics": [
        "ethics", "gift", "bribery",
        "corruption", "harassment",
        "conflict of interest", "conduct"
    ],
    "compliance": [
        "compliance", "fraud",
        "audit", "whistleblowing",
        "financial control"
    ],
}

POLICY_SIGNALS = [
    "policy", "policies", "eligibility",
    "requirements", "code of conduct",
    "code of ethics", "reimbursement",
    "regulations", "entitlement",
    "allowance", "procedure"
]

GUIDANCE_SIGNALS = [
    "tutorial", "template", "example",
    "training", "faq", "handbook guide",
    "meeting notes", "team overview",
    "case study", "checklist"
]


def sha256(data):

    return hashlib.sha256(data).hexdigest()


def parse_markdown(text):

    frontmatter = ""

    body = text

    if text.startswith("---\n"):

        parts = text.split("\n---", 1)

        if len(parts) == 2:

            frontmatter = parts[0]

            body = parts[1].strip()

    match = re.search(
        r"^title:\s*(.+)$",
        frontmatter,
        re.MULTILINE,
    )

    title = ""

    if match:

        title = match.group(1).strip(
            " \"'"
        )

    if not title:

        heading = re.search(
            r"^#\s+(.+)$",
            body,
            re.MULTILINE,
        )

        if heading:

            title = heading.group(1).strip()

    return title, body


def classify(value):

    normalized = re.sub(
        r"[-_/]+",
        " ",
        value.casefold(),
    )

    return [
        category
        for category, terms in CATEGORIES.items()
        if any(
            re.search(
                r"\b" + re.escape(term) + r"\w*\b",
                normalized,
            )
            for term in terms
        )
    ]


def assess(title, relative_path, body):

    label = (
        title + " " + relative_path
    ).casefold()

    label = label.replace("-", " ")

    categories = classify(label)

    words = len(body.split())

    if words < 100:

        status = "insufficient_content"

    elif not categories:

        status = "out_of_scope"

    elif any(
        term in label
        for term in GUIDANCE_SIGNALS
    ):

        status = "supplementary_guidance"

    elif any(
        term in label
        for term in POLICY_SIGNALS
    ):

        status = "high_priority_policy_candidate"

    else:

        status = "requires_policy_review"

    return categories, words, status


def source_url(relative):

    if relative.name in ("_index.md", "index.md"):

        route = relative.parent.as_posix()

    else:

        route = relative.with_suffix("").as_posix()

    if route == ".":

        route = ""

    base = "https://handbook.gitlab.com/handbook/"

    return base + (
        route.strip("/") + "/"
        if route else ""
    )


def main():

    if not HANDBOOK.is_dir():

        raise FileNotFoundError(HANDBOOK)

    commit = subprocess.check_output(
        [
            "git", "-C", str(REPOSITORY),
            "rev-parse", "HEAD"
        ],
        text=True,
    ).strip()

    rows = []

    known_hashes = set()

    files = sorted(HANDBOOK.rglob("*.md"))

    for file in files:

        relative = file.relative_to(HANDBOOK)

        raw = file.read_bytes()

        text = raw.decode("utf-8-sig")

        text = text.replace("\r\n", "\n")

        title, body = parse_markdown(text)

        if not title:

            title = relative.stem.replace(
                "-", " "
            ).title()

        categories, words, status = assess(
            title,
            relative.as_posix(),
            body,
        )

        normalized = " ".join(
            body.casefold().split()
        )

        content_hash = sha256(
            normalized.encode("utf-8")
        )

        if content_hash in known_hashes:

            status = "duplicate_content"

        known_hashes.add(content_hash)

        rows.append({
            "title": title,
            "organization": "GitLab",
            "repository_commit": commit,
            "repository_path": file.relative_to(
                REPOSITORY
            ).as_posix(),
            "source_url": source_url(relative),
            "directory": relative.parts[0],
            "categories": categories,
            "word_count": words,
            "raw_file_hash": sha256(raw),
            "normalized_content_hash": content_hash,
            "review_status": status,
            "is_index_file": (
                file.name == "_index.md"
            ),
        })

    counts = Counter(
        row["review_status"]
        for row in rows
    )

    category_counts = Counter(
        category
        for row in rows
        if row["review_status"] in (
            "high_priority_policy_candidate",
            "requires_policy_review",
        )
        for category in row["categories"]
    )

    report = {
        "version": "2.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "repository_url": (
            "https://gitlab.com/gitlab-com/"
            "content-sites/handbook"
        ),
        "repository_commit": commit,
        "total_markdown_files": len(rows),
        "status_counts": dict(counts),
        "category_counts": dict(category_counts),
        "validated_documents": 0,
        "hosted_ai_api_calls": 0,
        "documents": rows,
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n========== INVENTORY SUMMARY ==========")

    print("Repository commit:", commit)

    print("Total Markdown files:", len(rows))

    for status, count in sorted(counts.items()):

        print(status, ":", count)

    print("\n========== CATEGORY SUMMARY ==========")

    for category, count in sorted(
        category_counts.items()
    ):

        print(category, ":", count)

    print("\nValidated documents: 0")

    print("Hosted AI API calls: 0")

    print("Output:", OUTPUT)

    print("SUCCESS: Repository inventory created.")


if __name__ == "__main__":

    main()
