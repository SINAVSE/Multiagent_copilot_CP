
from pathlib import Path
from collections import Counter
import hashlib
import json
import re


ROOT = Path("data/real_corpus")

MANIFEST = (
    ROOT / "expanded_v2"
    / "provisional_manifest_v2.jsonl"
)

CLASSIFICATION = (
    ROOT / "metadata"
    / "policy_applicability_draft_v2.json"
)

OUTPUT = (
    ROOT / "expanded_v2"
    / "applicability_evidence_review_v2.json"
)

SUMMARY = (
    ROOT / "metadata"
    / "applicability_evidence_summary_v2.json"
)

PATTERNS = {
    "explicit_scope": (
        r"\b(?:this policy|this procedure|this guideline)"
        r"\s+(?:applies|is applicable|covers)\b"
    ),
    "employee_scope": (
        r"\b(?:applies to|applicable to|eligible employees|"
        r"eligible team members|covered employees)\b"
    ),
    "global_scope": (
        r"\b(?:globally|worldwide|all countries|"
        r"regardless of location|all locations)\b"
    ),
    "country_scope": (
        r"\b(?:country-specific|country specific|"
        r"local laws?|local legislation|"
        r"country of employment|employment country)\b"
    ),
    "employment_entity": (
        r"\b(?:employed by|employed in|"
        r"employment entity|legal entity|"
        r"entity employees|based in)\b"
    ),
    "exceptions": (
        r"\b(?:does not apply|not applicable|"
        r"except for|unless otherwise|"
        r"excluded from|subject to local)\b"
    ),
    "scope_heading": (
        r"^\s*#{1,6}\s*"
        r"(?:scope|applicability|eligibility|"
        r"who is eligible|purpose and scope)\b"
    ),
}

COUNTRY_PATTERNS = {
    "US": r"\b(?:United States|USA|U\.S\.A?\.?)\b",
    "IN": r"\bIndia\b",
    "DE": r"\bGermany\b",
    "GB": r"\b(?:United Kingdom|UK)\b",
    "AU": r"\bAustralia\b",
    "NZ": r"\bNew Zealand\b",
    "BE": r"\bBelgium\b",
    "NL": r"\bNetherlands\b",
    "ES": r"\bSpain\b",
    "JP": r"\bJapan\b",
    "KR": r"\b(?:South Korea|Korea)\b",
    "IL": r"\bIsrael\b",
    "CA": r"\bCanada\b",
    "FR": r"\bFrance\b",
    "IE": r"\bIreland\b",
    "SG": r"\bSingapore\b",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def extract_evidence(text):

    lines = text.splitlines()

    candidates = []

    weights = {
        "explicit_scope": 5,
        "employee_scope": 3,
        "global_scope": 4,
        "country_scope": 4,
        "employment_entity": 4,
        "exceptions": 3,
        "scope_heading": 5,
    }

    for index, line in enumerate(lines):

        if not line.strip():
            continue

        signals = [
            name
            for name, pattern in PATTERNS.items()
            if re.search(pattern, line, re.I)
        ]

        if not signals:
            continue

        start = max(0, index - 1)
        end = min(len(lines), index + 4)

        passage = "\n".join(lines[start:end])

        countries = [
            code
            for code, pattern in COUNTRY_PATTERNS.items()
            if re.search(pattern, passage, re.I)
        ]

        score = sum(
            weights[name]
            for name in signals
        )

        if countries:
            score += 2

        candidates.append({
            "line_start": start + 1,
            "line_end": end,
            "signals": signals,
            "country_mentions": countries,
            "priority_score": score,
            "passage": passage,
            "passage_sha256": sha256(
                passage.encode("utf-8")
            ),
        })

    candidates.sort(
        key=lambda item: (
            -item["priority_score"],
            item["line_start"],
        )
    )

    selected = []
    seen = set()

    for item in candidates:

        if item["passage_sha256"] in seen:
            continue

        seen.add(item["passage_sha256"])

        selected.append(item)

        if len(selected) >= 12:
            break

    return selected


def main():

    manifest_hash = sha256(
        MANIFEST.read_bytes()
    )

    draft = json.loads(
        CLASSIFICATION.read_text(
            encoding="utf-8"
        )
    )

    assert draft["manifest_sha256"] == manifest_hash

    documents = [
        json.loads(line)
        for line in MANIFEST.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    classifications = {
        item["document_id"]: item
        for item in draft["documents"]
    }

    assert len(documents) == 659
    assert len(classifications) == 659

    assert set(classifications) == {
        item["document_id"]
        for item in documents
    }

    records = []

    for document in documents:

        document_id = document["document_id"]

        raw = Path(
            document["processed_path"]
        ).read_bytes()

        assert sha256(raw) == (
            document["processed_file_hash"]
        ), "Integrity mismatch: " + document_id

        text = raw.decode("utf-8-sig")

        evidence = extract_evidence(text)

        classification = classifications[
            document_id
        ]

        if not evidence:

            priority = "requires_full_review"

        elif (
            classification["candidate_applicability"]
            == "COUNTRY_SPECIFIC"
        ):

            priority = "country_specific_review"

        elif evidence[0]["priority_score"] >= 5:

            priority = "high_priority_scope_review"

        else:

            priority = "standard_scope_review"

        records.append({
            "document_id": document_id,
            "organization": document["organization"],
            "title": document["title"],
            "source_url": document["source_url"],
            "processed_file_hash": (
                document["processed_file_hash"]
            ),
            "candidate_applicability": (
                classification["candidate_applicability"]
            ),
            "candidate_countries": (
                classification["candidate_countries"]
            ),
            "review_priority": priority,
            "evidence_count": len(evidence),
            "evidence": evidence,
            "verified_applicability": "UNKNOWN",
            "review_status": "pending_semantic_review",
        })

    priority_counts = Counter(
        item["review_priority"]
        for item in records
    )

    signal_counts = Counter(
        signal
        for record in records
        for evidence in record["evidence"]
        for signal in evidence["signals"]
    )

    organization_counts = {}

    for organization in sorted({
        item["organization"]
        for item in records
    }):

        organization_counts[organization] = dict(
            Counter(
                item["review_priority"]
                for item in records
                if item["organization"] == organization
            )
        )

    summary = {
        "manifest_sha256": manifest_hash,
        "total_documents": len(records),
        "integrity_checks_passed": len(records),
        "documents_with_evidence": sum(
            bool(item["evidence"])
            for item in records
        ),
        "total_evidence_passages": sum(
            item["evidence_count"]
            for item in records
        ),
        "review_priority_distribution": dict(
            priority_counts
        ),
        "signal_distribution": dict(signal_counts),
        "organization_distribution": (
            organization_counts
        ),
        "verified_applicability": 0,
        "hosted_ai_api_calls": 0,
        "limitations": [
            "Keyword matches are review candidates only.",
            "Country mentions do not prove employment scope.",
            "Extracted passages may omit relevant exceptions.",
            "No policy has been automatically approved."
        ],
    }

    OUTPUT.write_text(
        json.dumps({
            "summary": summary,
            "documents": records,
        }, indent=2),
        encoding="utf-8"
    )

    SUMMARY.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(
        "\n========== APPLICABILITY EVIDENCE REVIEW =========="
    )

    print("Documents scanned:", len(records))

    print(
        "Integrity checks passed:",
        summary["integrity_checks_passed"]
    )

    print(
        "Documents with evidence:",
        summary["documents_with_evidence"]
    )

    print(
        "Evidence passages:",
        summary["total_evidence_passages"]
    )

    print("\nReview priority:")

    for priority, count in priority_counts.items():
        print(priority, ":", count)

    print("\nVerified applicability: 0")
    print("Hosted AI API calls: 0")
    print("Detailed evidence:", OUTPUT)
    print("Summary:", SUMMARY)
    print("SUCCESS: Applicability evidence extracted.")


if __name__ == "__main__":
    main()
