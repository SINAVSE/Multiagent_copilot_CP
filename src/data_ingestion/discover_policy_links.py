import json
import re
import time

from collections import deque, Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from lxml import html


OUTPUT = Path(
    "data/real_corpus/metadata/policy_candidates_v2.json"
)

USER_AGENT = "AcademicPolicyResearchBot/1.0"

TIMEOUT = 30

DELAY = 2

MAX_INDEX_PAGES = 60

MAX_CANDIDATES_PER_SOURCE = 400

SOURCES = [
    {
        "id": "UTAH",
        "organization": "University of Utah",
        "url": "https://regulations.utah.edu/info/policyList.php",
    },
    {
        "id": "UW",
        "organization": "University of Washington",
        "url": "https://policy.uw.edu/directory/aps/",
    },
    {
        "id": "UC",
        "organization": "University of California",
        "url": "https://policy.ucop.edu/advanced-search.php?action=welcome&all=1&op=browse",
    },
    {
        "id": "IU",
        "organization": "Indiana University",
        "url": "https://policies.iu.edu/",
    },
    {
        "id": "GITLAB",
        "organization": "GitLab",
        "url": "https://handbook.gitlab.com/handbook/",
    },
]

CATEGORIES = {
    "travel": [
        "travel", "per diem", "lodging",
        "airfare", "mileage", "transportation"
    ],
    "expense": [
        "expense", "reimbursement", "reimburse",
        "meal allowance", "entertainment"
    ],
    "hr": [
        "employee", "employment", "personnel",
        "leave", "vacation", "absence",
        "benefits", "compensation",
        "remote work", "telework",
        "recruitment", "working hours",
        "payroll", "overtime"
    ],
    "ethics": [
        "ethics", "gift", "conflict of interest",
        "bribery", "corruption", "conduct",
        "discrimination", "harassment"
    ],
    "procurement": [
        "procurement", "purchasing", "spending",
        "purchase", "contract", "credit card"
    ],
    "compliance": [
        "fraud", "audit", "compliance",
        "financial control", "accounting",
        "whistleblower", "financial management"
    ],
}

EXCLUDED = (
    "/revisions_",
    "/archive/",
    "/archives/",
    "/regulation-update/",
    "/news/",
    "/events/",
    "/contact/",
    "/login/",
    "/search/",
)

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": USER_AGENT,
})

ROBOTS = {}

LAST_REQUEST = {}


def canonical_url(url):

    parts = urlsplit(url)

    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path,
        parts.query,
        "",
    ))


def same_host(first, second):

    a = urlsplit(first).hostname or ""

    b = urlsplit(second).hostname or ""

    return a.removeprefix("www.") == (
        b.removeprefix("www.")
    )


def wait_for_host(url, delay=DELAY):

    host = urlsplit(url).hostname

    previous = LAST_REQUEST.get(host, 0)

    remaining = delay - (
        time.monotonic() - previous
    )

    if remaining > 0:
        time.sleep(remaining)

    LAST_REQUEST[host] = time.monotonic()


def robots_allowed(url):

    parts = urlsplit(url)

    origin = f"{parts.scheme}://{parts.netloc}"

    if origin not in ROBOTS:

        robots_url = origin + "/robots.txt"

        try:

            wait_for_host(robots_url)

            response = SESSION.get(
                robots_url,
                timeout=TIMEOUT,
                allow_redirects=False,
            )

            parser = RobotFileParser()

            if response.status_code == 200:

                parser.parse(
                    response.text.splitlines()
                )

            elif response.status_code in (404, 410):

                parser.parse([])

            else:

                print(
                    "ROBOTS CHECK UNAVAILABLE:",
                    origin,
                    response.status_code,
                )

                ROBOTS[origin] = None

                return False

            ROBOTS[origin] = parser

        except requests.RequestException:

            ROBOTS[origin] = None

            return False

    parser = ROBOTS[origin]

    return (
        parser is not None
        and parser.can_fetch(USER_AGENT, url)
    )


def fetch_index(url):

    if not robots_allowed(url):

        print("SKIPPED BY ROBOTS CHECK:", url)

        return None

    parts = urlsplit(url)

    origin = f"{parts.scheme}://{parts.netloc}"

    parser = ROBOTS[origin]

    delay = max(
        DELAY,
        parser.crawl_delay(USER_AGENT) or 0,
    )

    wait_for_host(url, delay)

    response = SESSION.get(
        url,
        timeout=TIMEOUT,
        allow_redirects=False,
    )

    response.raise_for_status()

    if response.is_redirect:

        print("REDIRECT REQUIRES REVIEW:", url)

        return None

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if "html" not in content_type:

        return None

    return html.fromstring(response.content)


def classify(text, url):

    normalized = re.sub(
        r"[-_/]+",
        " ",
        (text + " " + url).lower(),
    )

    categories = []

    for category, terms in CATEGORIES.items():

        if any(
            re.search(
                r"\b" + re.escape(term) + r"\w*",
                normalized,
            )
            for term in terms
        ):

            categories.append(category)

    return categories


def is_candidate(source_id, url):

    parts = urlsplit(url)

    path = parts.path.lower()

    if any(
        excluded in path
        for excluded in EXCLUDED
    ):

        return False

    if source_id == "UTAH":

        return bool(
            re.search(
                r"/(?:r|g|p)?[1-9]-\d{3}[a-z]?"
                r"\.(?:php|pdf)$",
                path,
            )
        )

    if source_id == "UW":

        return bool(
            re.search(
                r"/aps-\d+",
                path,
            )
        )

    if source_id == "UC":

        return (
            path.endswith(".pdf")
            or (
                path.endswith("/doc/")
                and bool(parts.query)
            )
            or bool(
                re.search(r"/doc/\d+", path)
            )
        )

    if source_id == "IU":

        return (
            path.startswith("/policies/")
            and path.count("/") >= 3
            and path.endswith("/index.html")
        )

    if source_id == "GITLAB":

        return (
            path.startswith("/handbook/")
            and path.count("/") >= 3
            and not path.startswith(
                "/handbook/company/culture/all-remote/"
            )
            and any(
                term in path
                for term in (
                    "/policy",
                    "/policies",
                    "/expenses",
                    "/travel",
                    "/procurement",
                    "/leave",
                    "/time-off",
                    "/benefits",
                    "/anti-fraud",
                    "/anti-corruption",
                    "/gifts",
                    "/payroll",
                    "/compensation",
                )
            )
        )

    return False


def is_directory(source_id, url, categories):

    path = urlsplit(url).path.lower()

    if source_id == "UW":

        return bool(
            re.fullmatch(
                r"/directory/aps/section-[^/]+/?",
                path,
            )
        )

    if source_id == "UC":

        return (
            "op=browse" in url
            and "action=" in url
        )

    if source_id == "IU":

        return (
            (
                path.startswith("/policies/")
                and path.count("/") <= 2
            )
            or (
                path.endswith("index.html")
                and bool(categories)
                and not is_candidate(source_id, url)
            )
        )

    if source_id == "GITLAB":

        return (
            path.startswith("/handbook/")
            and bool(categories)
            and path.count("/") <= 4
        )

    return False


def discover(source):

    queue = deque([source["url"]])

    visited = set()

    candidates = {}

    while (
        queue
        and len(visited) < MAX_INDEX_PAGES
        and len(candidates) < MAX_CANDIDATES_PER_SOURCE
    ):

        index_url = canonical_url(queue.popleft())

        if index_url in visited:

            continue

        visited.add(index_url)

        print("SCANNING:", index_url)

        try:

            page = fetch_index(index_url)

            if page is None:

                continue

            for anchor in page.xpath("//a[@href]"):

                href = anchor.get("href", "").strip()

                if not href:

                    continue

                url = canonical_url(
                    urljoin(index_url, href)
                )

                if urlsplit(url).scheme != "https":

                    continue

                if not same_host(source["url"], url):

                    continue

                if url == index_url:

                    continue

                title = " ".join(
                    anchor.text_content().split()
                )

                if not title:

                    continue

                categories = classify(title, url)

                if not categories:

                    if is_directory(
                        source["id"],
                        url,
                        categories,
                    ) and url not in visited:

                        queue.append(url)

                    continue

                if is_candidate(source["id"], url):

                    candidates[url] = {
                        "source_id": source["id"],
                        "organization": source["organization"],
                        "title": title,
                        "source_url": url,
                        "discovered_from": index_url,
                        "categories": categories,
                        "country": (
                            None
                            if source["id"] == "GITLAB"
                            else "US"
                        ),
                        "status": "candidate",
                        "redistribution_status": "requires_review",
                    }

                    if len(candidates) >= MAX_CANDIDATES_PER_SOURCE:

                        break

                if is_directory(
                    source["id"],
                    url,
                    categories,
                ):

                    if url not in visited:

                        queue.append(url)

        except (
            requests.RequestException,
            ValueError,
        ) as error:

            print("INDEX FAILED:", str(error))

    return list(candidates.values())


def main():

    candidates = {}

    source_counts = {}

    for source in SOURCES:

        print("\n" + "=" * 60)

        print("SOURCE:", source["organization"])

        found = discover(source)

        source_counts[source["id"]] = len(found)

        for row in found:

            candidates[row["source_url"]] = row

        print("Candidates discovered:", len(found))

    rows = list(candidates.values())

    report = {
        "version": "2.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "candidate_count": len(rows),
        "source_counts": source_counts,
        "category_counts": dict(Counter(
            category
            for row in rows
            for category in row["categories"]
        )),
        "validated_documents": 0,
        "hosted_ai_api_calls": 0,
        "candidates": rows,
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

    print("\n========== DISCOVERY SUMMARY ==========")

    for name, count in source_counts.items():

        print(name, ":", count)

    print("Total unique candidate URLs:", len(rows))

    print("Validated documents: 0")

    print("Hosted AI API calls: 0")

    print("Output:", OUTPUT)

    print("\nSUCCESS: Candidate discovery completed.")


if __name__ == "__main__":

    main()
