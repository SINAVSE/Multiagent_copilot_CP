
"""
CAP-08: Context-aware Query Resolver.

Provides deterministic field-level conversational memory.

Supports:
- Independent request detection
- Topic switching
- Related follow-up questions
- Selective context inheritance
- Amount and currency replacement
- Headcount and duration updates
- Explicit references to previous amounts
- Ambiguous reference detection

No external dependencies or LLM calls.
"""

import re

from typing import Any


class QueryResolver:

    # Specific categories must precede broader categories.
    CATEGORY_PATTERNS = {
        "traveling_team_meals": (
            r"\b(?:travell?ing team meals?|"
            r"team meals? (?:while|during) travel)\b"
        ),

        "customer_dinner": (
            r"\bcustomer (?:dinner|meal|lunch)s?\b"
        ),

        "team_dinner": (
            r"\bteam (?:dinner|meal|lunch)s?\b"
        ),

        "team_outing": (
            r"\b(?:team outing|team event|"
            r"team building)s?\b"
        ),

        "hotel": (
            r"\b(?:hotel|lodging|accommodation)s?\b"
        ),

        "transport": (
            r"\b(?:taxi|cab|transport|"
            r"transportation|mileage)s?\b"
        ),

        "airfare": (
            r"\b(?:airfare|flight|flights)\b"
        ),

        "meal": (
            r"\b(?:meal|meals|food|"
            r"breakfast|per[- ]diem)\b"
        ),

        "dinner": r"\bdinner\b",

        "lunch": r"\blunch\b",

        "gift": r"\b(?:gift|gifts)\b",

        "entertainment": r"\bentertainment\b",

        "sick_leave": r"\bsick leave\b",

        "annual_leave": r"\bannual leave\b",

        "parental_leave": (
            r"\b(?:parental|maternity|"
            r"paternity) leave\b"
        ),

        "leave": (
            r"\b(?:leave|vacation|pto|holiday)s?\b"
        ),

        "remote_work": (
            r"\b(?:wfh|work from home|"
            r"remote work|hybrid work)\b"
        ),

        "receipt": (
            r"\b(?:receipt|invoice|bill)s?\b"
        ),

        "fraud": (
            r"\b(?:fraud|bribery|corruption)\b"
        ),

        "travel": (
            r"\b(?:travel|travell?ing|"
            r"travell?ed|trip)s?\b"
        ),

        "reimbursement": (
            r"\b(?:reimburse|reimbursement|"
            r"expense claim|expense report)\b"
        ),
    }

    NON_EXPENSE_CATEGORIES = {
        "leave",
        "annual_leave",
        "sick_leave",
        "parental_leave",
        "remote_work",
        "fraud",
    }

    FOLLOW_UP_PATTERN = re.compile(
        r"^(?:what about\b|how about\b|"
        r"what if\b|and\b|also\b|then\b|"
        r"in that case\b|actually\b|"
        r"instead\b|correction\b)",
        re.IGNORECASE,
    )

    SAME_AMOUNT_PATTERN = re.compile(
        r"\b(?:same amount|same cost|"
        r"same expense amount)\b",
        re.IGNORECASE,
    )

    REFERENCE_PATTERN = re.compile(
        r"\b(?:it|that|this|those|these|"
        r"the same|previous one)\b",
        re.IGNORECASE,
    )

    EXPENSE_PATTERN = re.compile(
        r"\b(?:claim|spent|spend|cost|"
        r"paid|reimburse|reimbursable|"
        r"eligible|expense|assess)\b",
        re.IGNORECASE,
    )

    POLICY_LIMIT_PATTERN = re.compile(
        r"\b(?:what is|what's|tell me|"
        r"how much is|explain)\b.*"
        r"\b(?:limit|allowance|budget|policy|cap)\b",
        re.IGNORECASE,
    )

    LOCATION_PATTERN = re.compile(
        r"\b(?:international|internationally|"
        r"domestic|domestically|overseas)\b",
        re.IGNORECASE,
    )

    HOTEL_LOCATION_PATTERN = re.compile(
        r"\b(?:non[- ]metro|metro)\b",
        re.IGNORECASE,
    )

    AMOUNT_PATTERN = re.compile(
        r"(?<!\w)"
        r"(?:(?P<prefix>USD|INR|EUR|GBP|"
        r"Rs\.?|[$₹€£])\s*"
        r"(?P<amount1>-?\d[\d,]*(?:\.\d+)?)"
        r"|(?P<amount2>-?\d[\d,]*(?:\.\d+)?)"
        r"\s*(?P<suffix>USD|INR|EUR|GBP|"
        r"dollars?|rupees?))"
        r"(?!\w)",
        re.IGNORECASE,
    )

    HEADCOUNT_PATTERN = re.compile(
        r"\b(\d+|one|two|three|four|five|"
        r"six|seven|eight|nine|ten)\s+"
        r"(?:eligible\s+)?"
        r"(?:employees?|people|persons?|"
        r"attendees?|participants?|guests?)\b",
        re.IGNORECASE,
    )

    DAYS_PATTERN = re.compile(
        r"\b(\d+|one|two|three|four|five|"
        r"six|seven|eight|nine|ten)\s+"
        r"(?:travel\s+)?days?\b",
        re.IGNORECASE,
    )

    NIGHTS_PATTERN = re.compile(
        r"\b(\d+|one|two|three|four|five|"
        r"six|seven|eight|nine|ten)\s+nights?\b",
        re.IGNORECASE,
    )

    CURRENCY_MAPPING = {
        "$": "USD",
        "₹": "INR",
        "€": "EUR",
        "£": "GBP",
        "RS": "INR",
        "RS.": "INR",
        "RUPEE": "INR",
        "RUPEES": "INR",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
    }

    NUMBER_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }

    @classmethod
    def parse_number(cls, value):

        value = value.lower()

        if value in cls.NUMBER_WORDS:
            return cls.NUMBER_WORDS[value]

        return int(value)

    @classmethod
    def detect_category(cls, query):

        for category, pattern in (
            cls.CATEGORY_PATTERNS.items()
        ):
            if re.search(
                pattern,
                query,
                re.IGNORECASE,
            ):
                return category

        return None

    @classmethod
    def detect_topic(cls, category):

        if category in {
            "annual_leave",
            "sick_leave",
            "parental_leave",
            "leave",
        }:
            return "leave"

        if category == "remote_work":
            return "remote_work"

        if category == "fraud":
            return "fraud"

        if category == "receipt":
            return "document"

        if category is not None:
            return "expense"

        return None

    @classmethod
    def extract_amount(cls, query):

        matches = list(
            cls.AMOUNT_PATTERN.finditer(query)
        )

        if not matches:
            return {}

        # Multiple monetary values require
        # separate line-item interpretation.
        if len(matches) != 1:
            return {
                "multiple_amounts": True,
            }

        match = matches[0]

        amount = (
            match.group("amount1")
            or match.group("amount2")
        )

        currency = (
            match.group("prefix")
            or match.group("suffix")
        ).upper()

        currency = cls.CURRENCY_MAPPING.get(
            currency,
            currency,
        )

        return {
            "claimed_amount": float(
                amount.replace(",", "")
            ),
            "currency": currency,
        }

    @classmethod
    def extract_fields(cls, query):

        fields = cls.extract_amount(query)

        headcount = cls.HEADCOUNT_PATTERN.search(
            query
        )

        if headcount:
            fields["headcount"] = cls.parse_number(
                headcount.group(1)
            )

        days = cls.DAYS_PATTERN.search(query)

        if days:
            fields["travel_days"] = cls.parse_number(
                days.group(1)
            )

        nights = cls.NIGHTS_PATTERN.search(query)

        if nights:
            fields["nights"] = cls.parse_number(
                nights.group(1)
            )

        travel_location = (
            cls.LOCATION_PATTERN.search(query)
        )

        if travel_location:

            value = travel_location.group(0).lower()

            fields["travel_type"] = (
                "international"
                if value in {
                    "international",
                    "internationally",
                    "overseas",
                }
                else "domestic"
            )

        hotel_location = (
            cls.HOTEL_LOCATION_PATTERN.search(query)
        )

        if hotel_location:

            fields["hotel_location"] = (
                hotel_location.group(0)
                .lower()
                .replace(" ", "-")
            )

        return fields

    @classmethod
    def detect_intent(cls, query, fields):

        if cls.POLICY_LIMIT_PATTERN.search(query):
            return "policy_query"

        if (
            "claimed_amount" in fields
            or fields.get("multiple_amounts")
            or cls.EXPENSE_PATTERN.search(query)
        ):
            return "expense_assessment"

        return "policy_query"

    @classmethod
    def is_field_update(cls, query, fields):

        # Standalone clarification answers such as
        # "3 employees" or "There were 3 employees."
        return bool(fields) and bool(
            re.match(
                r"^(?:there (?:were|are)\b|"
                r"we (?:were|are|had)\b|"
                r"it was\b|"
                r"for\s+\d+\b|"
                r"\d+\s+|"
                r"(?:one|two|three|four|five|"
                r"six|seven|eight|nine|ten)\s+)",
                query,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def format_field(key, value):

        labels = {
            "category": "Expense or policy category",
            "claimed_amount": "Claimed amount",
            "currency": "Currency",
            "headcount": "Number of people",
            "travel_days": "Travel days",
            "nights": "Hotel nights",
            "travel_type": "Travel type",
            "hotel_location": "Hotel location",
        }

        label = labels.get(key, key)

        if isinstance(value, str):
            value = value.replace("_", " ")

        return f"{label}: {value}"

    def resolve(
        self,
        query: str,
        previous_context: dict[str, Any] | None = None,
    ) -> dict:

        current_query = query.strip()

        previous = dict(previous_context or {})

        result = {
            "user_query": current_query,
            "resolved_query": current_query,
            "conversation_context": {},
            "inherited_fields": {},
            "context_used": False,
            "resolution_type": "standalone",
            "requires_clarification": False,
            "clarification_message": None,
        }

        if not current_query:

            result["resolution_type"] = "empty_query"

            return result

        category = self.detect_category(
            current_query
        )

        topic = self.detect_topic(category)

        fields = self.extract_fields(
            current_query
        )

        intent = self.detect_intent(
            current_query,
            fields,
        )

        follow_up = bool(
            self.FOLLOW_UP_PATTERN.search(
                current_query
            )
        )

        same_amount = bool(
            self.SAME_AMOUNT_PATTERN.search(
                current_query
            )
        )

        reference = bool(
            self.REFERENCE_PATTERN.search(
                current_query
            )
        )

        field_update = self.is_field_update(
            current_query,
            fields,
        )

        previous_category = previous.get(
            "category"
        )

        previous_topic = previous.get("topic")

        previous_intent = previous.get("intent")

        # A generic travel reference can refine a
        # specific prior category, such as hotel.
        if (
            follow_up
            and category == "travel"
            and "travel_type" in fields
            and previous_category in {
                "hotel",
                "meal",
                "transport",
                "airfare",
                "traveling_team_meals",
            }
        ):

            category = previous_category
            topic = "expense"

        # Multiple amounts must not inherit another
        # monetary value or reuse a single amount.
        if fields.get("multiple_amounts"):

            result.update({
                "resolution_type": (
                    "multiple_amounts"
                ),
                "requires_clarification": True,
                "clarification_message": (
                    "Please identify each expense "
                    "amount and its category."
                ),
                "conversation_context": {
                    "topic": topic,
                    "category": category,
                    "intent": intent,
                },
            })

            return result

        # Explicit non-expense topic changes
        # always isolate prior monetary facts.
        if (
            topic
            and previous_topic
            and topic != previous_topic
        ):

            result["resolution_type"] = (
                "topic_switch"
            )

            previous = {}

        inherited = {}

        # Explicit reference to the same amount.
        if (
            same_amount
            and "claimed_amount" not in fields
        ):

            if "claimed_amount" not in previous:

                result.update({
                    "resolution_type": (
                        "unresolved_amount_reference"
                    ),
                    "requires_clarification": True,
                    "clarification_message": (
                        "What amount and currency "
                        "would you like to assess?"
                    ),
                    "conversation_context": {
                        "topic": topic,
                        "category": category,
                        "intent": intent,
                        **fields,
                    },
                })

                return result

            if topic in {
                None,
                "expense",
            }:

                for key in [
                    "claimed_amount",
                    "currency",
                ]:

                    if key in previous:
                        inherited[key] = previous[key]

                intent = "expense_assessment"

        # A related field update is applicable only
        # to an existing expense assessment.
        related_expense_update = (
            bool(previous)
            and previous_intent
            == "expense_assessment"
            and topic in {None, "expense"}
            and (
                category is None
                or category == previous_category
            )
            and (
                follow_up
                or field_update
                or same_amount
                or reference
            )
            and (
                bool(fields)
                or same_amount
            )
        )

        related_policy_follow_up = (
            bool(previous)
            and (follow_up or reference)
            and (
                category == previous_category
                or (
                    category is None
                    and reference
                )
                or (
                    topic == "expense"
                    and category == previous_category
                    and "travel_type" in fields
                )
            )
        )

        if related_expense_update:

            for key in [
                "claimed_amount",
                "currency",
                "headcount",
                "travel_days",
                "nights",
                "travel_type",
                "hotel_location",
            ]:

                if (
                    key in previous
                    and key not in fields
                ):

                    inherited[key] = previous[key]

            intent = "expense_assessment"

            result["resolution_type"] = (
                "expense_follow_up"
            )

        elif same_amount and inherited:

            # New category: transfer the referenced
            # amount only, not category-specific facts.
            result["resolution_type"] = (
                "selective_amount_inheritance"
            )

        elif related_policy_follow_up:

            # Inherit applicable policy context,
            # never previous financial decisions.
            for key in [
                "travel_type",
                "hotel_location",
            ]:

                if (
                    key in previous
                    and key not in fields
                ):

                    inherited[key] = previous[key]

            result["resolution_type"] = (
                "policy_follow_up"
            )

        # Retain the relevant previous category
        # when the current request is a fragment.
        if (
            category is None
            and previous
            and (
                related_expense_update
                or related_policy_follow_up
                or (same_amount and inherited)
            )
        ):

            category = previous_category
            topic = previous_topic

        # A reference without usable context
        # should not invent an expense.
        if (
            category is None
            and reference
            and not related_expense_update
            and not related_policy_follow_up
            and not inherited
        ):

            result.update({
                "resolution_type": (
                    "unresolved_reference"
                ),
                "requires_clarification": True,
                "clarification_message": (
                    "Which expense or policy "
                    "are you referring to?"
                ),
            })

            return result

        # New explicit fields override inherited
        # values. No assistant-generated facts
        # are used in this process.
        context = {
            "topic": topic,
            "category": category,
            "intent": intent,
            **inherited,
            **fields,
        }

        context = {
            key: value
            for key, value in context.items()
            if value is not None
        }

        context_lines = []

        if category and (
            related_expense_update
            or related_policy_follow_up
            or same_amount
        ):

            context_lines.append(
                self.format_field(
                    "category",
                    category,
                )
            )

        for key, value in inherited.items():

            if key not in fields:

                context_lines.append(
                    self.format_field(
                        key,
                        value,
                    )
                )

        if context_lines:

            result["resolved_query"] = (
                current_query
                + "\n\n"
                + "Relevant user-provided context:\n"
                + "\n".join(context_lines)
                + "\nUse this context only for "
                  "the current request. "
                  "Do not treat it as policy "
                  "evidence or approval."
            )

            result["context_used"] = True

        result.update({
            "conversation_context": context,
            "inherited_fields": inherited,
        })

        return result
