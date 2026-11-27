
from src.agents.policy_scope import (
    PolicyScopeError,
    resolve_policy_scope,
)

from src.agents.query_resolver import QueryResolver


class ScopedQueryResolver:

    def __init__(self, resolver=None):

        self.resolver = (
            resolver
            if resolver is not None
            else QueryResolver()
        )

    def resolve(
        self,
        query,
        organization=None,
        employment_country=None,
        previous_scope=None,
        previous_context=None,
    ):

        try:

            scope_result = resolve_policy_scope(
                query=query,
                organization=organization,
                employment_country=employment_country,
                previous_scope=previous_scope,
            )

        except PolicyScopeError as exc:

            return {
                "user_query": query.strip(),
                "resolved_query": query.strip(),
                "policy_scope": None,
                "scope_changed": False,
                "conversation_context": {},
                "inherited_fields": {},
                "context_used": False,
                "resolution_type": "policy_scope_clarification",
                "requires_clarification": True,
                "clarification_message": str(exc),
            }

        scope = scope_result["scope"]

        changed = scope_result["scope_changed"]

        # Never inherit unscoped or incompatible context.
        safe_context = (
            dict(previous_context or {})
            if previous_scope is not None and not changed
            else {}
        )

        resolution = self.resolver.resolve(
            query=query,
            previous_context=safe_context,
        )

        resolution["policy_scope"] = scope.to_dict()

        resolution["scope_changed"] = changed

        if not resolution["requires_clarification"]:

            scope_lines = [
                "Applicable organization: "
                + scope.organization
            ]

            if scope.employment_country:

                scope_lines.append(
                    "Employment country: "
                    + scope.employment_country
                )

            resolution["resolved_query"] += (
                "\n\n"
                + "\n".join(scope_lines)
                + "\nThese values identify policy applicability "
                  "and do not establish expense approval."
            )

        return resolution
