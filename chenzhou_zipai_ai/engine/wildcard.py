"""Wildcard value planner helpers."""

from __future__ import annotations

from ai.pro_brain import WildcardValue, evaluate_wildcard_value


class WildcardPlanner:
    def evaluate(self, context, allocation) -> WildcardValue:
        return evaluate_wildcard_value(context, allocation)


__all__ = ["WildcardPlanner", "WildcardValue", "evaluate_wildcard_value"]
