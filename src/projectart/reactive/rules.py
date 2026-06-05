"""Rule engine: pick the first rule whose match applies to a tracked class.

A rule's ``match`` supports ``class`` as an exact name or ``"*"`` wildcard.
First match wins (ordered), so specific rules precede the wildcard fallback.
"""
from __future__ import annotations

from .config import Rule


def match_rule(rules: list[Rule], class_name: str) -> Rule | None:
    for rule in rules:
        want = rule.match.get("class", "*")
        if want == "*" or want == class_name:
            return rule
    return None


def parse_behaviors(specs: list[dict]) -> list[tuple[str, dict]]:
    """Convert config behavior dicts (``{"follow": {...}}``) to ``(name, params)``."""
    out: list[tuple[str, dict]] = []
    for spec in specs:
        for name, params in spec.items():
            out.append((name, dict(params)))
    return out
