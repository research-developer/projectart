from projectart.reactive.config import Rule
from projectart.reactive.rules import match_rule

RULES = [
    Rule(
        match={"class": "person"},
        action={"spawn": "box", "behaviors": [{"follow": {"gain": 0.5}}]},
    ),
    Rule(match={"class": "*"}, action={"spawn": "ball", "behaviors": []}),
]


def test_first_matching_rule_wins():
    r = match_rule(RULES, "person")
    assert r.action["spawn"] == "box"


def test_wildcard_fallback():
    r = match_rule(RULES, "cat")
    assert r.action["spawn"] == "ball"


def test_no_match_returns_none():
    r = match_rule([Rule(match={"class": "dog"}, action={})], "cat")
    assert r is None


def test_behaviors_parsed_to_tuples():
    from projectart.reactive.rules import parse_behaviors

    bs = parse_behaviors([{"follow": {"gain": 0.5}}, {"scale": {"min": 0.03}}])
    assert bs == [("follow", {"gain": 0.5}), ("scale", {"min": 0.03})]
