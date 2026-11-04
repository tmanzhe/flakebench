"""Python checker and tool examples bundled for reproducible example suites."""

import json


def sorted_unique(answer):
    try:
        values = json.loads(answer)
        return values == [2, 5, 9] and all(type(value) is int for value in values)
    except (ValueError, TypeError):
        return False


def reverse(text):
    return text[::-1]
