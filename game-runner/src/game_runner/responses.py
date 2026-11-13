"""Decode model JSON for gameplay and offline analysis."""

import json
import re


def decode_json(content: object):
    if not isinstance(content, str):
        raise ValueError("model content must be a JSON string")
    text = content.strip()
    fenced = re.fullmatch(
        r"(?P<fence>`{3,})(?:json)?\s*(?P<body>.*?)\s*(?P=fence)", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return json.loads(fenced.group("body") if fenced else text)
