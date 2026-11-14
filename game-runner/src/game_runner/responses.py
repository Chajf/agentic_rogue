"""Decode model JSON for gameplay and offline analysis."""

import json
import re

from langchain_core.messages import AIMessage


def decode_json(content: object):
    if isinstance(content, AIMessage):
        response = content
        content = response.content
        if isinstance(content, str) and not content.strip():
            if response.response_metadata.get("finish_reason") == "length":
                usage = response.usage_metadata or {}
                reasoning = (usage.get("output_token_details") or {}).get("reasoning")
                output = usage.get("output_tokens")
                message = "model reached its completion token limit before returning JSON"
                if reasoning == output and output:
                    message += f" ({reasoning} tokens spent on reasoning)"
                raise ValueError(message)
            raise ValueError("model returned empty content")
    if not isinstance(content, str):
        raise ValueError("model content must be a JSON string")
    text = content.strip()
    fenced = re.fullmatch(
        r"(?P<fence>`{3,})(?:json)?\s*(?P<body>.*?)\s*(?P=fence)", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return json.loads(fenced.group("body") if fenced else text)
