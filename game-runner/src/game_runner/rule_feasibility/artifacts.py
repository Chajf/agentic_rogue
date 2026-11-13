"""Canonical, versioned records for the offline evaluation."""

from hashlib import sha256
import json
from pathlib import Path

SCHEMA_VERSION = 1
FAMILIES = ("ambiguous_movement", "conditional_rest", "item_identity")


class FeasibilityError(ValueError):
    pass


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes())


def read_rows(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.splitlines()]


def write_artifact(path: Path, value: dict) -> str:
    data = canonical({"schema_version": SCHEMA_VERSION, **value})
    with path.open("xb") as output:
        output.write(data)
    return digest(data)


def read_artifact(path: Path, kind: str) -> tuple[dict, str]:
    data = path.read_bytes()
    value = json.loads(data)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != kind:
        raise FeasibilityError(f"expected {kind} artifact version {SCHEMA_VERSION}")
    return value, digest(data)
