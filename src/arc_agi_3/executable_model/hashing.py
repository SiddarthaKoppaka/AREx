"""Content hashing for candidate source, kept separate from the trace's
own `canonical_hash` since this hashes raw source text, not a JSON value."""

import hashlib


def source_sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
