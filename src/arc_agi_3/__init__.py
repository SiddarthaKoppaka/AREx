"""Offline-first ARC-AGI-3 cognitive harness."""

__version__ = "0.1.0"


def main() -> None:
    from arc_agi_3.cli import app

    app()
