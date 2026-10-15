"""The minimal builtins/import surface a sandboxed candidate may use.

Defense-in-depth only: this blocks the obvious, non-adversarial cases
(explicit `import os`, `open(...)`, `eval(...)`). It is not a claim that a
deliberately adversarial program cannot reach dangerous names through
object introspection - the process boundary and resource limits in
`sandbox_worker.py` are the real backstop regardless of what this misses.
"""

import builtins as _builtins
import math as _math

ALLOWED_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "len", "list", "map", "max", "min",
    "range", "round", "set", "sorted", "str", "sum", "tuple", "zip",
    "True", "False", "None", "ArithmeticError", "ValueError", "TypeError",
    "IndexError", "KeyError", "ZeroDivisionError", "StopIteration",
    "Exception",
)  # fmt: skip

ALLOWED_MODULES: dict[str, object] = {"math": _math}


def restricted_import(name: str, *_args: object, **_kwargs: object) -> object:
    if name not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed in a sandboxed candidate")
    return ALLOWED_MODULES[name]


def safe_builtins() -> dict[str, object]:
    allowed = {
        name: getattr(_builtins, name)
        for name in ALLOWED_BUILTIN_NAMES
        if hasattr(_builtins, name)
    }
    allowed["__import__"] = restricted_import
    return allowed
