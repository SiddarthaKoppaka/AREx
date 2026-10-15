"""Static syntax/interface check for candidate transition-model source.

Never executes the source - pure AST inspection, run before any sandboxed
call is attempted. This is a cheap first filter, not the security
boundary: the sandbox's resource limits and restricted builtins
(`sandbox_worker.py`) are the authoritative backstop regardless of what
this check misses.
"""

import ast

from .contracts import REQUIRED_FUNCTIONS

ALLOWED_IMPORTS = frozenset({"math"})
BLOCKED_CALL_NAMES = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "input", "vars", "globals"}
)


class InterfaceCheckError(ValueError):
    """Candidate source fails the syntax/interface gate."""


def check_interface(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        raise InterfaceCheckError(f"syntax error: {error}") from error
    defined = _check_top_level(tree)
    _check_nested(tree)
    missing = [name for name in REQUIRED_FUNCTIONS if name not in defined]
    if missing:
        raise InterfaceCheckError(f"missing required function(s): {missing}")
    _check_signature(defined["encode_observation"], 1)
    _check_signature(defined["transition"], 2)
    _check_signature(defined["terminal_or_goal"], 1)
    return tree


def _check_top_level(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    defined: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            defined[node.name] = node
        elif isinstance(node, ast.Import | ast.ImportFrom):
            _check_import(node)
        elif isinstance(node, ast.Assign):
            pass
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            pass  # a module docstring
        else:
            raise InterfaceCheckError(
                f"disallowed top-level statement: {type(node).__name__}"
            )
    return defined


def _check_nested(tree: ast.Module) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Global | ast.Nonlocal):
            raise InterfaceCheckError("global/nonlocal statements are not allowed")
        if isinstance(node, ast.Import | ast.ImportFrom):
            _check_import(node)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in BLOCKED_CALL_NAMES
        ):
            raise InterfaceCheckError(f"call to {node.func.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise InterfaceCheckError("dunder attribute access is not allowed")


def _check_import(node: ast.Import | ast.ImportFrom) -> None:
    modules = (
        [alias.name for alias in node.names]
        if isinstance(node, ast.Import)
        else [node.module or ""]
    )
    disallowed = [name for name in modules if name not in ALLOWED_IMPORTS]
    if disallowed:
        raise InterfaceCheckError(f"disallowed import(s): {disallowed}")


def _check_signature(fn: ast.FunctionDef, arity: int) -> None:
    args = fn.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        raise InterfaceCheckError(f"{fn.name} must take exactly {arity} argument(s)")
    if len(args.args) != arity:
        raise InterfaceCheckError(f"{fn.name} must take exactly {arity} argument(s)")
    if fn.decorator_list:
        raise InterfaceCheckError(f"{fn.name} may not use decorators")
