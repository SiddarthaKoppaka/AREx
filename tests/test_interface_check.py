"""The static syntax/interface gate rejects malformed candidates before
any subprocess is even spawned."""

import pytest

from arc_agi_3.executable_model.interface_check import (
    InterfaceCheckError,
    check_interface,
)

VALID = (
    "def encode_observation(o):\n    return o\n"
    "def transition(state, action):\n    return state\n"
    "def terminal_or_goal(state):\n    return False\n"
)


def test_a_well_formed_candidate_passes() -> None:
    check_interface(VALID)


def test_a_syntax_error_is_rejected() -> None:
    with pytest.raises(InterfaceCheckError, match="syntax error"):
        check_interface("def transition(state action):\n    return state\n")


def test_a_missing_required_function_is_rejected() -> None:
    source = "def transition(state, action):\n    return state\n"
    with pytest.raises(InterfaceCheckError, match="missing required function"):
        check_interface(source)


def test_a_disallowed_top_level_statement_is_rejected() -> None:
    source = VALID + "\nprint('side effect')\n"
    with pytest.raises(InterfaceCheckError, match="disallowed top-level statement"):
        check_interface(source)


def test_a_disallowed_import_is_rejected() -> None:
    source = "import os\n" + VALID
    with pytest.raises(InterfaceCheckError, match="disallowed import"):
        check_interface(source)


def test_an_allowed_import_passes() -> None:
    check_interface("import math\n" + VALID)


def test_a_class_definition_is_rejected() -> None:
    source = VALID + "\nclass Helper:\n    pass\n"
    with pytest.raises(InterfaceCheckError, match="disallowed top-level statement"):
        check_interface(source)


def test_wrong_arity_is_rejected() -> None:
    source = (
        "def encode_observation(o):\n    return o\n"
        "def transition(state):\n    return state\n"
        "def terminal_or_goal(state):\n    return False\n"
    )
    with pytest.raises(InterfaceCheckError, match="transition"):
        check_interface(source)


def test_a_decorated_function_is_rejected() -> None:
    source = "@staticmethod\n" + VALID
    with pytest.raises(InterfaceCheckError):
        check_interface(source)


def test_global_statement_is_rejected() -> None:
    source = VALID.replace(
        "def transition(state, action):\n    return state\n",
        "x = 1\ndef transition(state, action):\n    global x\n    return state\n",
    )
    with pytest.raises(InterfaceCheckError, match="global"):
        check_interface(source)


def test_a_blocked_call_name_is_rejected() -> None:
    source = VALID.replace("return state", "return eval('state')")
    with pytest.raises(InterfaceCheckError, match="eval"):
        check_interface(source)
