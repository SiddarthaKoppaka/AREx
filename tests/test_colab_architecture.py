"""Architecture banner names exactly which backends are wired in."""

from types import SimpleNamespace

import pytest

from arc_agi_3.research.colab_architecture import print_architecture


@pytest.mark.parametrize(
    "specialist,system1,architecture",
    [
        (None, None, "core-agent-only"),
        (SimpleNamespace(metadata={}), None, "core-agent-plus-exocortex-specialists"),
        (None, SimpleNamespace(metadata={}), "core-agent-plus-system1"),
    ],
)
def test_architecture_label_reflects_wired_backends(
    specialist: object,
    system1: object,
    architecture: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_architecture(specialist, system1)  # type: ignore[arg-type]
    assert f"Architecture: {architecture}" in capsys.readouterr().out


def test_system1_backend_metadata_is_printed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_system1 = SimpleNamespace(metadata={"provider": "jev"})
    print_architecture(None, fake_system1)  # type: ignore[arg-type]
    assert "System-1 backend: {'provider': 'jev'}" in capsys.readouterr().out
