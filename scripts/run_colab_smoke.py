"""Command-line entrypoint for the Colab/public-game smoke."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from arc_agi_3.research.colab_smoke import ColabSmokeConfig, run_colab_smoke
from arc_agi_3.trace.canonical import canonical_json


def main(
    dataset_root: Annotated[Path, typer.Option()] = Path(
        "/content/arc_data/environment_files"
    ),
    game_id: Annotated[str, typer.Option()] = "ls20",
    seed: Annotated[int, typer.Option()] = 0,
    model: Annotated[str, typer.Option()] = "Qwen/Qwen3-8B",
    output_dir: Annotated[Path, typer.Option()] = Path("/content/AREx/runs"),
    run_id: Annotated[str, typer.Option()] = "ls20-qwen3-8b-smoke-001",
    max_actions: Annotated[int, typer.Option()] = 12,
    max_model_calls: Annotated[int, typer.Option()] = 4,
    max_input_tokens: Annotated[int, typer.Option()] = 100_000,
    max_output_tokens: Annotated[int, typer.Option()] = 8_000,
    max_new_tokens: Annotated[int, typer.Option()] = 1_024,
    quantization: Annotated[Literal["auto", "none", "nf4"], typer.Option()] = "auto",
) -> None:
    settings = ColabSmokeConfig(
        dataset_root=dataset_root,
        game_id=game_id,
        seed=seed,
        model=model,
        output_dir=output_dir,
        run_id=run_id,
        max_actions=max_actions,
        max_model_calls=max_model_calls,
        max_input_tokens=max_input_tokens,
        max_output_tokens=max_output_tokens,
        max_new_tokens=max_new_tokens,
        quantization=quantization,
    )
    typer.echo(canonical_json(run_colab_smoke(settings)))


if __name__ == "__main__":
    typer.run(main)
