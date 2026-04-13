"""openvocab-tsdf CLI entry point.

Subcommands:
  info        — print env, GPU, and config summary
  fuse        — ingest a dataset, build a TSDF, save mesh
  encode      — run the VLM encoder over a dataset, cache features
  ground      — text query over a built map
  bench       — run named benchmarks and emit JSON
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.command()
def info(config: Path | None = typer.Option(None, "--config", "-c")) -> None:
    """Print environment and hardware summary."""
    import platform

    import torch

    table = Table(title="openvocab-tsdf environment")
    table.add_column("Item")
    table.add_column("Value")

    table.add_row("Python", platform.python_version())
    table.add_row("PyTorch", torch.__version__)
    table.add_row("CUDA available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA runtime", torch.version.cuda or "?")
        table.add_row("Device", torch.cuda.get_device_name(0))
        cap = torch.cuda.get_device_capability(0)
        table.add_row("Compute capability", f"sm_{cap[0]}{cap[1]}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        table.add_row("VRAM (total)", f"{total_gb:.1f} GB")

    console.print(table)

    if config is not None:
        from openvocab_tsdf.config import load_config

        cfg = load_config(config)
        console.print(cfg)


@app.command()
def fuse(
    config: Path = typer.Option(..., "--config", "-c"),
    output: Path = typer.Option(Path("outputs/mesh.ply"), "--output", "-o"),
) -> None:
    """Ingest a dataset and save a TSDF mesh. [Phase 1]"""
    raise typer.Exit(code=2)  # implemented in Phase 1


@app.command()
def encode(config: Path = typer.Option(..., "--config", "-c")) -> None:
    """Run the VLM encoder over a dataset, cache features. [Phase 2]"""
    raise typer.Exit(code=2)


@app.command()
def ground(
    config: Path = typer.Option(..., "--config", "-c"),
    query: str = typer.Option(..., "--query", "-q"),
    top_k: int = typer.Option(5, "--top-k"),
) -> None:
    """Text-to-3D grounding query. [Phase 3]"""
    raise typer.Exit(code=2)


@app.command()
def bench(
    name: str = typer.Argument(..., help="benchmark name, e.g. 'tsdf_fuse'"),
    config: Path = typer.Option(..., "--config", "-c"),
) -> None:
    """Run a named benchmark and write a JSON result. [Phase 4]"""
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
