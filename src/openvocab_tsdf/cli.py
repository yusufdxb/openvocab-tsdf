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
    """Ingest a dataset and save a TSDF mesh."""
    import logging

    from openvocab_tsdf.config import load_config
    from openvocab_tsdf.pipeline import fuse_dataset

    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(config)
    mesh = fuse_dataset(cfg, output)
    console.print(
        f"[green]fused[/green] {len(mesh.vertices)} verts / {len(mesh.triangles)} tris -> {output}"
    )


@app.command()
def encode(
    config: Path = typer.Option(..., "--config", "-c"),
    output: Path = typer.Option(Path("outputs/map.npz"), "--output", "-o"),
) -> None:
    """Run the VLM encoder over a dataset, fuse features into voxels, save map."""
    import logging

    from openvocab_tsdf.config import load_config
    from openvocab_tsdf.pipeline import build_dataset, encode_and_fuse

    logging.basicConfig(level="INFO", format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config(config)
    dataset = build_dataset(cfg)
    stats = encode_and_fuse(cfg, dataset, output)
    console.print(
        f"[green]encoded[/green] {stats['num_frames']} frames "
        f"(enc {stats['encode_s']:.2f}s, fuse {stats['fuse_s']:.2f}s) -> {output}"
    )


@app.command()
def ground(
    map_path: Path = typer.Option(..., "--map", "-m"),
    query: str = typer.Option(..., "--query", "-q"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    top_k: int = typer.Option(5, "--top-k"),
    score_threshold: float = typer.Option(0.22, "--score-threshold"),
    min_cluster_voxels: int = typer.Option(8, "--min-cluster"),
) -> None:
    """Text-to-3D grounding query against a saved map."""
    from openvocab_tsdf.config import load_config
    from openvocab_tsdf.pipeline import ground_text

    if config is not None:
        cfg = load_config(config)
        kwargs = {
            "model": cfg.semantics.model,
            "pretrained": cfg.semantics.pretrained,
            "device": cfg.semantics.device,
            "dtype": cfg.semantics.dtype,
        }
    else:
        kwargs = {}

    results = ground_text(
        map_path,
        query,
        **kwargs,
        score_threshold=score_threshold,
        min_cluster_voxels=min_cluster_voxels,
        top_k=top_k,
    )

    table = Table(title=f"grounding: {query!r}")
    table.add_column("#")
    table.add_column("center")
    table.add_column("score", justify="right")
    table.add_column("voxels", justify="right")
    for i, r in enumerate(results):
        cx, cy, cz = r.center_m
        table.add_row(
            str(i + 1), f"({cx:+.2f}, {cy:+.2f}, {cz:+.2f})", f"{r.score:.3f}", str(r.voxel_count)
        )
    console.print(table)
    if not results:
        console.print("[yellow]no clusters above threshold[/yellow]")


@app.command()
def bench(
    name: str = typer.Argument(..., help="benchmark name, e.g. 'tsdf_fuse'"),
    config: Path = typer.Option(..., "--config", "-c"),
) -> None:
    """Run a named benchmark and write a JSON result. [Phase 4]"""
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
