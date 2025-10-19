"""Console script for pyzma_idf_lightcurve."""

from typing import Annotated

import typer

from .utils import TyperCLI

cli = TyperCLI(
    cmd_name="idflc",
    name="IDF Lightcurve Processing Package",
    description="Build and analyze lightcurves from Spitzer IDF data",
)
app = cli.app
console = cli.console


@app.command()
def pipeline(
    dev_home: Annotated[
        str,
        typer.Argument(
            help=(
                "Directory for Dagster instance "
                "(e.g., scratch_dagster/idf_lightcurve)"
            ),
        ),
    ],
    port: Annotated[
        int, typer.Option("--port", "-p", help="Port for Dagster dev server"),
    ] = 3001,
    host: Annotated[
        str, typer.Option("--host", "-h", help="Host for Dagster dev server"),
    ] = "127.0.0.1",
    config: Annotated[
        str | None,
        typer.Option("--config", "-c", help="Path to YAML configuration file"),
    ] = None,
) -> None:
    """Start the Dagster development server for the IDF lightcurve pipeline."""
    from ..pipeline.dev_server import run as dev_server_run

    console.print("[bold green]Starting IDF Pipeline Development Server[/bold green]")
    console.print(f"Dev Home: {dev_home}")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    if config:
        console.print(f"Config: {config}")
    console.print(f"URL: http://{host}:{port}")

    dev_server_run(dev_home=dev_home, host=host, port=port, config_file=config)


@app.command()
def viz(
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to run the visualization server on"),
    ] = 8050,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to (default: 0.0.0.0 for all interfaces)"),
    ] = "0.0.0.0",
    storage_path: Annotated[
        str | None,
        typer.Option("--storage-path", "-s", help="Pre-populate storage path in the UI"),
    ] = None,
    *,
    debug: Annotated[
        bool, typer.Option("--debug/--no-debug", help="Run in debug mode with hot reload (default: enabled)"),
    ] = True,
) -> None:
    """Start the interactive Dash v3 lightcurve visualization web application.
    
    This command starts a modern Dash v3 application for exploring IDF lightcurve
    data stored in xarray/zarr format. Load your zarr storage through the web UI.
    
    By default, the server binds to 0.0.0.0 (all interfaces) so it's accessible
    from other machines. Use --host 127.0.0.1 to restrict to localhost only.
    """
    from ..lightcurve.dash import run_app

    console.print("[bold blue]Starting IDF Lightcurve Visualization (Dash v3)[/bold blue]")
    console.print(f"Host: {host}")
    console.print(f"Port: {port}")
    console.print(f"Debug mode: {debug}")
    if storage_path:
        console.print(f"Initial storage path: {storage_path}")
    if host == "0.0.0.0":
        console.print(f"URL: http://localhost:{port} (or use your machine's IP)")
    else:
        console.print(f"URL: http://{host}:{port}")
    console.print("\n[dim]Load your zarr storage through the web interface[/dim]")

    run_app(debug=debug, port=port, host=host, storage_path=storage_path)


@app.command()
def info():
    """Show information about the IDF lightcurve package."""
    cli.print_app_info()

    # Examples
    console.print("\n[bold]Examples:[/bold]")
    console.print(
        f"• [dim]{app.info.name} pipeline scratch_dagster/dev "
        "--config config.yaml[/dim]",
    )
    console.print(
        f"• [dim]{app.info.name} viz --port 8051 --debug[/dim]",
    )
    console.print(
        f"• [dim]{app.info.name} viz --storage-path scratch_dagster/idf_lightcurves.zarr[/dim]",
    )

    # Print configuration info
    console.print("\n[bold]Configuration:[/bold]")
    console.print("• See idf_pipeline_config.example.yaml for template")



if __name__ == "__main__":
    app()
