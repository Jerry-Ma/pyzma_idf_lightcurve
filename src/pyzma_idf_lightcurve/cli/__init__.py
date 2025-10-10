"""Console script for pyzma_idf_lightcurve."""

from typing import Annotated

import typer
from rich.console import Console
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
    db_path: Annotated[
        str, typer.Option("--db-path", "-d", help="Path to lightcurve database"),
    ] = "lightcurves.db",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to run the visualization server on"),
    ] = 8050,
    *,
    debug: Annotated[
        bool, typer.Option("--debug", help="Run in debug mode"),
    ] = False,
    binary: Annotated[
        bool, typer.Option("--binary", help="Use binary blob storage"),
    ] = True,
) -> None:
    """Start the interactive lightcurve visualization web application."""
    from ..lightcurve.dash.app import LightcurveVisualizationApp

    console.print("[bold blue]Starting IDF Lightcurve Visualization[/bold blue]")
    console.print(f"Database: {db_path}")
    console.print(f"Port: {port}")
    console.print(f"Binary storage: {binary}")
    console.print(f"Debug mode: {debug}")
    console.print(f"URL: http://localhost:{port}")

    app_viz = LightcurveVisualizationApp(db_path, use_binary=binary)
    app_viz.run_server(debug=debug, port=port)


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
        f"• [dim]{app.info.name} viz --port 8051 --db-path my.db[/dim]",
    )

    # Print configuration info
    console.print("\n[bold]Configuration:[/bold]")
    console.print("• See idf_pipeline_config.example.yaml for template")



if __name__ == "__main__":
    app()
