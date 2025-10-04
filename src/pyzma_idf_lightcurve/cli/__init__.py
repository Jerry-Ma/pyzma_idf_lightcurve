"""Console script for pyzma_idf_lightcurve."""

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for pyzma_idf_lightcurve."""
    console.print("This is pyzma_idf_lightcurve CLI.")


if __name__ == "__main__":
    app()
