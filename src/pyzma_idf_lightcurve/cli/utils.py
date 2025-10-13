import dataclasses

import typer
from rich.console import Console
from rich.panel import Panel


@dataclasses.dataclass
class TyperCLI:
    cmd_name: str
    name: str
    description: str
    app: None | typer.Typer = None
    console: None | Console = None

    def __post_init__(self):
        self.app = typer.Typer(
            name=self.cmd_name,
            help=f"{self.name} - {self.description}",
            rich_markup_mode="rich",
        )
        self.console = Console()

    def print_app_info(self) -> None:
        """General utility function to print app information."""
        from .. import __author__, __email__, __version__

        app = self.app
        console = self.console
        title_str = f"[bold]{self.name}[/bold]"
        body_str = (
            f"\nVersion: {__version__}\n"
            f"Author: {__author__}\n"
            f"Email: {__email__}\n"
        )
        console.print("")
        panel = Panel(body_str, title=title_str, style="bold green")
        console.print(panel, justify="left")

        # Dynamically get commands from the app
        console.print("\n[bold]Available Commands:[/bold]")
        for command_info in app.registered_commands:
            command_name = command_info.name or getattr(
                command_info.callback, "__name__", "unknown",
            )

            # Get command help from docstring or help parameter
            help_text = (
                command_info.help
                or (command_info.callback.__doc__ or "").strip().split("\n")[0]
            )

            # Add argument info for commands that need it
            command_display = command_name
            if command_name == "pipeline":
                command_display = "pipeline <dev_home>"

            console.print(f"• [cyan]{command_display}[/cyan] - {help_text}")
