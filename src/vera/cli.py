from __future__ import annotations

import typer
from rich import print


app = typer.Typer(
    help="Vera is a conversational research assistant grounded in evidence."
)


@app.command()
def hello():
    print("[bold green]Hello, it's Vera.[/bold green]")


if __name__ == "__main__":
    app()
