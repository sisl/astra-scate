import typer

app = typer.Typer()


@app.command(help="Train a new evaluator model.")  # type: ignore
def train() -> None:
    """
    Train a model using the ASTRA-RL toolbox.
    """
    typer.echo("Training model...")
