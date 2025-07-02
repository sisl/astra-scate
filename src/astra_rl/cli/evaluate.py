import typer

app = typer.Typer()


@app.command(help="Evaluate a model using an evaluator model.")  # type: ignore
def evaluate() -> None:
    """
    Evaluate a model using the ASTRA-RL toolbox.
    """

    typer.echo("Evaluating model...")
