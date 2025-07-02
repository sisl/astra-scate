import typer

app = typer.Typer()

@app.command(help="Evaluate a model using an evaluator model.")
def evaluate():
    """
    Evaluate a model using the Astral toolbox.
    """
    typer.echo("Evaluating model...")