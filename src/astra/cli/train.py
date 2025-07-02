import typer

app = typer.Typer()

@app.command(help="Train a new evaluator model.")
def train():
    """
    Train a model using the Astra Toolbox.
    """
    typer.echo("Training model...")