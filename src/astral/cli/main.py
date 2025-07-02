import typer

# Import commands & sub-commands
from .train import app as train_cmd
from .evaluate import app as evaluate_cmd

app = typer.Typer() 

# Directly add sub-commands to the main app
app.add_typer(train_cmd)
app.add_typer(evaluate_cmd)

# Entry point function needed by the package installation
def main():
    """
    Astral CLI - A command line interface for the Astral toolbox.
    """
    app()
