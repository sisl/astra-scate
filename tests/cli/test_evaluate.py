from typer.testing import CliRunner

runner = CliRunner()


def test_evaluate(cli_app):
    result = runner.invoke(cli_app, ["evaluate"])
    print(result.output)  # For debugging purposes
    assert result.exit_code == 0
    assert "Evaluating model" in result.output
