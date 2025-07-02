from typer.testing import CliRunner

runner = CliRunner()


def test_train(cli_app):
    result = runner.invoke(cli_app, ["train"])
    assert result.exit_code == 0
    assert "Training model" in result.output
