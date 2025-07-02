import pytest

@pytest.fixture()
def cli_app():
    from astra_rl.cli.main import app
    return app