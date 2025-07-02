import pytest

@pytest.fixture()
def cli_app():
    from astra.cli.main import app
    return app