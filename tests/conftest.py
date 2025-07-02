import pytest

@pytest.fixture()
def cli_app():
    from astral.cli.main import app
    return app