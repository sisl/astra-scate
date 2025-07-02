# ASTRA Toolbox

A Python library for training and evaluating language models and generative AI systems that
use textual inputs. It provides a set of tools for training, evaluating, and analyzing
language models, with a focus on applying reinforcement learning based refinement techniques
to improve evaluator model performance.

## Installation

To install the ASTRA Toolbox, you can use pip. The package is available on PyPI, so you can install it directly from there.

```bash
pip install astra
```

## Development

This section provides instructions for setting up the development environment and running tests.

To start, we _STRONGLY_ recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. This will ensure that you have the correct dependencies and versions installed.

### Setting Up the Development Environment

1. Clone the repository:

   ```bash
   git clone git@github.com:sisl/astra-toolbox.git
   cd astra
   ```

1. Sync package dependencies:

   ```bash
   uv sync --dev
   ```

   This will create a `.venv` directory in the project root with all the necessary dependencies installed.


### Running Tests

Assuming you've set up your environment using `uv`, you can run the tests using the following command:

```bash
pytest
```

or 

```bash
uv run pytest
```

To generate local coverage reports, you can use:

```bash
uv run coverage run -m pytest
uv run coverage report # Generate CLI report
uv run coverage html   # Generate HTML report
```

### Generating Documentation

To generate the documentation, you can use the following command:

```bash
uv run mkdocs serve
```

This will build the documentation and start a local server. You can then view the documentation in your web browser.