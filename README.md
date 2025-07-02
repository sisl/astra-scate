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

2. Create a virtual environment:

   ```bash
   uv venv -p 3.12 # Minimum Python version is 3.12
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

4. Install the package with development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```
