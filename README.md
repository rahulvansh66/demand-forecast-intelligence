# demand-forecast-intelligence
ML-driven demand forecasting, pricing insights, and risk analytics with an agentic decision copilot.

## Python Environment Setup

This project uses `pyproject.toml` as the main package definition and `uv.lock` for reproducible dependency resolution. Use a local `.venv` for development; it is intentionally ignored by git.

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create the Virtual Environment

```bash
uv venv
source .venv/bin/activate
```

### Install Dependencies

```bash
# Runtime dependencies only
uv sync

# Runtime + notebook dependencies
uv sync --extra jupyternotebook

# Runtime + linting/formatting/type-checking tools
uv sync --extra lint

# Runtime + test tools
uv sync --extra test

# Full local development environment
uv sync --all-extras
```

### Jupyter Kernel

After installing notebook dependencies, register the local environment as a Jupyter kernel:

```bash
uv run python -m ipykernel install --user --name demand-forecast-intelligence --display-name "Python (demand-forecast-intelligence)"
```

### Common Commands

```bash
uv run pytest
uv run ruff check .
uv run black --check .
uv run isort --check-only .
uv run mypy src
```
