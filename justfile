set shell := ['bash', '-ceuo', 'pipefail']

@default: lint test

@lint:
    ruff check
    ruff format --check
    ruff check
    pyright

@test:
    pytest

@build-package:
    uv build

@build-docs:
    mkdocs build

@serve-docs:
    mkdocs serve

@setup-env:
    uv sync
    direnv allow .
