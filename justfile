set shell := ['bash', '-ceuo', 'pipefail']

@default: lint test

@lint:
    ruff check
    ruff format --check
    ruff check
    pyright

@test:
    pytest

@fix:
    ruff check --fix
    ruff format

@build-package:
    uv build

@build-docs:
    mkdocs build

@serve-docs:
    mkdocs serve

@setup-env:
    uv sync
    direnv allow .
