#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p build/check
check_dir=$(mktemp -d "$PWD/build/check/run.XXXXXXXX")
trap 'rm -rf -- "$check_dir"' EXIT
export PYLINTHOME="$check_dir/pylint"

uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pyright
uv run pylint src/ppy_furiosa tests scripts
uv run pytest --basetemp "$check_dir/pytest" -o "cache_dir=$check_dir/pytest-cache"
uv run python scripts/doc_examples.py
uv run mkdocs build --strict
uv build --out-dir "$check_dir/dist"
uv run twine check "$check_dir"/dist/*
