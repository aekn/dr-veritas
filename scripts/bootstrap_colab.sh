#!/usr/bin/env bash
set -euxo pipefail

python -m pip install --upgrade pip
pip install poetry

poetry config virtualenvs.create false

poetry install --no-interaction
