#!/bin/bash

cd "$(dirname ${0})/.."
python -m ruff check netspresso --fix
