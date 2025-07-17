#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
python main.py
