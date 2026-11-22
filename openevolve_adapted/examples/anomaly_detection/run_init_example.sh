#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENEVOLVE_DIR="$(cd -- "$EXAMPLE_DIR/../.." && pwd)"

export OPENEVOLVE_CONFIG_PATH="$EXAMPLE_DIR/configs/config_anomaly_detection.yaml"
export PYTHONPATH="$OPENEVOLVE_DIR${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

"$PYTHON" "$EXAMPLE_DIR/evaluators/evaluator_modular.py" \
  "$EXAMPLE_DIR/initial_programs/initial_program.py"