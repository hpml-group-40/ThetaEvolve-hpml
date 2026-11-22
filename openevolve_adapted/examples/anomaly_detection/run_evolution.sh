#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage: bash run_evolution.sh <openrouter|vllm> <target_pooled_f1: 0-1> <max_runtime_seconds> [output_dir]

Examples:
	OPENAI_API_KEY=... bash run_evolution.sh openrouter 0.85 3600
  bash run_evolution.sh vllm 0.85 3600
EOF
}

if [[ $# -lt 3 || $# -gt 4 ]]; then
	usage >&2
	exit 2
fi

PROVIDER="$1"
TARGET_F1="$2"
if [[ ! "$TARGET_F1" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
	echo "Target pooled F1 must be between 0 and 1." >&2
	exit 2
fi

MAX_RUNTIME_SECONDS="$3"
if [[ ! "$MAX_RUNTIME_SECONDS" =~ ^[0-9]+$ ]]; then
	echo "Maximum runtime must be a positive number of seconds." >&2
	exit 2
fi
MAX_RUNTIME_SECONDS=$((10#$MAX_RUNTIME_SECONDS))
if (( MAX_RUNTIME_SECONDS < 1 )); then
	echo "Maximum runtime must be at least one second." >&2
	exit 2
fi

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OPENEVOLVE_DIR="$(cd -- "$EXAMPLE_DIR/../.." && pwd)"

case "$PROVIDER" in
	openrouter)
		CONFIG_PATH="$EXAMPLE_DIR/configs/config_anomaly_detection.yaml"
		if [[ -z "${OPENAI_API_KEY:-}" ]]; then
			echo "Set OPENAI_API_KEY before selecting the OpenRouter profile." >&2
			exit 2
		fi
		;;
	vllm)
		CONFIG_PATH="$EXAMPLE_DIR/configs/config_anomaly_detection_vllm.yaml"
		if grep -q "REPLACE_WITH_VLLM_SERVED_MODEL" "$CONFIG_PATH"; then
			echo "Replace the vLLM model placeholder in $CONFIG_PATH first." >&2
			exit 2
		fi
		;;
	*)
		echo "Provider must be 'openrouter' or 'vllm'." >&2
		usage >&2
		exit 2
		;;
esac

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
	echo "Python executable not found: $PYTHON" >&2
	exit 2
fi

OUTPUT_DIR="${4:-outputs/anomaly_detection_${PROVIDER}_$(date '+%Y%m%d_%H%M%S')}"
if [[ "$OUTPUT_DIR" != /* ]]; then
	OUTPUT_DIR="$OPENEVOLVE_DIR/$OUTPUT_DIR"
fi

cd "$OPENEVOLVE_DIR"

"$PYTHON" "$EXAMPLE_DIR/run_with_timeout.py" "$MAX_RUNTIME_SECONDS" -- \
	"$PYTHON" "$OPENEVOLVE_DIR/openevolve-run.py" \
	"$EXAMPLE_DIR/initial_programs/initial_program.py" \
	"$EXAMPLE_DIR/evaluators/evaluator_modular.py" \
	--config "$CONFIG_PATH" \
	--iterations 2147483647 \
	--target-score "$TARGET_F1" \
	--output "$OUTPUT_DIR"