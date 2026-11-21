#!/usr/bin/env bash
# Usage: bash run_baseline.sh [seed] [seconds] [output_directory]
# Start the pinned 7B server first. This script does not use held-out data.
set -euo pipefail
SCRIPT_PATH="$(realpath "$0")"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$PROJECT_ROOT/.venv-vllm/bin/activate"
cd "$PROJECT_ROOT/openevolve_adapted"
SEED=42
BUDGET=900
if [[ $# -ge 1 ]]; then SEED="$1"; fi
if [[ $# -ge 2 ]]; then BUDGET="$2"; fi
OUTPUT="$PWD/outputs/baseline_$(date -u +%Y%m%dT%H%M%SZ)_seed$SEED"
if [[ $# -ge 3 ]]; then OUTPUT="$(realpath -m "$3")"; fi
if [[ -e "$OUTPUT/evolution" ]]; then echo "Refusing to overwrite $OUTPUT/evolution" >&2; exit 2; fi
mkdir -p "$OUTPUT"
cp "$SCRIPT_PATH" "$OUTPUT/run_baseline.sh"
git rev-parse HEAD > "$OUTPUT/git_commit.txt"
git branch --show-current > "$OUTPUT/git_branch.txt"
git diff --binary > "$OUTPUT/code_changes.patch"
cp "$PROJECT_ROOT/vm-setup/server-launch.json" "$OUTPUT/server_launch.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUTPUT/gpu.csv"
python - "$OUTPUT" <<'PY'
import importlib.metadata as m, json, sys
from pathlib import Path
p=Path(sys.argv[1])
versions={name:m.version(name) for name in ('vllm','torch','transformers','openai','numpy','scipy','PyYAML')}
(p/'software_versions.json').write_text(json.dumps(versions,indent=2))
launch=json.loads((p/'server_launch.json').read_text())
cmd=launch['command']
assert cmd[cmd.index('--model')+1]=='Qwen/Qwen2.5-Coder-7B-Instruct'
assert cmd[cmd.index('--max-num-seqs')+1]=='1'
assert '--no-enable-prefix-caching' in cmd and '--enforce-eager' in cmd
assert 'VLLM_TORCH_PROFILER_DIR' not in launch['environment_overrides']
import yaml
config=yaml.safe_load(Path('examples/anomaly_detection/configs/config_anomaly_detection_vllm.yaml').read_text())
assert config['prompt']['use_system_message_sampling'] is True
PY
RC=0
python -u examples/anomaly_detection/single_gpu/run.py --mode benchmark --model Qwen/Qwen2.5-Coder-7B-Instruct --seed "$SEED" --target-f1 1.0 --budget-seconds "$BUDGET" --output "$OUTPUT/evolution" > "$OUTPUT/evolution.log" 2>&1 || RC=$?
printf '%s\n' "$RC" > "$OUTPUT/exit_code.txt"
python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
meta=json.loads((p/'evolution/run_metadata.json').read_text())
best_path=p/'evolution/best/best_program_info.json'
best=json.loads(best_path.read_text()) if best_path.exists() else None
summary={'experiment':'single_gpu_7b_baseline','run':meta,'best':best,'heldout_evaluated':False,'best_program_saved':(p/'evolution/best/best_program.py').is_file()}
(p/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if meta['exit_code'] not in (0,124) or best is None:
    raise SystemExit('Run failed or best program missing; inspect evolution.log')
PY
printf 'Baseline results: %s\n' "$OUTPUT"
