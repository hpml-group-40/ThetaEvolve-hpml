#!/usr/bin/env bash
# Usage: bash run_baseline.sh <server_launch.json> [seed] [seconds] [output_directory]
# The server (see run_baseline_suite.sh) must already be up. No held-out data is used.
set -euo pipefail
SCRIPT_PATH="$(realpath "$0")"
PROJECT_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
source "$PROJECT_ROOT/.venv-vllm/bin/activate"
cd "$PROJECT_ROOT/openevolve_adapted"
MODEL="${HPML_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
LAUNCH_RECORD="$(realpath "$1")"
SEED="${2:-42}"
BUDGET="${3:-900}"
OUTPUT="$PWD/outputs/baseline_$(date -u +%Y%m%dT%H%M%SZ)_seed$SEED"
if [[ $# -ge 4 ]]; then OUTPUT="$(realpath -m "$4")"; fi
if [[ -e "$OUTPUT/evolution" ]]; then echo "Refusing to overwrite $OUTPUT/evolution" >&2; exit 2; fi
mkdir -p "$OUTPUT"
cp "$SCRIPT_PATH" "$OUTPUT/run_baseline.sh"
git rev-parse HEAD > "$OUTPUT/git_commit.txt"
git branch --show-current > "$OUTPUT/git_branch.txt"
git diff --binary HEAD > "$OUTPUT/code_changes.patch"
cp "$LAUNCH_RECORD" "$OUTPUT/server_launch.json"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$OUTPUT/gpu.csv"
python - "$OUTPUT" "$MODEL" <<'PY'
import importlib.metadata as m, json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); model=sys.argv[2]
versions={name:m.version(name) for name in ('vllm','torch','transformers','openai','numpy','scipy','PyYAML')}
(p/'software_versions.json').write_text(json.dumps(versions,indent=2))
mem=dict(line.split(':',1) for line in open('/proc/meminfo'))
(p/'host.json').write_text(json.dumps({'cpu_count':os.cpu_count(),'mem_total':mem['MemTotal'].strip()},indent=2))
launch=json.loads((p/'server_launch.json').read_text())
cmd=launch['command']
# Baseline definition: one sequence, no prefix cache, eager, no kernel profiler.
assert cmd[cmd.index('--model')+1]==model
assert cmd[cmd.index('--max-num-seqs')+1]=='1'
assert '--no-enable-prefix-caching' in cmd and '--enforce-eager' in cmd
assert not launch['profiler_enabled'] and 'VLLM_TORCH_PROFILER_DIR' not in launch['environment']
import yaml
config=yaml.safe_load(Path('examples/anomaly_detection/configs/config_anomaly_detection_vllm.yaml').read_text())
assert config['prompt']['use_system_message_sampling'] is True
PY
RC=0
python -u examples/anomaly_detection/single_gpu/run.py --mode benchmark --model "$MODEL" --seed "$SEED" --target-f1 1.0 --budget-seconds "$BUDGET" ${HPML_WEIGHTS_GIB:+--weights-gib "$HPML_WEIGHTS_GIB"} --output "$OUTPUT/evolution" > "$OUTPUT/evolution.log" 2>&1 || RC=$?
printf '%s\n' "$RC" > "$OUTPUT/exit_code.txt"
python - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
meta=json.loads((p/'evolution/run_metadata.json').read_text())
best_path=p/'evolution/best/best_program_info.json'
best=json.loads(best_path.read_text()) if best_path.exists() else None
trace_path=p/'evolution/trace_summary.json'
trace=json.loads(trace_path.read_text()) if trace_path.exists() else None
summary={'experiment':'single_gpu_7b_baseline','run':meta,'best':best,'trace_summary':trace,'heldout_evaluated':False,'best_program_saved':(p/'evolution/best/best_program.py').is_file()}
(p/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
if meta['exit_code'] not in (0,124) or best is None:
    raise SystemExit('Run failed or best program missing; inspect evolution.log')
PY
printf 'Baseline results: %s\n' "$OUTPUT"
