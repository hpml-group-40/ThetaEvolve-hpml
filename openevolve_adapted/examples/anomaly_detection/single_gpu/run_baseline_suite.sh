#!/usr/bin/env bash
# Usage: bash run_baseline_suite.sh [absolute_output_directory]
# Starts the baseline vLLM server, runs the measured seeds, then one diagnostic
# kernel-profile run against a profiler-enabled server (excluded from timing).
# Environment: HPML_BUDGET (seconds per seed, default 900), HPML_SEEDS ("42 43 44"),
# HPML_MODEL, HPML_REVISION, HPML_SKIP_PROFILE=1 to skip the diagnostic run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT/.venv-vllm/bin/activate"
BUDGET="${HPML_BUDGET:-900}"
SEEDS="${HPML_SEEDS:-42 43 44}"
export HPML_MODEL="${HPML_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
REVISION="${HPML_REVISION:-c03e6d358207e414f1eca0bb1891e29f1db0e242}"
PORT=8000
SUITE="$ROOT/openevolve_adapted/outputs/baseline_suite_$(date -u +%Y%m%dT%H%M%SZ)"
if [[ $# -gt 0 ]]; then SUITE="$(realpath -m "$1")"; fi
mkdir "$SUITE"
cp "$SCRIPT_DIR/BASELINE_PROTOCOL.md" "$SCRIPT_DIR/run_baseline_suite.sh" "$SUITE/"
printf 'running\n' > "$SUITE/status.txt"
SERVER_PID=""

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap stop_server EXIT

# start_server <name> [extra serve.py arguments]; records startup seconds.
start_server() {
    local name="$1"; shift
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
        echo "Port $PORT already has a server; stop it first." >&2; exit 2
    fi
    local began; began="$(date +%s.%N)"
    setsid python "$SCRIPT_DIR/serve.py" --model "$HPML_MODEL" --revision "$REVISION" --gpu 0 --port "$PORT" \
        --launch-record "$SUITE/$name.launch.json" "$@" > "$SUITE/$name.log" 2>&1 &
    SERVER_PID=$!
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "Server exited during startup; see $SUITE/$name.log" >&2; tail -n 30 "$SUITE/$name.log" >&2; exit 1
        fi
        sleep 1
    done
    python - "$began" "$SUITE/$name.startup.json" <<'PY'
import json, sys, time
from pathlib import Path
json.dump({'startup_seconds': time.time() - float(sys.argv[1]),
           'note': 'Includes model download when the Hugging Face cache was cold.'},
          open(sys.argv[2], 'w'), indent=2)
PY
    printf '%s Server %s ready\n' "$(date -u +%FT%TZ)" "$name"
}

start_server server
for SEED in $SEEDS; do
    printf '%s Starting seed %s\n' "$(date -u +%FT%TZ)" "$SEED"
    if ! bash "$SCRIPT_DIR/run_baseline.sh" "$SUITE/server.launch.json" "$SEED" "$BUDGET" "$SUITE/seed_$SEED" > "$SUITE/seed_$SEED.launcher.log" 2>&1; then
        printf 'failed seed %s\n' "$SEED" > "$SUITE/status.txt"
        tail -n 20 "$SUITE/seed_$SEED.launcher.log"
        exit 1
    fi
    printf '%s Finished seed %s\n' "$(date -u +%FT%TZ)" "$SEED"
done
curl -sf "http://127.0.0.1:$PORT/metrics" > "$SUITE/server_metrics_final.txt" || true
stop_server

python - "$SUITE" "$BUDGET" $SEEDS <<'PY'
import csv, json, statistics, sys
from pathlib import Path
p=Path(sys.argv[1]); budget=int(sys.argv[2]); seeds=[int(s) for s in sys.argv[3:]]; rows=[]
for seed in seeds:
    s=json.loads((p/f'seed_{seed}/summary.json').read_text())
    m=s['best']['metrics']; r=s['run']; t=s.get('trace_summary') or {}
    assert m.get('validity') == 1, f'Invalid best program for seed {seed}'
    rows.append({'seed':seed,'search_f1':m['combined_score'],'precision':m['precision'],'recall':m['recall'],
                 'best_iteration':s['best']['iteration'],'budget_seconds':r['budget_seconds'],
                 'wall_seconds_including_cleanup':r['wall_seconds_including_cleanup'],'stop_reason':r['stop_reason'],
                 'candidates_evaluated':(t.get('loop') or {}).get('candidates_evaluated'),
                 'llm_requests':(t.get('llm') or {}).get('requests'),
                 'output_tokens_per_second_while_generating':(t.get('llm') or {}).get('output_tokens_per_second_while_generating'),
                 'mean_gpu_utilization_percent':(t.get('gpu') or {}).get('mean_utilization_percent'),
                 'peak_gpu_memory_mib':(t.get('gpu') or {}).get('peak_memory_used_mib')})
scores=[r['search_f1'] for r in rows]
summary={'budget_seconds_per_seed':budget,'seeds':seeds,'results':rows,'mean_search_f1':statistics.mean(scores),
         'sample_std_search_f1':statistics.stdev(scores) if len(scores)>1 else None,
         'server_startup':json.loads((p/'server.startup.json').read_text()),'heldout_evaluated':False}
(p/'baseline_summary.json').write_text(json.dumps(summary,indent=2))
with (p/'baseline_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(summary,indent=2))
PY

if [[ "${HPML_SKIP_PROFILE:-0}" != "1" ]]; then
    # Diagnostic only: CUDA kernel trace of the start of one generation call.
    printf '%s Starting diagnostic kernel profile\n' "$(date -u +%FT%TZ)"
    start_server profile_server --profile-dir "$SUITE/diagnostic_profile/gpu_kernel_trace"
    cd "$ROOT/openevolve_adapted"
    python -u examples/anomaly_detection/single_gpu/run.py --mode profile --model "$HPML_MODEL" \
        --seed "${SEEDS%% *}" --iterations 3 --budget-seconds 600 --target-f1 1.0 \
        --output "$SUITE/diagnostic_profile/evolution" > "$SUITE/diagnostic_profile/run.log" 2>&1 \
        || printf 'Diagnostic profile failed; see %s\n' "$SUITE/diagnostic_profile/run.log"
    stop_server
fi
printf 'complete\n' > "$SUITE/status.txt"
printf 'Suite results: %s\n' "$SUITE"
