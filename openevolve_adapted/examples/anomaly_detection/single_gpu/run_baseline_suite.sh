#!/usr/bin/env bash
# Usage: bash run_baseline_suite.sh [absolute_output_directory]
# Starts the baseline vLLM server, runs the measured seeds, then one diagnostic
# kernel-profile run against a profiler-enabled server (excluded from timing).
# Environment: HPML_BUDGET (seconds per seed, default 900), HPML_SEEDS ("42 43 44"),
# HPML_MODEL, HPML_REVISION, HPML_SKIP_PROFILE=1 to skip the diagnostic run,
# HPML_TARGET_F1 (search F1 whose first crossing is reported as time to target).
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
# Weight size as loaded by vLLM, for the hardware-limit summary.
export HPML_WEIGHTS_GIB="$(grep -oP 'Model loading took \K[0-9.]+(?= GiB)' "$SUITE/server.log" | head -n 1)"
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

HPML_TARGET_F1="${HPML_TARGET_F1:-}" python - "$SUITE" "$BUDGET" $SEEDS <<'PY'
import csv, json, os, statistics, sys
from pathlib import Path
p=Path(sys.argv[1]); budget=int(sys.argv[2]); seeds=[int(s) for s in sys.argv[3:]]; rows=[]; curves={}
for seed in seeds:
    s=json.loads((p/f'seed_{seed}/summary.json').read_text())
    m=s['best']['metrics']; r=s['run']; t=s.get('trace_summary') or {}
    assert m.get('validity') == 1, f'Invalid best program for seed {seed}'
    g=t.get('programs') or {}; lat=t.get('serving_latency') or {}; hw=t.get('hardware_limit') or {}
    curves[seed]=t.get('best_f1_curve') or []
    rows.append({'seed':seed,'search_f1':m['combined_score'],'precision':m['precision'],'recall':m['recall'],
                 'best_iteration':s['best']['iteration'],'budget_seconds':r['budget_seconds'],
                 'wall_seconds_including_cleanup':r['wall_seconds_including_cleanup'],'stop_reason':r['stop_reason'],
                 'candidates_evaluated':(t.get('loop') or {}).get('candidates_evaluated'),
                 'llm_requests':(t.get('llm') or {}).get('requests'),
                 'replies_discarded':g.get('replies_discarded'),'programs_graded':g.get('graded'),
                 'programs_valid':g.get('valid'),'programs_distinct_valid':g.get('distinct_valid_programs'),
                 'mean_time_to_first_token_seconds':(lat.get('time_to_first_token') or {}).get('mean_seconds'),
                 'mean_inter_token_latency_seconds':(lat.get('inter_token_latency') or {}).get('mean_seconds'),
                 'decode_tokens_per_second':hw.get('decode_tokens_per_s'),
                 'tokens_per_decode_step':hw.get('tokens_per_decode_step'),
                 'fraction_of_memory_bandwidth':hw.get('fraction_of_memory_bandwidth'),
                 'output_tokens_per_second_while_generating':(t.get('llm') or {}).get('output_tokens_per_second_while_generating'),
                 'mean_gpu_utilization_percent':(t.get('gpu') or {}).get('mean_utilization_percent'),
                 'peak_gpu_memory_mib':(t.get('gpu') or {}).get('peak_memory_used_mib')})
scores=[r['search_f1'] for r in rows]
# Time to target: a target fixed in advance, else the median best search F1 across the seeds.
target=float(os.environ['HPML_TARGET_F1']) if os.environ.get('HPML_TARGET_F1') else statistics.median(scores)
for r in rows:
    r['seconds_to_target_f1']=next((c['elapsed_seconds'] for c in curves[r['seed']] if c['best_f1']>=target),None)
reached=[r['seconds_to_target_f1'] for r in rows if r['seconds_to_target_f1'] is not None]
summary={'budget_seconds_per_seed':budget,'seeds':seeds,'results':rows,'mean_search_f1':statistics.mean(scores),
         'target_search_f1':target,'target_fixed_in_advance':bool(os.environ.get('HPML_TARGET_F1')),
         'seeds_reaching_target':len(reached),
         'median_seconds_to_target_counting_misses_as_budget':statistics.median(
             [r['seconds_to_target_f1'] if r['seconds_to_target_f1'] is not None else budget for r in rows]),
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
    python examples/anomaly_detection/single_gpu/analyze_kernel_trace.py "$SUITE/diagnostic_profile/gpu_kernel_trace" \
        --output "$SUITE/diagnostic_profile/kernel_summary.json" > /dev/null \
        || printf 'Kernel trace analysis failed\n'
    stop_server
fi
printf 'complete\n' > "$SUITE/status.txt"
# One file to download and pass around: summaries, traces, logs and best programs.
# Checkpoints and per-process raw trace fragments are left out to keep it small.
(cd "$(dirname "$SUITE")" && python - "$(basename "$SUITE")" <<'PY'
import sys, zipfile
from pathlib import Path
suite = Path(sys.argv[1])
skip = ('checkpoints', '__pycache__', 'visualizations')
with zipfile.ZipFile(f'{suite}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(suite.rglob('*')):
        if path.is_file() and not any(part in skip for part in path.parts) and not path.name.endswith('.jsonl'):
            archive.write(path)
print(f'Shareable archive: {Path.cwd() / (str(suite) + ".zip")}')
PY
)
printf 'Suite results: %s\n' "$SUITE"
