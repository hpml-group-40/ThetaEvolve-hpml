#!/usr/bin/env bash
# Usage: bash run_baseline_suite.sh [absolute_output_directory]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
source "$ROOT/.venv-vllm/bin/activate"
SUITE="$ROOT/openevolve_adapted/outputs/baseline_suite_$(date -u +%Y%m%dT%H%M%SZ)"
if [[ $# -gt 0 ]]; then SUITE="$(realpath -m "$1")"; fi
mkdir "$SUITE"
cp "$SCRIPT_DIR/BASELINE_PROTOCOL.md" "$SCRIPT_DIR/run_baseline_suite.sh" "$SUITE/"
printf 'running\n' > "$SUITE/status.txt"
for SEED in 42 43 44; do
    printf '%s Starting seed %s\n' "$(date -u +%FT%TZ)" "$SEED"
    if ! bash "$SCRIPT_DIR/run_baseline.sh" "$SEED" 900 "$SUITE/seed_$SEED" > "$SUITE/seed_$SEED.launcher.log" 2>&1; then
        printf 'failed seed %s\n' "$SEED" > "$SUITE/status.txt"
        tail -n 20 "$SUITE/seed_$SEED.launcher.log"
        exit 1
    fi
    printf '%s Finished seed %s\n' "$(date -u +%FT%TZ)" "$SEED"
done
python - "$SUITE" <<'PY'
import csv, json, statistics, sys
from pathlib import Path
p=Path(sys.argv[1]); rows=[]
for seed in (42,43,44):
    s=json.loads((p/f'seed_{seed}/summary.json').read_text())
    m=s['best']['metrics']; r=s['run']
    assert m.get('validity') == 1, f'Invalid best program for seed {seed}'
    rows.append({'seed':seed,'search_f1':m['combined_score'],'precision':m['precision'],'recall':m['recall'],'best_iteration':s['best']['iteration'],'budget_seconds':r['budget_seconds'],'wall_seconds_including_cleanup':r['wall_seconds_including_cleanup'],'stop_reason':r['stop_reason']})
scores=[r['search_f1'] for r in rows]
summary={'budget_seconds_per_seed':900,'seeds':[42,43,44],'results':rows,'mean_search_f1':statistics.mean(scores),'sample_std_search_f1':statistics.stdev(scores),'heldout_evaluated':False}
(p/'baseline_summary.json').write_text(json.dumps(summary,indent=2))
with (p/'baseline_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(summary,indent=2))
PY
printf 'complete\n' > "$SUITE/status.txt"
printf 'Suite results: %s\n' "$SUITE"
