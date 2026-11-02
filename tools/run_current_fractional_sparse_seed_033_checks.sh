#!/usr/bin/env bash
set -euo pipefail
repo="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
out="${1:?usage: run_current_fractional_sparse_seed_033_checks.sh OUTPUT_DIR}"
base_commit=09caf05528f9d81d2cb0a5faba751ca0915cf7aa
cxx="${CXX:-/usr/bin/clang++}"
mkdir -p "$out/bin" "$out/compile" "$out/run" "$out/baseline"
git -C "$repo" show "$base_commit:apps/ConservativeSpatialFractionalSemanticSupport.hpp" > "$out/baseline/ConservativeSpatialFractionalSemanticSupport.hpp"
common=(-std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -UNDEBUG)
compile_run() {
    local name="$1" source="$2" include_mode="$3" define_mode="${4:-}"
    local -a includes=()
    if [[ "$include_mode" == baseline ]]; then
        includes=(-I "$out/baseline" -I "$repo/apps")
    else
        includes=(-I "$repo/apps")
    fi
    local -a args=("${common[@]}")
    if [[ -n "$define_mode" ]]; then args+=(-D "$define_mode"); fi
    "$cxx" "${args[@]}" "${includes[@]}" -MMD -MF "$out/compile/$name.d" "$repo/tests/$source" -o "$out/bin/$name" >"$out/compile/$name.stdout.log" 2>"$out/compile/$name.stderr.log"
    "$out/bin/$name" >"$out/run/$name.stdout.log" 2>"$out/run/$name.stderr.log"
}
compile_run policy-033 CurrentFractionalSparseSeedPolicyCheck_033.cpp candidate
compile_run reduction-033 CurrentFractionalSparseSeedReductionCheck_033.cpp candidate
compile_run implicit-baseline-033b CurrentFractionalSparseSeedImplicitNonP1Check_033b.cpp baseline SPARSE_SEED_BASELINE_CONTROL
compile_run implicit-bounded-033b CurrentFractionalSparseSeedImplicitNonP1Check_033b.cpp candidate
python3 - "$out" <<'PY'
from pathlib import Path
import re,sys
root=Path(sys.argv[1]); run=root/'run'
base=(run/'implicit-baseline-033b.stderr.log').read_text()
bounded=(run/'implicit-bounded-033b.stderr.log').read_text()
base_out=(run/'implicit-baseline-033b.stdout.log').read_text()
bounded_out=(run/'implicit-bounded-033b.stdout.log').read_text()
def stats(text):
    rows=re.findall(r'shared_capacity_perf stage=complete seed_edges=(\d+) cut_rounds=(\d+) cut_edges_added=(\d+) augment_calls=(\d+) flow_augmentations=(\d+)',text)
    if len(rows)!=2 or rows[0]!=rows[1]: raise SystemExit(f'expected identical deterministic repeat counters, got {rows}')
    return tuple(map(int,rows[0]))
b=stats(base); q=stats(bounded)
if (b[0],b[2])!=(12,0): raise SystemExit(f'legacy baseline counters unexpected: {b}')
if (q[0],q[2])!=(10,1): raise SystemExit(f'bounded seed/cut counters unexpected: {q}')
if 'materialized_candidate_pairs=1' not in bounded or 'materialized_candidate_pairs=0' not in base:
    raise SystemExit('implicit candidate materialization counter mismatch')
if 'PASS dense baseline supports required non-P1 B-to-far flow' not in base_out:
    raise SystemExit('baseline fixture output missing')
if 'PASS bounded seed requires/materializes non-P1 implicit B-to-far flow' not in bounded_out:
    raise SystemExit('bounded fixture output missing')
print('PASS_SOURCE_ONLY_SPARSE_SEED_033_AND_REQUIRED_IMPLICIT_ROUTE')
print(f'baseline seed_edges={b[0]} cut_edges_added={b[2]}; bounded seed_edges={q[0]} cut_edges_added={q[2]}')
PY
