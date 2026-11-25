#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    printf '%s\n' 'Usage: tests/numi_mac_workbench_test.sh "/path/to/Numi Lab.app"' >&2
    exit 2
fi
app=$(CDPATH= cd "$1" && pwd -P)
repo=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
rollout="$app/Contents/MacOS/metalrobo_task_rollout"
compiler="$app/Contents/Helpers/metalrobo_numi_window_catalog_compile"
library="$app/Contents/Resources/MetalRobo.metallib"
work=$(mktemp -d "${TMPDIR:-/tmp}/numi-workbench.XXXXXX")
trap 'rm -rf "$work"' EXIT
trap 'exit 1' HUP INT TERM

source="$repo/assets/window/bench-pendulum.urdf"
"$compiler" --urdf "$source" --output "$work/Experiment.numiproject" --metallib "$library"
mv "$work/Experiment.numiproject" "$work/Moved Experiment.numiproject"
moved="$work/Moved Experiment.numiproject"
"$compiler" --urdf "$moved/source/model.urdf" --gravity 0 --effort 0 \
    --output "$work/Control.numiproject" --metallib "$library"
for trial in original repeat control; do
    project="$moved"
    if [ "$trial" = control ]; then project="$work/Control.numiproject"; fi
    "$rollout" --scene-file "$project/robot-project.numi-window.json" --headless \
        --steps 100 --state-trace "$work/$trial.tsv" --contact-trace "$work/$trial.contacts" \
        > "$work/$trial.json"
done
cmp "$work/original.tsv" "$work/repeat.tsv"
python3 - "$work" <<'PY'
import json
from pathlib import Path
import sys

folder = Path(sys.argv[1])
for name in ("original", "repeat", "control"):
    stats = json.loads((folder / f"{name}.json").read_text())
    assert stats["failed_environment_steps"] == 0, stats
    rows = (folder / f"{name}.tsv").read_text().splitlines()[1:]
    assert len(rows) == 100
    peak = max(abs(float(row.split()[1])) for row in rows)
    assert peak < 1e-7 if name == "control" else peak > 0.2, (name, peak)
PY
printf '%s\n' 'Numi Mac workbench passed import, relocation, edit, movement, and exact replay.'
