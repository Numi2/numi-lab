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
"$compiler" --urdf "$moved/source/model.urdf" --gravity 0 \
    --output "$work/Zero Gravity.numiproject" --metallib "$library"
for trial in original repeat control zero-gravity; do
    project="$moved"
    if [ "$trial" = control ]; then project="$work/Control.numiproject"; fi
    if [ "$trial" = zero-gravity ]; then project="$work/Zero Gravity.numiproject"; fi
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
project = folder / "Moved Experiment.numiproject"
for visual in (project / "visual").glob("*.mrvpack"):
    assert str(folder).encode() not in visual.read_bytes(), visual
positions = {}
for name in ("original", "repeat", "control", "zero-gravity"):
    stats = json.loads((folder / f"{name}.json").read_text())
    assert stats["failed_environment_steps"] == 0, stats
    rows = (folder / f"{name}.tsv").read_text().splitlines()[1:]
    assert len(rows) == 100
    positions[name] = [float(row.split()[1]) for row in rows]
    peak = max(map(abs, positions[name]))
    assert peak < 1e-7 if name == "control" else peak > 0.2, (name, peak)
assert max(abs(a - b) for a, b in zip(positions["original"], positions["zero-gravity"])) > 0.15
PY

for fixture in channels contact; do
    "$compiler" --mjcf "$repo/tests/fixtures/workbench_mjcf_$fixture.xml" --effort 0 \
        --output "$work/$fixture.numiproject" --metallib "$library"
done
sed '/<geom name="pedestal_sphere"/i\
      <joint name="pedestal_hinge"/>\
' "$repo/tests/fixtures/workbench_mjcf_contact.xml" > "$work/moving-parent.xml"
"$compiler" --mjcf "$work/moving-parent.xml" --effort 0 \
    --output "$work/moving-parent.numiproject" --metallib "$library"
PYTHONPATH="$repo/python/metalrobo" python3 - "$work" "$rollout" "$library" <<'PY'
import csv
import hashlib
from pathlib import Path
import struct
import subprocess
import sys

# The experiment wrapper is standard-library-only. Load it independently of
# the optional installed scientific stack when checking the packaged app.
from experiment import Experiment

work, rollout, library = map(Path, sys.argv[1:])

def open_experiment(fixture):
    return Experiment(work / f"{fixture}.numiproject", rollout=rollout, metallib=library)

def final_q(result, count):
    assert result.stats["failed_environment_steps"] == 0, result.stats
    return list(map(float, result.state_trace.read_text().splitlines()[-1].split()[1:count + 1]))

motors = open_experiment("channels")
first = motors.run(steps=4, actions=[(1, 0)], state_trace=work / "first.tsv")
second = motors.run(steps=4, actions=[(0, 1)], state_trace=work / "second.tsv")
alpha, beta = final_q(first, 2)
assert abs(alpha) < 1e-7 and abs(beta - 0.328) < 1e-5, (alpha, beta)
alpha, beta = final_q(second, 2)
assert abs(alpha - 0.328) < 1e-5 and abs(beta) < 1e-7, (alpha, beta)
assert first.action_clip == struct.pack("<2f", 1, 0)
assert first.stats["action_clip_sha256"] == hashlib.sha256(first.action_clip).hexdigest()
assert first.stats["action_clip_sha256"] != second.stats["action_clip_sha256"]
assert "action_clip" not in first.stats
saved_clip = work / "saved.f32"
saved_clip.write_bytes(first.action_clip)
replay = motors.run(steps=4, action_clip=saved_clip, state_trace=work / "replayed.tsv")
assert first.state_trace.read_bytes() == replay.state_trace.read_bytes()
assert first.stats["action_clip_sha256"] == replay.stats["action_clip_sha256"]

contact = open_experiment("contact").run(steps=20,
    state_trace=work / "contact.tsv", contact_trace=work / "force.tsv")
travel, = final_q(contact, 1)
assert abs(travel + 0.05) < 1e-4, travel
with contact.contact_trace.open() as stream:
    row = list(csv.DictReader(stream, delimiter="\t"))[-1]
assert row["active_contacts"] == "1", row
assert abs(float(row["peak_normal_force_n"]) - 9.81) < 0.2, row
assert {row["body_a"], row["body_b"]} == {"mover", "pedestal"}, row

moving_parent = open_experiment("moving-parent").run(steps=20,
    state_trace=work / "moving.tsv", contact_trace=work / "moving-force.tsv")
hinge, travel = final_q(moving_parent, 2)
assert abs(hinge) < 1e-7 and abs(travel + 0.5) < 1e-5, (hinge, travel)
assert moving_parent.stats["maximum_active_contacts"] == 0, moving_parent.stats

command = [
    rollout, "--headless", "--scene-file", open_experiment("contact").scene,
    "--metallib", library, "--steps", "20", "--contact-trace", work / "selected.tsv",
    "--contact-body",
]
subprocess.run([*command, "mover"], check=True, capture_output=True)
assert (work / "selected.tsv").read_bytes() == contact.contact_trace.read_bytes()
subprocess.run([*command, "numi_mjcf_world"], check=True, capture_output=True)
with (work / "selected.tsv").open() as stream:
    row = list(csv.DictReader(stream, delimiter="\t"))[-1]
assert row["active_contacts"] == "0" and float(row["peak_normal_force_n"]) == 0, row
unknown = subprocess.run([*command, "unknown"], capture_output=True, text=True)
assert unknown.returncode != 0 and "not present in the model" in unknown.stderr, unknown
PY
printf '%s\n' 'Numi Mac workbench passed relocation, motor routing, contact force, and exact replay.'
