#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    printf '%s\n' 'Usage: tests/numi_mac_mjcf_test.sh /path/to/catalog-compiler' >&2
    exit 2
fi
compiler=$1
repo=$(CDPATH= cd "$(dirname "$0")/.." && pwd -P)
source="$repo/tests/fixtures/workbench_pendulum.xml"
work=$(mktemp -d "${TMPDIR:-/tmp}/numi-mjcf.XXXXXX")
trap 'rm -rf "$work"' EXIT
trap 'exit 1' HUP INT TERM

"$compiler" --mjcf "$source" --output "$work/Original.numiproject"
mv "$work/Original.numiproject" "$work/Moved Project.numiproject"
moved="$work/Moved Project.numiproject"
cmp "$source" "$moved/source/model.xml"
"$compiler" --mjcf "$moved/source/model.xml" --gravity 0 --effort 0 \
    --output "$work/Control.numiproject"
"$compiler" --mjcf "$repo/tests/fixtures/workbench_mjcf_channels.xml" \
    --output "$work/Channels.numiproject"

python3 - "$work" "$repo/tests/fixtures/workbench_mjcf_channels.xml" <<'PY'
import json
from pathlib import Path
import sys
from xml.etree import ElementTree

folder = Path(sys.argv[1])
moved = folder / "Moved Project.numiproject"
settings = json.loads((moved / "project.json").read_text())
assert settings["source_format"] == "mjcf" and settings["gravity"] == 8.75
control = json.loads((folder / "Control.numiproject" / "project.json").read_text())
assert control["gravity"] == 0 and control["effort"] == 0
channels = json.loads((folder / "Channels.numiproject/project.json").read_text())["action_channels"]
assert [(channel["source_motor"], channel["joint"]) for channel in channels] == [
    ("motor_beta", "beta"), ("motor_alpha", "alpha"),
]
args = json.loads((moved / "robot-project.numi-window.json").read_text())["arguments"]
assert args[args.index("--physics-substeps") + 1] == "10"
robot = ElementTree.parse(moved / "source/model.urdf")
joint = robot.find("./joint[@name='pivot']")
assert joint is not None and joint.attrib["type"] == "revolute"
assert abs(float(joint.find("origin").attrib["xyz"].split()[2]) - 0.8) < 1e-12
assert abs(float(joint.find("origin").attrib["rpy"].split()[2]) - 0.7853981633974483) < 1e-10
assert abs(float(joint.find("limit").attrib["upper"]) - 2.530727415391778) < 1e-12
assert float(joint.find("limit").attrib["effort"]) == 12
assert len(robot.findall("./link[@name='stand']/collision")) == 1
assert len(robot.findall("./link[@name='stand']/visual")) == 4
assert robot.find("./transmission/joint").attrib["name"] == "pivot"
for visual in (moved / "visual").glob("*.mrvpack"):
    assert str(folder).encode() not in visual.read_bytes(), visual

source = (moved / "source/model.xml").read_text()
cases = {
    "doctype": '<!DOCTYPE mujoco [<!ENTITY unrelated "ignored">]>\n' + source,
    "include": source.replace("<worldbody>", '<include file="other.xml"/><worldbody>'),
    "free": source.replace('<joint name="pivot"', '<freejoint/><joint name="pivot"'),
    "two-joints": source.replace('<joint name="pivot"', '<joint name="second"/><joint name="pivot"'),
    "timestep": source.replace('timestep="0.002"', 'timestep="0.003"'),
    "asymmetric": source.replace('ctrlrange="-6 6"', 'ctrlrange="-5 6"'),
    "unknown-joint": source.replace('joint="pivot"', 'joint="missing"'),
}
for name, xml in cases.items():
    (folder / f"{name}.xml").write_text(xml)
(folder / "duplicate-motor.xml").write_text(
    Path(sys.argv[2]).read_text().replace('name="motor_alpha"', 'name="motor_beta"')
)
PY

reject() {
    if "$compiler" --mjcf "$work/$1.xml" --output "$work/$1.numiproject" > "$work/$1.log" 2>&1; then
        printf 'Unexpectedly accepted %s\n' "$1" >&2
        exit 1
    fi
    grep -F -- "$2" "$work/$1.log"
}
reject doctype 'without an XML document type'
reject include 'Unsupported content'
reject free 'Unsupported content'
reject two-joints 'Expected one <joint>'
reject timestep 'must divide the 20 ms control period'
reject asymmetric 'symmetric, enforced motor ctrlrange'
reject unknown-joint 'unknown joint'
reject duplicate-motor 'Motor names must be nonempty and unique'
printf '%s\n' 'Numi Mac MJCF passed translation, relocation, settings, and rejection checks.'
