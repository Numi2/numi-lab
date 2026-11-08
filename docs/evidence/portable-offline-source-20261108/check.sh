#!/bin/bash
set -euo pipefail
base=/Users/numi/numi-minimal-source-offline-20261108
source=/Users/numi/numi-minimal-source-with-fixture-20261108/source
bundle=/Users/numi/numi-minimal-source-with-fixture-20261108
pkg="$base/installed"
ctest --test-dir "$base/build" -R '^metalrobo_numi_policy_status$' --output-on-failure > "$base/ctest.txt" 2>&1
python3 -m unittest discover -s "$source/tests" -p policy_selection_test.py > "$base/python-tests.txt" 2>&1
swiftc "$source/apps/NumiPolicyStatus.swift" "$bundle/qualification/decision.swift" -o "$base/decision-check"
python3 - "$bundle/qualification/producer-consumer.py" "$base/producer-installed.py" "$pkg" <<'PY'
from pathlib import Path
import sys
source,target,pkg=sys.argv[1:]
s=Path(source).read_text().replace("str(Path(__file__).resolve().parents[3] / 'python')",repr(pkg+'/share/numi/python'))
Path(target).write_text(s)
PY
python3 "$base/producer-installed.py" "$base/decision-check" > "$base/producer-consumer.txt" 2>&1
"$pkg/bin/metalrobo_world_pack" --inspect "$bundle/qualification/fixture/world.mrworld" > "$base/world-inspect.txt" 2>&1
