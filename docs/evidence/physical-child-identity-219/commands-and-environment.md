# Commands and runtime inputs

## Native target build

cd /Users/numi/Work/skin-fat-native-integration-209
git diff --check
cmake --build build-native --target numi-human-native -j3

A fresh compile after the final app change completed with exit 0; see native-build-final-compile.log. The later incremental build check returned ninja no work to do in native-build-final-incremental.log.

## Standalone no-GPU importer and exact-frame check

cd /Users/numi/Work/skin-fat-native-integration-209
clang++ -std=c++23 -O2 -Wall -Wextra -Wpedantic -Werror -Iapps -Iinclude -Imatter/include /tmp/numi_physical_skin_child_identity_check.cpp -o /tmp/numi_physical_skin_child_identity_check
/tmp/numi_physical_skin_child_identity_check /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-label-capture-016/compiled-parent-cell-mass.csv /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-label-capture-016/compiled-parent-cell-mass.csv.labels.csv /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-tetgen-import-control-218/parent-reference.node /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-tetgen-import-control-218/parent-reference.ele /Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-tetgen-import-control-218/parent-reference.labels.csv

The harness imports the real capture-016 parent packet and identity TetGen tables, checks preloaded sidecars survive import, exercises distinct exact-F64 current/reference preservation with Float32 input, verifies exact captured-current F32 versus reference F64 agreement, and rejects node/tet/material/label mismatches and partial paths. Output is in identity-check.log. It performs no native app launch, GPU operation, or physical step.

## App control environment

Root owns the native capture-only preparation launch. The retained controls use these import and identity flags:

NUMI_HUMAN_UNPAIRED_IDENTITY_PREFLIGHT=1
NUMI_HUMAN_UNPAIRED_IDENTITY_INSTALL=1
NUMI_HUMAN_PHYSICAL_SKIN_CHILD_IDENTITY_IMPORT=1
NUMI_HUMAN_PHYSICAL_SKIN_CHILD_PARENT_MASS_CSV=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-label-capture-016/compiled-parent-cell-mass.csv
NUMI_HUMAN_PHYSICAL_SKIN_CHILD_PARENT_LABELS_CSV=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-label-capture-016/compiled-parent-cell-mass.csv.labels.csv
NUMI_HUMAN_PHYSICAL_SKIN_CHILD_NODE_FILE=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-tetgen-import-control-218/parent-reference.node
NUMI_HUMAN_PHYSICAL_SKIN_CHILD_ELE_FILE=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-tetgen-import-control-218/parent-reference.ele
NUMI_HUMAN_PHYSICAL_SKIN_CHILD_LABELS_CSV=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/native-parent-tetgen-import-control-218/parent-reference.labels.csv
NUMI_HUMAN_SKIN_FAT_PARENT_MASS_CAPTURE_ONLY=1

The full original invocations and complete environments are preserved in native-preparation-control-001-failure/launch.json and native-preparation-control-002-failure/launch.json. Those runs failed before final pass receipts and are retained. Root then ran native-preparation-control-003, whose complete launch/result/stdout/stderr and resting-coupled.csv are under native-preparation-control-003-pass/. It passed identity import/remap/install with zero advanced physical time. This worker did not launch a native control or GPU/physical run.
