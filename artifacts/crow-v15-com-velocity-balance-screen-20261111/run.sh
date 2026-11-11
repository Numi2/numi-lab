#!/bin/bash
set -eo pipefail
D=artifacts/crow-v15-com-velocity-balance-screen-20261111
CROW_SUBSTEP_TRACE=1 build/crow-v15-com-velocity-balance-screen-20261111/probe 2> "$D/stderr.txt" | gzip -n > "$D/results.txt.gz"
