#!/bin/bash
set -u
cd /Users/n/Developer/numi-crow-rl-20261107
build/crow-v15-grip-balance-deadband-20261110/probe > artifacts/crow-v15-grip-balance-deadband-20261110/results.txt 2> artifacts/crow-v15-grip-balance-deadband-20261110/stderr.txt
rc=$?
gzip -n -c artifacts/crow-v15-grip-balance-deadband-20261110/results.txt > artifacts/crow-v15-grip-balance-deadband-20261110/results.txt.gz
if [ "$rc" -eq 0 ]; then /Users/n/Developer/.venvs/crow-rl-20261107/bin/python artifacts/crow-v15-grip-balance-deadband-20261110/analyze.py > artifacts/crow-v15-grip-balance-deadband-20261110/analysis.log 2>&1; rc=$?; fi
printf 'exit %s\n' "$rc" > artifacts/crow-v15-grip-balance-deadband-20261110/run.log
exit "$rc"
