#!/bin/bash
set -u
cd /Users/n/Developer/numi-crow-rl-20261107
while pgrep -f '^build/crow-v15-toe-contact-limited-probe-20261110/probe$' >/dev/null; do sleep 5; done
build/crow-v15-supported-balance-identification-20261110/probe > artifacts/crow-v15-supported-balance-identification-20261110/results.txt 2> artifacts/crow-v15-supported-balance-identification-20261110/stderr.txt
rc=$?
gzip -n -c artifacts/crow-v15-supported-balance-identification-20261110/results.txt > artifacts/crow-v15-supported-balance-identification-20261110/results.txt.gz
printf 'exit %s\n' "$rc" > artifacts/crow-v15-supported-balance-identification-20261110/run.log
exit "$rc"
