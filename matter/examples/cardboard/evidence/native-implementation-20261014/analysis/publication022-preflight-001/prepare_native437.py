from pathlib import Path
import json, subprocess
B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
old_source=B/'snapshots/source-base-state-cauchy-armijo-133'
source=B/'snapshots/source-cache-projection-reuse-runtime-136'
old_case=B/'solver/base-cauchy-disabled-parity-434'
case=B/'solver/cache-reuse-full-history-parity-437'
cmd=json.loads((B/'native-434-command.json').read_text())
for i,s in enumerate(cmd):
    cmd[i]=s.replace(str(old_source),str(source)).replace(str(old_case),str(case)).replace('build-cardboard-provenance-133','build-cardboard-provenance-136')
cmd[cmd.index('--question')+1]='Does opt-in projection reuse preserve the complete native434 attempted coupon history?'
cmd[cmd.index('--prediction')+1]='Expect identical accepted/rejected numerical histories, all output geometry/material observations and control1 Newton48 rejection, after only implementation/provenance/timing changes. This is complete attempted-history parity, not a completed coupon or numerical-convergence qualification. Measure exact GPU step times as observations only.'
split=cmd.index('--')
cmd[split:split]=['--env','NM_FEM_CACHE_PROJECTION_REUSE=1','--input-file',str(B/'analysis/native434-disabled-history-parity-001/report.json')]
assert not case.exists()
(B/'native-437-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
print('prepared437')
