from pathlib import Path
import json
B=Path('/Users/home/cardboard-implementation-20261014')
old_source=B/'snapshots/source-cauchy-positive-mass-admission-135'
source=B/'snapshots/source-repeated-cauchy-fallback-137'
old_case=B/'solver/base-cauchy-first-failure-enabled-435'
case=B/'solver/repeated-cauchy-full-coupon-438'
cmd=json.loads((B/'native-435-command.json').read_text())
for i,s in enumerate(cmd):
    cmd[i]=s.replace(str(old_source),str(source)).replace(str(old_case),str(case)).replace('build-cardboard-provenance-135','build-cardboard-provenance-137')
cmd[cmd.index('--question')+1]='Can repeated native residual-descent fallbacks at control1 complete the unchanged64-Newton coupon attempt?'
cmd[cmd.index('--prediction')+1]='Use native p=tau*Atranspose*b after any exhausted healthy primary Armijo loop at control1, requiring a fresh certificate for every original Newton solve. Keep64Newton/24trials/restart16/physicalprotocol/finalgates unchanged. Expect primaryhistory0..47 to match435 and firstfallback48 to match435; laterbehavior is a prediction. Preserve whole27-control attempt and rollback on any rejection. Cache reuse is UNSET after437 initialization rejection. Source137 trace actualselectedcap512; two retained notice strings still say64, a documented observer-label erratum. No new-direction linearcertificate or completedcoupon claim from an acceptedprefix.'
idx=cmd.index('NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=1:48')
cmd[idx]='NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR=1:all'
split=cmd.index('--')
cmd[split:split]=['--input-file',str(B/'analysis/native435-enabled-fallback-adjudication-001/report.json'),
    '--input-file',str(B/'analysis/repeated-cauchy-selector-review-137-001/report.md')]
assert not case.exists()
(B/'native-438-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
print('prepared438')
