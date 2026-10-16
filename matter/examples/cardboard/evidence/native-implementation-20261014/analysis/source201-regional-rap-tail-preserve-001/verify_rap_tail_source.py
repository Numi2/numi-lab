#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, re

BASE=Path("/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-contact-damped-memory-198-001")
CANDIDATE=Path("/Users/n/cardboard-implementation-20261014-host-macmini/snapshots/source-regional-rap-tail-preserve-201")

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def body(text, name):
    marker="kernel void " + name + "("
    start=text.index(marker)
    brace=text.index("{", start)
    depth=0
    for i in range(brace, len(text)):
        if text[i]=="{": depth += 1
        elif text[i]=="}":
            depth -= 1
            if depth==0: return text[start:i+1]
    raise AssertionError("unterminated kernel: "+name)

base_metal=(BASE/"matter/src/metal/fgmres.metalinc").read_text()
new_metal=(CANDIDATE/"matter/src/metal/fgmres.metalinc").read_text()
base_runtime=(BASE/"matter/src/runtime.mm").read_text()
new_runtime=(CANDIDATE/"matter/src/runtime.mm").read_text()
checker=(CANDIDATE/"matter/tools/cardboard_regional_rap_tail_preserve_native_check.mm").read_text()
checks={}
for k in ("nm_fgmres_regional_build_rap", "nm_fgmres_regional_factor_rap"):
    checks["legacy_kernel_unchanged_"+k]=body(base_metal,k)==body(new_metal,k)
    assert checks["legacy_kernel_unchanged_"+k], k+" changed from source198"
match=re.search(r'dispatchGroups32\("nm_fgmres_regional_restart_build_rap",\s*([^,]+),',new_runtime)
assert match, "restart RAP host dispatch missing"
expr=" ".join(match.group(1).split())
checks["E2_restart_build_extent"]=(expr=="environments * augmentedModes * augmentedModes")
assert checks["E2_restart_build_extent"], "restart RAP must launch E*m^2 groups"
base_pos=new_runtime.index('dispatchGroups32("nm_fgmres_regional_build_rap"')
base_expr=re.search(r'dispatchGroups32\("nm_fgmres_regional_build_rap",\s*([^,]+),',new_runtime[base_pos:]).group(1)
checks["E2_initial_base_build_extent"]=(" ".join(base_expr.split())=="environments * regionalModes * regionalModes")
assert checks["E2_initial_base_build_extent"], "initial base RAP must launch E*m^2 groups"
for k in ("nm_fgmres_regional_restart_build_rap", "nm_fgmres_regional_restart_factor_rap"):
    b=body(new_metal,k)
    checks["restart_tail_guard_"+k]="tailPreserved[environment] != 0u" in b
    assert checks["restart_tail_guard_"+k], k+" missing tail guard"
assert '"nm_fgmres_regional_restart_build_rap"' in checker
assert '"nm_fgmres_regional_restart_factor_rap"' in checker
checks["checker_uses_restart_only_kernels"]=True
assert "kEnvironmentCount*kModeCount*kModeCount" in checker
checks["checker_dispatches_E2_m2"]=True
assert "next-Newton E2 RAP rebuild" in checker and "staleTail" in checker
checks["checker_rebuilds_after_reset"]=True
alloc=new_runtime.index("candidate->femRegionalRestartTailPreserved =")
preceding=new_runtime[max(0,alloc-3000):alloc]
checks["tail_buffer_conditional_allocation"]="if (candidate->femRegionalRestartAugmentationEnabled)" in preceding
assert checks["tail_buffer_conditional_allocation"]
assert "nm_fgmres_regional_restart_build_rap" not in new_runtime[new_runtime.index("dispatchGroups32(\"nm_fgmres_regional_build_rap")-100:new_runtime.index("dispatchGroups32(\"nm_fgmres_regional_build_rap")+100]
report={"schema":"numi.cardboard.rap-tail-source-contract.v1","checks":checks,"base_source_sha256":{"runtime.mm":sha(BASE/"matter/src/runtime.mm"),"fgmres.metalinc":sha(BASE/"matter/src/metal/fgmres.metalinc")},"candidate_source_sha256":{"runtime.mm":sha(CANDIDATE/"matter/src/runtime.mm"),"fgmres.metalinc":sha(CANDIDATE/"matter/src/metal/fgmres.metalinc"),"native_checker.mm":sha(CANDIDATE/"matter/tools/cardboard_regional_rap_tail_preserve_native_check.mm")},"scope":"Static source contract only; production kernel execution remains unrun."}
print(json.dumps(report,sort_keys=True,indent=2))
