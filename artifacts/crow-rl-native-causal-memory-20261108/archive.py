from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import mlx.core as mx
import numpy as np
r=Path.cwd();out=r/'artifacts/crow-rl-native-causal-memory-20261108';runs=r/'.numi/runs/american-crow';trial=runs/'native-causal-01'
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def record(p):return {'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777}
retained=r/'.numi/retained-runtime/native-causal-01'
for relative in ('bin/metalrobo_task_rollout','bin/metalrobo_task_train','bin/metalrobo_crow_memory_causal_check','lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/GRUMemory.metallib','shaders/RetinalStream.metallib'):
 p=r/'build'/relative;q=retained/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
old=json.loads((r/'artifacts/crow-rl-adaptive-optimization-20261108/runtime-recovery.json').read_text());m={'manifests':old['manifests']+['artifacts/crow-rl-native-causal-memory-20261108/manifest.json'],'native':{},'python_sources':{},'git_inputs':{},'checkpoint':{}}
for p in retained.rglob('*'):
 if p.is_file():m['native'][str(p.relative_to(retained))]={**record(p),'archive_source':str(p.relative_to(r))}
for p in (r/'python/metalrobo').rglob('*.py'):
 name=str(p.relative_to(r));m['python_sources'][name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for p in (r/'environments/american-crow').rglob('*'):
 if p.is_file():
  name=str(p.relative_to(r));m['git_inputs']['project/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
m['checkpoint']={'input/fixed-policy81.policypack':str((trial/'fixed-policy81.policypack').relative_to(r))}
ref=trial/'evaluation/native-evidence'
m['causal_reference']={str(p.relative_to(ref)):record(p) for p in ref.rglob('*') if p.is_file()}
m['recovery_scope']='Actual Git-restored fixed-policy native12-arm245760-transition evaluation, matching all native reference evidence bytes; matching device/OS required.'
analysis=json.loads((trial/'analysis/summary.json').read_text());analysis['source_commit']=a.git('rev-parse','HEAD').decode().strip()
subprocess.run([sys.executable,'tools/crow_evidence_archive.py','capture',str(out.relative_to(r))],check=True)
(out/'runtime-recovery.json').write_text(json.dumps(m,indent=2)+'\n');(out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(out/'README.md').write_text("""# Native causal crow cue memory, fixed fixture

The unchanged stochastic policy81 (SHA25621de3d792f6aafb4d1fdfdbdd7916f10179321030b31d069ba5f44687d9afbf4)
executes3 new allocation/action seeds x4 matched64-lane320-tick arms.245760
accepted transitions, no learning, retries, physics failures or terminations.
Arms: intact, GRU-only clear at40, absent rendered cue, swapped rendered cue.

Intact selects the food side157/192 (81.8%, Wilson95%75.7..86.6) and consumes
food38/192 (19.8%,14.8..26.0). Clearing GRU gives80/192 choices and12/192
intakes; absent cues82 and13; swapped cues25 and5. Paired intake p-values:
clear0.000306, absent0.000621, swapped0.000000250. All per-seed effects and
paired discordances remain in analysis.json and raw tabs. This supports causal
learned cue memory in this fixed head-reachable fixture and sampled native seeds.
It does NOT establish reliable feeding, new geometry, biological validity,
manipulation, social learning, natural ecology or the full lifelong objective.

The new C++ evaluation-only memory API requires the correct successful quiescent
resident checkpoint context and installed healthy recurrent policy. It clears
only GRU state/metadata. Foreign/pending ownership rejects; clearing is idempotent.
Actual complete resident bytes except the GRU envelope, full retina and food
checkpoints compare exact. The next native reset trace and hidden inputs verify
zero memory; no episode/physics/action clock is reset. Before/after worlds are
preserved. Cue interventions alter only two mesh visibility bindings; normal
rasterization/occlusion remains authoritative. They cannot publish or restore a
canonical retinal checkpoint. Input policy/semantic contracts are never rebound.

Every arm retains per-lane allocations/actions/intakes/bites, rendered cue truth
counts and input hashes. Pre-choice q/v and proprioception, postcue RGB and food
allocation are exact across paired arms. Truth counts are evidence only, never
actor input. Eight focused native CTests and full Release build pass. The first
harness startup lacked parent directories after verified cleanup; preserved
original driver and correction, with no native execution in that attempt.

Git recovery executes the complete native evaluation using restored exact binary,
library, shaders, policy and ports and compares all reference evidence bytes.
Only then may raw and duplicate checkout payloads be removed. No video captures.
""")
