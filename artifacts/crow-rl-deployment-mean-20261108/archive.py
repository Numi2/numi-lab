from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import mlx.core as mx
import numpy as np
r=Path.cwd();out=r/'artifacts/crow-rl-deployment-mean-20261108';runs=r/'.numi/runs/american-crow';trial=runs/'precision-evaluation-01'
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def record(p):return {'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777}
retained=r/'.numi/retained-runtime/deployment-mean-01'
for relative in ('bin/metalrobo_task_rollout','bin/metalrobo_task_train','bin/metalrobo_crow_memory_causal_check','lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/GRUMemory.metallib','shaders/RetinalStream.metallib'):
 p=r/'build'/relative;q=retained/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
old=json.loads((r/'artifacts/crow-rl-precision-training-20261108/runtime-recovery.json').read_text());m={'manifests':old['manifests']+['artifacts/crow-rl-deployment-mean-20261108/manifest.json'],'native':{},'python_sources':{},'git_inputs':{},'checkpoint':{}}
for p in retained.rglob('*'):
 if p.is_file():m['native'][str(p.relative_to(retained))]={**record(p),'archive_source':str(p.relative_to(r))}
for p in (r/'python/metalrobo').rglob('*.py'):
 name=str(p.relative_to(r));m['python_sources'][name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for p in (r/'environments/american-crow').rglob('*'):
 if p.is_file():
  name=str(p.relative_to(r));m['git_inputs']['project/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
m['checkpoint']={'input/'+mode+'.policypack':str((trial/(mode+'.policypack')).relative_to(r)) for mode in ('gaussian','mean')}
m['seed_base']=2651213502;m['evaluations']={}
for mode in ('gaussian','mean'):
 ref=trial/mode/'native-evidence';m['evaluations'][mode]={'original_root':str(ref),'policy':'input/'+mode+'.policypack','files':{str(p.relative_to(ref)):record(p) for p in ref.rglob('*') if p.is_file()}}
headers=[s for s in a.git('ls-files','include').decode().splitlines() if s.endswith(('.h','.hpp'))]+['apps/CrowCameraFixture.hpp']
for name in headers:
 p=r/name;m['git_inputs']['source/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
p=r/'artifacts/crow-rl-deployment-mean-20261108-identity_oracle.mm';m['git_inputs']['source/identity_oracle.mm']={**record(p),'git_blob':a.git('hash-object','-w',str(p)).decode().strip()}
m['recovery_scope']='Actual Git-restored complete Gaussian and deployment-mean12-arm native evaluation; behavior/checkpoint files byte-exact, path-bound visual manifest identities independently recomputed at original/restored locations with every other JSON field exact. Matching OS/device required.'
analysis=json.loads((trial/'qualification-summary.json').read_text());analysis['source_commit']=a.git('rev-parse','HEAD').decode().strip();analysis['mode_analyses']={mode:json.loads((trial/mode/'analysis/summary.json').read_text()) for mode in ('gaussian','mean')}
subprocess.run([sys.executable,'tools/crow_evidence_archive.py','capture',str(out.relative_to(r))],check=True)
(out/'runtime-recovery.json').write_text(json.dumps(m,indent=2)+'\n');(out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(out/'README.md').write_text("""# Deterministic crow97: native causal feeding and cue memory

Two exact actor-equivalent policy publications are tested on3 new allocation/action
seeds in four matched64-lane320-tick native arms each.491520 accepted transitions,
no learning, retries, physics rejection or termination. Seed base2651213502.
Input PolicyPacks are Git-restored, unchanged, never rebound to different contracts.

Deployment mean:192/192 correct-side choices and192/192 intakes,40000..45000ug
per lifetime,8140000ug total. GRU-only clear at40, absent cue and swapped cue
all give0/192 intakes; swapped cues give0 correct-side choices. This is a causal
learned memory-to-action-to-nutrition result in the fixed head-reachable fixture.
Mean mode has only two unique initial head commands: the192 allocation replicates
are repeated fixed cue conditions, not192 independent geometric challenges.
No broad reliability confidence interval or natural intelligence claim follows.

Gaussian exploration:173/192 correct choices,49/192 intakes. The memory/absent/
swapped controls give24/21/3 intakes. This qualifies causal memory in the same
fixed fixture but is less reliable than the actual deployment mean. The128-lane
training intake remained near20% while head std narrowed. Deployment evaluation,
not training exploration intake alone, resolves the useful behavior.

Native intervention checks preserve all resident state except GRU, full retina
and food bytes; foreign/pending ownership rejects, clearing is idempotent and
next hidden/reset trace is checked. Cue modes alter native rendered mesh visibility
only; physical q/v/proprioception, postcue RGB and allocations remain paired.
Food conservation holds. Every per-lane outcome and rendered cue count is retained.
Small canonical deployment97 is registered under the active environment policies
with exact SHA256 and scope in the project profile. It remains checked out for use.

Git recovery actually repeats both complete evaluations with restored exact binary,
library, shaders, policies and ports. Non-manifest evidence must be byte-exact;
visual-pack paths legitimately change render-scene identities, so the native owner
recomputes both original/restored fingerprints and all other JSON fields must match.
Only then may raw and duplicate archive payloads be pruned. No video/full rollouts.
Broader geometry, persistent ecology/physiology, object manipulation, social learning
and full lifelong intelligence remain unfinished. This is a development foundation.
""")
