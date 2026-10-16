from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import time

O=Path(__file__).resolve().parents[1]
B=O.parents[1]
R=Path('/Users/home/numi-cardboard-20261006')
base=B/'snapshots/source-cgls-block-right-preconditioned-149'
target=B/'snapshots/source-cgls-all-controls-153'
build=R/'build-cardboard-provenance-153'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
available=lambda:os.statvfs(B).f_bavail*os.statvfs(B).f_frsize
frozen=json.loads((base/'frozen-source.json').read_text())
support=json.loads((base/'configure-support-source.json').read_text())
assert len(frozen['source_sha256'])==787 and len(support['files'])==1034
for rel,h in frozen['source_sha256'].items(): assert sha(base/rel)==h,rel
for rel,h in support['files'].items(): assert sha(base/rel)==h,rel
prep=json.loads((O/'preparation.json').read_text())
assert sha(base/'matter/src/runtime.mm')==prep['base_sha256']
assert sha(O/'candidate/matter/src/runtime.mm')==prep['candidate_sha256']
assert sha(O/'runtime.patch')==prep['patch_sha256']
assert not target.exists() and not build.exists()
assert available()>=1536*1024**2
record={'schema':'numi.cardboard.source153_build.v1',
        'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'available_before_clone':available(),'base':str(base),'target':str(target),
        'base_manifest_sha256':sha(base/'frozen-source.json'),
        'source_count':787,'configure_support_count':1034,
        'physical_or_native_probe_execution':False}
subprocess.run(['/bin/cp','-cR',str(base),str(target)],check=True)
patch=subprocess.run(['patch','--batch','--forward','-p1','-d',str(target),
                      '-i',str(O/'runtime.patch')],text=True,capture_output=True)
(O/'validation/patch.stdout').write_text(patch.stdout)
(O/'validation/patch.stderr').write_text(patch.stderr)
assert patch.returncode==0,patch.stderr
assert sha(target/'matter/src/runtime.mm')==prep['candidate_sha256']
frozen['source_sha256']['matter/src/runtime.mm']=prep['candidate_sha256']
frozen['base_snapshot']=str(base)
frozen['scope']='Source149 plus exact opt-in all-control selectors. Existing bounded pure-FEM capability checks repeat independently at every selected control; no physical operator, constitutive law, acceptance tolerance, certificate or rollback change.'
(target/'frozen-source.json').write_text(json.dumps(frozen,indent=2)+'\n')
for rel,h in frozen['source_sha256'].items(): assert sha(target/rel)==h,rel
for rel,h in support['files'].items(): assert sha(target/rel)==h,rel
assert (target/'configure-support-source.json').read_bytes()==(base/'configure-support-source.json').read_bytes()
assert {str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}=={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
assert all(not p.is_symlink() for p in target.rglob('*'))
changed=[str(p.relative_to(base)) for p in base.rglob('*') if p.is_file() and sha(p)!=sha(target/p.relative_to(base))]
assert set(changed)=={'matter/src/runtime.mm','frozen-source.json'},changed
record.update(changed_files=changed,source_manifest_sha256=sha(target/'frozen-source.json'),available_after_clone=available())
command=['/Users/home/.pyenv/versions/3.10.12/bin/python3',
         str(target/'matter/tools/cardboard_source_bound_build.py'),
         '--source-snapshot',str(target),'--build-dir',str(build),
         '--cmake-source-relative','.','--jobs','4','--target','numi-matter-cardboard-probe']
record['command']=command
(O/'validation/build-preflight.json').write_text(json.dumps(record,indent=2)+'\n')
with (O/'validation/build.stdout').open('x') as out,(O/'validation/build.stderr').open('x') as err:
    proc=subprocess.Popen(command,stdout=out,stderr=err)
    minimum=available()
    aborted=False
    while proc.poll() is None:
        free=available(); minimum=min(free,minimum)
        if free<512*1024**2:
            proc.terminate(); aborted=True; break
        time.sleep(2)
    code=proc.wait()
record.update(exit_code=code,minimum_observed_available_bytes=minimum,space_abort=aborted,
              finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
if code==0:
    receipt=json.loads((build/'source-build.json').read_text())
    assert receipt['success']
    for filename,h in receipt['artifacts_sha256'].items(): assert sha(Path(filename))==h,filename
    for rel,h in frozen['source_sha256'].items(): assert sha(target/rel)==h,rel
    record['source_build_sha256']=sha(build/'source-build.json')
    record['artifacts_sha256']=receipt['artifacts_sha256']
(O/'validation/build-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
raise SystemExit(code)
