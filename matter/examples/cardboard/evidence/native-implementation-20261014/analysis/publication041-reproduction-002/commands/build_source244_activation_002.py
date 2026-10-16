from pathlib import Path
import hashlib,json,shutil,subprocess,os,datetime,sys
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
BASE=Q/'snapshots/source-strict-refined-contact-observers-241-004'
OVER=Q/'analysis/source244-activation-full-residual-control-002'
SNAP=Q/'snapshots/source-activation-full-residual-armijo-244-002'
BUILD=Q/'build-source244-activation-002'
OUT=Q/'analysis/source244-activation-build-002'
REL='matter/tools/cardboard_full_contact_armijo_check.mm'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if SNAP.exists() or BUILD.exists() or OUT.exists():
    raise SystemExit('refusing to overwrite source244-002 build paths')
OUT.mkdir()
base_frozen=sha(BASE/'frozen-source.json')
assert base_frozen=='f64e6c5b12a0d69b944e002eb4c499533ac9b6cd6b852a96d869ce053c41c1db'
f=json.loads((BASE/'frozen-source.json').read_text())
cfg=json.loads((BASE/'configure-support-source.json').read_text())
expected={**cfg['files'],**f['source_sha256']}
for name,digest in expected.items():
    if sha(BASE/name)!=digest:
        raise SystemExit('source241 input hash mismatch: '+name)
SNAP.mkdir()
for name,digest in expected.items():
    src=BASE/name; dst=SNAP/name
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)
    if sha(dst)!=digest:
        raise SystemExit('source241 copy mismatch: '+name)
candidate=OVER/'candidate'/REL
dst=SNAP/REL
if REL not in f['source_sha256']:
    raise SystemExit('changed path absent from frozen source map')
if sha(dst)!=f['source_sha256'][REL]:
    raise SystemExit('base source path mismatch')
shutil.copy2(candidate,dst)
cand_sha=sha(dst)
f['source_sha256'][REL]=cand_sha
f['source244_activation_full_residual']={
    'parent_snapshot':str(BASE),'parent_frozen_sha256':base_frozen,
    'overlay_manifest_sha256':sha(OVER/'source-overlay.json'),
    'patch_sha256':sha(OVER/'source-overlay.patch'),
    'physical_operator_changed':False,
    'scope':'checker-only selector for authored activation control 1199 plus exact host/result scene sidecars; default path unchanged'
}
(SNAP/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
(SNAP/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
pyexe=str(Q/'venvs/cardboard-reference-185/bin/python3')
argv=[pyexe,'-B',str(SNAP/'matter/tools/cardboard_source_bound_build.py'),
      '--source-snapshot',str(SNAP),'--build-dir',str(BUILD),
      '--cmake-source-relative','.','--jobs','2','--target',
      'numi-matter-cardboard-full-contact-armijo-check']
with (OUT/'build.log').open('xb') as log:
    build=subprocess.run(argv,cwd=SNAP,env=env,stdout=log,stderr=subprocess.STDOUT)
commands=[{'argv':argv,'exit_code':build.returncode}]
cpu=None
if build.returncode==0:
    cpuargv=[str(BUILD/'matter/numi-matter-cardboard-full-contact-armijo-check'),
             str(SNAP/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'),
             '--cpu-only']
    cpu=subprocess.run(cpuargv,cwd=SNAP,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    (OUT/'cpu.stdout').write_bytes(cpu.stdout);(OUT/'cpu.stderr').write_bytes(cpu.stderr)
    commands.append({'argv':cpuargv,'exit_code':cpu.returncode})
artifact_paths=[BUILD/'source-build.json',
                BUILD/'matter/numi-matter-cardboard-full-contact-armijo-check',
                BUILD/'matter/libnumi_matter_metal.metallib']
artifacts=[]
for p in artifact_paths:
    if p.exists() and p.is_file(): artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
all_inputs_unchanged=all(sha(SNAP/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items())
r={'schema':'numi.cardboard.source244.activation_build.v1','snapshot':str(SNAP),
   'source_frozen_sha256':sha(SNAP/'frozen-source.json'),
   'base_snapshot':str(BASE),'base_frozen_sha256':base_frozen,
   'changed_source':REL,'changed_source_sha256':cand_sha,
   'overlay_manifest_sha256':sha(OVER/'source-overlay.json'),
   'overlay_patch_sha256':sha(OVER/'source-overlay.patch'),
   'commands':commands,'artifacts':artifacts,'all_inputs_unchanged':all_inputs_unchanged,
   'all_pass':build.returncode==0 and cpu is not None and cpu.returncode==0 and all_inputs_unchanged,
   'native_gpu_execution':False,'physical_fem_transactions':0,
   'physical_qualification':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/'report.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps(r,sort_keys=True),flush=True)
sys.exit(int(not r['all_pass']))
