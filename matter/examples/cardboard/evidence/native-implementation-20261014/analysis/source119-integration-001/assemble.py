from pathlib import Path
import hashlib, importlib.util, json, shutil, subprocess, sys

B=Path('/Users/home/cardboard-implementation-20261014')
R=Path('/Users/home/numi-cardboard-20261006')
S=B/'snapshots/source-scaled-material-merit-118'
N=B/'snapshots/source-trial-summary-face-guard-119'
O=B/'analysis/source119-integration-001'
A=B/'analysis/armijo-all-trial-summary-overlay-001'
G=B/'analysis/shipping-damage-face-ownership-memory-audit-115-002'
D=B/'analysis/cmake-transitive-metal-deps-118-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert (A/'manifest.json').is_file(), 'Agent must seal the overlay before integration'
assert sha(A/'manifest.json')=='213de1d4113a33f62273e333499c61987586f39be6f3142a8fe9b14b0f128402'
for row in json.loads((A/'manifest.json').read_text())['overlay_files']:assert sha(A/row['path'])==row['sha256']
assert not N.exists() and not O.exists()
old=json.loads((S/'frozen-source.json').read_text())
support=json.loads((S/'configure-support-source.json').read_text())
for p,h in {**old['source_sha256'],**support['files']}.items(): assert sha(S/p)==h,p
assert sha(G/'manifest.json')=='483d94eedc5e36ffa8fffc31548e8a64062de8aa64ee774dff35bf92369c807e'
assert sha(D/'overlay.patch')=='d4da81c4041e12548995cbb1fe2d0c3723a0a5a6b1d87cabb6ef0cfa3f66192e'
for row in json.loads((G/'manifest.json').read_text())['artifacts']: assert sha(G/row['path'])==row['sha256']
O.mkdir()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
q=subprocess.run(['/bin/cp','-cR',str(S),str(N)],capture_output=True,text=True)
(O/'clone.log').write_text(q.stdout+q.stderr);assert q.returncode==0
for p,h in {**old['source_sha256'],**support['files']}.items(): assert sha(N/p)==h,p
refreshed={}
for p in {**old['source_sha256'],**support['files']}:
 if '__pycache__' in Path(p).parts:
  if (N/p).is_file(): (N/p).unlink()
  continue
 if (R/p).is_file():
  if sha(R/p)!=sha(N/p):refreshed[p]={'snapshot118':sha(N/p),'current_main':sha(R/p)}
  shutil.copy2(R/p,N/p)
# Keep root CMake's current, unrelated targets configurable without rolling
# another workflow's source back. Bind all tracked apps as configure support.
for p in subprocess.check_output(['git','ls-files','-z','apps'],cwd=R).decode().split('\0'):
 if p and (R/p).is_file():
  (N/p).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/p,N/p)
intended=[
 'matter/src/metal/common.metalinc',
 'matter/tools/cardboard_bilinear15_capture_diagnostic_check.mm',
 'matter/CMakeLists.txt',
 'matter/tools/cardboard_paper_check.metal',
 'matter/tools/cardboard_source370_material_point_check.mm']
for p in intended:shutil.copy2(S/p,N/p)
p='matter/tools/cardboard_source370_material_point_check.mm'
exact=B/'snapshots/source-exact380-material674-117'/p
if not exact.exists():
 matches=list((B/'snapshots').glob('*/'+p))
 matches=[x for x in matches if sha(x)=='f02bae340982ad889f5e48488d82c42d25394c82b5af96bc4a51338277c25f21']
 assert matches;exact=matches[0]
assert sha(exact)=='f02bae340982ad889f5e48488d82c42d25394c82b5af96bc4a51338277c25f21'
shutil.copy2(exact,N/p)
runtime=A/'overlay/matter/src/runtime.mm';assert runtime.is_file()
shutil.copy2(runtime,N/'matter/src/runtime.mm')
for package in (G,D):
 q=subprocess.run(['git','apply','--check',str(package/'overlay.patch')],cwd=N,capture_output=True,text=True)
 (O/(package.name+'-check.log')).write_text(q.stdout+q.stderr);assert q.returncode==0
 q=subprocess.run(['patch','-p1','--batch','--forward','-i',str(package/'overlay.patch')],cwd=N,capture_output=True,text=True)
 (O/(package.name+'-apply.log')).write_text(q.stdout+q.stderr);assert q.returncode==0
spec=importlib.util.spec_from_file_location('bound_build',N/'matter/tools/cardboard_source_bound_build.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
selected=mod.source_bindings(N)
extra={str(p.relative_to(N)):sha(p) for p in sorted(N.rglob('*')) if p.is_file() and str(p.relative_to(N)) not in selected and p.name not in ('frozen-source.json','configure-support-source.json') and '__pycache__' not in p.parts}
assert not any('__pycache__' in p for p in selected)
(N/'frozen-source.json').write_text(json.dumps(dict(schema='numi.cardboard.frozen-source.v1',parent_revision=head,base_snapshot=str(S),scope='Source118 accurate local dimensionless Armijo merit plus source117 exact F readback, retained damage strict face guard, opt-in full-contact all-trial summary, compiler Metal depfiles. Current-main configure context and convergence tools retained. No equation, threshold or material calibration change.',source_sha256=selected),indent=2)+'\n')
(N/'configure-support-source.json').write_text(json.dumps(dict(schema='numi.cardboard.configure-support-source.v1',scope='Current-main tracked apps and existing configure context; excludes generated pycache',files=extra),indent=2)+'\n')
main_changes={p:{'main':sha(R/p) if (R/p).is_file() else None,'candidate':h} for p,h in selected.items() if not (R/p).is_file() or sha(R/p)!=h}
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(N),current_main=head,refreshed_from_current_main=refreshed,proposed_main_changes=main_changes,agent_overlay_manifest_sha256=sha(A/'manifest.json'),guard_manifest_sha256=sha(G/'manifest.json'),depfile_validation_sha256=sha(D/'validation.json'),selected_source_count=len(selected),support_source_count=len(extra),source_sha256=selected,support_sha256=extra,physical_steps=0),indent=2)+'\n')
(O/'assemble.py').write_bytes(Path(__file__).read_bytes())
cmd=json.loads((B/'build-command-118-full.json').read_text())
cmd=[x.replace('source-scaled-material-merit-118','source-trial-summary-face-guard-119').replace('provenance-118-full','provenance-119') for x in cmd]
cmd+=['--target','numi-matter-cardboard-source370-material-point-check']
(B/'build-command-119.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build119-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(N/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-119/source-build.json')),indent=2)+'\n')
print('build119',code,flush=True);raise SystemExit(code)
