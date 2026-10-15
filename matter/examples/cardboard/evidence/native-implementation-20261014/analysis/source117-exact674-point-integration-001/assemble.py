from pathlib import Path
import hashlib,json,shutil,subprocess,difflib
B=Path('/Users/home/cardboard-implementation-20261014');R=Path('/Users/home/numi-cardboard-20261006');OLD=B/'snapshots/source-exact-material-point-112';NEW=B/'snapshots/source-exact-material674-point-117';O=B/'analysis/source117-exact674-point-integration-001';I=B/'solver/explicit32-certified-coupon-380'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'frozen-source.json').read_text());s=json.loads((OLD/'configure-support-source.json').read_text())
for p,h in {**a['source_sha256'],**s['files']}.items():assert sha(OLD/p)==h,p
assert not NEW.exists();O.mkdir(exist_ok=False);shutil.copytree(OLD,NEW)
rel='matter/tools/cardboard_source370_material_point_check.mm';text=(NEW/rel).read_text();before=text
def one(old,new):
 global text
 assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
one('kTargetTet=1820u','kTargetTet=674u')
bindings={}
for key,name in [('kPackageSHA','compiled.nmatterpack'),('kCandidateNodesSHA','rejected-conditioning-000002/candidate_nodes.bin'),('kConditioningManifestSHA','rejected-conditioning-000002/manifest.json'),('kAcceptedStateSHA','material_state_000002_bent.json'),('kRunManifestSHA','manifest.json')]:
 p=I/'output'/name;h=sha(p);prefix=f'constexpr char {key}[]="';start=text.index(prefix)+len(prefix);end=text.index('";',start);text=text[:start]+h+text[end:];bindings[str(p)]=h
one('7918250992036473017ull','8485105279886344921ull')
one('"medium compiled local iteration budget")==16u','"medium compiled local iteration budget")==32u')
one('package.materials[kTargetMaterial].localNewtonIterations==16u','package.materials[kTargetMaterial].localNewtonIterations==32u')
one('captured.localNewtonIterations==16u','captured.localNewtonIterations==32u')
one('immutable source-370 descriptor no longer records its actual 16-update budget','immutable source-380 descriptor no longer records its actual 32-update budget')
one('tetrahedron.nodes.x==584u&&tetrahedron.nodes.y==880u&&\n         tetrahedron.nodes.z==878u&&tetrahedron.nodes.w==881u','tetrahedron.nodes.x==228u&&tetrahedron.nodes.y==524u&&\n         tetrahedron.nodes.z==522u&&tetrahedron.nodes.w==525u')
one('immutable_run_package_medium_iterations\\\":16','immutable_run_package_medium_iterations\\\":32')
one('not the immutable 370 compiled world','not the immutable 380 compiled world')
one('numi.cardboard.source370-native-material-point.v1','numi.cardboard.source380-native-material-point.v1')
one('base_snapshot_manifest_sha256\\\":','historical_base_snapshot107_manifest_sha256\\\":')
one('tet_id\\\":1820','tet_id\\\":674')
one('run_compiled_local_iterations\\\":16','run_compiled_local_iterations\\\":32')
one('budget32_is_not_a_replay_of_the_full_370_trajectory','budget32_is_not_a_replay_of_the_full_380_trajectory')
one('parameter words verified against the 370 package','parameter words verified against the 380 package')
(NEW/rel).write_text(text)
changes={rel:sha(NEW/rel)}
(O/'overlay.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)))
selected={p:sha(NEW/p) for p in a['source_sha256']}
(NEW/'frozen-source.json').write_text(json.dumps({**a,'base_snapshot':str(OLD),'scope':'Diagnostic-only exact source380 tetrahedron674 production FEM/material-frame deformation readback and material projection16/32. Same production shader/helpers as built112; only host bindings and metadata changed.','source_sha256':selected},indent=2)+'\n')
extra={str(p.relative_to(NEW)):sha(p) for p in sorted(NEW.rglob('*')) if p.is_file() and str(p.relative_to(NEW)) not in selected and p.name not in ['frozen-source.json','configure-support-source.json']};assert extra==s['files']
(NEW/'configure-support-source.json').write_text(json.dumps({**s,'files':extra},indent=2)+'\n')
(O/'integration.json').write_text(json.dumps(dict(snapshot=str(NEW),changed_files=changes,base_files={rel:sha(OLD/rel)},exact_run_inputs=bindings,selected_source_files=len(selected),support_files=len(extra),production_material_and_shader_bytes_unchanged_from112=True,material_outcome_not_an_execution_success_condition=True,historical_target_name_retained='numi-matter-cardboard-source370-material-point-check',limits='No physical transaction. F and local validity are measured natively; no new derivative or global convergence verdict.'),indent=2)+'\n');shutil.copy2(__file__,O/'assemble.py')
cmd=[x.replace('source-exact-material-point-112','source-exact-material674-point-117').replace('provenance-112','provenance-117') for x in json.loads((B/'build-command-112.json').read_text())]
(B/'build-command-117.json').write_text(json.dumps(cmd,indent=2)+'\n')
with (B/'build117-runner.log').open('x') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
for p,h in {**selected,**extra}.items():assert sha(NEW/p)==h,p
(O/'post-build-check.json').write_text(json.dumps(dict(build_exit_code=code,source_inputs_unchanged=True,source_build_sha256=sha(R/'build-cardboard-provenance-117/source-build.json')),indent=2)+'\n');print('build117',code,flush=True);raise SystemExit(code)
