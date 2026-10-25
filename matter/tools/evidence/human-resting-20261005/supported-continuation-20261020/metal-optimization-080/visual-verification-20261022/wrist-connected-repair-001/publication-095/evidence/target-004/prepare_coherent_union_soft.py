from pathlib import Path
import importlib.util,json,ast,numpy as np,hashlib,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';A=D/'automatic-connected-continuation-169'
s=importlib.util.spec_from_file_location('adapt',A/'run_existing_continuation_002.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=m.read(Path(sys.argv[1]));candidate=Path(r['candidate']);point=Path(r['point_summary']);surface=Path(r['surface_report']);parent=Path(r['parent_fit']);ip=m.read(point);sr=m.read(surface)
src=Path(sys.argv[3]);tr=Path(sys.argv[4]);tjface=m.read(tr)
assert tjface['inputs']['candidate_sha256']==m.sha(candidate)
assert tjface['matched_159_complete'] is True and tjface['banks']==185
bz=np.load(src);z=np.load(candidate);F=z['source_O_faces'];X=z['points_world_m'].astype(float)
assert np.array_equal(F[bz['face_rows']],bz['face_node_rows'])
assert np.all(bz['target_exactly_outside_all185'])
samples=np.einsum('mi,mij->mj',bz['barycentric_weights'],X[bz['face_node_rows']])
targets=bz['target_world_f32_m'].astype(float)
normal_key='current_outward_world_unit' if 'current_outward_world_unit' in bz.files else 'source_outward_world_unit'
normals=bz[normal_key]
ahead=np.einsum('ij,ij->i',normals,targets-samples)>1e-7
assert np.any(ahead), 'all coherent soft targets reached'
out=Path(sys.argv[2]);out.mkdir(exist_ok=False);bt=out/'barycentric-targets.npz'
np.savez_compressed(bt,face_rows=bz['face_rows'][ahead],weights=bz['barycentric_weights'][ahead],normals_world=normals[ahead],targets_world_m=targets[ahead],bank_body=bz['bank_body'][ahead],bank_stable_bone_id=bz['bank_stable_bone_id'][ahead])
count=m.owner().summarize_surface_crossing_report(sr);receipt=out/'surface-count.json';m.write(receipt,count)
target=out/'point-targets';target.mkdir()
query=point.parent/'query';expected=dict(candidate=m.sha(candidate),**{'query_'+k:m.sha(query/(k+'.csv')) for k in ('points','pairs','nodes')})
targetpy=target/'build_targets.py'
m.emit(m.TARGET,targetpy,dict(CAND=candidate,QUERY=query,OUT=target,RPT=target/'nearest-union-exits.json',NPZ=target/'nearest-union-exits.npz',EXPECTED_QUERY_HASHES=expected),'target')
m.run([m.PY,targetpy],out/'point-targets.log')
tj=m.read(target/'nearest-union-exits.json')
assert tj['passing_targets']==ip['inside_nodes']
pins=next(ast.literal_eval(n.value) for n in ast.parse(m.DRIVER.read_text()).body if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name) and x.id=='PIN' for x in n.targets))
paths=dict(start=candidate,target_start=candidate,inside_report=point,target_json=target/'nearest-union-exits.json',target_npz=target/'nearest-union-exits.npz',old_trial=candidate,crossing_report=surface,crossing_count=receipt,parent_fit=parent)
pins.update({k:m.sha(v) for k,v in paths.items()})
pins['progress_helper']=m.sha(Path('/Users/numi/Work/numi-human-connected-surface-069/D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py'))
helper_override=Path(sys.argv[5]) if len(sys.argv)>5 else None
if helper_override is not None:pins['self_helper']=m.sha(helper_override)
overrides=dict(START=candidate,TARGET_START=candidate,INSIDE_REPORT=point,TARGET_JSON=paths['target_json'],TARGET_NPZ=paths['target_npz'],OLD_TRIAL=candidate,CROSSING_REPORT=surface,CROSSING_COUNT=receipt,OUT=out/'attempt-001',PARENT_FIT=parent,PIN=pins,TARGET_NODE_COUNT=ip['inside_nodes'],SURFACE_COUNT=count['total_crossing_pairs'],BARY_HELPER=H/'human_skin_fat_barycentric_constraints.py',BARY_TARGETS=bt,BARY_TARGET_SHA=m.sha(bt),BARY_HELPER_SHA=m.sha(H/'human_skin_fat_barycentric_constraints.py'),PROGRESS_HELPER=Path('/Users/numi/Work/numi-human-connected-surface-069/D/joint-clearance-merit-175/canonical/matter/tools/human_skin_fat_bone_audit.py'),BARY_WEIGHT=8.0,BEST_SURFACE_COUNT=count['total_crossing_pairs'],MAX_SCP=4,STEP_MM=0.25,MAX_SOLVE_S=20.0)
driver=out/'run_connected_barycentric_fit.py';m.emit(H/'joint_merit_barycentric_driver_template.py',driver,overrides,'driver')
if helper_override is not None:
 import shutil
 shutil.copy2(helper_override,driver.with_name('self_separation_026.py'))
m.write(out/'source-inputs.json',dict(candidate_sha256=m.sha(candidate),point_sha256=m.sha(point),surface_sha256=m.sha(surface),driver_sha256=m.sha(driver),self_helper_sha256=m.sha(driver.with_name('self_separation_026.py')),bary_targets_sha256=m.sha(bt),raw_bary_targets_sha256=m.sha(src),bary_scope='coherent orientation-verified direction first-empty-union targets; actual direction array recorded; all endpoints exactly outside185; final all185 exact candidate audit mandatory',best_prior_surface_pairs=count['total_crossing_pairs'],starting_surface_pairs=count['total_crossing_pairs'],fixed_target_scope='fresh matched face-interior samples and full-union endpoint certificates; soft construction only',retained_soft_rows=int(ahead.sum()),coherent_target_report_sha256=m.sha(tr),target_direction_array=normal_key,native_admission=False))
print(driver)
