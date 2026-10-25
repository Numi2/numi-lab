from pathlib import Path
import importlib.util,json,ast,numpy as np,hashlib,sys
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';A=D/'automatic-connected-continuation-169'
s=importlib.util.spec_from_file_location('adapt',A/'run_existing_continuation_002.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=m.read(Path(sys.argv[1]));candidate=Path(r['candidate']);point=Path(r['point_summary']);surface=Path(r['surface_report']);parent=Path(r['parent_fit']);ip=m.read(point);sr=m.read(surface)
src=A/'run-002/round-001/barycentric-virtual-point-172-corrected-raw-vector-transform/run002-round001-certified-inside-face-rows.npz'
assert m.sha(src)=='cd8213406ee11bd0adf759b9ff7e55aca9722887642aa4961ca239927ccff0d6'
bz=np.load(src);z=np.load(candidate);F=z['source_O_faces'];X=z['points_world_m'].astype(float)
assert np.array_equal(F[bz['face_rows']],bz['face_node_rows'])
samples=np.einsum('mi,mij->mj',bz['barycentric_weights'],X[bz['face_node_rows']])
targets=bz['contact_world_m']+bz['normal_world_unit']*50e-6
ahead=np.einsum('ij,ij->i',bz['normal_world_unit'],targets-samples)>1e-7
assert np.any(ahead), 'all frozen soft target planes reached'
out=Path(sys.argv[2]);out.mkdir(exist_ok=False);bt=out/'barycentric-targets.npz'
np.savez_compressed(bt,face_rows=bz['face_rows'][ahead],weights=bz['barycentric_weights'][ahead],normals_world=bz['normal_world_unit'][ahead],targets_world_m=targets[ahead],bank_body=bz['bank_body'][ahead],bank_stable_bone_id=bz['bank_stable_bone_id'][ahead])
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
overrides=dict(START=candidate,TARGET_START=candidate,INSIDE_REPORT=point,TARGET_JSON=paths['target_json'],TARGET_NPZ=paths['target_npz'],OLD_TRIAL=candidate,CROSSING_REPORT=surface,CROSSING_COUNT=receipt,OUT=out/'attempt-001',PARENT_FIT=parent,PIN=pins,TARGET_NODE_COUNT=ip['inside_nodes'],SURFACE_COUNT=count['total_crossing_pairs'],BARY_HELPER=H/'human_skin_fat_barycentric_constraints.py',BARY_TARGETS=bt,BARY_TARGET_SHA=m.sha(bt),BARY_HELPER_SHA=m.sha(H/'human_skin_fat_barycentric_constraints.py'),BARY_WEIGHT=8.0,BEST_SURFACE_COUNT=count['total_crossing_pairs'],MAX_SCP=4,STEP_MM=0.25,MAX_SOLVE_S=20.0)
driver=out/'run_connected_barycentric_fit.py';m.emit(H/'soft_barycentric_driver_template.py',driver,overrides,'driver')
m.write(out/'source-inputs.json',dict(candidate_sha256=m.sha(candidate),point_sha256=m.sha(point),surface_sha256=m.sha(surface),driver_sha256=m.sha(driver),bary_targets_sha256=m.sha(bt),raw_bary_targets_sha256=m.sha(src),bary_scope='per implicated closed bank only; final all185 exact audit mandatory',best_prior_surface_pairs=count['total_crossing_pairs'],starting_surface_pairs=count['total_crossing_pairs'],fixed_target_scope='original per-bank soft proposals; original inside classifications are stale after rebase',retained_soft_rows=int(ahead.sum()),native_admission=False))
print(driver)
