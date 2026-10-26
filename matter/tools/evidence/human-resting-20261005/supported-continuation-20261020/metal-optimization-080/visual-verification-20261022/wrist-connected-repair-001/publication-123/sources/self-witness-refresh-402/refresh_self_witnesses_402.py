from pathlib import Path
import numpy as np,json,hashlib,importlib.util
D=Path('/Users/numi/Work/numi-human-connected-surface-069');H=D/'barycentric-connected-clearance-170';c=H/'combined-clearance-394/run-001/candidate.npz';z=np.load(c,allow_pickle=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(c)=='251b7e0a8c39d2e1e3ef57b2654bf4912cfb3e62bc9cc38fb9ee4d7a9a52e8c3'
old=H/'combined-clearance-394/run-001/selected-state-003/retained-self-pair-seeds.npy';a=np.load(old,allow_pickle=False);sources=[]
for p in sorted((H/'postfit-backtrack-400/run-001/backtrack-373').glob('pass-*/audit-130/surface-self-pairs.npy')):
 summary=json.loads(p.with_name('summary.json').read_text());trial=p.parent.parent/'candidate.npz';t=np.load(trial,allow_pickle=False)
 assert summary['complete_surface_audit'] and sha(trial)==summary['candidate_sha256'];assert np.array_equal(t['source_O_faces'],z['source_O_faces'])
 pairs=np.load(p,allow_pickle=False);assert pairs.shape==(summary['exact_PLC_surface_pairs'],2);assert np.all((pairs>=0)&(pairs<len(z['source_O_faces'])))
 sources.append(dict(path=str(p),sha256=sha(p),trial_sha256=sha(trial),pair_count=len(pairs)));a=np.concatenate((a,pairs))
a=np.unique(np.sort(a,axis=1),axis=0);assert np.all(a[:,0]<a[:,1])
out=D/'D/self-witness-refresh-402';out.mkdir(exist_ok=False);np.save(out/'merged-self-pair-seeds.npy',a)
owner=H/'connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py';sp=importlib.util.spec_from_file_location('root402owner',owner);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows,records=m.build_pair_rows(z['points_world_m'].astype(float)*1000,z['source_O_faces'],a,margin_fraction=.25)
mx=max(r['lower'] for r in rows);assert mx<=2e-8,mx
j=dict(candidate_npz=str(c),candidate_sha256=sha(c),old_seeds=str(old),old_seeds_sha256=sha(old),old_pair_count=16527,sources=sources,merged_pair_count=len(a),new_pair_count=len(a)-16527,output_pairs_npz=str(out/'merged-self-pair-seeds.npy'),output_pairs_sha256=sha(out/'merged-self-pair-seeds.npy'),pair_owner=str(owner),pair_owner_sha256=sha(owner),rebuilt_row_count=len(rows),baseline_max_rhs_mm=mx,margin_fraction=.25,native_admission=False,accepted_physical_steps=0)
(out/'report.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps({k:v for k,v in j.items()if k not in['sources']}))
