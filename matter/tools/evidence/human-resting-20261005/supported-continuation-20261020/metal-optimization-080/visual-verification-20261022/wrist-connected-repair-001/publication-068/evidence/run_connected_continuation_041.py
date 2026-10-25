from pathlib import Path
import json,hashlib,subprocess,sys,shutil,csv,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");P=E/'connected-volume-continuation-041';P.mkdir()
M=P/'canonical/matter/tools';M.mkdir(parents=True)
shutil.copy2(E/'connected-volume-repair-036/canonical/matter/tools/human_skin_fat_connected_repair.py',M/'human_skin_fat_connected_repair.py')
PY="/Users/n/numi-human-prep-venv-20261005/bin/python"
from scipy import sparse
AP=np.load(E/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz')
BX=AP['points'].astype(np.float32).astype(float)
rr=[];cc=[];vv=[];bb=[]
for col in range(2411):
 for layer in [0,1]:
  a=layer*2411+col;b=a+2411;d=BX[a]-BX[b];length=np.linalg.norm(d);n=d/length
  for sign,bound in [(1,.5*length),(-1,-1.5*length)]:
   row=len(bb)
   for node,weight in [(a,sign),(b,-sign)]:
    for k in range(3):rr.append(row);cc.append(3*node+k);vv.append(weight*n[k])
   bb.append(bound)
sparse.save_npz(P/'global-layer-rows.npz',sparse.coo_matrix((vv,(rr,cc)),shape=(len(bb),3*len(BX))).tocsr())
np.save(P/'global-layer-targets.npy',np.asarray(bb))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def invoke(args,log):
 r=subprocess.run(args,capture_output=True,text=True);log.write_text(r.stdout+r.stderr)
 if r.returncode:raise RuntimeError(str(log)+' '+r.stderr[-1500:])
def metrics(candidate,query,audit):
 r=json.loads((audit/'tet-edge-report.json').read_text())
 rows=list(csv.DictReader((query/'all-bank-containment.csv').open()))
 return {'candidate_sha256':sha(candidate),'edge_bank_groups':r['edge_bank_groups_with_hits'],'edges':r['edges_with_hits'],'inside_nodes':len({r['local_node']for r in rows if r['classification']=='inside'}),'inside_pairs':sum(r['classification']=='inside'for r in rows),'ambiguous':sum(int(r['ambiguous'])for r in rows)}
base_template=(E/'connected-volume-repair-032/run_connected_032.py').read_text()
base_template=base_template.replace('patch=np.unique(T[np.isin(T,patch).any(axis=1)])','patch=np.unique(T[np.isin(T,patch).any(axis=1)])\npatch=np.unique(T[np.isin(T,patch).any(axis=1)])')
query_template=(E/'root_run_geometry_only_native_query_028.py').read_text()
previous=E/'connected-volume-continuation-040/iteration-005/candidate.npz';query=E/'connected-volume-continuation-040/iteration-005/native-query/source-preflight';audit=E/'connected-volume-continuation-040/iteration-005/exact-audit'
history=[metrics(previous,query,audit)];stalls=0;start=time.monotonic()
for it in range(1,6):
 D=P/('iteration-%03d'%it);D.mkdir()
 s=base_template.replace("D=E/'connected-volume-repair-032';M=D/'canonical/matter/tools'","D=Path("+repr(str(D))+");M=Path("+repr(str(M))+")")
 s=s.replace("X=ap['points'].astype(np.float32)","X=np.load("+repr(str(previous))+")['points_world_f32']")
 s=s.replace("query=E/'native-candidate020-all-bank-containment-035/source-preflight-003'","query=Path("+repr(str(query))+")")
 s=s.replace("E/'route-audit-counter-fix-030/baseline035/tet-edge-report.json'","Path("+repr(str(audit/'tet-edge-report.json'))+")")
 lo=s.index("columns=np.unique(patch%2411)");hi=s.index("(D/'contact-provenance.json')",lo)
 s=s[:lo]+"L=sparse.load_npz(Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-041/global-layer-rows.npz'));lb=np.load(Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-041/global-layer-targets.npy'))\n"+s[hi:]
 s=s.replace("time_limit_s=90.","time_limit_s=90.")
 prep=D/'run_connected.py';prep.write_text(s);invoke([PY,str(prep)],D/'construction.log')
 report=json.loads((D/'report.json').read_text())
 if not report['history'][-1].get('step_applied'):
  print(json.dumps({'iteration':it,'status':'NO_HARD_VALID_STEP_STOP','report':report}),flush=True);break
 C=D/'candidate.npz';Q=D/'native-query';csha=sha(C)
 qs=query_template.replace("E/'o12-outward-repair-027-f32-gated/candidate.npz'","Path("+repr(str(C))+")").replace('ccfeb27e6e939249e4c5146f77722bf2477d1f8377300bde8cc82086f44bbe1a',csha).replace("E/'native-source-only-rest-geometry-027-query-028'","Path("+repr(str(Q))+")").replace('027 proposed rest geometry','connected continuation static geometry')
 qs=qs.replace(" result['status']='SOURCE_QUERY_PASS_BANKS_IDENTICAL_TO_039_NOT_PHYSICAL_ADMISSION'"," import os\n fresh=OUT/'source-preflight/exact-emitted.triangles.csv';retained=mod.FROZEN/'exact-emitted.triangles.csv'\n assert mod.sha(fresh)==mod.sha(retained)\n fresh.unlink();os.link(retained,fresh)\n result['status']='SOURCE_QUERY_PASS_BANKS_IDENTICAL_TO_039_NOT_PHYSICAL_ADMISSION'")
 qr=D/'run_native_static_query.py';qr.write_text(qs);invoke([PY,str(qr)],D/'native-query.log')
 A=D/'exact-audit'
 args=[PY,str(E/'route-audit-counter-fix-030/canonical/matter/tools/human_skin_fat_bone_audit.py'),'--evidence-root',str(E/'route-audit-counter-fix-030/canonical/matter'),'--bank-export-dir',str(E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'),'--candidate-npz',str(C),'--candidate-sha256',csha,'--node-ids-npz',str(E/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'),'--node-ids-sha256','39d110a33b5c997575e75f2447ab14a2416a7d52d30bb369414b82d0416220cf','--matched-query-run-dir',str(Q/'source-preflight'),'--all-tet-edges','--output-dir',str(A)]
 invoke(args,D/'audit.log')
 m=metrics(C,Q/'source-preflight',A);m.update({'iteration':it,'quality':report['history'][-1],'candidate_path':str(C),'query_path':str(Q/'source-preflight'),'audit_path':str(A)})
 old=history[-1];score=m['edge_bank_groups']+4*m['inside_nodes'];oldscore=old['edge_bank_groups']+4*old['inside_nodes']
 m['composite_improvement_fraction']=1-score/oldscore
 history.append(m);(P/'continuation.json').write_text(json.dumps({'history':history,'elapsed_s':time.monotonic()-start,'accepted_physical_steps':0,'physical_qualification':False,'geometry_admitted':False},indent=2)+'\n')
 print(json.dumps({k:v for k,v in m.items()if k!='quality'}),flush=True)
 if not m['edge_bank_groups']and not m['inside_nodes']and not m['ambiguous']:break
 stalls=stalls+1 if m['composite_improvement_fraction']<.03 else 0
 previous=C;query=Q/'source-preflight';audit=A
 if stalls>=2:print('STALLED_TWO_REFRESHED_CONNECTED_STEPS_SWITCH_TO_REFINEMENT',flush=True);break
print('CONTINUATION_ENDED',flush=True)

