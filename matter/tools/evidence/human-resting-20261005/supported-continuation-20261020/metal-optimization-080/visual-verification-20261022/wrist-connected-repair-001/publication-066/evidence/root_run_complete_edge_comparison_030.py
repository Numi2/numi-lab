from pathlib import Path
import numpy as np,csv,subprocess,json,hashlib,sys
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'route-audit-counter-fix-030';M=D/'canonical/matter';AP=E/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(AP)=='39d110a33b5c997575e75f2447ab14a2416a7d52d30bb369414b82d0416220cf'
ap=np.load(AP,allow_pickle=False)
rs=list(csv.DictReader((E/'native-candidate020-all-bank-containment-035/source-preflight-003/initial-skin-fem-nodes.csv').open()))
X=np.array([[float(r[c+'_m']) for c in 'xyz']for r in rs],np.float32)
assert np.array_equal(X.view(np.uint32),ap['points'].astype(np.float32).view(np.uint32))
base=D/'baseline-035.npz';assert not base.exists()
np.savez_compressed(base,points_world_f32=X,global_node_ids=ap['global_nodes'],tetrahedra=ap['tetrahedra'],material_ids=ap['material_ids'])
cases=[('baseline035',base,E/'native-candidate020-all-bank-containment-035/source-preflight-003'),
       ('candidate014',E/'baseline-depth-transition-014/candidate.npz',None),
       ('candidate027',E/'o12-outward-repair-027-f32-gated/candidate.npz',E/'native-source-only-rest-geometry-027-query-028/source-preflight')]
rows=[]
for name,C,query in cases:
 O=D/name
 args=[sys.executable,str(M/'tools/human_skin_fat_bone_audit.py'),'--evidence-root',str(M),'--bank-export-dir',str(E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'),'--candidate-npz',str(C),'--candidate-sha256',sha(C),'--all-tet-edges','--output-dir',str(O)]
 if name!='baseline035':args+=['--node-ids-npz',str(AP),'--node-ids-sha256',sha(AP)]
 if query:args+=['--matched-query-run-dir',str(query)]
 p=subprocess.run(args,text=True,capture_output=True)
 (D/(name+'-command.json')).write_text(json.dumps(args,indent=2)+'\n')
 (D/(name+'-stdout.log')).write_text(p.stdout+p.stderr)
 assert p.returncode==0,p.stderr[-2000:]
 r=json.loads((O/'tet-edge-report.json').read_text())
 assert r['counts']['proper_hits']==sum(w['classification']=='proper_interior_hit'for w in r['witnesses'])
 assert r['counts']['boundary_hits']==sum(w['classification']=='boundary_hit'for w in r['witnesses'])
 rows.append({'case':name,'candidate_sha256':sha(C),'report_sha256':sha(O/'tet-edge-report.json'),'counts':r['counts'],'edges_with_hits':r['edges_with_hits'],'edge_bank_groups_with_hits':r['edge_bank_groups_with_hits']})
 print(json.dumps(rows[-1]),flush=True)
(D/'comparison.json').write_text(json.dumps({'cases':rows,'accepted_physical_steps':0,'physical_qualification':False,'source_sha256':sha(M/'tools/human_skin_fat_bone_audit.py'),'baseline_derived_from':'033 points verified bitwise against native035 baseline nodes'},indent=2)+'\n')

