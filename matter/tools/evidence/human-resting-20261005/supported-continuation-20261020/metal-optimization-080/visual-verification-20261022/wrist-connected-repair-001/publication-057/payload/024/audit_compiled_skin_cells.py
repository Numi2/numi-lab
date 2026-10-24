#!/usr/bin/env python3
import pathlib,json,csv,hashlib,numpy as np
E=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");O=E/'native-candidate020-mechanical-source-024';N=O/'source-preflight-001';D=E/'wholebody-attachment-source-rebind-037-candidate020-identity004-002'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
rows=read(N/'initial-skin-fem-nodes.csv');rr=read(D/'generated/nodes.csv');tt=read(D/'generated/tetrahedra.csv')
n=len(rr);assert n==2411 and len(rows)==3*n
assert [int(r['global_node']) for r in rows]==list(range(1666,1666+3*n))
assert [(int(r['layer']),int(r['row'])) for r in rows]==[(l,i) for l in range(3) for i in range(n)]
p=np.array([[r[k] for k in ('x_m','y_m','z_m')] for r in rows],dtype=np.float32)
mass=np.array([r['mass_kg'] for r in rows],dtype=np.float64)
t=np.array([[r[k] for k in 'abcd'] for r in tt],dtype=np.int64)
assert np.isfinite(p).all() and np.isfinite(mass).all() and (mass>0).all()
assert len(t)==28632 and t.min()>=0 and t.max()<3*n
ref=np.concatenate([np.array([[r[prefix+k] for k in 'xyz'] for r in rr],dtype=np.float64) for prefix in 'oid'])
def det(x,typ):
 x=x.astype(typ);z=x[t];a=z[:,1]-z[:,0];b=z[:,2]-z[:,0];c=z[:,3]-z[:,0]
 return np.sum(a*np.cross(b,c),axis=1)
d64=det(p,np.float64);d32=det(p,np.float32);dr=det(ref,np.float64)
edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
longest=np.max(np.stack([np.linalg.norm(p[t[:,a]].astype(float)-p[t[:,b]].astype(float),axis=1) for a,b in edges]),axis=0)
q32=d32.astype(float)/longest**3
report={'scope':'compiled Float32 initial skin/fat cells from successful source-only native preparation; no accepted physical step','status':'PASS_POSITIVE_NATIVE_EMITTED_CELLS' if (d64>0).all() and (d32>0).all() else 'FAIL_NONPOSITIVE_NATIVE_CELL','input_hashes':{str(x):sha(x) for x in (N/'initial-skin-fem-nodes.csv',D/'generated/nodes.csv',D/'generated/tetrahedra.csv',N/'execution-receipt.json')},'nodes':len(p),'tetrahedra':len(t),'positive_node_mass_count':int((mass>0).sum()),'node_mass_min_kg':float(mass.min()),'compiled_skin_plus_fat_mass_kg':float(mass.sum()),'source_fixed_skin_plus_fat_debit_kg':3.4525707761420956+10.487108964465744,'mass_difference_kg':float(mass.sum()-3.4525707761420956-10.487108964465744),'minimum_det_float64_m3':float(d64.min()),'minimum_det_float32_m3':float(d32.min()),'nonpositive_float64':int((d64<=0).sum()),'nonpositive_float32':int((d32<=0).sum()),'minimum_normalized_float32_det':float(q32.min()),'det_ratio_to_registered_reference_min':float((d64/dr).min()),'det_ratio_to_registered_reference_max':float((d64/dr).max()),'physical_qualification':False,'cross_structure_penetration_not_audited_by_this_check':True}
(N/'compiled-cells-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report));raise SystemExit(0 if report['status'].startswith('PASS') else 1)
