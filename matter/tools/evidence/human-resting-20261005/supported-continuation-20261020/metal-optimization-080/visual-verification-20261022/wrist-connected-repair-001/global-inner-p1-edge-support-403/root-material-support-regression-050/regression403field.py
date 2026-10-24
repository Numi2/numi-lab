from pathlib import Path
import importlib.util,json,hashlib,numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=E/'global-inner-p1-edge-support-403/root-material-support-regression-050';D.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=E/'global-inner-p1-field-402-absolute-variable-integration/absolute_field_402.py'
assert sha(source)=='3a3ccbb93efa463f15575c4e9d96aef50576be9784ca1928ec741e3fefd8f446'
spec=importlib.util.spec_from_file_location('field402',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=E/'global-inner-p1-field-405-tet88190-bank-triangle-clip/clip-receipt.json';assert sha(p)=='c7c868cd2b5aaf0c4f62e49d3a2e077758126c8fc73aa2da47b59119dba907e5'
r=json.loads(p.read_text());cp=E/'global-inner-p1-edge-support-403/build-002/candidate.npz';assert sha(cp)=='9c8c23c8229c46494d19f6fc12d6131667b901864dfab21cd1ef7de6a2fc6744'
z=np.load(cp);child=r['bank_triangle_clips'][0]['children'][1];ids=np.asarray(child['tet_node_rows']);x=z['points_world_m'][ids].astype(float);P=np.asarray(child['vertices_world_m'])
matrix=np.stack([x[1]-x[0],x[2]-x[0],x[3]-x[0]],axis=1);b3=np.linalg.solve(matrix,(P-x[0]).T).T;raw=np.column_stack([1-b3.sum(1),b3]);assert raw.min()>-1e-10
B=np.maximum(raw,0);B/=B.sum(1)[:,None]
T=np.asarray([[0,1,2,3]],np.int64);O=np.isin(ids,z['physical_outer_node_ids']);cost=np.full(4,abs(np.linalg.det(matrix))/24)
field,meta=m.solve_absolute_obstacle_field(np.zeros(4),T,np.zeros(len(B),np.int64),B,np.flatnonzero(O),np.zeros(O.sum()),cost,.341)
oldB=np.asarray([[.7473772446,0.,.2526227554,0.]]);oldids=np.asarray([9451,9471,9473,27269]);oldO=np.isin(oldids,z['physical_outer_node_ids'])
oldfield,oldmeta=m.solve_absolute_obstacle_field(np.zeros(4),T,np.zeros(1,np.int64),oldB,np.flatnonzero(oldO),np.zeros(oldO.sum()),np.ones(4),.341)
passed=field is not None and meta['pass'] and oldfield is None and oldmeta['status']=='fixed_support_conflict'
rep={'pass':bool(passed),'baseline_398_witness':oldmeta,'refined_403_actual_clipped_triangle':meta,'global_child_row':child['tet_row_403'],'typed_bank_key':'bank_0004','triangle_row':2562,'original_node_ids':ids.tolist(),'physical_O_mask':O.tolist(),'raw_barycentric_min':float(raw.min()),'construction_barycentric_clamp_max_change':float(np.max(abs(B-raw))),'local_solution':None if field is None else field.tolist(),'target':.341,'constraints_solved':len(B),'source402_sha256':sha(source),'source405_receipt_sha256':sha(p),'candidate403_sha256':sha(cp),'scope':'Only one actual clipped source triangle and its child tetrahedron; no full field or native admission.','native_admission':False}
(D/'result.json').write_text(json.dumps(rep,indent=2)+'\n');np.savez_compressed(D/'local-regression-arrays.npz',raw_barycentric=raw,construction_barycentric=B,local_field=field)
print(json.dumps(rep));assert passed

