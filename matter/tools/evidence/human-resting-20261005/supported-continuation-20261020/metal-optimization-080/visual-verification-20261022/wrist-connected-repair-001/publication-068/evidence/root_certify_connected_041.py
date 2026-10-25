from pathlib import Path
import hashlib,json,csv,sys,itertools
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
D=E/'connected-volume-continuation-041/iteration-005'
sys.path.insert(0,str(E/'connected-volume-repair-036/canonical/matter/tools'))
import human_skin_fat_connected_repair as m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
C=D/'candidate.npz';a=np.load(C);X=a['points_world_f32'];T=a['tetrahedra'];material=a['material_ids']
AP=E/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz';ap=np.load(AP)
RF=E/'bone-clearance-fit-040-shared-columns/attempt-004/fit-input.npz';R=np.load(RF)['reference_points_m']
assert np.array_equal(T,ap['tetrahedra']) and np.array_equal(material,ap['material_ids'])
J,G,q,Q=m.cell_values_gradients(X,R,T)
J32,_,_,_=m.cell_values_gradients(X,R.astype(np.float32),T)
A=ap['points'].astype(np.float32).astype(float)
layers=[]
for l in [0,1]:
 aa=l*2411;bb=aa+2411;d=A[aa:aa+2411]-A[bb:bb+2411];length=np.linalg.norm(d,axis=1);normal=d/length[:,None]
 ratio=np.einsum('ij,ij->i',X[aa:aa+2411].astype(float)-X[bb:bb+2411],normal)/length
 layers.append({'layer':l,'projected_thickness_ratio_percentiles':np.percentile(ratio,[0,1,50,99,100]).tolist(),'violations_with_1um_tolerance':int(np.sum((ratio*length<.5*length-1e-6)|(ratio*length>1.5*length+1e-6)))})
faces=np.concatenate([np.sort(T[:,[j for j in range(4)if j!=i]],axis=1)for i in range(4)])
uf,counts=np.unique(faces,axis=0,return_counts=True);boundary=uf[counts==1]
edges=np.concatenate([boundary[:,[0,1]],boundary[:,[0,2]],boundary[:,[1,2]]]);_,ec=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
edgeaudit=json.loads((D/'exact-audit/tet-edge-report.json').read_text())
rows=list(csv.DictReader((D/'native-query/source-preflight/all-bank-containment.csv').open()))
r={'status':'STATIC_CONNECTED_CANDIDATE_NOT_ADMITTED','accepted_physical_steps':0,'anatomy_qualified':False,'physical_qualification':False,'candidate_sha256':sha(C),'original_reference_sha256':sha(RF),'baseline_sha256':sha(AP),'script_sha256':sha(Path(__file__)),'nodes':len(X),'cells':len(T),'nonfinite_current_coordinates':int((~np.isfinite(X)).sum()),'minimum_emitted_quality':float(q.min()),'nonpositive_cells':int((q<=0).sum()),'quality_below_1e_5':int((q<1e-5).sum()),'J_authored_reference_range':[float(J.min()),float(J.max())],'J_cast_F32_reference_range':[float(J32.min()),float(J32.max())],'J_cast_F32_reference_outside_035_25':int(((J32<.35)|(J32>2.5)).sum()),'thickness':layers,'tet_faces_incidence_gt2':int((counts>2).sum()),'boundary_triangles':len(boundary),'boundary_edges_not_incidence2':int((ec!=2).sum()),'edge_bank_crossing_groups':edgeaudit['edge_bank_groups_with_hits'],'crossing_edges':edgeaudit['edges_with_hits'],'contained_nodes':len({r['local_node']for r in rows if r['classification']=='inside'}),'ambiguous_queries':sum(int(r['ambiguous'])for r in rows),'cumulative_displacement_mm_percentiles':np.percentile(np.linalg.norm(X.astype(float)-A,axis=1)*1000,[0,50,90,95,99,100]).tolist(),'unchanged_original_reference':True,'unchanged_material_ids':True,'unchanged_tetrahedra':True,'mass_note':'Current deformation does not reauthor rest geometry or assign rho times deformed volume. Actual compiled per-cell and donor mass admission remain separate.','limitations':['Exact edge/bank and contained-node audits do not certify full tet-solid or surface clearance.','Zero cell inversions here is static Float32 geometric consistency; no physical step, full material response, detailed294 transfer, supported-pose sweep or viewer confirmation is admitted.','Cast-F32 reference determinant ratio is diagnostic; compiled inverseRest arithmetic remains a separate native check.']}
(E/'root-connected-041-review.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
