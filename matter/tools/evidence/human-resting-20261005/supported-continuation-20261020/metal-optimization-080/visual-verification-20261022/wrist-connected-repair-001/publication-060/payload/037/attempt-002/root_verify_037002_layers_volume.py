import pathlib,numpy as np,json,hashlib
e=pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");d=e/'bone-clearance-fit-037-componentwise-035affine-002'
a=np.load(d/'candidate.npz');b=np.load(e/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz')
x=a['points_world_f32'].astype(float);x0=b['points'].astype(np.float32).astype(float); t=a['tetrahedra'];mat=a['material_ids'];ring=a['ring_columns']
v=x[t];v0=x0[t]
det=np.einsum('ij,ij->i',v[:,1]-v[:,0],np.cross(v[:,2]-v[:,0],v[:,3]-v[:,0]));det0=np.einsum('ij,ij->i',v0[:,1]-v0[:,0],np.cross(v0[:,2]-v0[:,0],v0[:,3]-v0[:,0]))
pairs=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
q=det/np.max(np.array([np.linalg.norm(v[:,i]-v[:,j],axis=1)for i,j in pairs]),axis=0)**3
out={}
for label,oo,ii in [('skin',ring,ring+2411),('fat',ring+2411,ring+4822)]:
 n=x0[oo]-x0[ii];L=np.linalg.norm(n,axis=1);n/=L[:,None];proj=np.einsum('ij,ij->i',x[oo]-x[ii],n)*1000;floor=np.minimum(.2*L*1000,.1)
 out[label]={'min_projection_mm':float(proj.min()),'min_floor_mm':float(floor.min()),'strict_floor_violations':int((proj<floor).sum()),'min_slack_mm':float((proj-floor).min()),'min_fraction_of_reference_projection':float((proj/(L*1000)).min())}
vol={}
for m in np.unique(mat):
 old=float(det0[mat==m].sum()/6);new=float(det[mat==m].sum()/6);vol[str(int(m))]={'baseline_m3':old,'candidate_m3':new,'relative_change':new/old-1,'fixed_reference_density_mass_change_kg':(new-old)*(1000 if m==1 else 900)}
result={'status':'POSITIVE_GEOMETRY_CANDIDATE_NOT_ADMITTED','candidate_sha256':hashlib.sha256((d/'candidate.npz').read_bytes()).hexdigest(),'cells':len(t),'nonpositive':int((det<=0).sum()),'q_below_1e_minus_5':int((q<1e-5).sum()),'q_below_construction_1_25e_minus_5_rows':np.flatnonzero(q<1.25e-5).tolist(),'min_q':float(q.min()),'layer_checks':out,'material_volume_mass_diagnostic':vol,'mass_mapping_regenerated':False,'thickness_physiology_qualified':False}
assert result['nonpositive']==0 and result['q_below_1e_minus_5']==0 and all(c['strict_floor_violations']==0 for c in out.values())
(e/'root-review-037002-layers-volume.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
