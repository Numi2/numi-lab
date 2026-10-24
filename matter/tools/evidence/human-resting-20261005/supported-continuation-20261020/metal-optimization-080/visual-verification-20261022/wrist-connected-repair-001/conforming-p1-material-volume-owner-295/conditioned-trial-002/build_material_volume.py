from pathlib import Path
import sys,json,time,hashlib,numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');O=E/'conforming-p1-material-volume-owner-295/conditioned-trial-002'
sys.path[:0]=[str(O/'tools'),str(O/'tools')]
from human_skin_p1_material_volume import split_material_bands,material_role_faces,exterior_source_face_rows
from human_skin_material_boundary import certify_material_boundaries
p=E/'tissue-aware-commonfield-280-cached-output/field-cache-left.npz'
start=time.monotonic()
tool_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (O/'tools').glob('*.py')}
with np.load(p,allow_pickle=False) as z:Z={k:z[k].copy() for k in ['field_points_world_m','field_tetrahedra','phi0','physical_outer_faces','temporary_cap_faces']}
r=split_material_bands(Z['field_points_world_m'],Z['field_tetrahedra'],Z['phi0'],0.06642863100720746,0.34079955258503536)
roles=material_role_faces(r['tetrahedra'],r['material_ids'])
srcfaces=np.vstack([Z['physical_outer_faces'],Z['temporary_cap_faces']])
lineage=exterior_source_face_rows(r,roles,srcfaces)
construction=[];physical={}
for k,faces in roles.items():
 mask=lineage[k]>=len(Z['physical_outer_faces'])
 construction.extend(faces[mask]);physical[k]=faces[~mask]
construction=np.asarray(construction,np.int64).reshape(-1,3)
x=r['points_world_m'][r['tetrahedra']].astype(float);e=x[:,1:]-x[:,:1]
det=np.einsum('ij,ij->i',e[:,0],np.cross(e[:,1],e[:,2]))
longest=np.max(np.stack([np.linalg.norm(x[:,i]-x[:,j],axis=1) for i in range(4) for j in range(i+1,4)]),axis=0)
quality=det/longest**3
report={'tool_hashes':tool_hashes,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_cache_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'points':len(r['points_world_m']),'tetrahedra':len(r['tetrahedra']),'minimum_float32_quality':float(quality.min()),'nonpositive_cells':int((det<=0).sum()),'below_owner_quality_floor':int((quality<1e-5).sum()),'roles':{k:len(v) for k,v in roles.items()},'construction_facets':len(construction),'material_volume_m3':{str(k):float(det[r['material_ids']==k].sum()/6) for k in [1,2]},'native_admission':False,'scope':'retained left280 construction-method diagnostic; source has retained anatomy crossings; not current wholebody candidate'}
try:
 cert=certify_material_boundaries(r['points_world_m'],r['tetrahedra'],r['material_ids'],**physical,construction_faces=construction,minimum_quality=1e-5)
 report['owner_quality_boundary_pass']=True
except ValueError as exc:report['owner_quality_boundary_pass']=False;report['failure']=str(exc)
out=O/'left280-conforming-material-volume.npz'
np.savez_compressed(out,**{k:v for k,v in r.items() if k!='native_admission'},**roles,construction_faces=construction,**{k+'_source_parent_rows':v for k,v in lineage.items()})
report['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();report['elapsed_s']=time.monotonic()-start
(O/'left280-construction-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)

