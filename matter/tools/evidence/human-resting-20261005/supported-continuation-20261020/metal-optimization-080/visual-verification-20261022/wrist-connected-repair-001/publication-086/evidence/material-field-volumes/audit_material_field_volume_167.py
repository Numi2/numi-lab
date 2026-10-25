from pathlib import Path
import numpy as np,sys,json,hashlib,time
D=Path('/Users/numi/Work/numi-human-connected-surface-069')
sys.path.insert(0,str(D/'publication-086/canonical/matter/tools'))
from human_skin_fat_layer_reconstruction import positive_linear_tetrahedron_volume_fractions
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
B=D/'connected-volume-extraction-157/true-o-foundation-159'
old=D/'whole-reference-tetgen-084-bounded/connectivity-001/run-001/candidate-tetrahedra-compacted.npz'
def tetvol(p,t):
 x=p[t].astype(float)
 return np.einsum('ij,ij->i',x[:,1]-x[:,0],np.cross(x[:,2]-x[:,0],x[:,3]-x[:,0]))/6
def boundvol(p,f):
 x=p[f].astype(float)
 return float(np.einsum('ij,ij->i',x[:,0],np.cross(x[:,1],x[:,2])).sum()/6)
z=np.load(old);v=tetvol(z['reference_points_m'],z['tetrahedra_zero_based']);ids=z['material_ids']
targets={str(int(k)):float(v[ids==k].sum())for k in np.unique(ids)}
r={'status':'EXACT_AFFINE_FIELD_VOLUME_AUDIT_ONLY','physical_or_native_admission':False,
   'native_physical_steps_accepted':0,'reference_domain_sha256':sha(old),'reference_material_volumes_m3':targets,
   'owner_sha256':sha(D/'publication-086/canonical/matter/tools/human_skin_fat_layer_reconstruction.py'),'cases':{}}
for name,cp,fp in [
 ('original',B/'run-003-conversion/true-o-tetgen-foundation.npz',D/'original-foundation-joint-interfaces-166/joint-facets/run-001/joint-interface-facets.npz'),
 ('refined',B/'run-013-conforming-field-refinement/refined-foundation-checkpoint.npz',D/'joint-interface-facet-extraction-164/run-013-joint-facets/run-003/joint-interface-facets.npz')]:
 t0=time.monotonic();z=np.load(cp);P=z['points_world_m'];T=z['tetrahedra_zero_based'];V=tetvol(P,T);assert np.all(V>0)
 zz=np.load(fp);I=zz['phi_inner_snapped_m'];F=zz['phi_deep_snapped_m']
 skin=float(np.dot(V,positive_linear_tetrahedron_volume_fractions(I[T])))
 total=float(np.dot(V,positive_linear_tetrahedron_volume_fractions(F[T])))
 facet=zz['facets_zero_based'];p=zz['points_world_m_f32'];ifs=zz['interface_ids']
 bs=float(V.sum())+boundvol(p,facet[ifs==1]);bt=float(V.sum())+boundvol(p,facet[ifs==2])
 # Affine field integration uses pre-rounding edge intersections; signed
 # facet integration uses emitted F32 points. Report the emission difference.
 r['cases'][name]={'foundation_sha256':sha(cp),'facets_sha256':sha(fp),'total_foundation_volume_m3':float(V.sum()),
    'skin_volume_m3':skin,'fat_volume_m3':total-skin,'combined_volume_m3':total,
    'emitted_surface_skin_volume_m3':bs,'emitted_surface_combined_volume_m3':bt,
    'surface_emission_skin_difference_m3':bs-skin,'surface_emission_combined_difference_m3':bt-total,
    'skin_relative_reference_volume_change':skin/targets['1']-1,'fat_relative_reference_volume_change':(total-skin)/targets['2']-1,
    'seconds':time.monotonic()-t0,'reference_mass_gate_pass':False}
out=D/'layer-field-volume-audit-167/report.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
