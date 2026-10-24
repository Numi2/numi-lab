from pathlib import Path
import numpy as np,csv,json,sys,hashlib
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");d=e/'bone-clearance-fit-040-shared-columns';sys.path.insert(0,str(d))
from neo_hookean_fit_support import prepare_reference_tetrahedra,evaluate_neo_hookean_batch,MATERIALS_PA
z=np.load(d/'attempt-004/candidate.npz');T=z['tetrahedra'];X=z['points_world_f32'].astype(float);Ref=np.empty_like(X)
for r in csv.DictReader((e/'root-reference-relative-j-062/source-preflight/identity-reference-points.csv').open()):Ref[int(r['local_volume_node'])]=[float(r[c+'_m']) for c in 'xyz']
rows=np.array([0,2107,4299,11828,13057,13630,15530]);V=X[T[rows]];R=Ref[T[rows]];mid=z['material_ids'][rows]
ref=prepare_reference_tetrahedra(R);a=evaluate_neo_hookean_batch(V,ref,mid,T[rows]%2411)
Dm=np.stack([R[:,k]-R[:,0] for k in [1,2,3]],axis=2);Ds=np.stack([V[:,k]-V[:,0] for k in [1,2,3]],axis=2);F=Ds@np.linalg.inv(Dm);J=np.linalg.det(F);mu=np.array([MATERIALS_PA[int(m)][0] for m in mid]);K=np.array([MATERIALS_PA[int(m)][1] for m in mid]);ln=np.log(J)
W=.5*mu*(np.sum(F*F,axis=(1,2))-3-2*ln)+.5*K*ln*ln
P=mu[:,None,None]*F+(K*ln-mu)[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
assert a['valid'].all()
assert np.allclose(a['energy_density_pa'],W,rtol=5e-13,atol=1e-7)
assert np.allclose(a['piola_pa'],P,rtol=5e-13,atol=1e-6)
rng=np.random.default_rng(4004);max_energy_rel=0.;max_P_rel=0.
for i in range(len(rows)):
 ids=T[rows[i]]%2411; unique=np.unique(ids);delta=rng.normal(size=(len(unique),3));delta/=np.linalg.norm(delta)
 dv=np.stack([delta[np.flatnonzero(unique==j)[0]] for j in ids]);eps=1e-10
 ap=evaluate_neo_hookean_batch((V[i]+eps*dv)[None],prepare_reference_tetrahedra(R[i:i+1]),mid[i:i+1],ids[None])
 am=evaluate_neo_hookean_batch((V[i]-eps*dv)[None],prepare_reference_tetrahedra(R[i:i+1]),mid[i:i+1],ids[None])
 assert ap['valid'].all() and am['valid'].all()
 D=np.zeros((4,3))
 for slot,col in enumerate(a['jacobian_column_ids'][i]):
  if col>=0:D[slot]=delta[np.flatnonzero(unique==col)[0]]
 we=float(np.sum(a['energy_jacobian_j_per_m'][i]*D))/ref.volume_m3[i]
 wp=float((ap['energy_density_pa'][0]-am['energy_density_pa'][0])/(2*eps))
 pe=np.einsum('ikj,kj->i',a['piola_component_jacobian_pa_per_m'][i],D)
 pp=(ap['piola_components_pa'][0]-am['piola_components_pa'][0])/(2*eps)
 er=abs(we-wp)/max(abs(we),abs(wp),1);pr=np.linalg.norm(pe-pp)/max(np.linalg.norm(pe),np.linalg.norm(pp),1)
 max_energy_rel=max(max_energy_rel,er);max_P_rel=max(max_P_rel,pr)
 assert er<2e-4 and pr<2e-4,(int(rows[i]),er,pr)
out={'status':'PASS_INDEPENDENT_REAL_CELL_ENERGY_PIOLA_AND_DIRECTIONAL_DERIVATIVES','cell_rows':rows.tolist(),'max_directional_energy_relative_error':max_energy_rel,'max_directional_Piola_relative_error':max_P_rel,'helper_sha256':hashlib.sha256((d/'neo_hookean_fit_support.py').read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((d/'attempt-004/candidate.npz').read_bytes()).hexdigest(),'limitation':'Offline descriptor-equation check, not native mixed-field stress parity or physical admission'}
(e/'root-independent-material-helper-040.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

