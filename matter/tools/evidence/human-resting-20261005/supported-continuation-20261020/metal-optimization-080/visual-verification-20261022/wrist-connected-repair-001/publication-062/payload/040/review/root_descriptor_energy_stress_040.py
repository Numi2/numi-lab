from pathlib import Path
import numpy as np,csv,json,hashlib,re
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");p=e/'bone-clearance-fit-040-shared-columns/attempt-004';z=np.load(p/'candidate.npz');X=z['points_world_f32'].astype(float);T=z['tetrahedra'];Ref=np.empty_like(X)
for r in csv.DictReader((e/'root-reference-relative-j-062/source-preflight/identity-reference-points.csv').open()):Ref[int(r['local_volume_node'])]=[float(r[c+'_m']) for c in 'xyz']
def D(P):
 V=P[T];return np.stack([V[:,k]-V[:,0] for k in [1,2,3]],axis=2)
F=D(X)@np.linalg.inv(D(Ref));J=np.linalg.det(F);valid=J>0
report={}
for mid,name,filename in [(1,'skin','human_skin_dermis_frequency_specific_effective.nmatter'),(2,'fat','human_subcutaneous_adipose_low_strain_effective.nmatter')]:
 file=Path("/Users/n/numi-human-aortic-contact-opt-20261019")/'matter/materials'/filename;txt=file.read_text()
 mu=float(re.search(r'parameter shear : MPa = ([\d.e+-]+)',txt)[1])*1e6;bulk=float(re.search(r'parameter bulk : MPa = ([\d.e+-]+)',txt)[1])*1e6
 ix=np.flatnonzero((z['material_ids']==mid)&valid);ff=F[ix];jj=J[ix];ll=np.log(jj);I1=np.sum(ff*ff,axis=(1,2))
 W=.5*mu*(I1-3-2*ll)+.5*bulk*ll*ll
 invt=np.linalg.inv(ff).transpose(0,2,1);P=mu*(ff-invt)+bulk*ll[:,None,None]*invt
 bad=ix[W>5e6];stress=np.max(np.abs(P),axis=(1,2));stressbad=ix[stress>5e6]
 report[name]={'descriptor_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'mu_Pa':mu,'bulk_Pa':bulk,'energy_max_J_per_m3':float(W.max()),'energy_above_5e6_count':len(bad),'energy_violation_local_volume_rows':bad.tolist(),'first_Piola_max_abs_Pa':float(stress.max()),'first_Piola_above_5e6_count':len(stressbad),'first_Piola_violation_local_volume_rows':stressbad.tolist(),'nonpositive_J_count':int(np.sum((z['material_ids']==mid)&~valid))}
r={'status':'DESCRIPTOR_EQUATION_OFFLINE_DIAGNOSTIC_NOT_NATIVE_ADMISSION','candidate_sha256':hashlib.sha256((p/'candidate.npz').read_bytes()).hexdigest(),'materials':report,'limits_Pa_or_J_per_m3':5e6,'limitation':'Evaluates stated isotropic Neo-Hookean expressions in Float64 from F32 world points and owner F64 reference; not bitwise Metal or mixed-pressure stress parity.'}
(e/'root-descriptor-energy-stress-040-attempt-004.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

