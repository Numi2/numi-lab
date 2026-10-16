#!/usr/bin/env python3
"""Corrected source555 ideal-FD assembly: apply deformation to paper and glue tets."""
import sys, pathlib, numpy as np, json, hashlib
Q=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
OUT=Q/'analysis/native555-global-ideal-fd-glue-corrected-001'
OUT.mkdir(parents=True,exist_ok=True)
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
REF=Q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools'
sys.path[:0]=[str(PREP),str(REF)]
import strict_layers
CAP=Q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture'
ARENA=Q/'analysis/native555-actual19-material-replay-001/environment-0/arrays.npz'
OLD=Q/'analysis/native555-global-ideal-fd-002/assembled_fd.npz'
OLDREPORT=Q/'analysis/native555-global-ideal-fd-002/report.json'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=np.load(ARENA); old=np.load(OLD)
header,nall,tall,_=strict_layers.geometry.capture_records(CAP)
t=sorted((x for x in tall if x['environment']==0),key=lambda x:x['tet_id'])
nodes=sorted((x for x in nall if x['environment']==0),key=lambda x:x['node_id'])
material=np.asarray(a['material']); glue=np.flatnonzero(material==2)
ids=np.asarray(a['tet_nodes'],np.int64); F0=np.asarray(a['F_center'],float)
dt=float(np.float32(header['dt_seconds']))
H=np.stack([strict_layers.f32(x['center']['F_direction_local_per_s']) for x in t])
rot=np.stack([strict_layers.geometry.rotation_from_xyzw(strict_layers.f32(x['material_frame_quaternion_xyzw'])) if x['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]==1 else np.eye(3) for x in t])
inv=np.stack([strict_layers.f32(x['inverse_rest_rows_per_m']) for x in t])
vol=strict_layers.f32([x['rest_volume_m3'] for x in t])
free=np.asarray(a['free_node_mask'],bool)
params=np.asarray(header['parameters_by_environment'][0],np.float32).astype(float)
spec={m['index']:m for m in header['materials']}
mu=np.full(len(glue),params[spec[2]['parameter_offset']+1])
lam=np.full(len(glue),params[spec[2]['parameter_offset']+2])
def glue_gather(Fg):
    local=strict_layers.ref.glue_stress(Fg,mu,lam)
    world=np.zeros((len(t),3,3),float)
    world[glue]=local@np.swapaxes(rot[glue],-1,-2)
    eforce=strict_layers.forces(world,inv,vol)
    return strict_layers.gather(eforce,ids,len(nodes))
g0=glue_gather(F0[glue])
center_diff=float(np.max(np.abs(strict_layers.ref.glue_stress(F0[glue],mu,lam)-a['s0_reference_stress'][glue])))
Aaction=np.asarray(old['A_action']); expected=-Aaction
original_fd={}; corrected_fd={}; corrected_gap={}; glue_secants={}; rows=[]
for power in (8,10,12,14,16,18,20,22,24,26):
    eps=2.0**(-power)
    dplus=dt*(glue_gather(F0[glue]+eps*dt*H[glue])-g0)
    dminus=dt*(glue_gather(F0[glue]-eps*dt*H[glue])-g0)
    Rp=np.asarray(old[f'p{power}_plus_R'])+dplus
    Rm=np.asarray(old[f'p{power}_minus_R'])+dminus
    sec=(Rp-Rm)/(2*eps); gsec=(dplus-dminus)/(2*eps); gap=sec-expected
    original=np.asarray(old[f'p{power}_secant'])
    original_fd[power]=original; corrected_fd[power]=sec; corrected_gap[power]=gap; glue_secants[power]=gsec
    rows.append({'power':power,'epsilon':eps,
      'original_incomplete_secant_l2_free':float(np.linalg.norm(original[free])),
      'exact_glue_secant_l2_free':float(np.linalg.norm(gsec[free])),
      'corrected_secant_l2_free':float(np.linalg.norm(sec[free])),
      'reference_dR_l2_free':float(np.linalg.norm(expected[free])),
      'corrected_gap_l2_free':float(np.linalg.norm(gap[free])),
      'corrected_relative_error_to_reference_A':float(np.linalg.norm(gap[free])/np.linalg.norm(expected[free])),
      'max_abs_corrected_gap_component':float(np.max(np.abs(gap[free])))})
# Per-node p20 decomposition of the invalid earlier operator comparison.
gap0=original_fd[20]-expected; gsec=glue_secants[20]; gap=corrected_gap[20]
nm=lambda x:np.linalg.norm(x,axis=1)
node_rank=np.argsort(np.where(free,nm(gap0),-1))[::-1][:12]; top=[]
for ni in node_rank:
    incident=np.flatnonzero(np.any(ids==ni,axis=1))
    top.append({'node':int(ni),'original_gap_xyz':gap0[ni].tolist(),
      'omitted_glue_endpoint_dR_xyz':gsec[ni].tolist(),
      'corrected_gap_xyz':gap[ni].tolist(),'original_gap_norm':float(nm(gap0)[ni]),
      'corrected_gap_norm':float(nm(gap)[ni]),
      'incident_paper_tets':[int(i) for i in incident if material[i]!=2],
      'incident_glue_tets':[int(i) for i in incident if material[i]==2]})
selected=set()
for ni in node_rank[:5]:
    selected.update(int(i) for i in np.flatnonzero(np.any(ids==ni,axis=1)) if material[i]!=2)
np.savez_compressed(OUT/'corrected_actions.npz',
  p20_original_secant=original_fd[20],p20_corrected_secant=corrected_fd[20],
  p20_original_gap=gap0,p20_corrected_gap=gap,p20_glue_dR=glue_secants[20],
  reference_dR=expected,free_mask=free,selected_paper_tets=np.asarray(sorted(selected),np.int32))
report={'schema':'numi.cardboard.native555-global-ideal-fd-glue-corrected.v1',
 'classification':'CPU audit correction only; prior paper-only endpoint FD is invalid for full operator comparison; no solver or physics change; corrected remainder is not a qualification',
 'capture_sha256':sha(CAP),'source_prior_run_report_sha256':sha(OLDREPORT),
 'source_prior_run_arrays_sha256':sha(OLD),'material_replay_arrays_sha256':sha(ARENA),
 'preparation_sha256':sha(PREP/'strict_layers.py'),
 'fp64_reference_sha256':sha(REF/'cardboard_bilinear_fp64_reference.py'),
 'dt_seconds':dt,'paper_tets':int(np.sum(material!=2)),'glue_tets':int(len(glue)),
 'directions':'same captured F_direction_local_per_s for all 1920 tets and same delta-v; glue has no plastic projection, but its neo-Hookean stress must still be evaluated at F0 +/- epsilon*dt*H.',
 'glue_center_stress_max_abs_difference_vs_existing_reference':center_diff,
 'old_run_bug':'run.py perturbed endpoint F only on paper rows, but formed the all-tet tangent/action including glue rows. Endpoint residuals held glue stress fixed while the reference A included glue stiffness.',
 'correction':'For each old endpoint residual, add dt*[g_glue(F0 + sign*eps*dt*H)-g_glue(F0)] to gathered internal-force residual using captured exact glue law and geometry. No paper endpoint or projection was altered.',
 'corrected_rows':rows,
 'p20_norms':{'original_gap_l2_free':float(np.linalg.norm(gap0[free])),
  'omitted_glue_endpoint_dR_l2_free':float(np.linalg.norm(gsec[free])),
  'corrected_gap_l2_free':float(np.linalg.norm(gap[free])),
  'relative_corrected_gap_to_A':float(np.linalg.norm(gap[free])/np.linalg.norm(expected[free])),
  'cosine_original_gap_vs_omitted_glue':float(np.sum(gap0[free]*gsec[free])/(np.linalg.norm(gap0[free])*np.linalg.norm(gsec[free]))),
  'corrected_gap_fraction_of_original_gap':float(np.linalg.norm(gap[free])/np.linalg.norm(gap0[free]))},
 'p20_top_node_map':top,'selected_incident_paper_tets_for_followup':sorted(selected),
 'projection_gates':'Inherited from prior run: endpoint rows passed its original project criteria. Separate tight run 003 did not admit a complete endpoint set at its stricter diagnostic target, so corrected endpoint FD remains exploratory.'}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
manifest={'schema':'numi.cardboard.native555-global-ideal-fd-glue-corrected-files.v1',
 'files_sha256':{x.name:sha(x) for x in sorted(OUT.iterdir()) if x.is_file() and x.name!='manifest.json'}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'out':str(OUT),'rows':rows,'p20':report['p20_norms'],'selected':sorted(selected)},separators=(',',':')))
