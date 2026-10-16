#!/usr/bin/env python3
"""Map the omitted glue endpoint derivative to per-tet nodal residual contributions."""
import pathlib,sys,json,hashlib,numpy as np
Q=pathlib.Path('/Users/n/cardboard-implementation-20261014-host-macmini')
OUT=Q/'analysis/native555-glue-node-tet-attribution-001';OUT.mkdir(parents=True,exist_ok=True)
CAP=Q/'solver/control19-newton3-resume-diagnostic-555/diagnostics/capture-operator-fd-capture/capture'
ARENA=Q/'analysis/native555-actual19-material-replay-001/environment-0/arrays.npz'
CORR=Q/'analysis/native555-global-ideal-fd-glue-corrected-001'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
REF=Q/'snapshots/source-stable-green-reference-206-buildfix-002/matter/tools'
sys.path[:0]=[str(PREP),str(REF)]
import strict_layers
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=np.load(ARENA); corr=np.load(CORR/'corrected_actions.npz')
h,nall,tall,_=strict_layers.geometry.capture_records(CAP)
t=sorted((x for x in tall if x['environment']==0),key=lambda x:x['tet_id'])
nodes=sorted((x for x in nall if x['environment']==0),key=lambda x:x['node_id'])
material=np.asarray(a['material']); ids=np.asarray(a['tet_nodes'],int); free=np.asarray(a['free_node_mask'],bool)
H=np.stack([strict_layers.f32(x['center']['F_direction_local_per_s']) for x in t])
C=np.asarray(a['reference_C'],float)
dP=np.einsum('tij,tj->ti',C,H.reshape(len(t),9)).reshape(-1,3,3)
rot=np.stack([strict_layers.geometry.rotation_from_xyzw(strict_layers.f32(x['material_frame_quaternion_xyzw'])) if x['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]==1 else np.eye(3) for x in t])
inv=np.stack([strict_layers.f32(x['inverse_rest_rows_per_m']) for x in t])
vol=strict_layers.f32([x['rest_volume_m3'] for x in t])
dt=float(np.float32(h['dt_seconds']))
ef=strict_layers.forces(dP@np.swapaxes(rot,-1,-2),inv,vol)
gl=np.flatnonzero(material==2)
nodeids=[856,854,564,145,443]
orig=corr['p20_original_gap']; corrected=corr['p20_corrected_gap']; data=[]
for ni in nodeids:
    terms=[]
    for ti in gl:
        slots=np.flatnonzero(ids[ti]==ni)
        for slot in slots:
            dR=dt*dt*ef[ti,slot]
            terms.append({'tet_id':int(ti),'slot':int(slot),'dR_glue_xyz':dR.tolist(),'omitted_gap_xyz':(-dR).tolist(),'omitted_gap_l2':float(np.linalg.norm(dR))})
    terms.sort(key=lambda x:x['omitted_gap_l2'],reverse=True)
    vec=sum((np.asarray(x['omitted_gap_xyz']) for x in terms),np.zeros(3))
    data.append({'node':ni,'original_gap_xyz':orig[ni].tolist(),'original_gap_l2':float(np.linalg.norm(orig[ni])),
      'sum_glue_omission_xyz':vec.tolist(),'sum_glue_omission_l2':float(np.linalg.norm(vec)),
      'corrected_gap_xyz':corrected[ni].tolist(),'corrected_gap_l2':float(np.linalg.norm(corrected[ni])),
      'top_glue_tet_terms':terms[:6]})
report={'schema':'numi.cardboard.native555-glue-node-tet-attribution.v1',
 'classification':'per-tet decomposition of the invalid paper-only endpoint comparison; no solver or physics changes',
 'capture_sha256':sha(CAP),'material_replay_arrays_sha256':sha(ARENA),
 'prior_corrected_report_sha256':sha(CORR/'report.json'),'prior_corrected_actions_sha256':sha(CORR/'corrected_actions.npz'),
 'all_tet_center_tangent_sha256':sha(REF/'cardboard_bilinear_fp64_reference.py'),
 'formula':'For a glue tet, endpoint R derivative is dt^2 times its captured-frame internal nodal force from (C_t H_t). The earlier run held glue F fixed; its comparison gap therefore contains the negative of this derivative. Listed terms sum by captured node slot.',
 'nodes':data}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
manifest={'schema':'numi.cardboard.native555-glue-node-tet-attribution-files.v1','files_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='manifest.json'}}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps({'out':str(OUT),'nodes':[{'node':x['node'],'original':x['original_gap_l2'],'glue_sum':x['sum_glue_omission_l2'],'corrected':x['corrected_gap_l2'],'terms':[(t['tet_id'],t['omitted_gap_l2']) for t in x['top_glue_tet_terms'][:4]]} for x in data]},separators=(',',':')))
