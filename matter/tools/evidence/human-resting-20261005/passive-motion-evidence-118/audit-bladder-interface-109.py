from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'bladder-interface-audit-109';out.mkdir(exist_ok=False)
source=E/'bladder-interface-conditioned-107/surface-462.npz';d=np.load(source)
v=d['vertices'];f=d['faces'];tri=v[f].astype(float);fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=np.zeros_like(v,dtype=float)
for k in range(3):np.add.at(normals,f[:,k],fn)
normals/=np.linalg.norm(normals,axis=1)[:,None]
prepared={462:(np.column_stack((v,normals)),f.copy(),{i:[i] for i in range(len(f))},{})}
helper=E/'passive-interface-helper-069.py';spec=importlib.util.spec_from_file_location('helper',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
quality=module.improve_registered_surface_sliver_faces(prepared,minimum_altitude_m=1.2e-6,maximum_nonplanarity_m=1e-5)
v6,f,_,_=prepared[462];candidate=out/'surface-462.npz';np.savez_compressed(candidate,vertices=v6[:,:3].astype('<f4'),normals=v6[:,3:].astype('<f4'),faces=f.astype('<i4'))
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools');import accepted_mrvpack_surface_audit as audit
predicate=Path('/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py');pred=audit.predicate_module(predicate)
base=E/'passive-neighbor-depth-062';data=np.load(base/'source-surfaces.npz');meta=json.loads((base/'input.json').read_text())
meshes={sid:(data[f'v{sid}'],data[f'f{sid}']) for sid in meta['stable_ids']}
for sid in [3,13]:
    d=np.load(E/f'passive-organ-interface-conditioned-071/surface-{sid}.npz');meshes[sid]=(d['vertices'],d['faces'])
d=np.load(E/'taenia-motion-margin-audit-103/surface-457.npz');meshes[457]=(d['vertices'],d['faces'])
meshes[462]=(v6[:,:3].astype('<f4'),f)
scale=audit.coordinate_lattice_scale(float(c) for v,f in meshes.values() for p in v for c in p)
records={}
def rec(sid):
    if sid not in records:
        v,f=meshes[sid];points=[tuple(audit.lattice_integer(float(c),scale) for c in p) for p in v];records[sid]=pred._records(points,f.tolist())
    return records[sid]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Exact static self and all passive-neighbor intersections for inferred bladder source candidate; crossing classification and native cycle remain separate.',
        'driver_sha256':sha(__file__),'source_sha256':sha(source),'helper_sha256':sha(helper),'predicate_sha256':sha(predicate),
        'quality':quality,'candidate_sha256':sha(candidate),'self':None,'pairs':[]}
start=time.monotonic();report['self']=pred._audit_pair(rec(462),rec(462),same_surface=True)
print('self',report['self']['count'],flush=True)
v,_=meshes[462]
for sid,(w,g) in sorted(meshes.items()):
    if sid==462:continue
    if np.any(v.max(0)<w.min(0)) or np.any(w.max(0)<v.min(0)):
        result={'count':0,'aabb_disjoint':True,'audit_complete':True}
    else:
        try:result=pred._audit_pair(rec(462),rec(sid),same_surface=False)
        except ValueError as error:result={'invalid_geometry':str(error),'audit_complete':False}
    report['pairs'].append({'ids':[462,sid],**result});(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if result.get('count') or result.get('invalid_geometry'):print(sid,result.get('count'),result.get('invalid_geometry'),flush=True)
report['elapsed_seconds']=time.monotonic()-start;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',report['elapsed_seconds'],flush=True)
