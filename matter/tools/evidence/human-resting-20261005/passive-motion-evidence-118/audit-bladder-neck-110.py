"""Verify the repaired bladder preserves the specific source prostate interface."""
from pathlib import Path
import hashlib,json,sys,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'bladder-neck-interface-110';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-conforming-source-009/matter/tools');import accepted_mrvpack_surface_audit as audit
predicate=Path('/Users/n/numi-human-neighbor-source-001/src/numilab_human/cardiac_cavity_intersections.py');pred=audit.predicate_module(predicate)
source=E/'passive-neighbor-depth-062/source-surfaces.npz';d=np.load(source)
v,f=d['v462'],d['f462'];w,g=d['v463'],d['f463']
candidate=E/'bladder-interface-audit-109/surface-462.npz';c=np.load(candidate);cv,cf=c['vertices'],c['faces']
scale=audit.coordinate_lattice_scale(float(x) for points in [v,w,cv] for p in points for x in p)
points=lambda vertices:[tuple(audit.lattice_integer(float(x),scale) for x in p) for p in vertices]
vp,wp,cp=points(v),points(w),points(cv)
original=pred._audit_pair(pred._records(vp,f.tolist()),pred._records(wp,g.tolist()),same_surface=False)
report=json.loads((E/'bladder-interface-audit-109/report.json').read_text());actual=next(p for p in report['pairs'] if p['ids']==[462,463])
def key(points,face):
    p=tuple(points[int(i)] for i in face);return min(p,p[1:]+p[:1],p[2:]+p[:2])
source_faces={key(vp,face):i for i,face in enumerate(f)}
mapped=[]
for a,b in actual['triangle_pairs']:
    assert key(cp,cf[a]) in source_faces,('source neck triangle was modified',a)
    mapped.append([source_faces[key(cp,cf[a])],b])
assert sorted(mapped)==sorted(original['triangle_pairs'])
witnesses=[]
for a,b in actual['triangle_pairs']:
    witnesses += [[float(x)/scale for x in p] for p in pred.triangle_intersection_points(tuple(cp[int(i)] for i in cf[a]),tuple(wp[int(i)] for i in g[b]))]
xyz=np.asarray(witnesses);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result={'scope':'The specific source bladder/prostate interface is retained by exact oriented coordinate-triangle identity. This does not waive other organ pairs or infer a simulated urinary lumen.',
        'driver_sha256':sha(__file__),'predicate_sha256':sha(predicate),'source_npz_sha256':sha(source),
        'candidate_sha256':sha(candidate),'exact_audit_sha256':sha(E/'bladder-interface-audit-109/report.json'),
        'source_members':['FJ3149','FJ3139'],'source_interface_count':original['count'],
        'source_triangle_pairs':original['triangle_pairs'],'candidate_triangle_pairs':actual['triangle_pairs'],
        'candidate_to_original_interface_pairs':mapped,'source_interface_triangles_preserved_exactly':True,
        'witness_bounds_m':[xyz.min(0).tolist(),xyz.max(0).tolist()],
        'distance_from_bladder_inferior_extent_m':[float(xyz[:,1].min()-v[:,1].min()),float(xyz[:,1].max()-v[:,1].min())],
        'distance_from_prostate_superior_extent_m':[float(w[:,1].max()-xyz[:,1].max()),float(w[:,1].max()-xyz[:,1].min())],
        'anatomical_interpretation':'The passive closed bladder envelope meets the prostate at the bladder neck. Source connectivity/inspection layers are retained; urine flow and urethral wall mechanics are not simulated.',
        'anatomical_source':'https://mapkmc.manipal.edu/specimen/anat301/'}
(out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
