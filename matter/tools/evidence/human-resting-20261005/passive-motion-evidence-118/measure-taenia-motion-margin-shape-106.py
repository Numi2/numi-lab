from pathlib import Path
import hashlib,json,numpy as np
from mathutils.bvhtree import BVHTree

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'taenia-motion-margin-shape-106';out.mkdir(exist_ok=False)
source=E/'taenia-mesocolica-exact-weld-001/source-component.npz';candidate=E/'taenia-motion-margin-audit-103/surface-457.npz'
def read(p):
    d=np.load(p);return d['vertices'],d['faces']
v,f=read(source);w,g=read(candidate)
def distance(a,b,c):
    tree=BVHTree.FromPolygons(b.tolist(),c.tolist(),all_triangles=True)
    values=np.asarray([tree.find_nearest(tuple(map(float,x)))[3] for x in a])
    return {'max_m':float(values.max()),'rms_m':float(np.sqrt(np.mean(values*values))),'p99_m':float(np.quantile(values,.99)),'sample_count':len(values)}
def volume(v,f):
    p=v[f].astype(float);p-=p.mean((0,1));return float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Final candidate reference-shape samples; all vertices and triangle centroids in both directions. This is not a global Hausdorff certificate.',
        'driver_sha256':sha(__file__),'source_sha256':sha(source),'candidate_sha256':sha(candidate),
        'source_to_candidate':distance(np.concatenate([v,v[f].mean(1)]),w,g),
        'candidate_to_source':distance(np.concatenate([w,w[g].mean(1)]),v,f),
        'source_volume_m3':volume(v,f),'candidate_volume_m3':volume(w,g),
        'sampled_distance_bound_m':.0005,'relative_volume_bound':.04}
report['passed']=max(report[k]['max_m'] for k in ['source_to_candidate','candidate_to_source'])<=.0005 and abs(report['candidate_volume_m3']/report['source_volume_m3']-1)<=.04
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
assert report['passed']
