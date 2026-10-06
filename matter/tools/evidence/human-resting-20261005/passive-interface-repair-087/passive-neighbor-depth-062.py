"""Blender CPU diagnostic of source passive interfaces; not collision acceptance."""
from pathlib import Path
import hashlib, json, sys, time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

E=Path('/Users/n/numi-human-resting-evidence-20261005')
folder=E/'passive-neighbor-depth-062'
input_file=folder/'source-surfaces.npz'
metadata=json.loads((folder/'input.json').read_text())
data=np.load(input_file)
cache={}
for sid in metadata['stable_ids']:
    v=data[f'v{sid}'].astype(np.float64); f=data[f'f{sid}']
    p=v[f]
    volume=float(np.sum(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])))/6)
    cache[sid]=(v,f,BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True),volume)

def directional(a,b):
    v,_,_,_=cache[a]; w,_,tree,volume=cache[b]
    selected=np.flatnonzero(np.all((v>=w.min(0)) & (v<=w.max(0)),axis=1))
    depths=[]
    for i in selected:
        point=Vector(v[i]); hit=tree.find_nearest(point)
        if hit[0] is None: continue
        location,normal,face,distance=hit
        sign=(point-location).dot(normal)*(1 if volume>0 else -1)
        if sign<0:
            depths.append((float(distance),int(i),int(face)))
    depths.sort(reverse=True)
    return {'from':a,'into':b,'aabb_vertices':len(selected),
            'nearest_normal_negative_vertices':len(depths),
            'approx_max_depth_m':depths[0][0] if depths else 0,
            'deepest_witnesses':[{'distance_m':d,'source_vertex':i,'target_face':f,
                                 'point_m':v[i].tolist()} for d,i,f in depths[:8]]}

start=time.monotonic()
report={'scope':'Source geometry nearest-triangle signed-distance diagnostic. Sign uses the local nearest outward triangle normal, not a robust containment predicate. Not a collision clearance or anatomical interface waiver.',
        'source':metadata,'input_npz_sha256':hashlib.sha256(input_file.read_bytes()).hexdigest(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mesh_signed_volume_m3':{str(s):row[3] for s,row in cache.items()},'pairs':[]}
for a,b in metadata['pairs']:
    row={'ids':[a,b],'names':[metadata['source_id_map'][str(x)]['name'] for x in (a,b)],
         'members':[metadata['source_id_map'][str(x)]['source_member'] for x in (a,b)],
         'directions':[directional(a,b),directional(b,a)]}
    report['pairs'].append(row)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'ids':[a,b],'max_depth_m':[x['approx_max_depth_m'] for x in row['directions']]}),flush=True)
report['elapsed_seconds']=time.monotonic()-start
(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',report['elapsed_seconds'],flush=True)
