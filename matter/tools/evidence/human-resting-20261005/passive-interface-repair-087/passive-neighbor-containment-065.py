"""CPU source-interface diagnostic. Ray parity is exploratory, not exact acceptance."""
from pathlib import Path
import hashlib, json, time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

E=Path('/Users/n/numi-human-resting-evidence-20261005')
folder=E/'passive-neighbor-containment-065'
data=np.load(folder/'source-surfaces.npz')
metadata=json.loads((folder/'input.json').read_text())
cache={}
for sid in metadata['stable_ids']:
    v=data[f'v{sid}'].astype(np.float64);f=data[f'f{sid}']
    cache[sid]=(v,f,BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True))
directions=[Vector(x).normalized() for x in [(1,.317,.791),(-.523,1,.271),(.147,-.613,1)]]

def parity(tree,point,direction):
    origin=point.copy(); count=0
    for _ in range(128):
        location,normal,face,distance=tree.ray_cast(origin,direction,10.0)
        if location is None:return count%2
        if abs(normal.dot(direction))<1e-5:return None
        count+=1; origin=location+direction*1e-6
    return None

def directional(a,b):
    v,_,_=cache[a];w,_,tree=cache[b]
    selected=np.flatnonzero(np.all((v>=w.min(0)) & (v<=w.max(0)),axis=1))
    inside=[];ambiguous=0;near=0
    for i in selected:
        point=Vector(v[i]); hit=tree.find_nearest(point)
        if hit[0] is None: continue
        location,normal,face,distance=hit
        if distance<=2e-6:near+=1;continue
        votes=[parity(tree,point,d) for d in directions]
        if all(x==1 for x in votes):inside.append((float(distance),int(i),int(face)))
        elif not all(x==0 for x in votes):ambiguous+=1
    inside.sort(reverse=True)
    return {'from':a,'into':b,'aabb_vertices':len(selected),'near_boundary_vertices_2um':near,
            'unanimous_inside_vertices':len(inside),'ambiguous_vertices':ambiguous,
            'max_inside_distance_m':inside[0][0] if inside else 0,
            'deepest_witnesses':[{'distance_m':d,'source_vertex':i,'target_face':f,'point_m':v[i].tolist()} for d,i,f in inside[:8]]}

start=time.monotonic()
report={'scope':'Registered source anatomy exploratory three-ray containment and nearest-surface distance on complete closed shells. Eight liver surface patches are audited as their closed anatomical aggregate. Blender Float32 BVH with1um ray restart and2um boundary exclusion; this is not the exact-predicate acceptance check. Ambiguous containment remains unresolved.',
        'input':metadata,'input_npz_sha256':hashlib.sha256((folder/'source-surfaces.npz').read_bytes()).hexdigest(),
        'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'pairs':[]}
for a,b in metadata['pairs']:
    row={'ids':[a,b],'names':[metadata['names'][str(x)] for x in (a,b)],'directions':[directional(a,b),directional(b,a)]}
    report['pairs'].append(row)
    (folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'ids':[a,b],'max_inside_distance_m':[d['max_inside_distance_m'] for d in row['directions']],'inside_count':[d['unanimous_inside_vertices'] for d in row['directions']]}),flush=True)
report['elapsed_seconds']=time.monotonic()-start
(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('DONE',report['elapsed_seconds'],flush=True)
