"""Source-bound Boolean volume diagnostic for distinct passive organs."""
from pathlib import Path
import collections, hashlib, json, time
import bpy
import numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005')
folder=E/'bladder-neighbor-overlap-volumes-104';folder.mkdir(exist_ok=False)
inputs=E/'passive-neighbor-containment-065'
data=np.load(inputs/'source-surfaces.npz')
meta=json.loads((inputs/'input.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def topology(v,f):
    edges=collections.Counter(tuple(sorted((int(t[i]),int(t[j])))) for t in f for i,j in ((0,1),(1,2),(2,0)))
    p=v[f]
    area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    return {'boundary_edges':sum(n==1 for n in edges.values()),'nonmanifold_edges':sum(n>2 for n in edges.values()),
            'zero_area_faces':np.flatnonzero(area==0).tolist(),'signed_volume_ml':float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6*1e6),
            'vertices':len(v),'triangles':len(f)}

def object_from(name,v,f):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(v.tolist(),[],f.tolist());mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj);return obj

bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
result={'scope':'Source geometry Boolean overlap diagnostic for distinct passive organs and continuity interfaces; no input mutation and no admission claim. Closed nondegenerate output is required to interpret volume.',
        'driver_sha256':sha(__file__),'source_payload_sha256':meta['payload_sha256'],'input_npz_sha256':sha(inputs/'source-surfaces.npz'),
        'blender_version':bpy.app.version_string,'pairs':[]}
start=time.monotonic()
for a,b in [(415,462),(416,462),(455,462),(459,462),(462,463)]:
    v=data[f'v{a}'].astype(np.float64);f=data[f'f{a}'];w=data[f'v{b}'].astype(np.float64);g=data[f'f{b}']
    obj=object_from(str(a),v,f);other=object_from(str(b),w,g)
    mod=obj.modifiers.new('intersection','BOOLEAN');mod.operation='INTERSECT';mod.solver='EXACT';mod.object=other
    bpy.context.view_layer.objects.active=obj;bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.data.calc_loop_triangles()
    ov=np.asarray([tuple(x.co) for x in obj.data.vertices],dtype=np.float64).reshape(-1,3)
    of=np.asarray([tuple(x.vertices) for x in obj.data.loop_triangles],dtype=np.int64).reshape(-1,3)
    p=folder/f'intersection-{a}-{b}.npz';np.savez_compressed(p,vertices=ov,faces=of)
    row={'ids':[a,b],'names':[meta['names'][str(s)] for s in (a,b)],'input_topology':[topology(v,f),topology(w,g)],
         'intersection':topology(ov,of) if len(of) else {'empty':True},'output_sha256':sha(p)}
    result['pairs'].append(row);(folder/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
    bpy.data.objects.remove(obj,do_unlink=True);bpy.data.objects.remove(other,do_unlink=True)
result['elapsed_seconds']=time.monotonic()-start
(folder/'report.json').write_text(json.dumps(result,indent=2)+'\n')
