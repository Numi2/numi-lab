"""Bounded numerical cleanup of Boolean reference interfaces; requires later exact audit."""
from pathlib import Path
import collections,hashlib,json
import bpy,bmesh,numpy as np
from mathutils.bvhtree import BVHTree
from mathutils import Vector

E=Path('/Users/n/numi-human-resting-evidence-20261005')
source=E/'bladder-interface-candidate-105-50um'
out=E/'bladder-interface-conditioned-107';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'scope':'Offline inferred passive organ interfaces. Numerical conditioning of the Boolean triangulation; no native or exact clearance admission yet.',
        'input_report_sha256':sha(source/'report.json'),'driver_sha256':sha(__file__),'merge_distance_m':1e-6,'dissolve_distance_m':1e-6,'coplanar_angle_rad':1e-5,'surfaces':[]}

for sid in [462]:
    path=source/f'surface-{sid}.npz';d=np.load(path);v=d['vertices'].astype(np.float64);f=d['faces']
    tree=BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
    mesh=bpy.data.meshes.new(str(sid));mesh.from_pydata(v.tolist(),[],f.tolist());mesh.update();bm=bmesh.new();bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm,edges=list(bm.edges),dist=1e-6)
    bmesh.ops.dissolve_limit(bm,angle_limit=1e-5,use_dissolve_boundaries=False,verts=list(bm.verts),edges=list(bm.edges))
    bmesh.ops.triangulate(bm,faces=list(bm.faces),quad_method='BEAUTY',ngon_method='BEAUTY')
    bm.to_mesh(mesh);bm.free();mesh.calc_loop_triangles()
    w=np.asarray([tuple(x.co) for x in mesh.vertices],dtype='<f4');g=np.asarray([tuple(x.vertices) for x in mesh.loop_triangles],dtype='<i4')
    edges=collections.Counter(tuple(sorted((int(t[i]),int(t[j])))) for t in g for i,j in ((0,1),(1,2),(2,0)))
    p=w[g].astype(np.float64);cross=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
    length=np.maximum.reduce([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)])
    dest=out/f'surface-{sid}.npz';np.savez_compressed(dest,vertices=w,faces=g)
    row={'stable_id':sid,'input_sha256':sha(path),'output_sha256':sha(dest),'vertices':len(w),'triangles':len(g),
         'boundary_edges':sum(n==1 for n in edges.values()),'nonmanifold_edges':sum(n>2 for n in edges.values()),'zero_area_faces':np.flatnonzero(cross==0).tolist(),
         'minimum_altitude_m':float(np.min(cross/np.maximum(length,1e-30))),
         'signed_volume_ml':float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6*1e6),
         'max_vertex_distance_to_unconditioned_m':max(tree.find_nearest(Vector(x))[3] for x in w)}
    report['surfaces'].append(row);print(json.dumps(row),flush=True)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
