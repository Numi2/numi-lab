"""Offline localized passive-organ reference repair in the existing geometry pipeline."""
from pathlib import Path
import collections, hashlib, json, sys, time
import bpy
import numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005')
margin=float(sys.argv[sys.argv.index('--')+1]) if '--' in sys.argv else 0.00005
assert margin in (0.000025,0.00005,0.0001)
folder=E/f'passive-organ-interface-candidate-067-{round(margin*1e6)}um';folder.mkdir(exist_ok=False)
inputs=E/'passive-neighbor-containment-065';data=np.load(inputs/'source-surfaces.npz')
meta=json.loads((inputs/'input.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stats(v,f):
    edges=collections.Counter(tuple(sorted((int(t[i]),int(t[j])))) for t in f for i,j in ((0,1),(1,2),(2,0)))
    p=v[f];cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);twice=np.linalg.norm(cross,axis=1)
    lengths=np.maximum.reduce([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)])
    return {'boundary_edges':sum(n==1 for n in edges.values()),'nonmanifold_edges':sum(n>2 for n in edges.values()),
            'zero_area_faces':np.flatnonzero(twice==0).tolist(),'minimum_altitude_m':float(np.min(twice/np.maximum(lengths,1e-30))),
            'signed_volume_ml':float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6*1e6),'vertices':len(v),'triangles':len(f)}

def normals(v,f):
    p=v[f];n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);out=np.zeros_like(v)
    for col in range(3):np.add.at(out,f[:,col],n)
    length=np.linalg.norm(out,axis=1);assert np.all(length>0);return out/length[:,None]

def obj(name,v,f):
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(v.tolist(),[],f.tolist());mesh.update()
    item=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(item);return item

def arrays(item):
    item.data.calc_loop_triangles()
    v=np.asarray([tuple(x.co) for x in item.data.vertices],dtype=np.float64)
    f=np.asarray([tuple(x.vertices) for x in item.data.loop_triangles],dtype=np.int64)
    v,indices=np.unique(v,axis=0,return_inverse=True);return v,indices[f]

bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
report={'scope':'Inferred passive reference surface repair; source organ identities retained. Difference removes only source overlaps and a declared numerical separation from stationary registered neighbors. The pancreas and spleen remain passive geometry and add no mass, force or physiological state. Not measured subject geometry or native full-cycle qualification.',
        'input_payload_sha256':meta['payload_sha256'],'input_npz_sha256':sha(inputs/'source-surfaces.npz'),'driver_sha256':sha(__file__),
        'blender_version':bpy.app.version_string,'separation_margin_m':margin,'method':'Source neighbor surfaces expanded along area-weighted outward normals by the declared numerical margin, then existing Blender exact Boolean difference. Original pancreas/spleen exterior retained outside the local cut. Difference priority preserves stomach, left kidney and duodenum source geometry.',
        'surfaces':[]}
resolved={}
for sid,neighbors in [(13,[2,5]),(3,[2,398,13])]:
    v=data[f'v{sid}'].astype(np.float64);f=data[f'f{sid}'];subject=obj(str(sid),v,f)
    row={'stable_id':sid,'neighbors':neighbors,'source_stats':stats(v,f),'stages':[]}
    for other_id in neighbors:
        w,g=resolved.get(other_id,(data[f'v{other_id}'].astype(np.float64),data[f'f{other_id}']))
        expanded=w+margin*normals(w,g);other=obj(str(other_id)+'expanded',expanded,g)
        mod=subject.modifiers.new('reference_interface','BOOLEAN');mod.operation='DIFFERENCE';mod.solver='EXACT';mod.object=other
        bpy.context.view_layer.objects.active=subject;bpy.ops.object.modifier_apply(modifier=mod.name)
        ov,of=arrays(subject);row['stages'].append({'neighbor_id':other_id,'stats':stats(ov,of)})
        bpy.data.objects.remove(other,do_unlink=True)
    ov,of=arrays(subject);resolved[sid]=(ov,of);p=folder/f'surface-{sid}.npz'
    np.savez_compressed(p,vertices=ov.astype('<f4'),faces=of.astype('<i4'))
    row['output_stats']=stats(ov,of);row['output_sha256']=sha(p)
    row['removed_volume_ml']=row['source_stats']['signed_volume_ml']-row['output_stats']['signed_volume_ml']
    report['surfaces'].append(row);(folder/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row),flush=True);bpy.data.objects.remove(subject,do_unlink=True)
