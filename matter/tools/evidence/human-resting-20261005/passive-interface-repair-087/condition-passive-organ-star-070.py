"""Small source-bound closed-shell conditioning diagnostic, offline only."""
from pathlib import Path
import hashlib,json,sys,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');source=E/'passive-organ-interface-flipped-069';out=E/'passive-organ-interface-star-070';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-anatomy-source-004/src')
from numilab_human.resting_anatomy_interface_patch import topology_report
target=1.2e-6;movement_bound=5e-6;volume_bound=1e-9
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def geometry(v,f):
    p=v[f];normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);length=np.maximum.reduce([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)])
    return normal,np.linalg.norm(normal,axis=1)/np.maximum(length,1e-30),float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)
report={'scope':'Bounded inferred numerical cleanup of isolated passive-organ reference interfaces; no accepted runtime qualification. Every collapse retains closed oriented topology, positive old/new face orientation, source ancestry displacement≤5um and shell volume delta≤1uL.',
        'driver_sha256':sha(__file__),'minimum_altitude_target_m':target,'movement_bound_m':movement_bound,'volume_bound_m3':volume_bound,'surfaces':[]}
for sid in [3,13]:
    p=source/f'surface-{sid}.npz';d=np.load(p);original=d['vertices'].astype(float);v=original.copy();f=d['faces'].copy();ancestry={i:[i] for i in range(len(v))};changes=[];base_volume=geometry(v,f)[2]
    for iteration in range(20):
        normal,altitude,volume=geometry(v,f);bad=np.flatnonzero(altitude<target)
        if not len(bad):break
        proposals=[]
        for face_id in bad:
            tri=f[face_id]
            for ia,ib in [(0,1),(1,2),(2,0)]:
                a,b=map(int,(tri[ia],tri[ib]));bound=ancestry[a]+ancestry[b]
                for point in [v[a],v[b],(v[a]+v[b])/2]:
                    point=point.astype('<f4').astype(float)
                    displacement=float(np.max(np.linalg.norm(original[bound]-point,axis=1)))
                    if displacement>movement_bound:continue
                    trial_v=v.copy();trial_v[a]=point;trial_f=f.copy();trial_f[trial_f==b]=a
                    keep=np.array([len(set(map(int,t)))==3 for t in trial_f]);trial_f=trial_f[keep]
                    if np.count_nonzero(~keep)!=2:continue
                    if len(np.unique(np.sort(trial_f,axis=1),axis=0))!=len(trial_f):continue
                    n,h,vol=geometry(trial_v,trial_f)
                    if np.any(np.einsum('ij,ij->i',n,normal[keep])<=0):continue
                    if abs(vol-base_volume)>volume_bound:continue
                    if np.count_nonzero(h<target)>=len(bad):continue
                    t=topology_report(trial_f);t.pop('boundary_edges',None)
                    if any(t[k] for k in ['boundary_edge_count','nonmanifold_edge_count','orientation_error_edge_count']):continue
                    proposals.append((len(bad)-int(np.count_nonzero(h<target)),float(h.min()),a,b,point,trial_v,trial_f,bound,displacement,vol,t))
        if not proposals:break
        best=max(proposals,key=lambda x:(x[0],x[1]));_,minimum,a,b,point,v,f,bound,displacement,vol,t=best
        ancestry[a]=bound;del ancestry[b]
        changes.append({'from_vertex':b,'to_vertex':a,'point_m':point.tolist(),'maximum_ancestry_displacement_m':displacement,'volume_delta_m3':vol-base_volume,'minimum_altitude_m':minimum,'topology':t})
    used=np.unique(f);lookup=np.full(len(v),-1);lookup[used]=np.arange(len(used));v=v[used];f=lookup[f]
    n,h,vol=geometry(v,f);summed=np.zeros_like(v)
    for k in range(3):np.add.at(summed,f[:,k],n)
    lengths=np.linalg.norm(summed,axis=1);assert np.all(lengths>0)
    dest=out/f'surface-{sid}.npz';np.savez_compressed(dest,vertices=v.astype('<f4'),normals=(summed/lengths[:,None]).astype('<f4'),faces=f.astype('<i4'))
    t=topology_report(f);t.pop('boundary_edges',None)
    row={'stable_id':sid,'source_sha256':sha(p),'output_sha256':sha(dest),'changes':changes,'remaining_below_target':int(np.count_nonzero(h<target)),
         'minimum_altitude_m':float(h.min()),'volume_m3':vol,'volume_delta_m3':vol-base_volume,'topology':t}
    report['surfaces'].append(row);print(json.dumps(row),flush=True)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
