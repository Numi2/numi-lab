from pathlib import Path
import hashlib,json,numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005');source=E/'passive-organ-interface-star-070';out=E/'passive-organ-interface-conditioned-071';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def measures(v,f):
    p=v[f];n=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);length=np.maximum.reduce([np.linalg.norm(p[:,1]-p[:,0],axis=1),np.linalg.norm(p[:,2]-p[:,1],axis=1),np.linalg.norm(p[:,0]-p[:,2],axis=1)])
    return n,np.linalg.norm(n,axis=1)/length,float(np.einsum('ij,ij->i',p[:,0],np.cross(p[:,1],p[:,2])).sum()/6)
report={'scope':'Bounded in-plane numerical opening of one residual passive-organ sliver; inferred surface conditioning only. No native acceptance yet.',
        'source_report_sha256':sha(source/'report.json'),'driver_sha256':sha(__file__),'surfaces':[]}
for sid in [3,13]:
    path=source/f'surface-{sid}.npz';d=np.load(path);v=d['vertices'].astype(float);f=d['faces'];before=v.copy();n,h,volume=measures(v,f);changes=[]
    for face in np.flatnonzero(h<1.2e-6):
        tri=f[face];best=None
        for tip in range(3):
            a,b=map(int,[tri[(tip+1)%3],tri[(tip+2)%3]]);i=int(tri[tip]);edge=v[b]-v[a]
            projection=v[a]+edge*np.dot(v[i]-v[a],edge)/np.dot(edge,edge);perp=v[i]-projection
            proposal=(projection+perp/np.linalg.norm(perp)*1.6e-6).astype('<f4').astype(float)
            movement=float(np.linalg.norm(proposal-v[i]))
            if movement>1e-6:continue
            candidate=v.copy();candidate[i]=proposal;nn,hh,vv=measures(candidate,f)
            if np.any(np.einsum('ij,ij->i',n,nn)<=0) or hh.min()<1.2e-6 or abs(vv-volume)>1e-9:continue
            if best is None or movement<best[0]:best=(movement,i,candidate,nn,hh,vv)
        assert best is not None,('no admissible opening',sid,int(face))
        movement,i,v,n,h,vv=best;changes.append({'face':int(face),'vertex':i,'from_m':before[i].tolist(),'to_m':v[i].tolist(),'displacement_m':movement})
    summed=np.zeros_like(v)
    for k in range(3):np.add.at(summed,f[:,k],n)
    lengths=np.linalg.norm(summed,axis=1);assert np.all(lengths>0)
    p=out/f'surface-{sid}.npz';np.savez_compressed(p,vertices=v.astype('<f4'),normals=(summed/lengths[:,None]).astype('<f4'),faces=f.astype('<i4'))
    row={'stable_id':sid,'input_sha256':sha(path),'output_sha256':sha(p),'changes':changes,'minimum_altitude_m':float(h.min()),'volume_m3':measures(v,f)[2],'volume_delta_m3':measures(v,f)[2]-volume}
    report['surfaces'].append(row);print(json.dumps(row),flush=True)
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
