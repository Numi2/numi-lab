from pathlib import Path
import numpy as np,csv,json
D=Path(__file__).parent;S=D.parent
with (S/'self-audit-055/O.bin').open('rb') as f:
 n,nf=map(int,np.fromfile(f,dtype='<u8',count=2));p=np.fromfile(f,dtype='<f8',count=n*3).reshape(-1,3);t=np.fromfile(f,dtype='<u4',count=nf*3).reshape(-1,3)
donors=np.array([int(r['donor']) for r in csv.DictReader((S/'dominant-frame-registration-042/registered-O.csv').open())]);source=np.array([-.30670645833015442,-.88691836595535278,.12189969420433044]);tri=p[t];a,b,c=tri[:,0],tri[:,1],tri[:,2];ab=b-a;ac=c-a;nrm=np.cross(ab,ac);nn=np.einsum('ij,ij->i',nrm,nrm);q=source-((nrm@source-np.einsum('ij,ij->i',nrm,a))/nn)[:,None]*nrm
v=q-a;d00=np.einsum('ij,ij->i',ab,ab);d01=np.einsum('ij,ij->i',ab,ac);d11=np.einsum('ij,ij->i',ac,ac);d20=np.einsum('ij,ij->i',v,ab);d21=np.einsum('ij,ij->i',v,ac);den=d00*d11-d01*d01;u=(d11*d20-d01*d21)/den;v=(d00*d21-d01*d20)/den;dist=np.linalg.norm(q-source,axis=1);dist[(u<0)|(v<0)|(u+v>1)]=np.inf;bary=np.stack([1-u-v,u,v],axis=1)
for k in range(3):
 x=tri[:,k];e=tri[:,(k+1)%3]-x;alpha=np.clip(np.einsum('ij,ij->i',source-x,e)/np.einsum('ij,ij->i',e,e),0,1);r=x+alpha[:,None]*e;d=np.linalg.norm(r-source,axis=1);sel=d<dist;q[sel]=r[sel];dist[sel]=d[sel];bary[sel]=0;bary[sel,k]=1-alpha[sel];bary[sel,(k+1)%3]=alpha[sel]
results={}
for name,mask in [('all',np.ones(len(t),dtype=bool)),('all_three_donor62',np.all(donors[t]==62,axis=1)),('any_corner_donor62',np.any(donors[t]==62,axis=1))]:
 rows=np.flatnonzero(mask);j=rows[np.argmin(dist[rows])];results[name]={'face':int(j),'vertices':t[j].tolist(),'donors':donors[t[j]].tolist(),'barycentric':bary[j].tolist(),'point':q[j].tolist(),'distance_m':float(dist[j]),'same_donor_bary_sum':float(bary[j][donors[t[j]]==62].sum()),'vertex_positions':tri[j].tolist()}
(D/'mixed-owner-surface-witness.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
