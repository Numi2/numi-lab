#!/usr/bin/env python3
"""Restore original material volumes after the bounded quality fit; never move the exterior."""
import pathlib,csv,json,hashlib,numpy as np
OUT=pathlib.Path(__file__).resolve().parent;ROOT=pathlib.Path.cwd()
src=ROOT/"tools/evidence/human-wholebody-skinfat-20261019/c296"
raw=np.fromstring((src/"candidate-024-safe.txt").read_text(),sep=" ",dtype=int);n,nf=raw[:2]
Fraw=raw[2+n:2+n+3*nf].reshape(-1,3);F=np.sort(Fraw,axis=1);a,b,c=F.T
P=np.load(OUT.parent/"mesh-quality-017/candidate-points-mm.npy")
rows=list(csv.DictReader((src/"admitted-layer-coordinates.csv").open()))
P0=np.array([[[float(r[p+"_"+a])*1000 for a in "xyz"] for r in rows] for p in ("outer","skin","fat")]).reshape(-1,3)
for band,lo,hi in ((1,.5002,2.4998),(2,.5002,9.9998)):
 A=P[(band-1)*n:band*n];B=P[band*n:(band+1)*n]
 v=B-A;l=np.linalg.norm(v,axis=1);B[:]=A+v*(np.clip(l,lo,hi)/l)[:,None]
Ts=[]
for k in range(2):
 Ts.extend([np.stack([a+k*n,b+k*n,c+k*n,c+(k+1)*n],1),
 np.stack([a+k*n,b+k*n,b+(k+1)*n,c+(k+1)*n],1),
 np.stack([a+k*n,a+(k+1)*n,b+(k+1)*n,c+(k+1)*n],1)])
T=np.concatenate(Ts);parity=np.array([(-1)**sum(f[i]>f[j] for i in range(3) for j in range(i+1,3)) for f in Fraw])
sg=np.tile(np.concatenate([-parity,parity,-parity]),2)
def calc(P,gradient=False):
 p=P[T];u=p[:,1]-p[:,0];v=p[:,2]-p[:,0];w=p[:,3]-p[:,0]
 det=np.einsum("ij,ij->i",u,np.cross(v,w));signed=det*sg
 vols=signed.reshape(2,-1).sum(axis=1)/6
 if not gradient:return vols
 G=np.zeros((2,len(P),3));gu=np.cross(v,w)*sg[:,None]/6;gv=np.cross(w,u)*sg[:,None]/6;gw=np.cross(u,v)*sg[:,None]/6
 for band in range(2):
  ix=slice(band*3*nf,(band+1)*3*nf)
  for slot,g in enumerate((-gu-gv-gw,gu,gv,gw)):np.add.at(G[band],T[ix,slot],g[ix])
 return vols,G
target=calc(P0)
# Compensate on elements far from the conditioning limit and thickness bounds.
p=P[T];E=p[:,1:]-p[:,:1];sv=np.linalg.svd(E,compute_uv=False);cond=sv[:,0]/sv[:,-1]
worst=np.zeros(len(P))
for j in range(4):np.maximum.at(worst,T[:,j],cond)
st=np.linalg.norm(P[n:2*n]-P[:n],axis=1);ft=np.linalg.norm(P[2*n:]-P[n:2*n],axis=1)
eligible=np.zeros(len(P),bool)
eligible[n:2*n]=(worst[n:2*n]<500)&(st>.55)&(st<2.49)&(ft>.55)&(ft<9.95)
eligible[2*n:]=(worst[2*n:]<500)&(ft>.55)&(ft<9.95)
before=P.copy();history=[]
for iteration in range(100):
 V,G=calc(P,True);err=V-target
 history.append((err/target).tolist())
 if np.max(np.abs(err/target))<1e-8:break
 G[:,~eligible,:]=0
 flat=G.reshape(2,-1)
 multiplier=np.linalg.solve(flat@flat.T,-err)
 delta=(multiplier@flat).reshape(-1,3)
 if np.linalg.norm(delta,axis=1).max()>.05:raise RuntimeError("volume correction exceeds 50um per iteration")
 P+=delta
 P=(P/1000).astype(np.float32).astype(float)*1000
assert np.array_equal(P[:n],P0[:n])
assert np.max(np.abs((calc(P)-target)/target))<1e-7
# Update existing CSV fields and exact donor-local positions, no new format.
reg=pathlib.Path("/Users/numi/numi-human-separation-20261018/C/wholebody-skin-weight-repair-293/native-native-root-pose-003/registered-skin.csv")
donors={int(r["vertex"]):r for r in csv.DictReader(reg.open())}
for i,r in enumerate(rows):
 d=donors[int(r["source_vertex"])];M=np.array([[float(d[f"m{j}{k}"]) for k in range(3)] for j in range(3)])
 for band,prefix in ((1,"skin"),(2,"fat")):
  old=np.array([float(r[prefix+"_"+a]) for a in "xyz"]);oldlocal=np.array([float(r[prefix+"_local_"+a]) for a in "xyz"])
  local=oldlocal+M@(P[band*n+i]/1000-old)
  for a,v,vl in zip("xyz",P[band*n+i]/1000,local):r[prefix+"_"+a]=format(v,".17g");r[prefix+"_local_"+a]=format(vl,".17g")
 r["skin_m"]=format(np.linalg.norm(P[n+i]-P[i])/1000,".17g");r["fat_m"]=format(np.linalg.norm(P[2*n+i]-P[n+i])/1000,".17g")
with (OUT/"candidate-layers.csv").open("w") as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n");w.writeheader();w.writerows(rows)
p=P[T];E=p[:,1:]-p[:,:1];sv=np.linalg.svd(E,compute_uv=False)
det=np.einsum("ij,ij->i",E[:,0],np.cross(E[:,1],E[:,2]))*sg
L=np.linalg.norm(p[:,:,None]-p[:,None,:],axis=3).max(axis=(1,2));q=det/L**3
st=np.linalg.norm(P[n:2*n]-P[:n],axis=1)/1000;ft=np.linalg.norm(P[2*n:]-P[n:2*n],axis=1)/1000
assert st.min()>=.0005 and st.max()<=.0025 and ft.min()>=.0005 and ft.max()<=.010
assert q.min()>4e-5
report=dict(scope="Exact Float32 bounded quality correction candidate; not native qualified",outer_unchanged=True,
 original_volumes_m3=(target/1e9).tolist(),candidate_volumes_m3=(calc(P)/1e9).tolist(),volume_residual_history=history,
 max_volume_compensation_mm=float(np.linalg.norm(P-before,axis=1).max()),
 max_total_displacement_mm=float(np.linalg.norm(P-P0,axis=1).max()),min_normalized_determinant=float(q.min()),max_condition=float((sv[:,0]/sv[:,-1]).max()),
 thickness_bounds_m=dict(skin=[float(st.min()),float(st.max())],fat=[float(ft.min()),float(ft.max())]),
 coordinates_sha256=hashlib.sha256((OUT/"candidate-layers.csv").read_bytes()).hexdigest())
(OUT/"quality.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))

