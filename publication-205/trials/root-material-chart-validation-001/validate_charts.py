#!/usr/bin/env python3
"""Independently validate a remesh against retained material charts.

This verifies reference-coordinate coverage and declared 806-face ancestry. It
does not validate current physical volume, donor-cell conservation, geometry
clearance, or simulation admission.
"""
import argparse, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np

def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for x in iter(lambda:f.read(1<<20),b""): h.update(x)
 return h.hexdigest()

def load(path):
 with np.load(path,allow_pickle=False) as a:
  return {k:a[k].copy() for k in a.files}

def cross(a,b): return float(a[0]*b[1]-a[1]*b[0])
def area(poly):
 if len(poly)<3:return 0.
 return abs(sum(cross(poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly))))/2.

def clip(subject,clipper):
 out=[np.array(v,dtype=np.float64) for v in subject]
 for i in range(3):
  a,b=clipper[i],clipper[(i+1)%3]; edge=b-a; inp=out; out=[]
  if not inp:break
  p=inp[-1]; dp=cross(edge,p-a)
  for q in inp:
   dq=cross(edge,q-a)
   pin=dp>=-1e-14; qin=dq>=-1e-14
   if pin!=qin:
    den=dp-dq
    if abs(den)<1e-30:raise ValueError("unstable polygon clipping")
    out.append(p+(q-p)*(dp/den))
   if qin:out.append(q)
   p,dp=q,dq
 return out

def csr(z,ptrkey,rowkey,count,limit,weights=None):
 p=np.asarray(z[ptrkey]); r=np.asarray(z[rowkey])
 if p.shape!=(count+1,) or not np.issubdtype(p.dtype,np.integer) or not np.issubdtype(r.dtype,np.integer) or p[0]!=0 or p[-1]!=len(r) or np.any(np.diff(p)<=0) or np.any(r<0) or np.any(r>=limit):raise ValueError("invalid "+ptrkey)
 if weights is not None:
  w=np.asarray(z[weights])
  if w.shape!=r.shape or not np.isfinite(w).all() or (w < -1e-12).any():raise ValueError("invalid "+weights)
 else:w=None
 for i in range(count):
  a,b=map(int,p[i:i+2])
  if len(np.unique(r[a:b]))!=b-a:raise ValueError("duplicate row in "+rowkey)
  if w is not None and abs(float(sum(w[a:b]))-1.)>1e-11:raise ValueError("non-convex source row")
 return p,r,w

def scene(z):
 f=np.asarray(z["source_O_faces"]); x=np.asarray(z["points_world_m"])
 p=np.asarray(z["refinement_face_parent_face_row"]); b=np.asarray(z["refinement_face_parent_barycentric"])
 if f.ndim!=2 or f.shape[1]!=3 or np.any(f<0) or np.any(f>=len(x)) or p.shape!=(len(f),) or (p<0).any() or b.shape!=(len(f),3,3) or not np.isfinite(b).all():raise ValueError("invalid face domain or chart")
 if (b < -1e-12).any() or (b > 1+1e-12).any() or np.max(np.abs(b.sum(axis=2)-1))>1e-11:raise ValueError("invalid chart simplex")
 xy=b[:,:,1:3]; u=xy[:,1]-xy[:,0]; v=xy[:,2]-xy[:,0]; a=(u[:,0]*v[:,1]-u[:,1]*v[:,0])/2.
 if np.any(a<=1e-12):raise ValueError("nonpositive chart face")
 return x,f,p,b,xy,a

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("baseline",type=Path);ap.add_argument("candidate",type=Path);ap.add_argument("output",type=Path)
 ap.add_argument("--expected-baseline",required=True);ap.add_argument("--expected-candidate",required=True)
 ap.add_argument("--identity",action="store_true",help="independent parity check without remesh metadata")
 a=ap.parse_args()
 if a.output.exists() or digest(a.baseline)!=a.expected_baseline or digest(a.candidate)!=a.expected_candidate:raise ValueError("output exists or hash mismatch")
 x=load(a.baseline);y=load(a.candidate);xp,xf,xi,xb,xc,xa=scene(x);yp,yf,yi,yb,yc,ya=scene(y)
 if a.identity:
  if a.expected_baseline!=a.expected_candidate:raise ValueError("identity requires equal file hashes")
  rev=np.arange(len(yp)); nmap=np.arange(len(xp));fp=np.arange(len(yf)+1);fr=np.arange(len(yf))
 else:
  pin=str(np.asarray(y["remesh806_baseline_sha256"]).reshape(-1)[0])
  if pin!=a.expected_baseline:raise ValueError("baseline pin mismatch")
  nmap=np.asarray(y["remesh806_old_to_new_node_rows"]);rev=np.asarray(y["remesh806_new_to_old_node_rows"])
  if nmap.shape!=(len(xp),) or rev.shape!=(len(yp),) or np.any(nmap < -1) or np.any(nmap>=len(yp)) or np.any(rev < -1) or np.any(rev>=len(xp)):raise ValueError("bad node map")
  active=nmap[nmap>=0]
  if len(np.unique(active))!=len(active):raise ValueError("ambiguous mapped nodes")
  expected=np.full(len(yp),-1,dtype=int); old=np.flatnonzero(nmap>=0);expected[nmap[old]]=old
  if not np.array_equal(expected,rev):raise ValueError("node map inverse mismatch")
  fp,fr,_=csr(y,"remesh806_face_parent_indptr","remesh806_face_parent_face_rows",len(yf),len(xf))
 if np.any(rev<0):raise ValueError("new nodes require an explicitly validated material insertion rule")
 # The top-level CSR indexes the 13,518-node intermediate domain. Its
 # two retained ancestors compose that domain into the 8,776-node source.
 # Equal top-level rows alone do not prove the source field is unchanged.
 for prefix in ("prior_input_006__","prior_run028__"):
  for suffix in ("source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights"):
   key=prefix+suffix
   if (key in x)!=(key in y) or (key in x and not np.array_equal(x[key],y[key])):raise ValueError("historical source mapping changed: "+key)
 p0,r0,w0=csr(x,"source_prolongation_indptr","source_prolongation_node_rows",len(xp),13518,"source_prolongation_weights")
 p1,r1,w1=csr(y,"source_prolongation_indptr","source_prolongation_node_rows",len(yp),13518,"source_prolongation_weights")
 for new,old in enumerate(rev):
  a0,b0=map(int,p0[old:old+2]);a1,b1=map(int,p1[new:new+2])
  o=sorted(zip(map(int,r0[a0:b0]),map(float,w0[a0:b0])));n=sorted(zip(map(int,r1[a1:b1]),map(float,w1[a1:b1])))
  if len(o)!=len(n) or any(oi!=ni or abs(ow-nw)>1e-12 for (oi,ow),(ni,nw) in zip(o,n)):raise ValueError("material source row changed on retained node "+str(new))
 oldgroups=defaultdict(list);newgroups=defaultdict(list)
 for i,p in enumerate(xi):oldgroups[int(p)].append(i)
 for i,p in enumerate(yi):newgroups[int(p)].append(i)
 if oldgroups.keys()!=newgroups.keys():raise ValueError("historical face chart was added or removed")
 altered=0; pairs=0
 for parent, olds in oldgroups.items():
  news=newgroups[parent]
  unchanged=len(olds)==len(news)
  if unchanged:
   lookup={int(o):1 for o in olds}
   for n in news:
    pp=fr[fp[n]:fp[n+1]]
    if len(pp)!=1 or int(pp[0]) not in lookup:unchanged=False;break
    o=int(pp[0])
    if not np.array_equal(xf[o],rev[yf[n]]) or not np.array_equal(xb[o],yb[n]):unchanged=False;break
    del lookup[o]
  if unchanged:continue
  altered+=1
  prior={}
  for o in olds:
   for k,oldnode in enumerate(xf[o]):
    key=int(oldnode); val=xb[o,k]
    if key in prior and not np.allclose(prior[key],val,rtol=0,atol=1e-11):raise ValueError("baseline material chart disagrees across old faces")
    prior[key]=val
  for n in news:
   for k,newnode in enumerate(yf[n]):
    oldnode=int(rev[newnode])
    if oldnode not in prior or not np.allclose(prior[oldnode],yb[n,k],rtol=0,atol=1e-11):raise ValueError("stale or invented chart on new face "+str(n))
  overlap=np.zeros((len(olds),len(news)))
  for i,o in enumerate(olds):
   for j,n in enumerate(news):overlap[i,j]=area(clip(xc[o],yc[n]))
  tol=1e-10
  if not np.allclose(overlap.sum(axis=1),xa[olds],atol=tol,rtol=1e-10):raise ValueError("old chart area lost or multiply covered in parent "+str(parent))
  if not np.allclose(overlap.sum(axis=0),ya[news],atol=tol,rtol=1e-10):raise ValueError("new chart area has missing/duplicate old ancestry in parent "+str(parent))
  for i in range(len(news)):
   for j in range(i):
    if area(clip(yc[news[i]],yc[news[j]]))>tol:raise ValueError("new chart faces overlap")
  for j,n in enumerate(news):
   actual={int(v) for v in fr[fp[n]:fp[n+1]]}
   expected={int(olds[i]) for i in range(len(olds)) if overlap[i,j]>tol}
   if actual!=expected:raise ValueError("declared old-face ancestry differs from geometric overlap on new face "+str(n))
   pairs+=len(expected)
 report={"status":"MATERIAL_CHART_VALIDATED","baseline_sha256":a.expected_baseline,"candidate_sha256":a.expected_candidate,"validator_sha256":digest(Path(__file__)),"nodes_before":len(xp),"nodes_after":len(yp),"faces_before":len(xf),"faces_after":len(yf),"historical_charts":len(oldgroups),"altered_charts":altered,"overlap_ancestry_pairs_checked":pairs,"deleted_nodes":int(np.sum(nmap<0)),"unmapped_new_nodes":int(np.sum(rev<0)),"limitations":["Reference material-chart coverage is not physical-volume ownership.","Removed inside-node witnesses remain unresolved until physical containment is checked.","No simulation or native admission was performed."],"physical_steps":0}
 a.output.write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps(report))
if __name__=="__main__":main()
