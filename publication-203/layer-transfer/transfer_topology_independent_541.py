#!/usr/bin/env python3
"""Topology-independent 541 surface-layer transfer with typed ancestry and cell gates.

Transfers only geometric surface offsets/depths. It does not assign volume-cell
mass or claim physical/native admission.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np
from scipy import sparse

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069")
D=ROOT/"D"
S=D/"surface-clearance-546/component-sat-selector-638"
DEFAULT_C=S/"per-bank-workers-20261027/combined-d071-b109-b61-001/candidate.npz"
DEFAULT_L=S/"per-bank-workers-20261027/combined-d071-b109-b61-001/skin-fat-layers-541.npz"
R0=ROOT/"full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz"
A146=ROOT/"connected-source-unfold-146/run-002-corrected-metadata/candidate.npz"
L128=ROOT/"coupled-layer-trial-128/run-004-full146-baseline-offset/layer-candidate.npz"
OWNER408=ROOT/"publication-127/sources/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
OWNER541=S/"per-bank-workers-20261027/physical-layer-diagnostic-001/transfer_current.py"
CHART_OWNER=S/"per-bank-workers-20261027/root-material-chart-validation-001/validate_charts.py"
PFX=("","prior_input_006__","prior_run028__")
PIN={
 "806":"806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756",
 "layer806":"eb41bf0d1c11f1ee0a94672058063c51350e8e0dca1483a6ae7568cb03a78b57",
 "r0":"70215460195063d68eaa13e2031ba9cc538ce52aede248c43a3b7679e8e68a31",
 "146":"4a393af65cc12181e00f2b578f73ad209f87e5f2107f4808e9ce083f46a2b3da",
 "layer128":"301328142aae18bef98b1dfd9fff974533728aa5e084317d53fcda9dad5b7950",
 "owner408":"5b259785752568963cad0f0738260d6f4889b84243815c842d02173ea638ebd2",
 "owner541":"14cc312781348947992b1ecd7b375224fc2006a648ca45501cf0c7ad0a46feb6",
 "chartOwner":"d46adbce768094f9c2d9a0c39cbb56f42700b1f915ec881678f9d3f640d8ad25",
}
NODE_TYPED=("node_kind","node_source_row","node_source_virtual_row","node_parent_coarse_face_row",
 "node_parent_coarse_edge_a","node_parent_coarse_edge_b","node_parent_coarse_edge_parameter")
FACE_TYPED=("face_kind","parent_coarse_face_row","parent_fine_detail_face_row","parent_source_face_marker",
 "parent_fine_subface_row","refinement_face_parent_face_row","refinement_face_parent_barycentric")
FACE_CSR=(("face_source_ancestry_indptr","face_source_ancestry_face_rows",True),
 ("face_fine_subface_ancestry_indptr","face_fine_subface_ancestry_rows",False),
 ("face_fine_detail_ancestry_indptr","face_fine_detail_ancestry_rows",False),
 ("refinement_face_parent_ancestry_indptr","refinement_face_parent_ancestry_face_rows",True))

class TransferError(RuntimeError): pass
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def req(ok,msg):
 if not ok: raise TransferError(msg)
def load_dict(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in z.files}
def import_owner(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None: raise TransferError(f"cannot import owner {path}")
 m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def bytes_equal(a,b):
 a,b=np.asarray(a),np.asarray(b)
 return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()

def csr(indptr,indices,weights,nrow,ncol,name,nonempty=True):
 ip=np.asarray(indptr,dtype=np.int64); ix=np.asarray(indices,dtype=np.int64); w=np.asarray(weights,dtype=np.float64)
 req(ip.shape==(nrow+1,) and ip[0]==0 and ip[-1]==len(ix),f"{name}: invalid pointer shape/endpoints")
 req(np.all(np.diff(ip)>=0) and (not nonempty or np.all(np.diff(ip)>0)),f"{name}: empty/nonmonotone CSR row")
 req(ix.shape==w.shape and np.all(ix>=0) and np.all(ix<ncol),f"{name}: invalid indices/weights shape or range")
 req(np.isfinite(w).all() and np.all(w>=0),f"{name}: nonfinite or negative weights")
 M=sparse.csr_matrix((w,ix,ip),shape=(nrow,ncol))
 sums=np.asarray(M.sum(axis=1)).ravel()
 err=float(np.max(np.abs(sums-1),initial=0.0)) if nonempty else 0.0
 if nonempty:req(err<=2e-10,f"{name}: row-sum error {err}")
 return M,err

def compose(c,root_n):
 n=len(c["points_world_m"]);P=sparse.eye(n,format="csr",dtype=np.float64);levels=[]
 for lev,prefix in enumerate(PFX):
  ik=prefix+"source_prolongation_indptr";xk=prefix+"source_prolongation_node_rows";wk=prefix+"source_prolongation_weights"
  req(all(k in c for k in (ik,xk,wk)),f"missing source ancestry CSR level {lev}: {prefix}")
  ip=np.asarray(c[ik],dtype=np.int64);nr=len(ip)-1
  nc=(len(c[PFX[lev+1]+"source_prolongation_indptr"])-1) if lev+1<len(PFX) else int(root_n)
  req(nr==P.shape[1],f"source ancestry level {lev} domain mismatch")
  M,err=csr(ip,c[xk],c[wk],nr,nc,f"source ancestry level {lev}")
  P=(P@M).tocsr()
  levels.append({"level":lev,"prefix":prefix,"rows":nr,"columns":nc,"nnz":int(M.nnz),"max_row_sum_error":err})
 req(P.shape==(n,int(root_n)),f"composed source ancestry shape mismatch {P.shape}")
 req(np.isfinite(P.data).all() and np.all(P.data>=-1e-14),"composed source ancestry has invalid values")
 err=float(np.max(np.abs(np.asarray(P.sum(axis=1)).ravel()-1),initial=0.0))
 req(err<=3e-10,f"composed source ancestry is nonaffine: {err}")
 return P,levels

def validate_face_ancestry(c,m):
 for ik,vk,nonempty in FACE_CSR:
  req(ik in c and vk in c,f"missing face ancestry CSR {ik}")
  ip=np.asarray(c[ik],dtype=np.int64);v=np.asarray(c[vk])
  req(ip.shape==(m+1,) and ip[0]==0 and ip[-1]==len(v) and np.all(np.diff(ip)>=0),f"malformed {ik}")
  if nonempty:req(np.all(np.diff(ip)>0),f"{ik} has an empty output-face row")
  req(v.ndim==1 and v.dtype.kind in "iu" and np.all(v>=0),f"invalid {vk}")
  for r in range(m):req(len(np.unique(v[ip[r]:ip[r+1]]))==ip[r+1]-ip[r],f"duplicate {ik} row {r}")

def validate_node_ancestry(c,n,parent_face_count):
 for k in NODE_TYPED:
  req(k in c and np.asarray(c[k]).shape==(n,),f"missing/misaligned typed node array {k}")
 req("refinement_node_kind" in c and np.asarray(c["refinement_node_kind"]).shape==(n,),"missing/misaligned refinement_node_kind")
 ip=np.asarray(c.get("refinement_node_parent_face_indptr",[]),dtype=np.int64)
 ix=np.asarray(c.get("refinement_node_parent_face_rows",[]),dtype=np.int64)
 b=np.asarray(c.get("refinement_node_parent_face_barycentric",[]),dtype=np.float64)
 req(ip.shape==(n+1,) and ip[0]==0 and ip[-1]==len(ix) and np.all(np.diff(ip)>=0),"missing/malformed node parent-face CSR")
 req(np.all(ix>=0) and np.all(ix<parent_face_count), "node parent-face ancestry outside declared input domain")
 req(b.shape==(len(ix),3) and np.isfinite(b).all(),"node parent-face barycentrics missing/malformed")
 if len(b):req(np.max(np.abs(b.sum(axis=1)-1))<=2e-9 and b.min()>=-2e-9 and b.max()<=1+2e-9,"node parent-face barycentrics invalid")

def validate_candidate(c,cp,b,base_sha):
 X=np.asarray(c["points_world_m"]);F=np.asarray(c["source_O_faces"])
 req(X.dtype==np.float32 and X.ndim==2 and X.shape[1]==3 and np.isfinite(X).all(),"candidate points must be finite emitted Float32 Nx3")
 req(F.dtype.kind in "iu" and F.ndim==2 and F.shape[1]==3 and len(F)>0,"candidate faces must be integer Fx3")
 n,m=len(X),len(F);F=F.astype(np.int64)
 req(np.all(F>=0) and np.all(F<n),"candidate face indices out of range")
 req(len({tuple(sorted(map(int,t))) for t in F})==len(F),"candidate contains duplicate triangle rows")
 fs=np.sort(F,axis=1);req(np.all(fs[:,1:]!=fs[:,:-1]),"degenerate face repeats a node")
 req("area_reference_037_source_O_faces" in c and np.array_equal(F,c["area_reference_037_source_O_faces"]),"area-reference connectivity differs")
 for k in FACE_TYPED:req(k in c and np.asarray(c[k]).shape[:1]==(m,),f"missing/misaligned typed face array {k}")
 bary=np.asarray(c["refinement_face_parent_barycentric"],dtype=np.float64)
 req(bary.shape==(m,3,3) and np.isfinite(bary).all(),"affine face-parent barycentrics malformed")
 berr=float(np.max(np.abs(bary.sum(axis=2)-1),initial=0.0))
 req(berr<=2e-9 and bary.min(initial=0)>=-2e-9 and bary.max(initial=0)<=1+2e-9,"affine face-parent barycentrics invalid")
 for k in ("parent_coarse_face_row","parent_fine_detail_face_row","parent_source_face_marker","parent_fine_subface_row","refinement_face_parent_face_row"):
  a=np.asarray(c[k]);req(a.dtype.kind in "iu" and np.isfinite(a).all(),f"invalid typed face field {k}")
 validate_face_ancestry(c,m)

 parent_hash=str(np.asarray(c.get("refinement_input_candidate_sha256",np.array([""]))).reshape(-1)[0])
 if parent_hash==base_sha: parent_face_count=len(b["source_O_faces"])
 else: parent_face_count=len(np.asarray(c.get("prior_input_parent__refinement_face_parent_face_row",[])))
 req(parent_face_count>0,"cannot determine declared affine parent-face domain")
 pfr=np.asarray(c["refinement_face_parent_face_row"],dtype=np.int64)
 req(np.all(pfr>=0) and np.all(pfr<parent_face_count),"affine parent-face row outside declared parent domain")
 validate_node_ancestry(c,n,parent_face_count)
 # Validate optional immediate node ancestry under the declared parent domain.
 q=None
 if all(k in c for k in ("immediate_refinement_prolongation_indptr","immediate_refinement_prolongation_node_rows","immediate_refinement_prolongation_weights")):
  ip=np.asarray(c["immediate_refinement_prolongation_indptr"],dtype=np.int64)
  parent_n=len(b["points_world_m"]) if parent_hash==base_sha else len(np.asarray(c.get("prior_input_parent__refinement_node_kind",[])))
  req(parent_n>0,"cannot determine immediate refinement node domain")
  q,_=csr(ip,c["immediate_refinement_prolongation_node_rows"],c["immediate_refinement_prolongation_weights"],n,parent_n,"immediate refinement ancestry")
 csha=sha(cp)
 if csha==base_sha:
  old_to_new=np.arange(len(b["points_world_m"]),dtype=np.int64)
  new_to_old=np.arange(n,dtype=np.int64)
  dptr=drows=None
 else:
  req("remesh806_old_to_new_node_rows" in c and "remesh806_new_to_old_node_rows" in c,"changed candidate lacks remesh806 node correspondence")
  old_to_new=np.asarray(c["remesh806_old_to_new_node_rows"],dtype=np.int64)
  new_to_old=np.asarray(c["remesh806_new_to_old_node_rows"],dtype=np.int64)
  n0=len(b["points_world_m"])
  req(old_to_new.shape==(n0,) and new_to_old.shape==(n,),"remesh806 node correspondence shapes invalid")
  req(np.all((old_to_new>=-1)&(old_to_new<n)) and np.all((new_to_old>=-1)&(new_to_old<n0)),"remesh806 node correspondence out of range")
  for old,newrow in enumerate(old_to_new):
   if newrow>=0:req(new_to_old[newrow]==old,"remesh806 old/new node maps are not inverse")
  for newrow,old in enumerate(new_to_old):
   if old>=0:req(old_to_new[old]==newrow,"remesh806 new/old node maps are not inverse")
  req(len(np.unique(new_to_old[new_to_old>=0]))==np.count_nonzero(new_to_old>=0),"remesh806 surviving node map is not injective")
  req(np.all(new_to_old>=0),"existing chart owner lacks a validated insertion rule for new nodes")
  direct_sha=str(np.asarray(c.get("remesh806_baseline_sha256",np.array([""]))).reshape(-1)[0])
  req(direct_sha==base_sha,"remesh806 direct ancestry baseline hash mismatch")
  ik="remesh806_face_parent_indptr";vk="remesh806_face_parent_face_rows"
  req(ik in c and vk in c,"changed candidate lacks exact remesh806 face ancestry CSR")
  dptr=np.asarray(c[ik],dtype=np.int64);drows=np.asarray(c[vk],dtype=np.int64)
  req(dptr.shape==(m+1,) and dptr[0]==0 and dptr[-1]==len(drows) and np.all(np.diff(dptr)>0),"remesh806 face-parent CSR malformed/empty")
  req(np.all(drows>=0) and np.all(drows<len(b["source_O_faces"])),"remesh806 face-parent rows outside baseline")
  if parent_hash==base_sha:
   # Explicitly rebased affine chart rows must name a baseline 806 face.
   for f in range(m):req(int(pfr[f]) in set(map(int,drows[dptr[f]:dptr[f+1]])),f"face {f} affine 806 parent absent from direct ancestry")
   if q is not None:
    q0=sparse.csr_matrix((np.asarray(c["immediate_refinement_prolongation_weights"],dtype=np.float64),
                          np.asarray(c["immediate_refinement_prolongation_node_rows"],dtype=np.int64),
                          np.asarray(c["immediate_refinement_prolongation_indptr"],dtype=np.int64)),
                         shape=(n,n0))
    m0=sparse.csr_matrix((np.asarray(b["source_prolongation_weights"],dtype=np.float64),
                          np.asarray(b["source_prolongation_node_rows"],dtype=np.int64),
                          np.asarray(b["source_prolongation_indptr"],dtype=np.int64)),
                         shape=(n0,len(np.asarray(b["prior_input_006__source_prolongation_indptr"]))-1))
    expected=(q0@m0).tocsr()
    current=sparse.csr_matrix((np.asarray(c["source_prolongation_weights"],dtype=np.float64),
                               np.asarray(c["source_prolongation_node_rows"],dtype=np.int64),
                               np.asarray(c["source_prolongation_indptr"],dtype=np.int64)),
                              shape=(n,len(np.asarray(c["prior_input_006__source_prolongation_indptr"]))-1))
    diff=(expected-current).tocsr()
    req(diff.nnz==0 or float(np.max(np.abs(diff.data),initial=0.0))<=2e-12,"top source CSR != immediate remesh map times baseline CSR")
  else:
   # Preserved-legacy affine fields remain in their pinned 41,745-face domain.
   base_parent=str(np.asarray(b.get("refinement_input_candidate_sha256",np.array([""]))).reshape(-1)[0])
   req(parent_hash==base_parent,"changed candidate changed its legacy affine parent hash")
   for key in ("refinement_input_candidate_sha256","refinement_input_face_report_sha256"):
    req(key in c and key in b and np.array_equal(c[key],b[key]),"legacy face-parent pin changed: "+key)
   # The two lower source maps are immutable. Every surviving node keeps its exact
   # top-level source and immediate-parent CSR row from 806.
   req(q is not None,"legacy-preserving remesh omitted immediate-parent node CSR")
   for prefix in ("prior_input_006__","prior_run028__"):
    for suffix in ("source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights"):
     key=prefix+suffix
     req(key in c and key in b and np.array_equal(c[key],b[key]),"historical source lineage changed: "+key)
   for newrow,old in enumerate(new_to_old):
    for prefix,idxkey,wkey in (("", "source_prolongation_node_rows","source_prolongation_weights"),
                               ("", "immediate_refinement_prolongation_node_rows","immediate_refinement_prolongation_weights")):
     ip=np.asarray(c[prefix+"source_prolongation_indptr" if idxkey=="source_prolongation_node_rows" else "immediate_refinement_prolongation_indptr"],dtype=np.int64)
     bip=np.asarray(b[prefix+"source_prolongation_indptr" if idxkey=="source_prolongation_node_rows" else "immediate_refinement_prolongation_indptr"],dtype=np.int64)
     ia,ib=map(int,ip[newrow:newrow+2]);ba,bb=map(int,bip[int(old):int(old)+2])
     require_equal=(np.array_equal(c[idxkey][ia:ib],b[idxkey][ba:bb]) and np.array_equal(c[wkey][ia:ib],b[wkey][ba:bb]))
     req(require_equal,f"retained node {newrow} source ancestry row changed")
    for key in NODE_TYPED+("refinement_node_kind","refinement_node_parent_edge_rows","refinement_node_parent_edge_weights"):
     req(key in c and key in b,"missing retained-node typed provenance: "+key)
     cv=np.asarray(c[key])[newrow];bv=np.asarray(b[key])[int(old)]
     equal=np.array_equal(cv,bv,equal_nan=True) if cv.dtype.kind=="f" else np.array_equal(cv,bv)
     req(equal,f"retained node {newrow} typed provenance changed: {key}")
    nip=np.asarray(c["refinement_node_parent_face_indptr"],dtype=np.int64)
    bip=np.asarray(b["refinement_node_parent_face_indptr"],dtype=np.int64)
    ca,cb=map(int,nip[newrow:newrow+2]);ba,bb=map(int,bip[int(old):int(old)+2])
    req(np.array_equal(c["refinement_node_parent_face_rows"][ca:cb],b["refinement_node_parent_face_rows"][ba:bb])
        and np.array_equal(c["refinement_node_parent_face_barycentric"][ca:cb],b["refinement_node_parent_face_barycentric"][ba:bb]),
        f"retained node {newrow} face-parent ancestry changed")
   # Each output face's typed source ancestry is the union of its exact 806 overlap parents.
   for f in range(m):
    parents=drows[dptr[f]:dptr[f+1]]
    for ik0,vk0,_ in FACE_CSR[:3]:
     ip0=np.asarray(b[ik0],dtype=np.int64);iv0=np.asarray(b[vk0])
     expected_rows=set()
     for oldface in parents:expected_rows.update(map(int,iv0[ip0[int(oldface)]:ip0[int(oldface)+1]]))
     ip1=np.asarray(c[ik0],dtype=np.int64);iv1=np.asarray(c[vk0])
     actual=set(map(int,iv1[ip1[f]:ip1[f+1]]))
     req(actual==expected_rows,f"face {f} {ik0} does not equal the direct-806-parent ancestry union")
    for key in ("face_kind","parent_coarse_face_row","parent_fine_detail_face_row","parent_source_face_marker","parent_fine_subface_row"):
     values=np.asarray(b[key])[parents]
     if len(values) and np.all(values==values[0]):
      req(np.asarray(c[key])[f]==values[0],f"face {f} typed scalar provenance disagrees with direct 806 parents: {key}")
 return F,old_to_new,new_to_old,dptr,drows,parent_hash,berr

def topology(F):
 from collections import defaultdict,Counter
 e=defaultdict(list)
 for fi,t in enumerate(np.asarray(F,dtype=np.int64)):
  for a,b in zip(t,np.roll(t,-1)):
   a,b=int(a),int(b);e[(min(a,b),max(a,b))].append((a,b))
 hist=Counter(map(len,e.values()))
 return {"edges":len(e),"incidence_histogram":{str(k):int(v) for k,v in sorted(hist.items())},
         "nonmanifold_edges":sum(1 for v in e.values() if len(v)>2),
         "same_direction_shared_edges":sum(1 for v in e.values() if len(v)==2 and v[0]==v[1]),
         "boundary_edges":int(hist.get(1,0))}
def boundary_edges(F):
 from collections import defaultdict
 e=defaultdict(list)
 for t in np.asarray(F,dtype=np.int64):
  for a,b in zip(t,np.roll(t,-1)):
   a,b=int(a),int(b);e[(min(a,b),max(a,b))].append((a,b))
 return {k for k,v in e.items() if len(v)==1}
def cyclic_same(a,b):
 a=tuple(map(int,a));b=tuple(map(int,b))
 return any(a==b[i:]+b[:i] for i in range(3))

def face_flags(outer,inner,faces,field):
 src=np.asarray(faces,dtype=np.int64);order=np.argsort(src,axis=1);f=np.take_along_axis(src,order,axis=1)
 inv=((src[:,0]>src[:,1]).astype(np.int8)+(src[:,0]>src[:,2]).astype(np.int8)+(src[:,1]>src[:,2]).astype(np.int8))
 parity=(inv&1).astype(bool);a,b,c=f.T;O=np.asarray(outer,dtype=np.float64);I=np.asarray(inner,dtype=np.float64)
 def vol(x,y,z,w):return np.einsum("ij,ij->i",np.cross(y-x,z-x),w-x)/6.0
 v0=vol(O[a],O[b],O[c],I[c]);v1=vol(O[a],O[b],I[b],I[c]);v2=vol(O[a],I[a],I[b],I[c])
 scalar=np.asarray(field,dtype=np.float64)
 tol=np.maximum(1e-8,np.maximum.reduce([scalar[a],scalar[b],scalar[c]]))**3*1e-12
 expected=np.where(parity,1.0,-1.0)
 authored=(np.isfinite(v0)&np.isfinite(v1)&np.isfinite(v2)&(np.abs(v0)>tol)&(np.abs(v1)>tol)&(np.abs(v2)>tol)&(np.sign(v0)==expected)&(np.sign(v1)==-expected)&(np.sign(v2)==expected))
 ids=np.stack([np.stack([a,b,c,len(O)+c],1),np.stack([a,b,len(O)+b,len(O)+c],1),np.stack([a,len(O)+a,len(O)+b,len(O)+c],1)],axis=1).reshape(-1,4)
 xyz64=np.concatenate([O,I],0);q=xyz64[ids]
 v64=np.einsum("ij,ij->i",np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])/6.0
 rev=v64<0;ids[rev,0],ids[rev,1]=ids[rev,1].copy(),ids[rev,0].copy()
 xyz32=np.concatenate([np.asarray(outer,dtype=np.float32).astype(np.float64),np.asarray(inner,dtype=np.float32).astype(np.float64)],0)
 p=xyz32[ids];vols=np.einsum("ij,ij->i",np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),p[:,3]-p[:,0])/6.0
 longest=np.zeros(len(p))
 for i in range(4):
  for j in range(i+1,4):longest=np.maximum(longest,np.linalg.norm(p[:,i]-p[:,j],axis=1))
 ndet=6*vols/np.maximum(longest,1e-300)**3
 return {"authored_bad":~authored,"nonpositive":(vols<=0).reshape(-1,3),
         "conditioned":(ndet<1e-5).reshape(-1,3),"normalized_det":ndet.reshape(-1,3),"volumes":vols.reshape(-1,3)}

def compare_flags(new,base,face_map,exact_geometry):
 inherited=exact_geometry.copy();changed=~inherited;bad={
  "authored_bad_faces":np.flatnonzero(new["authored_bad"]&changed).astype(int).tolist(),
  "nonpositive_tets":[[int(f),int(t)] for f,t in np.argwhere(new["nonpositive"]&changed[:,None])],
  "conditioned_tets":[[int(f),int(t)] for f,t in np.argwhere(new["conditioned"]&changed[:,None])]}
 mismatch=[]
 for nf in np.flatnonzero(inherited):
  bf=int(face_map[nf])
  for key in ("authored_bad","nonpositive","conditioned"):
   if not np.array_equal(np.asarray(new[key][nf]),np.asarray(base[key][bf])):
    mismatch.append({"output_face":int(nf),"baseline_face":bf,"flag":key})
 return bad,mismatch

def self_checks():
 # A missing output-face ancestry row must reject.
 try:
  validate_face_ancestry({"face_source_ancestry_indptr":np.array([0,0]),"face_source_ancestry_face_rows":np.array([],np.int32),
                          "face_fine_subface_ancestry_indptr":np.array([0,0]),"face_fine_subface_ancestry_rows":np.array([],np.int32),
                          "face_fine_detail_ancestry_indptr":np.array([0,0]),"face_fine_detail_ancestry_rows":np.array([],np.int32),
                          "refinement_face_parent_ancestry_indptr":np.array([0,0]),"refinement_face_parent_ancestry_face_rows":np.array([],np.int32)},1)
  missing_rejected=False
 except TransferError:missing_rejected=True
 req(missing_rejected,"missing-ancestry rejection self-check failed")
 # A zero-thickness prism must be caught by both authored and Float32-positive-volume guards.
 O=np.array([[0,0,0],[0,1,0],[1,0,0]],np.float64);I=np.array([[0,0,0],[0,1,0],[1,0,0]],np.float64)
 F=np.array([[0,2,1]],np.int64);d=face_flags(O,I,F,np.ones(3)*0.001)
 zero_rejected=bool(d["authored_bad"][0] or d["nonpositive"][0].any())
 req(zero_rejected,"invalid-cell rejection self-check failed")
 return {"missing_face_ancestry_rejected":True,"zero_thickness_cell_rejected":True}

def run(candidate_path,outdir,baseline_path,baseline_layer_path,parity_ref=None):
 candsha=sha(candidate_path);basesha=sha(baseline_path)
 outdir.mkdir(parents=True,exist_ok=False)
 for name,path,key in (("r0",R0,"r0"),("source146",A146,"146"),("layer128",L128,"layer128"),("owner408",OWNER408,"owner408"),("owner541",OWNER541,"owner541")):
  req(sha(path)==PIN[key],f"pinned {name} SHA mismatch")
 selfcheck=self_checks()
 cfile=np.load(candidate_path,allow_pickle=False);bfile=np.load(baseline_path,allow_pickle=False)
 c={k:np.asarray(cfile[k]).copy() for k in cfile.files};b={k:np.asarray(bfile[k]).copy() for k in bfile.files}
 l=load_dict(L128);bl=load_dict(baseline_layer_path)
 with np.load(R0,allow_pickle=False) as r,np.load(A146,allow_pickle=False) as a:
  for k in NODE_TYPED:req(np.array_equal(a[k],r[k],equal_nan=True),f"146/R0 typed mismatch {k}")
  for k in ("node_kind","node_source_row"):req(np.array_equal(a[k],l[k],equal_nan=True),f"146/128 typed mismatch {k}")
  req(np.array_equal(np.sort(a["source_O_faces"],axis=1),np.sort(l["outer_faces"],axis=1)),"128/146 outer topology mismatch")
  req(np.array_equal(np.asarray(a["points_world_m"],np.float64),np.asarray(l["outer_points_f64"],np.float64)),"128/146 outer point mismatch")
  root_n=len(r["points_before_joint_self_pair_repair_m"])
  F,old_to_new,new_to_old,dptr,drows,parent_hash,baryerr=validate_candidate(c,candidate_path,b,basesha)
  chart_validation=None
  if candsha!=basesha:
   req(sha(CHART_OWNER)==PIN["chartOwner"],"pinned material-chart validator hash mismatch")
   chart_out=outdir/"material-chart-validation.json"
   proc=subprocess.run([sys.executable,str(CHART_OWNER),str(baseline_path),str(candidate_path),str(chart_out),
                        "--expected-baseline",basesha,"--expected-candidate",candsha],capture_output=True,text=True)
   if proc.returncode!=0:
    chart_fail=outdir/"material-chart-validation-failure.json"
    chart_fail.write_text(json.dumps({"status":"REJECTED_MATERIAL_CHART","returncode":proc.returncode,
      "stdout":proc.stdout,"stderr":proc.stderr,"validator_sha256":sha(CHART_OWNER)},indent=2)+"\n")
    raise TransferError("independent material-chart validator rejected candidate; see "+str(chart_fail))
   chart_validation={"path":str(chart_out),"sha256":sha(chart_out),
      "status":json.loads(chart_out.read_text()).get("status"),"validator_sha256":sha(CHART_OWNER)}
  P,levels=compose(c,root_n);n=len(c["points_world_m"]);m=len(F)
  owner408=import_owner("surface_ref_408",OWNER408)
  owner_ref,owner_receipts=owner408.compose_source_reference(cfile,R0)
  source_ref=np.asarray(P@np.asarray(r["points_before_joint_self_pair_repair_m"],dtype=np.float64))
  req(owner_ref.shape==source_ref.shape and np.allclose(owner_ref,source_ref,rtol=0,atol=2e-12),"408 owner source reference differs from composed CSR")
  O=np.asarray(c["points_world_m"],dtype=np.float32).astype(np.float64)
  di=np.asarray(l["inner_points_f64"]-l["outer_points_f64"],dtype=np.float64)
  dd=np.asarray(l["deep_points_f64"]-l["inner_points_f64"],dtype=np.float64)
  sd0=np.asarray(l["skin_depth_m"],dtype=np.float64);fd0=np.asarray(l["fat_depth_m"],dtype=np.float64)
  req(di.shape==(root_n,3) and dd.shape==(root_n,3) and len(sd0)==root_n and len(fd0)==root_n,"layer source root shape mismatch")
  skin_off=np.asarray(P@di);fat_off=np.asarray(P@dd)
  skin_depth=np.asarray(P@sd0).ravel();fat_depth=np.asarray(P@fd0).ravel()
  I=O+skin_off;Deep=I+fat_off
  req(np.isfinite(O).all() and np.isfinite(I).all() and np.isfinite(Deep).all() and np.isfinite(skin_depth).all() and np.isfinite(fat_depth).all(),"transferred layer has nonfinite values")
  skin_norm=np.linalg.norm(I-O,axis=1);fat_norm=np.linalg.norm(Deep-I,axis=1)
  req(np.all(skin_norm>0) and np.all(fat_norm>0),"transferred layer has zero nodal thickness")
  rawvol=float(np.einsum("ij,ij->i",O[F[:,0]],np.cross(O[F[:,1]],O[F[:,2]])).sum()/6)
  lf=F[:,[0,2,1]].copy()
  layer_vol=float(np.einsum("ij,ij->i",O[lf[:,0]],np.cross(O[lf[:,1]],O[lf[:,2]])).sum()/6)
  req(rawvol<0 and layer_vol>0,"candidate winding violates 541 raw-inward/reversed-layer convention")
  top=topology(F)
  bt=topology(b["source_O_faces"])
  req(top["nonmanifold_edges"]==0 and top["same_direction_shared_edges"]==0,"candidate surface has nonmanifold or inconsistent shared edges")
  oldbe=boundary_edges(b["source_O_faces"]);newbe=boundary_edges(F);mapped=set()
  for a0,b0 in oldbe:
   x,y=int(old_to_new[a0]),int(old_to_new[b0]);req(x>=0 and y>=0,"candidate removed a pinned surface boundary node")
   mapped.add((min(x,y),max(x,y)))
  req(mapped==newbe,"candidate changed the whole-surface boundary edge set")
  base_raw=np.asarray(b["source_O_faces"],dtype=np.int64)
  req(np.array_equal(np.asarray(bl["outer_faces"]),base_raw[:,[0,2,1]]),"baseline layer winding/topology does not match baseline candidate")
  baseO=np.asarray(bl["outer_points_f64"],dtype=np.float64);baseI=np.asarray(bl["inner_points_f64"],dtype=np.float64);baseD=np.asarray(bl["deep_points_f64"],dtype=np.float64)
  baseSF=np.asarray(bl["skin_thickness_transferred_m"],dtype=np.float64);baseFF=np.asarray(bl["fat_thickness_transferred_m"],dtype=np.float64)
  base_idx={}
  for i,t in enumerate(base_raw):
   key=tuple(sorted(map(int,t)));req(key not in base_idx,"baseline contains duplicate faces; face identity ambiguous");base_idx[key]=i
  face_to_base=np.full(m,-1,dtype=np.int64);exact_geom=np.zeros(m,dtype=bool)
  for fi,t in enumerate(F):
   mappednodes=new_to_old[t]
   if np.any(mappednodes<0):continue
   br=base_idx.get(tuple(sorted(map(int,mappednodes))),-1)
   if br<0 or not cyclic_same(mappednodes,base_raw[br]):continue
   face_to_base[fi]=br
   if (np.array_equal(O[t],baseO[mappednodes]) and np.array_equal(I[t],baseI[mappednodes]) and np.array_equal(Deep[t],baseD[mappednodes])):
    exact_geom[fi]=True
  owner541=import_owner("surface_layer_541",OWNER541)
  skinm=owner541.layer_metrics(O,I,lf,skin_depth)
  fatm=owner541.layer_metrics(I,Deep,lf,fat_depth)
  skinf=face_flags(O,I,lf,skin_depth);fatf=face_flags(I,Deep,lf,fat_depth)
  req(int(skinf["authored_bad"].sum())==skinm["authored_prism_orientation_fail_count"],"skin authored-face identity reduction mismatch")
  req(int(skinf["nonpositive"].sum())==skinm["nonpositive_f32_tet_count"],"skin nonpositive-tet identity reduction mismatch")
  req(int(skinf["conditioned"].sum())==skinm["f32_conditioned_fail_count_lt_1e-5"],"skin conditioning identity reduction mismatch")
  req(int(fatf["authored_bad"].sum())==fatm["authored_prism_orientation_fail_count"],"fat authored-face identity reduction mismatch")
  req(int(fatf["nonpositive"].sum())==fatm["nonpositive_f32_tet_count"],"fat nonpositive-tet identity reduction mismatch")
  req(int(fatf["conditioned"].sum())==fatm["f32_conditioned_fail_count_lt_1e-5"],"fat conditioning identity reduction mismatch")
  bsk=face_flags(baseO,baseI,np.asarray(bl["outer_faces"]),baseSF)
  bft=face_flags(baseI,baseD,np.asarray(bl["outer_faces"]),baseFF)
  skin_bad,skin_mismatch=compare_flags(skinf,bsk,face_to_base,exact_geom)
  fat_bad,fat_mismatch=compare_flags(fatf,bft,face_to_base,exact_geom)
  cell_gate=not any(skin_bad.values()) and not any(fat_bad.values()) and not skin_mismatch and not fat_mismatch
  payload={
   "outer_points_f64":O,"inner_points_f64":I,"deep_points_f64":Deep,
   "outer_points_f32":O.astype(np.float32),"inner_points_f32":I.astype(np.float32),"deep_points_f32":Deep.astype(np.float32),
   "outer_faces":lf,"inner_faces":lf,"deep_faces":lf,"source_outer_faces_raw":F,
   "face_winding_permutation":np.array([0,2,1],dtype=np.int8),
   "skin_thickness_transferred_m":skin_depth,"fat_thickness_transferred_m":fat_depth,
   "skin_thickness_actual_norm_m":skin_norm,"fat_thickness_actual_norm_m":fat_norm,
   "node_kind":c["node_kind"],"node_source_row":c["node_source_row"],
   "source_face_ancestry_indptr":c["face_source_ancestry_indptr"],
   "source_face_ancestry_rows":c["face_source_ancestry_face_rows"],
   "node_source_virtual_row":c["node_source_virtual_row"],
   "node_parent_coarse_face_row":c["node_parent_coarse_face_row"],
   "node_parent_coarse_edge_a":c["node_parent_coarse_edge_a"],
   "node_parent_coarse_edge_b":c["node_parent_coarse_edge_b"],
   "node_parent_coarse_edge_parameter":c["node_parent_coarse_edge_parameter"],
   "full_source_prolongation_indptr":P.indptr.astype(np.int64),
   "full_source_prolongation_rows":P.indices.astype(np.int64),
   "full_source_prolongation_weights":P.data.astype(np.float64),
  }
  keep=set(NODE_TYPED+FACE_TYPED)
  for ik,vk,_ in FACE_CSR:keep|={ik,vk}
  keep|={"source_prolongation_indptr","source_prolongation_node_rows","source_prolongation_weights","source_prolongation_semantics",
   "refinement_node_kind","refinement_node_kind_semantics","refinement_node_parent_edge_rows","refinement_node_parent_edge_weights",
   "refinement_node_parent_face_indptr","refinement_node_parent_face_rows","refinement_node_parent_face_barycentric",
   "refinement_node_parent_face_barycentric_semantics","immediate_refinement_prolongation_indptr",
   "immediate_refinement_prolongation_node_rows","immediate_refinement_prolongation_weights",
   "immediate_refinement_prolongation_semantics","refinement_face_parent_barycentric_semantics",
   "refinement_face_parent_ancestry_semantics","face_source_ancestry_semantics",
   "face_fine_subface_ancestry_semantics","face_fine_detail_ancestry_semantics",
   "refinement_input_candidate_sha256","refinement_input_face_report_sha256",
   "remesh806_old_to_new_node_rows","remesh806_new_to_old_node_rows",
   "remesh806_baseline_sha256","remesh806_face_parent_indptr","remesh806_face_parent_face_rows"}
  for k in keep:
   if k in c:payload[k]=np.asarray(c[k])
  parity={"requested":parity_ref is not None,"pass":None,"keys_compared":0,"mismatches":[]}
  if parity_ref:
   with np.load(parity_ref,allow_pickle=False) as q:
    for k in q.files:
     parity["keys_compared"]+=1
     if k not in payload or not bytes_equal(payload[k],q[k]):
      row={"key":k,"present":k in payload}
      if k in payload and payload[k].shape==q[k].shape:
       row["output_dtype"]=str(payload[k].dtype);row["reference_dtype"]=str(q[k].dtype)
       if payload[k].dtype.kind in "fiu":row["max_abs_delta"]=float(np.max(np.abs(payload[k].astype(np.float64)-q[k].astype(np.float64)),initial=0))
      parity["mismatches"].append(row)
   parity["pass"]=not parity["mismatches"]
  gate=cell_gate and (parity["pass"] is not False)
  report={
   "schema":"numi.human.skin-fat-topology-independent-transfer.v1",
   "status":"PARITY_PASS_DIAGNOSTIC" if parity["pass"] and gate else ("TRANSFER_READY_DIAGNOSTIC_ONLY" if gate else "REJECTED_CHANGED_CELL_OR_PARITY_GATE"),
   "candidate_path":str(candidate_path),"candidate_sha256":candsha,
   "baseline_candidate_path":str(baseline_path),"baseline_candidate_sha256":basesha,
   "baseline_layer_path":str(baseline_layer_path),"baseline_layer_sha256":sha(baseline_layer_path),
   "pins":{"r0_sha256":sha(R0),"source146_sha256":sha(A146),"layer128_sha256":sha(L128),"owner408_sha256":sha(OWNER408),"owner541_sha256":sha(OWNER541),"material_chart_validator_sha256":sha(CHART_OWNER)},
   "lineage":{"prefix_order":list(PFX),"levels":levels,"composed_shape":list(P.shape),"composed_nnz":int(P.nnz),
    "owner408_source_reference_max_abs_delta_m":float(np.max(np.abs(owner_ref-source_ref),initial=0)),
    "owner408_receipts":owner_receipts,"immediate_parent_hash":parent_hash,"face_parent_barycentric_max_sum_error":baryerr,
    "source_face_ancestry_rows":int(len(c["face_source_ancestry_face_rows"])),
    "direct_remesh806_face_parent_rows":None if drows is None else int(len(drows)),
    "typed_node_and_face_ancestry_preserved":True},
   "material_chart_validation":chart_validation,
   "topology":{"nodes":n,"faces":m,"baseline_nodes":len(b["points_world_m"]),"baseline_faces":len(base_raw),
    "edge_metrics":top,"baseline_edge_metrics":bt,"boundary_edges_preserved":True,
    "bitwise_unchanged_cell_faces":int(exact_geom.sum()),"changed_or_new_faces":int((~exact_geom).sum())},
   "winding":{"candidate_raw_signed_volume_m3":rawvol,"layer_reversed_signed_volume_m3":layer_vol,"per_face_permutation":[0,2,1]},
   "transfer":{"method":"P = output-to-13,518 CSR × 13,518-to-10,145 CSR × 10,145-to-8,776 CSR; transfer 128 skin/fat displacement vectors and depth fields; reanchor at emitted Float32 output surface",
    "composed_source_root_nodes":P.shape[1],"skin_scalar_depth_range_m":[float(skin_depth.min()),float(skin_depth.max())],
    "fat_scalar_depth_range_m":[float(fat_depth.min()),float(fat_depth.max())],
    "skin_actual_norm_range_m":[float(skin_norm.min()),float(skin_norm.max())],
    "fat_actual_norm_range_m":[float(fat_norm.min()),float(fat_norm.max())]},
   "cell_checks":{"skin":skinm,"fat":fatm,
    "gate":"Only exact bitwise-inherited face geometry may retain its baseline cell flags; every changed/new face must pass authored orientation, positive emitted-Float32 tet volume, and normalized determinant >=1e-5.",
    "changed_face_invalids":{"skin":skin_bad,"fat":fat_bad},
    "inherited_cell_flag_mismatches":{"skin":skin_mismatch,"fat":fat_mismatch},
    "changed_face_cell_gate_pass":cell_gate,
    "strict_zero_invalid_global_cell_pass":all(skinm[k]==0 and fatm[k]==0 for k in ("authored_prism_orientation_fail_count","nonpositive_f32_tet_count","f32_conditioned_fail_count_lt_1e-5"))},
   "parity":parity,
   "material_mass":{"conservative_cell_mass_transfer":False,"cell_ownership_available":False,
    "reason":"The pinned source has surface offsets/ancestry only; no volume cells, parent-tet IDs, or cell mass/material fields are supplied."},
   "physical_or_native_admission":False,"accepted_physical_steps":0,
   "rejection_self_checks":selfcheck,
   "limits":["Surface offset transfer and deterministic prism validity checks only; no volume-cell mass ownership.","No all-bank bone/self audit or physical/native admission."]
  }
  report["candidate_transfer_gate_pass"]=bool(gate)
  if gate:
   out=outdir/"skin-fat-layers-541.npz"
   with out.open("xb") as f:np.savez_compressed(f,**payload)
   report["output"]={"path":str(out),"sha256":sha(out),"bytes":out.stat().st_size}
  rp=outdir/"report.json";rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
  cfile.close();bfile.close()
  return report,rp

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--candidate",type=Path,default=DEFAULT_C)
 ap.add_argument("--outdir",type=Path,required=True)
 ap.add_argument("--baseline-candidate",type=Path,default=DEFAULT_C)
 ap.add_argument("--baseline-layer",type=Path,default=DEFAULT_L)
 ap.add_argument("--parity-reference",type=Path,default=None)
 a=ap.parse_args()
 try:
  report,rp=run(a.candidate,a.outdir,a.baseline_candidate,a.baseline_layer,a.parity_reference)
  print(json.dumps({"report":str(rp),"status":report["status"],"pass":report["candidate_transfer_gate_pass"],
    "parity":report["parity"],"skin":report["cell_checks"]["skin"],"fat":report["cell_checks"]["fat"],
    "output":report.get("output")},indent=2))
  if not report["candidate_transfer_gate_pass"]:raise SystemExit(2)
 except SystemExit:raise
 except Exception as e:
  a.outdir.mkdir(parents=True,exist_ok=True)
  failure={"schema":"numi.human.skin-fat-topology-independent-transfer.failure.v1",
   "status":"REJECTED_INPUT_OR_TRANSFER","candidate_path":str(a.candidate),
   "candidate_sha256":sha(a.candidate) if a.candidate.exists() else None,
   "baseline_candidate_path":str(a.baseline_candidate),
   "baseline_candidate_sha256":sha(a.baseline_candidate) if a.baseline_candidate.exists() else None,
   "error":repr(e),"physical_or_native_admission":False}
  p=a.outdir/"failure.json"
  if not p.exists():p.write_text(json.dumps(failure,indent=2,sort_keys=True)+"\n")
  print(json.dumps(failure,indent=2));raise SystemExit(2)
if __name__=="__main__":main()
