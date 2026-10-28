#!/usr/bin/env python3
"""Connected offline obstacle reconstruction. Native physics admission is separate."""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, subprocess, sys, time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001')
OUT=Path(__file__).resolve().parent
PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def query(points,folder,bank_directory=None):
 folder.mkdir()
 pts=np.asarray(points,dtype=np.float32);bits=pts.view(np.uint32)
 p=folder/'points.csv'
 with p.open('w') as f:
  f.write('query_row,x_m,y_m,z_m,x_bits_u32,y_bits_u32,z_bits_u32\n')
  for i,(v,b) in enumerate(zip(pts,bits)):f.write(','.join([str(i),*[format(float(x),'.9g') for x in v],*[str(int(x)) for x in b]])+'\n')
 bank=Path(bank_directory) if bank_directory is not None else ROOT/'inputs/bank-export-039/source-preflight';binary=ROOT/'tools/numi-matter-human-skin-fat-static-query'
 args=[str(binary),str(bank/'exact-emitted.triangles.csv'),str(bank/'exact-emitted.bodies.csv'),str(p),str(folder/'pairs.csv'),str(folder/'nodes.csv'),'185']
 cp=subprocess.run(args,capture_output=True,text=True,timeout=120);(folder/'run.log').write_text(cp.stdout+cp.stderr)
 if cp.returncode:raise RuntimeError(cp.stdout+cp.stderr)
 rows=list(csv.DictReader((folder/'pairs.csv').open()));nodes=list(csv.DictReader((folder/'nodes.csv').open()))
 assert len(nodes)==len(pts)
 assert all(int(r['ambiguous_banks'])==0 and int(r['invalid_banks'])==0 for r in nodes)
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='run-001');ap.add_argument('--iterations',type=int,default=12);ap.add_argument('--rings',type=int,default=6);a=ap.parse_args()
 out=OUT/a.run;out.mkdir(exist_ok=False);t=time.monotonic()
 base=ROOT/'inputs/baseline-806.npz';z=np.load(base,allow_pickle=False);payload={k:z[k].copy() for k in z.files};X0=payload['points_world_m'];X=X0.copy();F=payload['source_O_faces'];n=len(X)
 doc=json.load(open(ROOT/'baseline/audit-159/report.json'));seeds=sorted({int(f) for r in doc['banks'] for f,b in (r.get('result') or {}).get('triangle_pairs',[])})
 edges=np.unique(np.sort(np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]),axis=1),axis=0)
 adj=sparse.csr_matrix((np.ones(2*len(edges)),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
 support=set(map(int,F[seeds].ravel()));front=set(support)
 for _ in range(a.rings):front=set(map(int,adj[list(front)].indices))-support;support.update(front)
 fixed=set(map(int,payload['typed_coarse_anchor_output_rows_018']));support=np.array(sorted(support-fixed),dtype=np.int64);lookup=np.full(n,-1,dtype=np.int64);lookup[support]=np.arange(len(support));ns=len(support)
 faces=np.flatnonzero(np.any(np.isin(F,support),axis=1));touched=F[faces]
 L=sparse.diags(np.asarray(adj.sum(1)).ravel())-adj
 # Uniform graph energy avoids negative cotangent weights on inherited slivers.
 Ln=sparse.diags(1/np.maximum(np.asarray(adj.sum(1)).ravel(),1))@L
 Ls=Ln[:,support];Q=(Ls.T@Ls+.025*L[:,support].T@L[:,support]+1e-5*sparse.eye(ns)).tocsc()
 # Fixed barycentric stencil includes all vertices, midpoints, centroid, and quarter-edge points.
 bary=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3,1/3,1/3],[.75,.25,0],[.25,.75,0],[.75,0,.25],[.25,0,.75],[0,.75,.25],[0,.25,.75]],dtype=float)
 nb=len(bary);sf=np.repeat(faces,nb);weights=np.tile(bary,(len(faces),1));sv=F[sf]
 # Matrix contains only free columns; fixed nodes have zero displacement.
 sr=np.repeat(np.arange(len(sf)),3);sc=lookup[sv].ravel();sd=weights.ravel();valid=sc>=0
 B=sparse.csr_matrix((sd[valid],(sr[valid],sc[valid])),shape=(len(sf),ns))
 helper=mod('obstacle_bankhelper',ROOT/'tools/human_skin_fat_bone_audit.py');_,_,_,_,M,off,pos,_,_,_=helper.read_bank_inputs(ROOT/'inputs/bank-export-039/source-preflight',ROOT/'tools')
 invM={k:np.linalg.inv(v) for k,v in M.items()}
 report={'baseline_sha256':sha(base),'script_sha256':sha(__file__),'host':'numi.local','support_nodes':ns,'affected_faces':len(faces),'sampling_rows':len(sf),'construction':'Full signed exit-depth targets; connected quadratic displacement regularization; fixed outside and exact anchors. Offline geometry preparation only.','native_admission':False,'iterations':[]}
 oldtri=X0.astype(float)[touched];oldnorm=np.cross(oldtri[:,1]-oldtri[:,0],oldtri[:,2]-oldtri[:,0]);oldlen=np.linalg.norm(oldnorm,axis=1)
 for it in range(a.iterations):
  itout=out/f'iteration-{it:03d}';itout.mkdir();sample=np.einsum('ij,ijk->ik',weights,X.astype(float)[sv]).astype(np.float32)
  qr=query(sample,itout/'query');inside=[r for r in qr if r['classification']=='inside'];print(json.dumps({'iteration':it,'inside_samples':len(inside),'elapsed_s':time.monotonic()-t}),flush=True)
  if not inside:break
  rows=[];target=[];depths=[]
  for r in inside:
   i=int(r['query_row']);body=int(r['bank_body']);p=np.array([float(r['query_local_'+c]) for c in 'xyz']);c=np.array([float(r['closest_'+c]) for c in 'xyz']);v=c-p;dep=np.linalg.norm(v)
   if dep<1e-12:continue
   # Exact inverse of raw matrix, rather than transpose assumption.
   d=(v*(1+.00015/dep))@invM[body].T
   rows.append(i);target.append(d*1000);depths.append(dep*1000)
  C=B[rows];target=np.array(target);H=(Q+80*(C.T@C)).tocsc();rhs=80*C.T@target
  delta=splu(H).solve(rhs)/1000
  # Bound only each continuation increment, not final reconstructed anatomy.
  norm=np.linalg.norm(delta,axis=1);delta*=min(1.,.0008/max(float(norm.max()),1e-30))
  trial=None;trials=[]
  for alpha in [1.,.5,.25,.125,.0625,.03125]:
   Y=X.copy();Y[support]=(X[support].astype(float)+alpha*delta).astype(np.float32)
   tri=Y.astype(float)[touched];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);length=np.linalg.norm(normal,axis=1);cos=np.einsum('ij,ij->i',normal,oldnorm)/np.maximum(length*oldlen,1e-40)
   # Reject actual degeneracy or reversal relative to the fixed source chart.
   good=bool(np.all(length>1e-14) and np.all(cos>.1) and np.all(length>.1*oldlen))
   trials.append({'alpha':alpha,'pass_orientation':good,'min_area_ratio':float(np.min(length/np.maximum(oldlen,1e-40))),'min_normal_cosine':float(cos.min())})
   if good:trial=(Y,alpha);break
  rec={'iteration':it,'inside_samples':len(inside),'maximum_depth_mm':max(depths),'median_depth_mm':float(np.median(depths)),'max_increment_mm':float(np.max(np.linalg.norm(delta,axis=1))*1000),'line_search':trials}
  if trial is None:rec['status']='SOURCE_ORIENTATION_LIMIT';report['iterations'].append(rec);break
  X,alpha=trial;rec['alpha']=alpha;report['iterations'].append(rec)
  # Small checkpoint reuses the native existing NPZ field conventions.
  np.savez_compressed(itout/'points.npz',points_world_m=X,support_nodes=support,source_O_faces=F)
  (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
 payload['points_world_m']=X;payload['connected_obstacle_207_delta_m']=X.astype(float)-X0.astype(float)
 np.savez_compressed(out/'candidate.npz',**payload)
 report.update(candidate_sha256=sha(out/'candidate.npz'),changed_nodes=int(np.count_nonzero(np.any(X.view(np.uint32)!=X0.view(np.uint32),axis=1))),max_displacement_mm=float(np.max(np.linalg.norm(X.astype(float)-X0.astype(float),axis=1))*1000),outside_bitwise_unchanged=bool(np.array_equal(X[np.setdiff1d(np.arange(n),support)].view(np.uint32),X0[np.setdiff1d(np.arange(n),support)].view(np.uint32))),anchors_bitwise_unchanged=bool(np.array_equal(X[list(fixed)].view(np.uint32),X0[list(fixed)].view(np.uint32))),elapsed_s=time.monotonic()-t)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='iterations'}),flush=True)
if __name__=='__main__':main()
