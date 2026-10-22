#!/usr/bin/env python3
"""Preparation regression: all prism face parities and free/fixed point maps."""
from pathlib import Path
import argparse,hashlib,itertools,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hand_layer_barrier_jacobian import prism_ratio_jacobian,width_constraints_jacobian
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rng=np.random.default_rng(44021);worst=0.;rel=0.;counts=0
for fixed in (False,True):
 for case in range(12):
  o=rng.normal(size=(5,3))*2;i=o+rng.normal(size=(5,3))*.7;d=i+rng.normal(size=(5,3))*.9
  faces=np.array(list(itertools.permutations(range(3))),int);om=np.full(5,-1);im=np.arange(5)*3;dm=15+np.arange(5)*3;nd=30
  if fixed:im[0]=-1;dm[0]=-1
  x=np.r_[i.ravel(),d.ravel()];N=rng.normal(size=(5,3));N/=np.linalg.norm(N,axis=1)[:,None];rows=np.arange(5);lo=np.full(5,.3);hi=np.full(5,3.);av=np.full(5,5.)
  def fn(x):
   ip=x[:15].reshape(5,3).copy();dp=x[15:].reshape(5,3).copy()
   if fixed:ip[0]=i[0];dp[0]=d[0]
   aa,aj=prism_ratio_jacobian(o,ip,faces,om,im,nd);bb,bj=prism_ratio_jacobian(ip,dp,faces,im,dm,nd);cc,cj=width_constraints_jacobian(o,ip,dp,rows,N,lo,hi,lo,hi,av,im,dm,nd)
   return np.r_[aa,bb,cc],np.vstack((aj,bj,cj))
  v,j=fn(x);h=2e-6
  num=np.column_stack([(fn(x+np.eye(nd)[k]*h)[0]-fn(x-np.eye(nd)[k]*h)[0])/(2*h) for k in range(nd)])
  worst=max(worst,float(np.max(np.abs(num-j))));rel=max(rel,float(np.linalg.norm(num-j)/max(np.linalg.norm(num),1e-30)))
  assert np.isfinite(j).all() and np.max(np.abs(num-j))<1e-7
  counts+=1
report={'status':'PASS','seed':44021,'cases':counts,'all_six_face_permutations':True,'fixed_and_free_point_maps':True,'width_and_inward_constraints':True,'central_difference_step_mm':2e-6,'max_absolute_derivative_error':worst,'max_relative_frobenius_error':rel,'helper_sha256':hashlib.sha256(Path(__file__).with_name('hand_layer_barrier_jacobian.py').read_bytes()).hexdigest(),'scope':'CPU geometry preparation derivatives; not native dynamics, anatomy or physical qualification'}
a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
