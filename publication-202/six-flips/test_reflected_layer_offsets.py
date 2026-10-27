#!/usr/bin/env python3
"""Bounded offline offset-direction diagnostic; never a physical or geometric admission."""
from pathlib import Path
import hashlib,importlib.util,json,sys,numpy as np
here=Path(__file__).parent;src=here/"layer-diagnostic/skin-fat-layers-541.npz";out=here/"layer-reflection-diagnostic-001";out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location("pinned_layer_metric",here/"transfer_layers.py");m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
with np.load(src,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
f=d["outer_faces"].astype(int);O=d["outer_points_f64"];I=d["inner_points_f64"];D=d["deep_points_f64"]
def normals(x):
  t=x[f];v=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);n=np.zeros_like(x)
  for col in range(3):np.add.at(n,f[:,col],v)
  length=np.linalg.norm(n,axis=1);good=length>1e-20;n[good]/=length[good,None]
  return n,good
# Reflect outward components across local area-weighted tangent planes; preserve offset lengths.
def reflect(offset,n,valid):
  dot=np.einsum("ij,ij->i",offset,n);mask=(dot>0)&valid;v=offset.copy();v[mask]-=2*dot[mask,None]*n[mask];return v,mask
skin_before=m.layer_metrics(O,I,f,d["skin_thickness_transferred_m"]);fat_before=m.layer_metrics(I,D,f,d["fat_thickness_transferred_m"])
n,good=normals(O);skin,sm=reflect(I-O,n,good);J=O+skin
nf,goodf=normals(J);fat,fm=reflect(D-I,nf,goodf);K=J+fat
skin_after=m.layer_metrics(O,J,f,d["skin_thickness_transferred_m"]);fat_after=m.layer_metrics(J,K,f,d["fat_thickness_transferred_m"])
r={"status":"EXPLORATORY_DIRECTION_DIAGNOSTIC_NOT_ADMITTED","source_sha256":sha(src),"skin_reflected_vertices":int(sm.sum()),"fat_reflected_vertices":int(fm.sum()),"skin_thickness_norm_max_abs_change_m":float(np.max(np.abs(np.linalg.norm(skin,axis=1)-np.linalg.norm(I-O,axis=1)))),"fat_thickness_norm_max_abs_change_m":float(np.max(np.abs(np.linalg.norm(fat,axis=1)-np.linalg.norm(D-I,axis=1)))),"skin_before":skin_before,"skin_after":skin_after,"fat_before":fat_before,"fat_after":fat_after,"global_layer_collision_checked":False,"donor_mass_transfer":False,"physical_admission":False,"limitation":"This reflection is an exploratory local orientation test, not a validated tissue model or accepted repair."}
(out/"report.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r,indent=2))
