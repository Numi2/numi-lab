#!/usr/bin/env python3
"""Arithmetic-only snapshot of pinned baseline face 22027; no flip replay, fitting, or audit."""
from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
BASE=ROOT/'D/surface-clearance-546/topology/round-007-wide-chain-025/round-004/candidate.npz'
CONDITIONER=ROOT/'D/area-quality-548/condition_area_quality_548.py'
QUALITY=ROOT/'D/focused-clearance-541/topology/repair_focused_clearance_541.py'
BASE_SHA='eadcd3838ec5bf4f2d72d45ba5d19c775e9fef4361f92ebea6f49e18d4bfe3a5'
CONDITIONER_SHA='dcc5ad2d6aa138f3a727171b1369d87ae4c5b7c2d52f67d8491a1252e9e65beb'
QUALITY_SHA='26930be05518834f02724948cffa5ff74719f070a4ac4e4917b0965f42a412aa'
FACE=22027
EXPECTED_NODES=[17266,9765,20308]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def arr_sha(a):
 x=np.ascontiguousarray(a); h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(repr(x.shape).encode());h.update(x.tobytes());return h.hexdigest()
def load_quality():
 spec=importlib.util.spec_from_file_location('quality541_no_flip_22027',QUALITY)
 if spec is None or spec.loader is None: raise RuntimeError('could not load pinned 541 quality owner')
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod

def main():
 pins={'candidate':sha(BASE),'conditioner548':sha(CONDITIONER),'quality_owner541':sha(QUALITY)}
 expected={'candidate':BASE_SHA,'conditioner548':CONDITIONER_SHA,'quality_owner541':QUALITY_SHA}
 if pins!=expected: raise SystemExit(f'input pin mismatch: {pins}')
 q=load_quality()
 with np.load(BASE,allow_pickle=False) as z: d={k:z[k] for k in z.files}
 F=np.asarray(d['source_O_faces']); RF=np.asarray(d['area_reference_037_source_O_faces'])
 X=np.asarray(d['points_world_m'],dtype=np.float64); R=np.asarray(d['area_reference_037_points_world_m_f64'],dtype=np.float64)
 if not np.array_equal(F,RF): raise SystemExit('fixed-037 topology differs from candidate topology')
 if FACE>=len(F): raise SystemExit('witness face row outside mesh')
 nodes=list(map(int,F[FACE]))
 if nodes!=EXPECTED_NODES: raise SystemExit(f'face 22027 nodes changed: {nodes}')
 metrics=q.face_quality(X*1000,R*1000,F,RF)
 cur=np.cross(X[F[FACE,1]]-X[F[FACE,0]],X[F[FACE,2]]-X[F[FACE,0]])
 ref=np.cross(R[RF[FACE,1]]-R[RF[FACE,0]],R[RF[FACE,2]]-R[RF[FACE,0]])
 cur_area=.5*float(np.linalg.norm(cur));ref_area=.5*float(np.linalg.norm(ref))
 anchors=np.asarray(d['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
 if anchors.size!=237: raise SystemExit(f'expected 237 anchors, got {anchors.size}')
 doc={'schema':'numi.human.area_quality_548.baseline_22027_arithmetic.v1',
      'status':'PINNED_BASELINE_ARITHMETIC_ONLY',
      'candidate_path':str(BASE),'candidate_sha256':pins['candidate'],
      'arithmetic_script_path':str(Path(__file__)),'arithmetic_script_sha256':sha(Path(__file__)),
      'conditioner548_path':str(CONDITIONER),'conditioner548_sha256':pins['conditioner548'],
      'quality_owner541_path':str(QUALITY),'quality_owner541_sha256':pins['quality_owner541'],
      'operation_performed':'baseline face arithmetic only; no operation applied',
      'flip_replay':'NOT_RUN','flip_eligibility':'NOT_ASSESSED_BY_THIS_ARTIFACT',
      'heavy_geometry_audit_or_fit_run':False,'candidate_written':False,
      'fixed037_topology_equal':True,'face_row':FACE,'face_node_rows':nodes,
      'face_22027':{'current_double_area_m2':cur_area,'fixed037_double_area_m2':ref_area,
                    'area_ratio':float(metrics['area_ratio'][FACE]),
                    'normal_cosine':float(metrics['normal_cos'][FACE]),
                    'minimum_angle_deg':float(metrics['min_angle_deg'][FACE])},
      'whole_mesh_fixed037':{'face_count':int(len(F)),
         'minimum_area_ratio':float(np.min(metrics['area_ratio'])),
         'minimum_normal_cosine':float(np.min(metrics['normal_cos'])),
         'minimum_angle_deg':float(np.min(metrics['min_angle_deg'])),
         'faces_below_area_ratio_0_10':int(np.sum(metrics['area_ratio']<0.10))},
      'anchors':{'count':int(anchors.size),'rows_sha256':arr_sha(anchors)},
      'coordinate_array_sha256':arr_sha(d['points_world_m']),
      'current_face_array_sha256':arr_sha(F),
      'fixed037_reference_points_sha256':arr_sha(d['area_reference_037_points_world_m_f64']),
      'interpretation':'Arithmetic-only snapshot of the pinned baseline face. No 402/405 quad replay, candidate flip, exact self/bone audit, or fitter was run. The prior no-eligible-flip assessment has no retained result in this folder; this artifact does not reproduce it.'}
 out=Path(__file__).with_name('baseline_22027_arithmetic.json')
 out.write_text(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps(doc,indent=2,sort_keys=True,allow_nan=False))
if __name__=='__main__': main()
