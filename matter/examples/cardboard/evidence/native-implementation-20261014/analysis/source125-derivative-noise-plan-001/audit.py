#!/usr/bin/env python3
"""Recheck exact captured-F representation and signed A-work decomposition for run419.
CPU-only, read-only. This does not rerun a solver or claim derivative qualification.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np

B=Path('/Users/home/cardboard-implementation-20261014')
RUN=B/'solver/two-stage-first-failure-rich-fd-419'
CAP=RUN/'operator-capture'
ADAPTER=B/'analysis/operator-fd-e2-adapter-001/coupon419-003'
NPZ=ADAPTER/'environment-0.npz'
SWEEP=B/'analysis/source125-fp64-smooth-direction-sweep-006/result-004/report.json'
REFERENCE=B/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
SOURCE=B/'snapshots/source-status5-vtccd-diagnostics-125'
OUT=Path(__file__).resolve().parent
RECEIPT=RUN/'receipt.json'
PREDICTION=RUN/'prediction.json'
PACKAGE=RUN/'output/compiled.nmatterpack'
PROVENANCE=Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-125/source-build.json')
ADAPTER_REPORT=ADAPTER/'report.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1<<20),b''): h.update(x)
 return h.hexdigest()
def f32(x): return np.asarray(x,dtype=np.float32).astype(np.float64)
def pair(rec,key):
 hi=np.asarray(rec[key],dtype=np.float32).astype(np.float64)
 lo=np.asarray(rec.get(key+'_low',np.zeros_like(hi)),dtype=np.float32).astype(np.float64)
 if not np.all(np.isfinite(hi)) or not np.all(np.isfinite(lo)): raise ValueError('nonfinite FP32 input')
 return hi+lo
def summary(xs):
 xs=np.asarray(xs,dtype=np.float64)
 if not np.all(np.isfinite(xs)): raise ValueError('nonfinite metric input')
 return {'count':int(xs.size),'max':float(xs.max()) if xs.size else None,
         'median':float(np.median(xs)) if xs.size else None}
def main():
 args=argparse.ArgumentParser()
 args.add_argument('--output',type=Path,default=OUT/'recomputed-report.json')
 output=args.parse_args().output.resolve()
 if output.exists(): raise RuntimeError(f'refusing to overwrite existing output: {output}')
 expected={
  str(CAP):'f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a',
  str(NPZ):'e20eeb559f5976e6b9ddb2ae605ea6860908cde3a1d255ea80832b24840c8353',
  str(SWEEP):'2a2ffd1bb9205b847f7b34119ccd38d98c2d5cc91d4801792017195f831798ba',
  str(RECEIPT):'0826a69a65ea4a0d76ef95e56c5d0d7f5238f4de746dd6daf81f11a87907f67d',
  str(PREDICTION):'41b37af2747cfa551fae60eaddd7681cd7c7dbc859a402e641677ec23bd957d6',
  str(PACKAGE):'cafbf36052bbfa72e8be49fdba8b23e900469c9f15136a14a4ec1e4c0b5ccf7c',
  str(PROVENANCE):'35b821184ca36ad57c529bc0e33e4f20519996def1c312f8ac0f8028ecadb8e6',
  str(ADAPTER_REPORT):'c918a287f0de7966d50813a70fa7d9d3917746a6fc5de8dca57283fbf375ba76',
  str(REFERENCE):'c17010564a98fe63da0c8b88be81313685a65cfbd57ff82292e797cab5a0fdf3',
  str(SOURCE/'matter/src/runtime.mm'):'1719f14310d643ca96c520eb47b3028f8259f85e3da0d263460921b06d94ac74',
  str(SOURCE/'matter/include/numi/matter/shared.h'):'cd173c076905f660aa8aecc900e6f6aba8d69406996f3cccf2e0656c753ff871',
  str(SOURCE/'matter/src/metal/common.metalinc'):'441a3f86f21979a086a0356f5fd2b91f692ba94dbd09a28d07ea9756c1246a5a',
  str(SOURCE/'matter/src/metal/fem.metalinc'):'70c2f396045f9abdcfdc32f5714e6e3a400f5566e5a7824353e11b307de52b7d',
  str(SOURCE/'matter/src/metal/fgmres.metalinc'):'1c423e02e189df17c925e0b77ded87465098278f836011c16aad1bb8f92bcec0',
 }
 hashes={}
 for name,exp in expected.items():
  p=Path(name); actual=sha(p)
  if actual!=exp: raise RuntimeError(f'hash mismatch: {p}: {actual} != {exp}')
  hashes[name]=actual
 header=None; nodes={}; tets={}
 with CAP.open() as f:
  for line in f:
   r=json.loads(line)
   kind=r.get('record')
   if kind=='header': header=r; continue
   if int(r.get('environment',-1))!=0: continue
   if kind=='node': nodes[int(r['node_id'])]=r
   elif kind=='tetrahedron': tets[int(r['tet_id'])]=r
 if not header or (header['root'],header['newton_iteration'])!=(1,48): raise RuntimeError('wrong capture selection')
 if len(nodes)!=882 or len(tets)!=1920: raise RuntimeError('incomplete env0 capture')
 if header.get('production_inputs_bitwise_unchanged') is not True: raise RuntimeError('input-integrity flag false')
 dt=float(np.float32(header['dt_seconds']))
 eps_values=[float(np.float32(x)) for x in header['epsilon_values']]
 eps_report=[]
 for pair_id,eps in enumerate(eps_values):
  abs_endpoint=[]; rel_direction=[]; valid_rows=0; total_rows=0
  native_status_counts={}; native_valid_counts={}
  for tet_id in sorted(tets):
   t=tets[tet_id]
   if int(t['material_index']) not in (0,1): continue
   c=t['center']; fc=pair(c,'F_local'); h=f32(c['F_direction_local_per_s'])
   plus,minus=t['sides'][2*pair_id],t['sides'][2*pair_id+1]
   fp,fm=pair(plus,'F_local'),pair(minus,'F_local')
   abs_endpoint.extend(np.abs(fp-(fc+eps*dt*h)).ravel())
   abs_endpoint.extend(np.abs(fm-(fc-eps*dt*h)).ravel())
   realized=(fp-fm)/(2*eps*dt)
   denom=np.linalg.norm(h)
   if denom>0: rel_direction.append(float(np.linalg.norm(realized-h)/denom))
   for side in (plus,minus):
    total_rows+=1
    # flags_status_determinant[2] is the environment status copied into the
    # observational record, not a per-tet projection result. Keep it separate.
    diag=side['flags_status_determinant']
    code=int(np.float32(diag[2])); native_status_counts[str(code)]=native_status_counts.get(str(code),0)+1
    valid=int(side['projection_diagnostics'][3]) == 1 and int(side['constitutive_kind_state_count_flags_valid'][3]) == 1
    native_valid_counts[str(valid)]=native_valid_counts.get(str(valid),0)+1
    if valid: valid_rows+=1
  eps_report.append({'epsilon':eps,'paper_side_rows':total_rows,
   'environment_status_word_counts_at_capture':native_status_counts,
   'native_local_projection_valid_counts':native_valid_counts,
   'native_local_projection_valid_rows':valid_rows,
   'max_abs_F_wordpair_vs_ideal_requested_endpoint':float(max(abs_endpoint)),
   'realized_centered_direction_relative_error':summary(rel_direction),
   'interpretation':'endpoint input representation diagnostic only; not a derivative noise floor'})
 # Reconstruct v^T K v elementwise from the independent elemental force action.
 z=np.load(NPZ)
 dv=np.asarray(z['correction_delta_v_m_s'],dtype=np.float64)
 force=np.asarray(z['independent_element_operator_force_node_world_N_per_s'],dtype=np.float64)
 material=np.asarray(z['material_index'],dtype=np.int32)
 if dv.shape!=(882,3) or force.shape!=(1920,4,3) or material.shape!=(1920,): raise RuntimeError('unexpected adapter shapes')
 per={0:[],1:[],2:[]}; signs={0:{'positive':0,'negative':0,'zero':0},1:{'positive':0,'negative':0,'zero':0},2:{'positive':0,'negative':0,'zero':0}}
 for tid in sorted(tets):
  nodes4=[int(i) for i in tets[tid]['nodes']]
  if len(set(nodes4))!=4 or any(i<0 or i>=882 for i in nodes4): raise RuntimeError('bad tet connectivity')
  terms=[float(dv[n,c]*force[tid,slot,c]) for slot,n in enumerate(nodes4) for c in range(3)]
  w=math.fsum(terms); mi=int(material[tid])
  per[mi].append(w)
  signs[mi]['positive' if w>0 else 'negative' if w<0 else 'zero']+=1
 dt64=float(header['dt_seconds'])
 work_by_material={str(mi):{'elements':len(vals),'v_dot_Kv_J':math.fsum(vals),
    'v_dot_minus_dt2_Kv_J':-dt64*dt64*math.fsum(vals),
    'element_directional_work_sign_counts':signs[mi]} for mi,vals in per.items()}
 sweep=json.loads(SWEEP.read_text())
 terms=sweep['center_A_terms']
 total_k=math.fsum(v['v_dot_Kv_J'] for v in work_by_material.values())
 computed=-dt64*dt64*total_k
 reported=terms['stiffness_action_minus_dt2_K_Ns']['signed_work_dv_dot']
 if abs(computed-reported)>max(1e-10,1e-7*abs(reported)):
  raise RuntimeError(f'element work failed consistency: {computed} vs {reported}')
 result={'schema':'numi.cardboard.source125.native-derivative-input-audit.v1',
  'claim_scope':'source125 root1/Newton48 captured operator/material input audit; no physical coupon qualification',
  'capture':{'path':str(CAP),'sha256':hashes[str(CAP)],'root':1,'newton_iteration':48,
   'environment':0,'node_count':882,'tetrahedron_count':1920,'paper_tetrahedra':1728,
   'dt_seconds_binary32_promoted':dt,'epsilon_values':eps_values,
   'F_encoding':'F_local high FP32 + F_local_low FP32, each word widened before sum',
   'fixed_accepted_history':True,'production_inputs_bitwise_unchanged':True},
      'endpoint_representation':eps_report,
  'stress_projection_same_input_reference':{
   'source_report':str(ADAPTER/'report.json'),'source_report_sha256':sha(ADAPTER/'report.json'),
   'pair_results':[]},
  'operator_quadratic_form':{
   'source_sweep_report':str(SWEEP),'source_sweep_sha256':hashes[str(SWEEP)],
   'definition':'K is the derivative of assembled internal force with respect to displacement; source action is A*dv=M*dv-dt^2*K*dv.',
   'reported_mass_signed_work_J':terms['mass_action_M_dv_Ns']['signed_work_dv_dot'],
   'reported_minus_dt2_K_signed_work_J':reported,
   'reported_total_A_signed_work_J':terms['total_A_dv_Ns']['signed_work_dv_dot'],
   'independent_elementwise_material_breakdown':work_by_material,
   'independent_minus_dt2_K_sum_J':computed,
   'breakdown_matches_reported_minus_dt2_K':True,
   'interpretation':'The negative contribution is in the material-tangent internal-force derivative after the -dt^2 factor; the audited assembly has no separate geometric-stiffness matrix. It is not a residual-sign bookkeeping mismatch. Negative directional work alone does not establish a physical-energy violation or software defect.'},
  'source_hashes':hashes}
 rep=json.loads((ADAPTER/'report.json').read_text())
 env=next(x for x in rep['environments'] if x['environment']==0)
 for p in env['finite_difference_pairs']:
  pairid=0 if p['epsilon']==eps_values[0] else 1
  mats={}
  for key,value in p['independent_side_projection'].items():
   mats[key]=value
  result['stress_projection_same_input_reference']['pair_results'].append({
   'epsilon':p['epsilon'],'native_global_FD_admissible':p['finite_difference_admission']['comparison_admissible'],
   'materials':{k:{'native_rows':v['rows'],'fp64_projection_success':v['reference_projection_success'],
       'native_stress_vs_fp64_global_relative':v['stress_vs_independent_reference_Pa']['global_l2_relative'],
       'native_stress_vs_fp64_abs_component_max_Pa':v['stress_vs_independent_reference_Pa']['maximum_component_absolute'],
       'plastic_multiplier_branch_crossings':v['plastic_multiplier_branch_comparison']['branch_crossing_rows']}
       for k,v in mats.items()}})
 result['source_hashes']=hashes
 output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(output)
 print(json.dumps({'endpoint_representation':eps_report,'work_by_material':work_by_material,
  'reported_work':reported,'recomputed_work':computed},indent=2,allow_nan=False))
if __name__=='__main__': main()
