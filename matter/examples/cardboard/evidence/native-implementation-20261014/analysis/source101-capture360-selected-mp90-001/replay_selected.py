#!/usr/bin/env python3
"""Bounded MP90 vs FP64 projection/tangent/FD replay for exact capture360 rows.
CPU-only. Preserves accepted history, exact captured float32 words, dt, and Fdot relation.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, pathlib, struct, sys, time, warnings
import numpy as np
import mpmath as mp
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
CAP_SHA='548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
MP_HELPER=BASE/'analysis/medium-tangent-state-sensitivity-1634-002/mp90_endpoint_fd.py'
FP_HELPER=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
SOURCE=BASE/'snapshots/source-boundary-reset-order-101'
OUT=pathlib.Path(__file__).resolve().parent/'output'
POINTS=[(1635,1,'medium_worst_FD'),(679,1,'medium_worst_native_action'),(457,0,'liner_representative_FD_miss')]
TARGETS=[1e-5,3e-6,1e-6,3e-7,1e-7,3e-8,1e-8,3e-9,1e-9,3e-10]
DPS=90

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load_module(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def parse_int(tok): return -0.0 if tok=='-0' else int(tok)
def f32(x): return float(np.float32(x))
def flatten32(x): return [f32(v) for v in np.asarray(x).reshape(-1)]
def pair_words(d,k):
 hi=flatten32(d[k]);lo=flatten32(d.get(k+'_low',np.zeros_like(d[k])))
 return hi,lo,[a+b for a,b in zip(hi,lo)]
def mpf(x):
 if isinstance(x,mp.mpf): return x
 n,d=float(x).as_integer_ratio();return mp.mpf(n)/d
def norm(v): return float(np.linalg.norm(np.asarray(v,dtype=float).reshape(-1)))
def relabs(a,b):
 aa=np.asarray(a,dtype=float).reshape(-1);bb=np.asarray(b,dtype=float).reshape(-1);delta=aa-bb;den=float(np.linalg.norm(bb));
 return {'absolute_l2':float(np.linalg.norm(delta)),'max_component_abs':float(np.max(np.abs(delta))),
         'reference_denominator_l2':den,'relative_l2':float(np.linalg.norm(delta)/den) if den>0 else None}
def f32words(xs): return [f'0x{struct.unpack("<I",struct.pack("<f",float(x)))[0]:08x}' for x in xs]
def branch_mp(helper,p,old,z): return helper.state_branch(p,old,z)
def safe_np_branch(ref,p,old,z):
 p=np.asarray(p,dtype=float)
 dl=float(z[12]-old[12]);q=float(ref.hill(p,z[6:12])[0]);gap=float(1+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]-q)
 return {'delta_multiplier':dl,'yield_gap':gap,'multiplier_positive':bool(dl>0),'fb_selection':'elastic_smoothed' if gap>dl else 'plastic_smoothed'}
def endpoint_mp(helper,p,F,old,zseed):
 pmp,Fmp,oldmp,_,residual,stress=helper.make_equations(p,F,old)
 z,diag=helper.project_endpoint(residual,pmp,oldmp,zseed,mp.mpf('1e-70'),max_iter=48)
 return z,diag,mp.matrix([stress(*list(z))[i] for i in range(9)])
def endpoint_fp64(ref,p,F,old):
 # The stock independent reference preserves its original FP64 equations and 2e-10 scaled residual stop.
 p=np.asarray(p,dtype=float)
 z,counts,residual=ref.project(p[None,:],np.asarray(F,float).reshape(1,3,3),
                                np.asarray(old,float)[None,:],maxiter=80)
 return z[0],int(counts[0]),float(residual[0])
def max_scaled_residual(ref,p,F,old,z):
 scales=np.array([1e4]*6+[1]*6+[1e4]*3)
 return float(np.max(np.abs(ref.residual(np.asarray(p,float),np.asarray(F,float).reshape(3,3),np.asarray(old,float),np.asarray(z,float)))*scales))
def main():
 if OUT.exists(): raise RuntimeError(f'refusing to overwrite {OUT}')
 if sha(CAP)!=CAP_SHA: raise RuntimeError('capture SHA mismatch')
 fd=load_module(FP_HELPER,'fd360_helper');helper=load_module(MP_HELPER,'mp90_1634_helper');ref=load_module(REF,'ref267')
 records=[json.loads(s,parse_int=parse_int) for s in CAP.read_text().splitlines() if s.strip()]
 header=records[0];assert header['record']=='header' and header['root']==2 and header['newton_iteration']==23
 assert records[-1]['record']=='integrity' and records[-1]['production_inputs_bitwise_unchanged'] is True
 byid={(int(r['environment']),int(r['tet_id'])):r for r in records if r.get('record')=='tetrahedron'}
 started=time.monotonic();outrows=[]
 for tet_id,material_index,label in POINTS:
  row=byid[(0,tet_id)];c=row['center']
  assert int(row['material_index'])==material_index and int(row['material_state_count'])==15
  p=flatten32(header['parameters_by_environment'][0][material_index*28:(material_index+1)*28])
  old=flatten32(c['accepted_state'])
  Fhi,Flo,Ff=pair_words(c,'F_local');F=np.asarray(Ff,float).reshape(3,3)
  zh,zl,zf=pair_words(c,'projected_state');zcap=np.asarray(zf,float)
  h=np.asarray(flatten32(c['F_direction_local_per_s']),float).reshape(3,3)
  dt=f32(c['dt_seconds_bulk_modulus_kind_frame_active'][0]);dFbase=dt*h;dFnorm=norm(dFbase)
  nativeC=np.asarray(flatten32(row['cached_algorithmic_tangent_local']['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']),float).reshape(9,9)
  if not np.isfinite(dFnorm) or dFnorm==0: raise RuntimeError(f'tet{tet_id} has zero/nonfinite captured rate direction')
  # MP90 exact equation replay from captured projected state; equations and accepted history are unchanged.
  mp.mp.dps=DPS
  zmp,centerdiag,Pmp=endpoint_mp(helper,p,Ff,old,zcap.tolist())
  Cmp=helper.refmod.mp_tangent_matrix(p,Ff,old,zmp)
  hmp=mp.matrix([mpf(v) for v in h.reshape(-1)])
  amp=Cmp*hmp
  ampnp=np.array([float(v) for v in amp],float).reshape(3,3)
  Pmpnp=np.array([float(v) for v in Pmp],float).reshape(3,3)
  zmpnp=np.array([float(v) for v in zmp],float)
  # FP64 independent root at the same widened inputs.
  z64,count64,res64=endpoint_fp64(ref,p,F,old)
  P64=ref.stress(np.asarray(p,float),F,z64);C64=ref.tangent(np.asarray(p,float),F,np.asarray(old,float),z64,h)
  Pcap=np.asarray(flatten32(c['P_local_Pa']),float).reshape(3,3)
  # derivative from exact captured native cached 9x9 matrix along same captured Fdot.
  anative=(nativeC@h.reshape(9)).reshape(3,3)
  center={
    'mp90_projection':{'status':centerdiag['status'],'converged':centerdiag['converged'],'iterations':centerdiag['iterations'],
      'scaled_residual':centerdiag['terminal_scaled_residual'],'newton_correction_inf':centerdiag['terminal_newton_correction_inf'],
      'branch':centerdiag['terminal_branch']},
    'fp64_projection':{'iterations':int(count64),'scaled_residual':max_scaled_residual(ref,p,F,old,z64),'branch':safe_np_branch(ref,p,old,z64)},
    'captured_candidate_vs_mp90_state':relabs(zcap,zmpnp),
    'fp64_state_vs_mp90_root':relabs(z64,zmpnp),
    'captured_stress_vs_mp90_Pa':relabs(Pcap,Pmpnp),
    'fp64_stress_vs_mp90_Pa':relabs(P64,Pmpnp),
    'fp64_directional_tangent_vs_mp90_Pa_per_s':relabs(C64,ampnp),
    'captured_native_cached_action_vs_mp90_Pa_per_s':relabs(anative,ampnp),
    'native_matrix_vs_mp90_full_matrix':relabs(nativeC,np.array([[float(Cmp[i,j]) for j in range(9)] for i in range(9)])),
    'captured_raw_inputs':{'F_high_f32_bits':f32words(Fhi),'F_low_f32_bits':f32words(Flo),'accepted_state_f32_bits':f32words(old),
      'projected_seed_high_f32_bits':f32words(zh),'projected_seed_low_f32_bits':f32words(zl),'parameters_f32_bits':f32words(p),
      'direction_f32_bits':f32words(h.reshape(-1)),'dt_f32_bits':f32words([dt]),'raw_negative_zero_preserved':True}}
  eps_results=[]
  for target in TARGETS:
   eps=float(target/dFnorm)
   Fp=F+eps*dFbase;Fm=F-eps*dFbase
   # MP90 endpoint projection starts from the MP90 center root and reuses the exact accepted history.
   zmp_p,dmp_p,Pmp_p=endpoint_mp(helper,p,[mpf(x) for x in Fp.reshape(-1)],old,zmp)
   zmp_m,dmp_m,Pmp_m=endpoint_mp(helper,p,[mpf(x) for x in Fm.reshape(-1)],old,zmp)
   # Stock FP64 reference endpoint projection from the same captured accepted history.
   z64p,n64p,r64p=endpoint_fp64(ref,p,Fp,old);z64m,n64m,r64m=endpoint_fp64(ref,p,Fm,old)
   P64p=ref.stress(np.asarray(p,float),Fp,z64p);P64m=ref.stress(np.asarray(p,float),Fm,z64m)
   Pmp_p_np=np.array([float(v) for v in Pmp_p],float).reshape(3,3);Pmp_m_np=np.array([float(v) for v in Pmp_m],float).reshape(3,3)
   zmp_p_np=np.array([float(v) for v in zmp_p],float);zmp_m_np=np.array([float(v) for v in zmp_m],float)
   denom=2*eps*dt
   fdmp=(Pmp_p_np-Pmp_m_np)/denom;fd64=(P64p-P64m)/denom
   branch0=centerdiag['terminal_branch'];bpm=dmp_p['terminal_branch'];bmm=dmp_m['terminal_branch']
   bp64=safe_np_branch(ref,p,old,z64p);bm64=safe_np_branch(ref,p,old,z64m)
   err_mp=relabs(fdmp,ampnp);err_fp64=relabs(fd64,ampnp);err_fp64_mpfd=relabs(fd64,fdmp)
   # One-sided results help distinguish asymmetry/branch crossings from symmetric truncation.
   fdpmp=(Pmp_p_np-Pmpnp)/(eps*dt);fdmmp=(Pmpnp-Pmp_m_np)/(eps*dt)
   fdp64=(P64p-P64)/(eps*dt);fdm64=(P64-P64m)/(eps*dt)
   eps_results.append({'target_deltaF_frobenius':target,'epsilon_scalar':eps,'actual_deltaF_plus_norm':norm(Fp-F),'actual_deltaF_minus_norm':norm(Fm-F),
    'mp90':{'plus_projection':{'converged':dmp_p['converged'],'iterations':dmp_p['iterations'],'scaled_residual':dmp_p['terminal_scaled_residual'],'newton_correction_inf':dmp_p['terminal_newton_correction_inf'],'branch':bpm},
      'minus_projection':{'converged':dmp_m['converged'],'iterations':dmp_m['iterations'],'scaled_residual':dmp_m['terminal_scaled_residual'],'newton_correction_inf':dmp_m['terminal_newton_correction_inf'],'branch':bmm},
      'center_branch_stable':(bpm['fb_selection']==branch0['fb_selection'] and bmm['fb_selection']==branch0['fb_selection'] and bpm['yield_sign']==branch0['yield_sign'] and bmm['yield_sign']==branch0['yield_sign']),
      'central_fd_vs_mp90_tangent':err_mp,'plus_one_sided_vs_mp90_tangent':relabs(fdpmp,ampnp),'minus_one_sided_vs_mp90_tangent':relabs(fdmmp,ampnp),
      'one_sided_gap':relabs(fdpmp,fdmmp)},
    'fp64_default':{'plus_iterations':int(n64p),'minus_iterations':int(n64m),'plus_scaled_residual':max_scaled_residual(ref,p,Fp,old,z64p),
      'minus_scaled_residual':max_scaled_residual(ref,p,Fm,old,z64m),'plus_branch':bp64,'minus_branch':bm64,
      'center_branch_stable':(bp64['fb_selection']==safe_np_branch(ref,p,old,z64)['fb_selection'] and bm64['fb_selection']==safe_np_branch(ref,p,old,z64)['fb_selection']),
      'plus_root_vs_mp90_root':relabs(z64p,zmp_p_np),'minus_root_vs_mp90_root':relabs(z64m,zmp_m_np),
      'plus_stress_vs_mp90_Pa':relabs(P64p,Pmp_p_np),'minus_stress_vs_mp90_Pa':relabs(P64m,Pmp_m_np),
      'central_fd_vs_mp90_tangent':err_fp64,'central_fd_vs_mp90_central_fd':err_fp64_mpfd,
      'plus_one_sided_vs_mp90_tangent':relabs(fdp64,ampnp),'minus_one_sided_vs_mp90_tangent':relabs(fdm64,ampnp),
      'one_sided_gap':relabs(fdp64,fdm64)},
    'absolute_rate_action_denominator_Pa_per_s':norm(ampnp)})
  outrows.append({'label':label,'environment':0,'tet_id':tet_id,'material_index':material_index,'source_material':row['material_name'] if 'material_name' in row else ('hajali2009_liner_hill_bilinear_fit' if material_index==0 else 'hajali2009_medium_hill_bilinear_fit'),
   'capture_identity':{'root':2,'newton_iteration':23,'world_fingerprint_hex':header['world_fingerprint_hex'],'execution_fingerprint_hex':header['execution_fingerprint_hex'],'row_material_state_count':15},
   'input_relation':{'F_center':'captured float32 high+low widened exactly to FP64/MP90','accepted_history':'captured float32 state held fixed across all projections','direction':'captured local grad(delta_v) per second','perturbation':'F±=Fcenter±epsilon*dt*captured grad(delta_v); finite difference divided by 2 epsilon dt','native_tangent':'captured read-only full local 9x9 cache; action compared along same captured rate direction'},
   'center':center,'epsilon_sweep':eps_results})
  print(f'completed tet {tet_id} ({label})',flush=True)
 report={'schema':'numi.cardboard.source101-capture360-selected-mp90-adjudication.v1',
  'classification':'Bounded CPU-only high-precision adjudication of three exact source101 capture360 rows. No native execution, production edits, physical acceptance, or full solver qualification.',
  'binding':{'capture_path':str(CAP),'capture_sha256':sha(CAP),'source_snapshot':str(SOURCE),'frozen_source_json_sha256':sha(SOURCE/'frozen-source.json'),
    'mp90_helper_path':str(MP_HELPER),'mp90_helper_sha256':sha(MP_HELPER),'mp90_equation_reference_path':str(BASE/'analysis/medium-tangent-state-sensitivity-1634-001/replay.py'),
    'mp90_equation_reference_sha256':sha(BASE/'analysis/medium-tangent-state-sensitivity-1634-001/replay.py'),'fp64_reference_path':str(REF),'fp64_reference_sha256':sha(REF),
    'directional_sweep_source_path':str(FP_HELPER),'directional_sweep_source_sha256':sha(FP_HELPER)},
  'numerics':{'mpmath_dps':DPS,'mp90_root_tolerance':'1e-70 scaled residual and Newton correction','fp64_reference':'bilinear15_reference.py stock FP64 projector; local stop 2e-10 scaled residual, maxiter 80','target_deltaF_frobenius_norms':TARGETS,
    'all_equations_and_timestep_unchanged':True,'same_accepted_history_every_endpoint':True,'branch_fields':'multiplier sign, yield sign, and smoothed FB branch selection retained per endpoint','finite_difference':'Perturbed F by epsilon*dt*captured grad(delta_v), then divided stress difference by 2*epsilon*dt to compare with rate-direction tangent.'},
  'points':outrows,'wall_seconds':time.monotonic()-started,
  'limits':['Only the named three tetrahedra from one rejected Newton capture are analyzed.','No all-nine-direction gate or native FP32 endpoint noise floor is inferred.','MP90/FP64 agreement at these points cannot qualify the material pair, coupled matrix, or a physical run.','Every projection failure and branch change is retained in the epsilon records; no row is filtered by error size.']}
 OUT.mkdir(parents=True,exist_ok=False)
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'report':str(OUT/'report.json'),'sha256':sha(OUT/'report.json'),'wall_seconds':report['wall_seconds'],'points':[{'tet_id':x['tet_id'],'center_mp90':x['center']['mp90_projection']['converged'],'sweep_count':len(x['epsilon_sweep'])} for x in outrows]},indent=2))
if __name__=='__main__':main()
