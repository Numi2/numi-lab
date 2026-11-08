from pathlib import Path
import hashlib,json,os,struct,subprocess,time
import numpy as np
from motor_reference import solve,residual
R=Path.cwd();B=R/'build/crow-bounded-drive-integration-build-01';A=R/'artifacts/crow-rl-bounded-drive-crow-20261108/witness';T=R/'.numi/retained-runtime/bounded-drive-witness-20261108'
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((A/'plan.json').read_text())
for p,h in plan['inputs'].items():assert sha(R/p)==h,p
assert not subprocess.check_output(['git','status','--porcelain'])
seal=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
commands=[]
def execute(name,cmd,expected):
 start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(p) for p in cmd],stdout=out,stderr=err,env=env,timeout=180).returncode
  except subprocess.TimeoutExpired:code=None
 rec={'name':name,'command':[str(p) for p in cmd],'exit_code':code,'seconds':time.monotonic()-start,'sealed_head':seal};commands.append(rec);write(B/'witness-execution.json',commands);print(json.dumps(rec),flush=True)
 assert code==expected,rec
execute('witness-constitutive',[T/'probe',T/'metalrobo/MetalRobo.metallib',B/'witness-constitutive.tsv'],0)
assert (B/'witness-constitutive.tsv').read_bytes()==(B/'native-v2.tsv').read_bytes()
env['METALROBO_BOUNDED_FAILURE_CAPTURE']=str(B/'crow-motor-failure.motorsnp')
execute('crow-witness',[B/'fixture-bounded',B/'crow-witness','0','1','32'],1)
checks={f:(B/'crow-witness'/f).read_bytes()==(B/'crow-bounded32'/f).read_bytes() for f in ('trace.tsv','failure.json','failure-q.f32','failure-v.f32','failure-scene.bin','failure-contact-status.bin')}
old=(B/'crow-bounded32/failure-world-status.bin').read_bytes();new=(B/'crow-witness/failure-world-status.bin').read_bytes()
oldrows=[struct.unpack_from('<8I4f',old,48*i) for i in range(4)];newrows=[struct.unpack_from('<8I4f',new,48*i) for i in range(4)]
checks['world_statuses_equal_except_diagnostic_flag']=all(a[:7]==b[:7] and a[7]==(b[7]&~(1<<9)) and a[8:]==b[8:] for a,b in zip(oldrows,newrows))
report={'sealed_head':seal,'commands':commands,'constitutive976rows_exact':True,'crow_observation_equivalence':checks,'diagnostic_capture_available':(B/'crow-motor-failure.motorsnp').is_file(),'policy_or_profile_promoted':False}
if not all(checks.values()) or not report['diagnostic_capture_available']:
 report['verdict']='inconclusive_instrument_equivalence_or_capture_failure';write(A/'results.json',report);raise SystemExit(1)
data=(B/'crow-motor-failure.motorsnp').read_bytes();assert len(data)==13968
header=struct.unpack_from('<8I4f',data);assert header[0]==0x42444631 and header[1]==0 and header[3]==27 and header[4]==22 and header[6]==256 and header[7]==11
n=header[4];h=header[8];values=np.frombuffer(data,dtype='<f4',offset=48).astype(np.float64)
L=np.tril(values[:1600].reshape(40,40)[:n,:n]);W=values[1600:3200].reshape(40,40)[:n,:n];free,b,c,cap,friction,tau,a=[v[:n] for v in values[3200:].reshape(7,40)]
M=L@L.T;physicalW=np.linalg.solve(M,np.eye(n))
reference={}
for name,matrix in [('captured_gpu_inverse_fp64_evaluation',W),('captured_factor_fp64_inverse',physicalW)]:
 tr,iterations,e=solve(matrix,free,b,c,cap,friction);dv=h*(matrix@(tr-tau));native_e=np.abs(residual(matrix,free,b,c,cap,friction,tau))
 reference[name]={'maximum_absolute_residual_at_native_torque_Nm':float(native_e.max()),'maximum_scaled_residual_at_native_torque':float(np.max(native_e/np.maximum(cap,1e-6))),'maximum_torque_difference_Nm':float(np.max(np.abs(tr-tau))),'maximum_velocity_difference_rad_s':float(np.max(np.abs(dv))),'reference_iterations':iterations,'reference_final_residual_Nm':e,'torque_reference':tr.tolist(),'velocity_difference_per_coordinate':dv.tolist()}
report.update(verdict='protected_failure_diagnosis_complete',protected_header=list(header),mass_condition_number=float(np.linalg.cond(M)),inverse_maximum_asymmetry=float(np.max(np.abs(W-W.T))),inverse_maximum_element_error=float(np.max(np.abs(W-physicalW))),inverse_maximum_relative_frobenius_error=float(np.linalg.norm(W-physicalW)/np.linalg.norm(physicalW)),native_acceleration_reconstruction_error=float(np.max(np.abs(a-(free+W@tau)))),references=reference,limits='Captured failed iterate only; FP64 conditional reference is numerical analysis, not a physical-state publication. Contact/motor coupling and full crow qualification remain unresolved.')
write(A/'results.json',report);write(B/'witness-analysis.json',report);print(json.dumps(report),flush=True)
