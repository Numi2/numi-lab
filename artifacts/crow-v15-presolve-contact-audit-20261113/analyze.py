from pathlib import Path
import collections,gzip,hashlib,json,math,os,statistics,subprocess
R=Path('/Users/n/Developer/numi-crow-rl-20261107');A=R/'artifacts/crow-v15-presolve-contact-audit-20261113';trial=A/'push-nowings-20261127-rebound-presolve';ref=R/'artifacts/crow-v15-launch-ground-contact-audit-20261113/push-nowings-20261127-rebound-ground';exe=R/'build/crow-v15-presolve-contact-audit-20261113/kinematics';lib=R/'build/crow-v15-presolve-contact-audit-20261113/libmetalrobo.dylib'
def load(path,extra):
 d={'state':{},'contact':{},'kin':{},'presolve':{},'ground':[],'errors':[]}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);key=(x['lane'],x['step']);d['state'].setdefault(key,[]).append(json.dumps(x,sort_keys=True,separators=(',',':')))
   elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=line.split();t,k,e=map(int,z[1:4]);key=(e,t,k);assert key not in d['contact'];d['contact'][key]=z[4:]
    if int(z[4])>int(z[5]):d['errors'].append(['overflow',e,t,k])
   elif extra and line.startswith('CROW_PRESOLVE_V '):
    z=line.split();t,k,e,nv=map(int,z[1:5]);v=list(map(float,z[5:]));assert nv==40 and len(v)==nv and all(map(math.isfinite,v));key=(e,t,k);assert key not in d['presolve'];d['presolve'][key]=v
   elif extra and line.startswith('CROW_KIN_STATE '):
    z=line.split();t,k,e,nq,nv=map(int,z[1:6]);assert nq==41 and nv==40;v=list(map(float,z[6:]));assert len(v)==nq+nv and all(map(math.isfinite,v));key=(e,t,k);assert key not in d['kin'];d['kin'][key]=v
   elif extra and line.startswith('CROW_GROUND_CONTACT '):
    z=line.split();assert len(z)==24,len(z);t,k,e,a,b=map(int,z[1:6]);x=list(map(float,z[8:]));assert all(map(math.isfinite,x));d['ground'].append((e,t,k,a,b,int(z[6]),int(z[7]),x))
   elif line.startswith('CROW_SUBSTEP_ERROR'):d['errors'].append(line.strip())
 return d
old=load(ref,False);new=load(trial,True)
def cmp(a,b):
 both=a.keys()&b.keys();bad=[k for k in sorted(both) if a[k]!=b[k]];return {'reference':len(a),'observed':len(b),'shared':len(both),'mismatches':len(bad),'missing_reference':len(b.keys()-a.keys()),'missing_observed':len(a.keys()-b.keys()),'first_mismatch':list(bad[0]) if bad else None}
parity={'state':cmp(old['state'],new['state']),'contact':cmp(old['contact'],new['contact']),'reference_errors':old['errors'],'audit_errors':new['errors'],'kin_states':len(new['kin']),'ground_contacts':len(new['ground'])};(A/'parity.json').write_text(json.dumps(parity,indent=2)+'\n');print('Recorded-field parity',json.dumps(parity))
assert not old['errors'] and not new['errors'] and len(new['kin'])==8*28*16 and len(new['ground'])>0 and len(new['presolve'])==8*28*16
assert all(parity[k]['mismatches']==0 and parity[k]['missing_reference']==0 and parity[k]['missing_observed']==0 for k in ('state','contact'))
inputtext=''.join(f'{e} {t} {k} '+' '.join(map(str,v))+'\n' for (e,t,k),v in sorted((key,value[:41]+new['presolve'][key]) for key,value in new['kin'].items()));env={**os.environ,'DYLD_LIBRARY_PATH':str(lib.parent)};p=subprocess.run([str(exe)],input=inputtext,text=True,capture_output=True,check=True,env=env);body={}
for l in p.stdout.splitlines():
 z=l.split();assert z[0]=='B';e,t,k,b=map(int,z[1:5]);v=list(map(float,z[5:]));assert len(v)==9 and all(map(math.isfinite,v));body[e,t,k,b]=v
assert len(body)==len(new['kin'])*18,(len(body),len(new['kin']))
# The first microstep state must agree with the preceding recorded control endpoint.
qerr=verr=0.
for e in range(8):
 for t in range(348,376):
  k=new['kin'][(e,t,0)];a=json.loads(new['state'][(e,t-1)][0]);qerr=max(qerr,max(abs(x-y) for x,y in zip(k[:41],a['q'])));verr=max(verr,max(abs(x-y) for x,y in zip(k[41:],a['v'])))
assert qerr<1e-6 and verr<1e-6,(qerr,verr)
with gzip.open(trial/'replay.json.gz','rt') as f:replay=json.load(f)['payload']
assert replay['environment']==5;frames={x['step']:x for x in replay['frames']};names=replay['body_names'];qreplay=poserr=velerr=angerr=0.
for t in (348,352,360,372):
 f=frames[t-2];q=new['kin'][(5,t,0)][:41];qreplay=max(qreplay,max(abs(x-y) for x,y in zip(q,f['q'])))
 for b in (11,15,19,25,27,33):
  z=body[(5,t,0,b)];i=names.index(names[b])*13;rec=f['body_states'][i:i+13];poserr=max(poserr,math.dist(z[:3],rec[:3]));velerr=max(velerr,math.dist(z[3:6],rec[7:10]));angerr=max(angerr,math.dist(z[6:9],rec[10:13]))
assert qreplay<1e-6 and poserr<1e-5,(qreplay,poserr)
# Body ordering is the v15 model ordering used by the preexisting ground observer.
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
vals=[];invalid=[]
for e,t,k,a,b,pair,feature,x in new['ground']:
 if e<2:continue
 if not ((a<35 and b==35) or (b<35 and a==35)):invalid.append('ground pair');continue
 which=a if a!=35 else b
 if (e,t,k,which) not in body:invalid.append('nonfoot body '+str(which));continue
 point=x[:3];normal=x[4:7];tangent=x[7:10];mus,mud,jn,ju,jv,solver=x[10:16];v=body[e,t,k,which];r=[point[i]-v[i] for i in range(3)];wcr=cross(v[6:9],r);pointv=[v[3+i]+wcr[i] for i in range(3)];vn=dot(pointv,normal);vt=math.sqrt(max(0.,dot(pointv,pointv)-vn*vn));util=math.hypot(ju,jv)/(mus*jn) if mus>0 and jn>1e-7 else None
 if jn>1e-7:vals.append({'lane':e,'step':t,'body':which,'jn':jn,'mu_s':mus,'mu_d':mud,'util':util,'solver_normal':solver,'predicted_normal':vn,'predicted_tangent_speed':vt})
assert not invalid,collections.Counter(invalid)
def quant(a,p):
 a=sorted(a);x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def stats(a):return {'n':len(a),'p50':quant(a,.5),'p90':quant(a,.9),'p99':quant(a,.99),'max':max(a)}
assert vals
nplus=math.sqrt(sum((v['predicted_normal']-v['solver_normal'])**2 for v in vals)/len(vals));nminus=math.sqrt(sum((-v['predicted_normal']-v['solver_normal'])**2 for v in vals)/len(vals));solverrms=math.sqrt(sum(v['solver_normal']**2 for v in vals)/len(vals));util=[v['util'] for v in vals if v['util'] is not None];speeds=[v['predicted_tangent_speed'] for v in vals];total_j=sum(v['jn'] for v in vals);dynamic=[v['util']*v['mu_s']/v['mu_d'] for v in vals if v['util'] is not None and v['mu_d']>0];dynamic_near=sum(v['jn'] for v in vals if v['util'] is not None and v['mu_d']>0 and v['util']*v['mu_s']/v['mu_d']>=.95)/total_j;near=sum(v['jn'] for v in vals if v['util'] is not None and v['util']>=.95)/total_j
# Descriptive posthoc associations, normal-impulse weighted; not an intervention.
lane_summary={}
for lane in range(2,8):
 w=[v for v in vals if v["lane"]==lane];denom=sum(v["jn"] for v in w)
 lane_summary[str(lane)]={"normal_impulse_ns":denom,"fraction_impulse_tangent_speed_at_least_0_02_m_s":sum(v["jn"] for v in w if v["predicted_tangent_speed"]>=.02)/denom,"fraction_impulse_dynamic_util_at_least_0_95_and_speed_at_least_0_02":sum(v["jn"] for v in w if v["predicted_tangent_speed"]>=.02 and v["util"]*v["mu_s"]/v["mu_d"]>=.95)/denom}
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();out={'parity':parity,'validation':{'first_substep_vs_prior_endpoint_q_max_error':qerr,'first_substep_vs_prior_endpoint_v_max_error':verr,'replay_q_max_error':qreplay,'replay_body_position_max_error_m':poserr,'candidate_vs_endpoint_body_linear_velocity_difference_max_m_s':velerr,'candidate_vs_endpoint_body_angular_velocity_difference_max_rad_s':angerr,'predicted_vs_solver_normal_velocity_rmse_same_sign_m_s':nplus,'predicted_vs_solver_normal_velocity_rmse_opposite_sign_m_s':nminus,'solver_normal_velocity_rms_m_s':solverrms,'prespecified_timing_check_passed':min(nplus,nminus)<=.005 and min(nplus,nminus)<=.1*solverrms},'contacts_n':len(vals),'static_mu_values':sorted(set(v['mu_s'] for v in vals)),'dynamic_mu_values':sorted(set(v['mu_d'] for v in vals)),'friction_utilization':stats(util),'dynamic_friction_utilization_posthoc':stats(dynamic),'normal_impulse_weighted_fraction_dynamic_utilization_at_least_095_posthoc':dynamic_near,'normal_impulse_weighted_fraction_utilization_at_least_095':near,'predicted_tangent_speed_m_s':stats(speeds),'kinematics_sha256':h(exe),'observer_host_sha256':h(lib),'scope':'Used-seed in-range launch; six correlated lanes, positive normal impulses above 1e-7 Ns, contact samples not independent. Ratio uses static friction and recorded impulse basis. Point velocity combines beginning-of-substep position with the captured candidate velocity and stored world contact point; interpret as estimated only to extent normal, replay and timing checks support it. Near-limit ratio does not prove sliding or a biological friction coefficient. No takeoff or mission qualification.'};out['lane_tangential_diagnostics_posthoc']=lane_summary;out['step_lane_tangential_diagnostics_posthoc']={str(lane)+':'+str(step):{'count':len(w),'normal_impulse_ns':sum(v['jn'] for v in w),'tangent_speed_m_s':stats([v['predicted_tangent_speed'] for v in w]),'dynamic_utilization':stats([v['util']*v['mu_s']/v['mu_d'] for v in w])} for lane in range(2,8) for step in range(348,376) if (w:=[v for v in vals if v['lane']==lane and v['step']==step])};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('Validation',out['validation']);print('Static/dynamic coefficients',out['static_mu_values'],out['dynamic_mu_values']);print('Friction utilization',out['friction_utilization'],'normal-impulse weighted >=.95',near);print('Dynamic-limit utilization (posthoc)',out['dynamic_friction_utilization_posthoc'],'normal-impulse weighted >=.95',dynamic_near);print('Estimated tangential speed',out['predicted_tangent_speed_m_s'])
