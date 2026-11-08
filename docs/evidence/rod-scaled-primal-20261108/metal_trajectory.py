#!/usr/bin/env python3
import json,struct,subprocess,sys
from pathlib import Path
import numpy as np
b=Path('docs/evidence/rod-scaled-primal-20261108')
ns={};exec((b/'metal_heldout.py').read_text().split("if sys.argv[1]=='generate':")[0],ns)
refsolve=ns['refsolve'];evaluate=ns['evaluate'];H=ns['H']
base={};exec((b/'vector_full_newton.py').read_text().rsplit('\nspec=[',1)[0],base)
make=base['make_case']

def initial(kind,n):
    scene='straight' if kind=='straight_moving' else 'curved'
    q,hard,_,rest,k,m=make(n,scene,np.float64);v=np.zeros_like(q)
    if kind=='straight_moving':q[n//2,0]-=.0005;v[n//2,0]=.01
    soft=[]
    if kind=='moving_support':soft=[(n//2,q[n//2]+np.array([0.,.0001,0.]),50.)]
    return dict(q=q,v=v,hard=hard,soft=soft,rest=float(rest),k=float(k),m=m,
                work=0.,impulses={str(i):np.zeros(3) for i in hard},previous_target={i:v.copy() for i,v in hard.items()})

def target_for(s,kind,step):
    hard={i:v.copy() for i,v in s['hard'].items()}
    if kind=='moving_support':hard[len(s['q'])-1]+=np.array([1e-6*step,2.5e-7*step,0.])
    return hard

def energy(q,v,soft,rest,k,m):
    value=.5*np.sum(m[:,None]*v*v)+.5*k*np.sum((np.linalg.norm(q[1:]-q[:-1],axis=1)-rest)**2)
    for i,target,stiffness in soft:value+=.5*stiffness*np.dot(q[i]-target,q[i]-target)
    return float(value)

def pack(p,hard,soft,rest,k,m):
    n=len(p);nodes=np.zeros((17,4),dtype=np.float32);tar=np.zeros((17,4),dtype=np.float32)
    nodes[:n,:3]=p;nodes[:n,3]=m
    for i,v in hard.items():tar[i,:3]=v;tar[i,3]=-1
    for i,v,stiffness in soft:tar[i,:3]=v;tar[i,3]=stiffness
    return nodes.astype('<f4').tobytes()+tar.astype('<f4').tobytes()+struct.pack('<4f4I',rest,k,H,1e-6,n,0,0,0)

spec=[('straight_moving',9),('curved_fixed',9),('moving_support',17)]
gpu=[initial(*x) for x in spec];reference=[initial(*x) for x in spec]
for s in gpu:s['q']=s['q'].astype(np.float32);s['v']=s['v'].astype(np.float32)
input_capture=bytearray();output_capture=bytearray();rows=[];completed=True
for step in range(1,21):
    batch=bytearray();context=[]
    for (kind,n),s in zip(spec,gpu):
        pred=np.asarray(s['q']+np.float32(H)*s['v'],dtype=np.float32)
        hard=target_for(s,kind,step)
        data=pack(pred,hard,s['soft'],s['rest'],s['k'],s['m']);batch.extend(data)
        context.append((pred,hard))
    input_capture.extend(batch)
    Path('/tmp/numi-rod-trajectory-input.bin').write_bytes(batch)
    r=subprocess.run(['/tmp/numi-rod-block-host','/tmp/numi-rod-block.metallib','/tmp/numi-rod-trajectory-input.bin','/tmp/numi-rod-trajectory-output.bin'],capture_output=True,text=True)
    if r.returncode:raise RuntimeError(f'host failure at {step}: {r.stderr}')
    raw=Path('/tmp/numi-rod-trajectory-output.bin').read_bytes();output_capture.extend(raw)
    outs=np.frombuffer(raw,dtype='<f4').reshape((3,35,4))
    for i,((kind,n),s,t) in enumerate(zip(spec,gpu,reference)):
        pred,hard=context[i];out=outs[i];status=int(out[34,2]);refhard=target_for(t,kind,step)
        refpred=t['q']+H*t['v'];ref=refsolve(refpred,refhard,t['soft'],t['rest'],t['k'],t['m'])
        if status!=0 or ref['status']!='converged':
            rows.append(dict(step=step,case=kind,status=status,reference_status=ref['status'],passed=False));completed=False;continue
        q=out[:n,:3].copy();force=out[17:17+n,:3].astype(np.float64)
        refq=np.array(ref['positions']);refv=(refq-t['q'])/H
        v=np.asarray((q-s['q'])/np.float32(H),dtype=np.float32)
        poserr=float(np.max(np.linalg.norm(q.astype(np.float64)-refq,axis=1)))
        p64=pred.astype(np.float64);soft32=[(j,np.asarray(x,dtype=np.float32).astype(np.float64),float(np.float32(z))) for j,x,z in s['soft']]
        _,g,_,_=evaluate(q.astype(np.float64),p64,soft32,float(np.float32(s['rest'])),float(np.float32(s['k'])),s['m'].astype(np.float32).astype(np.float64),False)
        free=[j for j in range(n) if j not in hard];local=float(np.max(np.linalg.norm(g[free],axis=1)))
        errors={};limits={}
        for name,rf in ref['reactions'].items():
            j=int(name.split(':')[1]);errors[name]=float(np.linalg.norm(force[j]-rf));limits[name]=max(1e-6,.01*float(np.linalg.norm(rf)))
        for j,target in hard.items():
            current=np.asarray(target,dtype=np.float32).astype(np.float64);previous=np.asarray(s['previous_target'][j],dtype=np.float32).astype(np.float64)
            s['work']-=float(np.dot(force[j],current-previous));s['previous_target'][j]=current
            s['impulses'][str(j)]+=H*force[j]
            rforce=np.array(ref['reactions'][f'hard:{j}']);t['work']-=float(np.dot(rforce,refhard[j]-t['previous_target'][j]));t['previous_target'][j]=refhard[j].copy()
            t['impulses'][str(j)]+=H*rforce
        s['q']=q;s['v']=v;t['q']=refq;t['v']=refv
        passed=bool(poserr<=1e-6 and local<=1e-5 and all(errors[key]<=limits[key] for key in errors))
        rows.append(dict(step=step,case=kind,status=status,reference_status=ref['status'],position_error_m=poserr,
            local_residual_N=local,reaction_error_N=errors,reaction_limit_N=limits,passed=passed))
        if not passed:completed=False
    if any(int(o[34,2])!=0 for o in outs) or any(row['reference_status']!='converged' for row in rows[-3:]):break
summaries=[]
for (kind,n),s,t in zip(spec,gpu,reference):
    error=abs(s['work']-t['work']);limit=max(1e-9,.02*abs(t['work']))
    summaries.append(dict(case=kind,work_J=s['work'],reference_work_J=t['work'],work_error_J=error,work_limit_J=limit,
        work_pass=bool(error<=limit),final_energy_J=energy(s['q'].astype(np.float64),s['v'].astype(np.float64),s['soft'],s['rest'],s['k'],s['m']),
        reference_final_energy_J=energy(t['q'],t['v'],t['soft'],t['rest'],t['k'],t['m']),
        support_impulse_Ns={k:v.tolist() for k,v in s['impulses'].items()},
        reference_support_impulse_Ns={k:v.tolist() for k,v in t['impulses'].items()}))
result=dict(all_pass=bool(completed and len(rows)==60 and all(r['passed'] for r in rows) and all(x['work_pass'] for x in summaries)),steps=rows,summaries=summaries)
(b/'metal-trajectory-input.bin').write_bytes(input_capture);(b/'metal-trajectory-output.bin').write_bytes(output_capture)
print(json.dumps(result,indent=2,sort_keys=True))
sys.exit(0 if result['all_pass'] else 1)
