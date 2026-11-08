#!/usr/bin/env python3
"""Research-only coupled trajectories with a native-energy Newton reference."""
import ast
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
p=Path(__file__).resolve().parent
base=p.parent/'rod-coupled-bend-20261108'
started=time.monotonic()
tangent=ast.parse((base/'tangent.py').read_text())
bfn=next(x for x in tangent.body if isinstance(x,ast.FunctionDef) and x.name=='grad_and_hv')
ns={'np':np};exec(compile(ast.Module(body=[bfn],type_ignores=[]),'tangent.py','exec'),ns)
bend=ns['grad_and_hv']
src=ast.parse((base/'solve.py').read_text()); funcs=['setup','grad_hv','energy_delta','solve']
ns.update({'bend':bend,'L':.11758755785232791,'EA':2.3818718058345718,'R':2e-5,'RHO':1100.,'POISSON':.35,'H':.001})
ns['EI']=ns['EA']*ns['R']**2/4;ns['GJ']=ns['EI']/(1+ns['POISSON']);ns['AREA']=np.pi*ns['R']**2
for name in funcs:
    node=next(x for x in src.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'solve.py','exec'),ns)
solve=ns['solve'];setup=ns['setup'];H=ns['H']; exe=sys.argv[1]; probe=sys.argv[2]; shader=sys.argv[3]
rawcases=json.loads((p.parent/'rod-coupled-metal-holdout-20261108'/'cases.json').read_text());indices=[0,1,2]
def make(index):
    c=rawcases[index].copy()
    for k in ('q','v','theta','omega','target'):c[k]=np.array(c[k],dtype=np.float64)
    return c

def encode(c,q,t):
    return ' '.join(format(float(a),'.17g') for a in [c['n'],*q.ravel(),*t])
trace=open(p/'trajectory-native-trace.gz','wb')
gz=gzip.GzipFile(fileobj=trace,mode='wb',mtime=0)
def native(c,states):
    if time.monotonic()-started>300:raise TimeoutError('study exceeded 300 seconds')
    inp='\n'.join(encode(c,q,t) for q,t in states)+'\n'
    r=subprocess.run([exe],input=inp,text=True,capture_output=True,timeout=30)
    gz.write(('INPUT\n'+inp+'OUTPUT\n'+r.stdout+'STDERR\n'+r.stderr+'EXIT '+str(r.returncode)+'\n').encode());gz.flush()
    if r.returncode:raise RuntimeError(f'native exit {r.returncode}: {r.stderr}')
    rows=[json.loads(x) for x in r.stdout.splitlines()]
    if len(rows)!=len(states):raise ValueError('native record count')
    return rows

def metal(c):
    arrays=[np.asarray(c[k],dtype=np.float64).ravel() for k in ('q','v','theta','omega','target')]
    inp=' '.join(format(float(a),'.17g') for a in [c['n'],30,*np.concatenate(arrays)])+'\n'
    r=subprocess.run([probe,shader],input=inp,text=True,capture_output=True,timeout=30)
    gz.write(('METAL INPUT\n'+inp+'METAL OUTPUT\n'+r.stdout+'METAL STDERR\n'+r.stderr+'METAL EXIT '+str(r.returncode)+'\n').encode());gz.flush()
    if r.returncode or r.stderr:raise RuntimeError(f'Metal probe exit {r.returncode}: {r.stderr}')
    a=json.loads(r.stdout)
    if not (a['status']==0 and a['gpu']==0 and a['published'] and a['states']==1 and a['reactions']==2):
        return {'status':'gpu_failure_'+str(a['candidate_status']),'iterations':a['iterations']}
    a['status']='converged'
    return a

def energygrad(r,n):
    bg=np.array(r['bendGradient1']);out=bg.copy()
    out[:3*n]+=r['stretchGradient1'];out[3*n:]+=r['twistGradient1']
    return out

def state_metrics(c,q,t,row):
    n=c['n'];ell,mass,inertia=setup(c);g=energygrad(row,n)
    d=q-c['q'];td=t-c['theta'];vel=d/H;omega=td/H
    rv=vel-c['v']+H*g[:3*n].reshape(n,3)/mass[:,None]
    rw=omega-c['omega']+H*g[3*n:]/inertia
    force=-(mass[:,None]/H**2*(d-H*c['v'])+g[:3*n].reshape(n,3))[[0,-1]]
    kinetic=.5*np.sum(mass[:,None]*vel**2)+.5*np.sum(inertia*omega**2)
    total=float(kinetic+row['bend']+row['stretch']+row['twist'])
    return vel,omega,rv,rw,force,total

def reference(c):
    n=c['n'];N=4*n-1;m=3*n;ell,mass,inertia=setup(c)
    free=np.r_[np.arange(3,3*(n-1)),np.arange(m,N)]
    d=H*c['v'];dt=H*c['omega'];d[[0,-1]]=c['target']-c['q'][[0,-1]]
    for iteration in range(20):
        q=c['q']+d;t=c['theta']+dt
        states=[(q,t)]
        for j in free:
            h=1e-6 if j<m else 1e-4
            for sign in (1.,-1.):
                qq=q.copy();tt=t.copy()
                if j<m:qq[j//3,j%3]+=sign*h
                else:tt[j-m]+=sign*h
                states.append((qq,tt))
        rr=native(c,states);g=energygrad(rr[0],n)
        vel,omega,rv,rw,force,total=state_metrics(c,q,t,rr[0])
        if max(np.max(np.abs(rv[1:-1])),np.max(np.abs(rw)))<=1e-4:
            return {'status':'converged','q':q,'theta':t,'v':vel,'omega':omega,'reaction':force,'energy':total,'iterations':iteration+1,'rv':float(np.max(np.abs(rv[1:-1]))),'rw':float(np.max(np.abs(rw)))}
        J=np.empty((len(free),len(free)))
        for k,j in enumerate(free):
            h=1e-6 if j<m else 1e-4
            J[:,k]=(energygrad(rr[1+2*k],n)[free]-energygrad(rr[2+2*k],n)[free])/(2*h)
        inertia_diag=np.r_[np.repeat(mass/H**2,3),inertia/H**2]
        J+=np.diag(inertia_diag[free]);J=(J+J.T)*.5
        if np.any(np.diag(J)<=0):return {'status':'reference_curvature','iterations':iteration+1}
        D=1/np.sqrt(np.diag(J));A=D[:,None]*J*D[None,:]
        if np.min(np.linalg.eigvalsh(A))<=0:return {'status':'reference_curvature','iterations':iteration+1}
        fullg=g+inertia_diag*np.r_[d.ravel()-H*c['v'].ravel(),dt-H*c['omega']]
        update=np.zeros(N);update[free]=D*np.linalg.solve(A,-D*fullg[free])
        if not np.isfinite(update).all():return {'status':'reference_nonfinite','iterations':iteration+1}
        d+=update[:m].reshape(n,3);dt+=update[m:];d[[0,-1]]=c['target']-c['q'][[0,-1]]
    return {'status':'reference_iterations','iterations':20}

def native_info(c,r):
    q=np.array(r['q']);t=np.array(r['theta']);row=native(c,[(q,t)])[0]
    return state_metrics(c,q,t,row)

def outstate(state,r):
    state['q']=np.asarray(r['q']);state['v']=np.asarray(r['v']);state['theta']=np.asarray(r['theta']);state['omega']=np.asarray(r['omega'])

def err(a,b):return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
results=[];good=True
try:
    with open(p/'trajectory-progress.jsonl','w') as progress:
        for index in indices:
            cand=make(index);ref=make(index);qbase=cand['q'].copy()
            cand['target']=cand['q'][[0,-1]].copy()
            ref['target']=ref['q'][[0,-1]].copy()
            base_target=cand['target'].copy()
            if not np.array_equal(base_target,qbase[[0,-1]]):
                raise ValueError('initial anchor targets must match endpoints')
            direction=qbase[1]-qbase[0];direction/=np.linalg.norm(direction)
            cw=rw=0.
            for step in range(1,11):
                tgt=base_target.copy()
                if index==1:tgt[-1]+=direction*(step*1e-6)
                cand['target']=tgt.copy();ref['target']=tgt.copy()
                a=metal(cand)
                if a['status']!='converged':
                    row={'case':index,'step':step,'candidate_status':a['status'],'passed':False};results.append(row);good=False;progress.write(json.dumps(row)+'\n');progress.flush();break
                b=reference(ref)
                if b['status']!='converged':
                    row={'case':index,'step':step,'candidate_status':a['status'],'reference_status':b['status'],'passed':False};results.append(row);good=False;progress.write(json.dumps(row)+'\n');progress.flush();break
                cv,co,crv,crw,cf,ce=native_info(cand,a)
                fa=np.array(a['reaction']);fb=np.array(b['reaction']);fe=np.linalg.norm(fa-fb,axis=1);fl=np.maximum(1e-6,.01*np.linalg.norm(fb,axis=1))
                wa=float(np.sum(fa*(tgt-cand['q'][[0,-1]])));wb=float(np.sum(fb*(tgt-ref['q'][[0,-1]])));cw+=wa;rw+=wb
                qerr=float(np.max(np.linalg.norm(np.array(a['q'])-b['q'],axis=1)))
                verr=float(np.max(np.linalg.norm(np.array(a['v'])-b['v'],axis=1)))
                terr=err(a['theta'],b['theta']);oerr=err(a['omega'],b['omega'])
                lm=float(np.max(np.abs(crv[1:-1])));am=float(np.max(np.abs(crw)))
                we=abs(wa-wb);wec=abs(cw-rw);el=abs(ce-b['energy'])
                pass_step=bool(qerr<=1e-6 and verr<=.001 and terr<=1e-4 and oerr<=.001 and np.all(fe<=fl) and lm<=.001 and am<=.001 and we<=max(1e-9,.02*abs(wb)) and wec<=max(1e-9,.02*abs(rw)) and el<=max(1e-9,.02*abs(b['energy'])))
                row={'case':index,'step':step,'candidate_status':a['status'],'reference_status':b['status'],'candidate_iterations':a['iterations'],'reference_iterations':b['iterations'],'position_error_m':qerr,'velocity_error_m_s':verr,'twist_error_rad':terr,'twist_rate_error_rad_s':oerr,'linear_residual_m_s':lm,'angular_residual_rad_s':am,'reference_linear_residual_m_s':b['rv'],'reference_angular_residual_rad_s':b['rw'],'force_error_N':fe.tolist(),'force_limit_N':fl.tolist(),'candidate_work_J':wa,'reference_work_J':wb,'candidate_cumulative_work_J':cw,'reference_cumulative_work_J':rw,'candidate_energy_J':ce,'reference_energy_J':b['energy'],'energy_error_J':el,'passed':pass_step}
                results.append(row);good=good and pass_step;progress.write(json.dumps(row,allow_nan=False)+'\n');progress.flush()
                if not pass_step: break
                outstate(cand,a);outstate(ref,b)
except Exception as e:
    good=False;results.append({'error':type(e).__name__+': '+str(e),'passed':False})
finally:
    gz.close();trace.close()
result={'all_pass':bool(good and len(results)==30),'completed_steps':len([r for r in results if 'step' in r]),'cases':results}
(p/'trajectory-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({'all_pass':result['all_pass'],'completed_steps':result['completed_steps'],'failures':[r for r in results if not r['passed']]},indent=2))
sys.exit(0 if result['all_pass'] else 1)
