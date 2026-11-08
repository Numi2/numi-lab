from pathlib import Path
from fractions import Fraction
import hashlib,json,math,os,struct,subprocess,time
import numpy as np
R=Path.cwd();A=R/'artifacts/crow-rl-split-precision-20261108';B=R/'build/crow-split-precision-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((A/'plan.json').read_text());runtime=json.loads((A/'runtime.json').read_text())
for name,f in plan['sources'].items():assert sha(A/name)==f['sha256']
for name,f in runtime['files'].items():assert sha(R/name)==f['sha256']
for f in plan['cases']:assert sha(R/f['path'])==f['sha256']
out=B/'observations';out.mkdir();records=[];analysis=[]
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
def launch(name,mode,input=None):
    dest=out/name;cmd=[B/'probe',mode,B/'split.metallib',dest]
    if input is not None:cmd.append(input)
    start=time.monotonic()
    with (out/(name+'.stdout')).open('xb') as stdout,(out/(name+'.stderr')).open('xb') as stderr:
        try:p=subprocess.run([str(x) for x in cmd],stdout=stdout,stderr=stderr,env=env,timeout=30);code=p.returncode
        except subprocess.TimeoutExpired:code=None
    rec=dict(name=name,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start);records.append(rec);write(A/'execution.json',records);assert code==0,rec
    rec.update(json.loads((out/(name+'.stdout')).read_text()));write(A/'execution.json',records)
    print(json.dumps(rec),flush=True);return dest
reference={};exec(compile((A/'independent-reference-source.py').read_text().split('calibration=calibrate()')[0],'independent-reference','exec'),reference)
reference_calibration=reference['calibrate']()
for repeat in range(2):
    dest=launch('calibration-'+str(repeat),'calibrate')
    x=np.fromfile(dest/'arithmetic.f32x2',dtype='<f4').reshape(-1,2)
    f=[Fraction(float(h))+Fraction(float(l)) for h,l in x]
    expected=[1+Fraction(1,2**30),Fraction(1,2**30),1-Fraction(1,2**48),Fraction(1,10),None,Fraction(1),Fraction(-7,3),-Fraction(1,2**30)]
    errors=[abs(float(a-b)) if b is not None else abs(float(a)-math.sqrt(2)) for a,b in zip(f,expected)]
    assert max(errors)<2e-14,(errors,x)
    analysis.append(dict(name='calibration-'+str(repeat),arithmetic_abs_errors=errors,passed=True))
assert (out/'calibration-0/arithmetic.f32x2').read_bytes()==(out/'calibration-1/arithmetic.f32x2').read_bytes()
def load(path):
    raw=path.read_bytes();h=struct.unpack_from('<8I',raw);nv,rows,nb=h[2:5];offset=32;blocks=[]
    for _ in range(nb):
        z=struct.unpack_from('<8I20f',raw,offset);offset+=128;at,n,kind=z[:3]
        blocks.append(dict(at=at,n=n,kind=kind,s=np.array(z[8:16])[:n],r=np.array(z[16:24])[:n],bounds=z[24:28]))
    def take(n):
        nonlocal offset
        x=np.frombuffer(raw,dtype='<f4',count=n,offset=offset).astype(float);offset+=4*n;return x
    mass,jac,bias,free,start=[take(n) for n in [nv*nv,rows*nv,rows,nv,nv]]
    assert offset==len(raw);return nv,rows,blocks,mass.reshape(nv,nv),jac.reshape(rows,nv),bias,free,start
def evaluate(case,v,hessian=False):
    nv,rows,blocks,mass,jac,bias,free,start=case
    w=jac@v+bias;p=np.zeros(rows);h=mass.copy();kinds={};feas=0.
    for b in blocks:
        at=slice(b['at'],b['at']+b['n']);local=w[at]
        if b['kind']==0:
            x=-local[0]/b['r'][0];lo,hi=b['bounds'][:2];pp=np.array([np.clip(x,lo,hi)]);d=np.array([[1/b['r'][0] if lo<x<hi else 0.]])
            kind='scalar_free' if lo<x<hi else 'scalar_bound';feas=max(feas,float(max(lo-pp[0],pp[0]-hi,0)))
        else:
            pp,d,kind=reference['cone'](local,b['r'],b['s']);feas=max(feas,float(max(0,np.linalg.norm(pp[1:]/b['s'][1:])-pp[0],-pp[0])))
        p[at]=pp;h+=jac[at].T@d@jac[at];kinds[kind]=kinds.get(kind,0)+1
    av=mass@(v-free);jp=jac.T@p;g=av-jp;res=float(np.max(np.abs(g))/(1+np.max(np.abs(av))+np.max(np.abs(jp))))
    return dict(g=g,p=p,w=w,h=h,res=res,kinds=kinds,feas=feas)
def oracle(case):
    v=case[-1].copy()
    for _ in range(4):
        e=evaluate(case,v)
        if e['res']<1e-13:break
        scale=1/np.sqrt(np.diag(e['h']));v+=scale*np.linalg.solve(e['h']*np.outer(scale,scale),-e['g']*scale)
    assert evaluate(case,v)['res']<1e-13
    return v
for c in plan['cases']:
    case=load(R/c['path']);nv,rows=case[:2];repeats=2 if c['name'].startswith('substep') else 1
    for repeat in range(repeats):
        name=c['name']+'-'+str(repeat);dest=launch(name,'solve',R/c['path'])
        si=struct.unpack('<8I8f',(dest/'status.bin').read_bytes());x=np.fromfile(dest/'scratch.f32x2',dtype='<f4').reshape(-1,2)
        split=x.astype(float).sum(axis=1);v=split[:nv];w=split[nv:nv+rows];p=split[nv+rows:nv+2*rows];g=split[nv+2*rows:2*nv+2*rows]
        rec=dict(name=name,case=c['name'],code=si[0],iterations=si[1],reported_scaled_residual=si[8],reported_max_gradient=si[9],projection_counts=dict(interior=si[2],vertex=si[3],scalar_free=si[4],scalar_bound=si[5]),unsupported_block=si[6],rounding=c['rounding'])
        if c['name']=='unsupported-boundary':
            assert si[0]==3 and np.array_equal(v,case[-1]);rec['negative_control_typed_and_unchanged']=True
        else:
            assert si[0] in (0,7) and np.isfinite(v).all() and np.isfinite(p).all() and np.isfinite(g).all()
            e=evaluate(case,v);rv=v.astype(np.float32).astype(float);er=evaluate(case,rv);ref=oracle(case)
            rec.update(independent_scaled_residual=e['res'],canonical_rounded_scaled_residual=er['res'],velocity_reference_max_error=float(np.max(np.abs(v-ref))),impulse_abs_error=float(np.max(np.abs(p-e['p']))) if rows else 0.,row_abs_error=float(np.max(np.abs(w-e['w']))) if rows else 0.,gradient_abs_error=float(np.max(np.abs(g-e['g']))),independent_feasibility=e['feas'],independent_projection_branches=e['kinds'],state_low_max=float(np.max(np.abs(x[:nv,1]))),native_gate_pass=si[0]==0,independent_gate_pass=e['res']<=1e-10)
            assert abs(si[8]-e['res'])<2e-12 and rec['gradient_abs_error']<2e-12
            assert (si[0]==0)==(e['res']<=1e-10)
        if repeat:
            prev=out/(c['name']+'-0')
            rec['repeat_bytes_exact']=all((dest/n).read_bytes()==(prev/n).read_bytes() for n in ['scratch.f32x2','status.bin'])
            assert rec['repeat_bytes_exact']
        analysis.append(rec);write(A/'analysis.json',analysis);print(json.dumps(rec),flush=True)
assert len(records)==14
pred=plan['predictions'];capture_records=[r for r in analysis if r.get('case','').startswith('substep')]
split_records=[r for r in capture_records if not r['rounding']]
rounded7=[r for r in capture_records if r['case']=='substep7-rounded']
results=dict(calibration_pass=all(r['passed'] for r in analysis if r['name'].startswith('calibration')),
 split_captures_gate_pass=all(r['native_gate_pass'] and r['independent_gate_pass'] for r in split_records),
 split_independent_scaled_residual_max=max(r['independent_scaled_residual'] for r in split_records),
 split_velocity_reference_max_error=max(r['velocity_reference_max_error'] for r in split_records),
 split_reported_gradient_abs_error_max=max(r['gradient_abs_error'] for r in split_records),
 substep7_rounded_gate_pass=all(r['native_gate_pass'] for r in rounded7),
 repeated_binary_outputs_exact=all(r.get('repeat_bytes_exact',True) for r in capture_records),
 unsupported_boundary_code=next(r['code'] for r in analysis if r.get('case')=='unsupported-boundary'))
matched={k:(v==pred[k] if isinstance(pred[k],bool) or k=='unsupported_boundary_code' else v<=pred[k]) for k,v in results.items()}
bytes_used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());assert bytes_used<=plan['budget']['raw_bytes_max']
report=dict(schema='crow-native-split-precision-result-v1',observations=results,predictions_supported=matched,all_predictions_supported=all(matched.values()),reference_calibration=reference_calibration,native_submissions=len(records),raw_bytes=bytes_used,limits=plan['limits'],canonical_state_or_production_changed=False,goal_complete=False)
write(A/'result.json',report);print(json.dumps(report),flush=True)
