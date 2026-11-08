"""Native CCD sphere/box distance, witnesses and replay against an FP64 oracle."""
from pathlib import Path
import argparse, hashlib, json, subprocess
import numpy as np
def rotate(q,v):
    u=q[:,:3];return v+2*np.cross(u,np.cross(u,v)+q[:,3,None]*v)
def generate(seed):
    rng=np.random.default_rng(seed);n=8192;x=np.zeros((n,7,4),np.float32)
    x[:,0,:3]=np.exp(rng.uniform(np.log(.002),np.log(.08),(n,3)))
    x[:n//2,0,:3]=[.025,.009,.004];x[:,0,3]=rng.uniform(.002,.02,n);x[:n//2,0,3]=.008
    q=rng.normal(size=(n,4));q/=np.linalg.norm(q,axis=1)[:,None]
    q[:128]=[0,0,0,1];x[:,3]=q;x[:,4]=q
    # Half use moving/rotating end transforms, including quaternion sign flips.
    delta=rng.normal(size=(n//2,4));delta/=np.linalg.norm(delta,axis=1)[:,None]
    x[n//2:,4]=delta;x[::7,4]*=-1
    x[:,1,:3]=rng.uniform(-1,1,(n,3));x[:,2,:3]=x[:,1,:3]+rng.uniform(-.003,.003,(n,3))
    x[:,1,3]=rng.uniform(0,1,n);x[:64,1,3]=0;x[64:128,1,3]=1
    local=rng.uniform(-1.6,1.6,(n,3))*x[:,0,:3]
    # Explicit thin-face, edge and corner approaches at signed gaps down to1nm.
    for i in range(n//2):
        k=i%3;local[i]=rng.uniform(-.8,.8,3)*x[i,0,:3]
        normal=np.zeros(3);normal[k]=(-1 if i%2 else 1)
        if i%4==1:normal[(k+1)%3]=normal[k]
        if i%4==2:normal[:]=normal[k]
        normal/=np.linalg.norm(normal)
        base=np.clip(local[i],-x[i,0,:3],x[i,0,:3])
        base[normal!=0]=np.sign(normal[normal!=0])*x[i,0,:3][normal!=0]
        gap=(-1 if i%5==0 else 1)*10**rng.uniform(-9,-2)
        local[i]=base+normal*(float(x[i,0,3])+gap)
    sphere=x[:,1,:3].astype(float)+rotate(x[:,3].astype(float),local)
    x[:,5,:3]=sphere;x[:,6,:3]=sphere+rng.uniform(-.003,.003,(n,3))
    x[:64,5,:3]=x[:64,1,:3];x[:64,6,:3]=x[:64,2,:3] # enclosed centers
    swapped=x.copy();swapped[:,2,3]=1
    return np.concatenate([x,swapped])
def analyze(cases,result):
    n=len(cases);raw=result if isinstance(result,bytes) else result.read_bytes()
    if len(raw)!=64*n:raise RuntimeError('Incomplete native result')
    floats=np.frombuffer(raw,dtype='<f4').reshape(n,16)[:,:12].astype(float)
    status=np.frombuffer(raw,dtype='<u4').reshape(n,16)[:,12:]
    if not np.isfinite(floats).all():raise RuntimeError('Non-finite native query output')
    c=cases.astype(float);alpha=c[:,1,3,None]
    q0=c[:,3];q1=c[:,4].copy();q1[np.sum(q0*q1,axis=1)<0]*=-1
    q=q0*(1-alpha)+q1*alpha;q/=np.linalg.norm(q,axis=1)[:,None]
    box=c[:,1,:3]*(1-alpha)+c[:,2,:3]*alpha;sphere=c[:,5,:3]*(1-alpha)+c[:,6,:3]*alpha
    qi=q.copy();qi[:,:3]*=-1;local=rotate(qi,sphere-box);h=c[:,0,:3];r=c[:,0,3]
    outside=np.maximum(np.abs(local)-h,0);dist=np.linalg.norm(outside,axis=1)
    expected=dist+np.minimum(np.max(np.abs(local)-h,axis=1),0)-r
    valid=(status[:,0]==0)&(status[:,1]==1)
    error=np.abs(floats[:,3]-expected)
    a=floats[:,4:7];b=floats[:,8:11];normal=floats[:,:3]
    sphere_point=np.where(c[:,2,3,None]!=0,a,b);box_point=np.where(c[:,2,3,None]!=0,b,a)
    box_local=rotate(qi,box_point-box)
    witness_sphere=np.abs(np.linalg.norm(sphere_point-sphere,axis=1)-r)
    witness_box=np.maximum(np.max(np.abs(box_local)-h,axis=1),0)+np.abs(np.minimum(np.max(np.abs(box_local)-h,axis=1),0))
    identity=np.abs(np.sum((b-a)*normal,axis=1)-floats[:,3]);normal_error=np.abs(np.linalg.norm(normal,axis=1)-1)
    half=n//2;swapped_norm=float(np.max(np.abs(normal[:half]+normal[half:])))
    failed=np.flatnonzero(~valid | (error>8e-6) | (witness_sphere>8e-6) | (witness_box>8e-6) | (identity>8e-6) | (normal_error>1e-4))
    return {'cases':n,'status_successes':int(valid.sum()),'distance_max_abs_error_m':float(error.max()),
      'valid_distance_max_abs_error_m':float(error[valid].max()) if valid.any() else None,
      'sphere_surface_witness_max_error_m':float(witness_sphere.max()),'box_surface_witness_max_error_m':float(witness_box.max()),
      'normal_unit_max_error':float(normal_error.max()),'witness_separation_identity_max_error_m':float(identity.max()),
      'swapped_normal_max_abs_error':swapped_norm,'swapped_separation_max_abs_error_m':float(np.max(np.abs(floats[:half,3]-floats[half:,3]))),
      'all_requirements_passed':bool(len(failed)==0 and swapped_norm<=1e-4),'failed_cases':failed.tolist(),
      'status_counts':{str(int(code)):int(np.count_nonzero(status[:,0]==code)) for code in np.unique(status[:,0])}}
def calibration():
    # Two hand-solvable witnesses, then their swapped-order twins. No GPU or
    # production geometry implementation is used to manufacture these answers.
    cases=np.zeros((4,7,4),np.float32);cases[:,0]=[.025,.009,.004,.008]
    cases[:,3,3]=cases[:,4,3]=1;cases[0,5,0]=cases[0,6,0]=.04
    cases[2:]=cases[:2];cases[2:,2,3]=1
    answer=np.zeros((4,16),np.float32)
    answer[0,:4]=[1,0,0,.007];answer[0,4:7]=[.025,0,0];answer[0,8:11]=[.032,0,0]
    answer[1,:4]=[0,0,1,-.012];answer[1,4:7]=[0,0,.004];answer[1,8:11]=[0,0,-.008]
    answer[2:]=answer[:2];answer[2:,:3]*=-1
    answer[2:,4:7]=answer[:2,8:11];answer[2:,8:11]=answer[:2,4:7]
    answer.view(np.uint32)[:,13]=1
    result=analyze(cases,answer.tobytes())
    if not result['all_requirements_passed']:raise RuntimeError('Known face/enclosed witness calibration failed')
    wrong=answer.copy();wrong[:2,4:7]=answer[:2,8:11];wrong[:2,8:11]=answer[:2,4:7]
    if analyze(cases,wrong.tobytes())['all_requirements_passed']:raise RuntimeError('Witness-order negative control failed')
    return {'status':'passed','hand_solved_face_and_enclosed_cases_both_orders':4,'wrong_witness_order_rejected':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--metallib',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2651319004)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    save('plan.json',{'cases':16384,'seed':args.seed,'requirements':'All status success; FP64 distance/surface/separation-identity errors<=8um; unit/swap normal errors<=1e-4; exact native replay',
        'oracle_calibration':calibration(),'probe_sha256':hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        'metallib_sha256':hashlib.sha256(args.metallib.read_bytes()).hexdigest(),'scope':'Native query correctness only, not trajectory or biological qualification'})
    cases=generate(args.seed);(out/'cases.f32').write_bytes(cases.tobytes())
    for name in ('results','repeat'):
        command=[str(args.probe.resolve()),str(args.metallib.resolve()),str(out/'cases.f32'),str(out/(name+'.bin'))]
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        (out/(name+'.stdout')).write_text(result.stdout);(out/(name+'.stderr')).write_text(result.stderr)
        save(name+'.receipt.json',{'command':command,'returncode':result.returncode})
        if result.returncode:raise RuntimeError('Native CCD query probe failed')
    if (out/'results.bin').read_bytes()!=(out/'repeat.bin').read_bytes():raise RuntimeError('Native replay differs')
    result=analyze(cases,out/'results.bin');result['native_replay_exact']=True
    save('analysis.json',result)
    if not result['all_requirements_passed']:raise RuntimeError('Native CCD distance/witness oracle failed')
    print(json.dumps({k:v for k,v in result.items() if k!='failed_cases'},indent=2))
if __name__=='__main__':main()
