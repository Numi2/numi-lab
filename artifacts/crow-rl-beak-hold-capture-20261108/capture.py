#!/usr/bin/env python3
"""Observational replication of retained failed native contact trials."""
import hashlib,json,os,shutil,struct,subprocess,sys,time
from pathlib import Path
R=Path('/Users/n/Developer/numi-crow-rl-20261107')
B=R/'build/crow-beak-hold-build-01'
A=R/'artifacts/crow-rl-beak-hold-capture-20261108'
cases=[('clamp',0,32),('open',1,32),('zero-friction',3,32),('refinement',0,64)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def seal():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=R)
    A.mkdir(exist_ok=False)
    files=['build/crow-beak-hold-build-01/probe-v5','build/crow-beak-hold-build-01/probe-v5.mm',
       'build/bin/metalrobo_ccd_sweep_probe','build/shaders/CCDSweepProbe.metallib',
       'build/lib/libmetalrobo.dylib','build/shaders/MetalRobo.metallib']
    write(A/'plan.json',{'purpose':'diagnostic replication, no new mechanical prediction or success selection',
      'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
      'parent':'artifacts/crow-rl-beak-hold-repair-20261108',
      'question':'Which protected native projected sweep caused each retained failure?',
      'revision':'No mechanics, controls, tolerances, solver budgets or native binary changes. Enable existing failure-only observation; fix loader logging by setting DYLD variables after protected /usr/bin/time has launched.',
      'prediction':'Each command exits1 at same accepted tick, and accepted trace bytes remain identical. Failure snapshot <=1MiB, v2 protected first-failure identity; isolated native resolver reproduces stored candidate byte exactly using protected event. If not, capture cannot explain parent failure.',
      'runtime':{p:{'sha256':sha(R/p),'bytes':(R/p).stat().st_size} for p in files},
      'cases':[{'name':n,'arm':a,'substeps':s,'chunk':1,'parent_trace_sha256':sha(B/f'repair-{n}/trace.tsv')} for n,a,s in cases],
      'calibration':'Existing protected capture ABI and overwrite/null controls, artifacts/crow-rl-ccd-protected-sweep-20261108. This use requires native isolated replay of every new captured candidate.',
      'analysis':'All paired accepted traces byte compared; replay records and protected stamp checked by native owning probe. Sample distance along captured projection is descriptive, not swept-volume proof.',
      'budget':{'runs':4,'timeout_seconds_each':180,'replay_timeout_seconds_each':60,'raw_trial_bytes_maximum':8388608,
         'stopping':'One run per listed parent condition; no retry or physics repair. Retain every failure/missing snapshot. No policy promotion.'},
      'limits':'Failure diagnosis only. Parent is invalid for qualification (native rejections and unsupported direct-action final checkpoint); partial lift is not a completed contact capability.'})
    shutil.copy2(__file__,A/'capture.py')
    print(sha(A/'plan.json'))
def run():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=R)
    plan=json.loads((A/'plan.json').read_text())
    for p,v in plan['runtime'].items():assert sha(R/p)==v['sha256']
    results=[]
    for n,arm,sub in cases:
        prefix=B/f'capture-{n}';assert not prefix.exists()
        capture=B/f'capture-{n}.ccdsweep';assert not capture.exists()
        command=['/usr/bin/time','-lp','/usr/bin/env','DYLD_PRINT_LIBRARIES=1',
          f'DYLD_LIBRARY_PATH={R}/build/lib',f'METALROBO_METALLIB={R}/build/shaders/MetalRobo.metallib',
          f'METALROBO_CCD_FAILURE_CAPTURE={capture}',str(B/'probe-v5'),str(prefix),str(arm),'1',str(sub)]
        t=time.monotonic()
        with (B/f'capture-{n}.stdout').open('xb') as out,(B/f'capture-{n}.stderr').open('xb') as err:
            code=subprocess.run(command,cwd=R,stdout=out,stderr=err,timeout=180).returncode
        row={'name':n,'exit_code':code,'seconds':time.monotonic()-t,'command':command,
             'trace_exact':sha(prefix/'trace.tsv')==sha(B/f'repair-{n}/trace.tsv'),
             'snapshot_present':capture.exists()}
        if capture.exists():
            data=capture.read_bytes();assert len(data)<=1024**2
            row.update(snapshot_bytes=len(data),snapshot_sha256=sha(capture),header=list(struct.unpack_from('<6I',data,8)))
            replay=[str(R/'build/bin/metalrobo_ccd_sweep_probe'),str(R/'build/shaders/MetalRobo.metallib'),
              str(R/'build/shaders/CCDSweepProbe.metallib'),str(capture),str(B/f'capture-{n}-replay.bin')]
            with (B/f'capture-{n}-replay.json').open('xb') as out,(B/f'capture-{n}-replay.stderr').open('xb') as err:
                row['replay_exit_code']=subprocess.run(replay,cwd=R,stdout=out,stderr=err,timeout=60).returncode
            if row['replay_exit_code']==0:
                row['replay_matches']=json.loads((B/f'capture-{n}-replay.json').read_text())['stored_record_matches']
        logs=(B/f'capture-{n}.stderr').read_text()
        row['actual_loader_lines']=[s for s in logs.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
        results.append(row);write(B/'capture-execution.json',{'plan_sha256':sha(A/'plan.json'),'results':results})
        print(json.dumps(row),flush=True)
    write(A/'analysis.json',{'qualification':'failed parent, not qualified',
        'capture_equivalence_passed':all(x['exit_code']==1 and x['trace_exact'] and x['snapshot_present'] and x.get('replay_matches',[False])[0] and x['actual_loader_lines'] for x in results),
        'results':results})
os.chdir(R)
{'seal':seal,'run':run}[sys.argv[1]]()
