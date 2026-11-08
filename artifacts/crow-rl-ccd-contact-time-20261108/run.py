from pathlib import Path
import hashlib,json,os,shutil,struct,subprocess,time
import numpy as np
from oracle import HALF,RADIUS,TOL,DT,local_segment
R=Path.cwd();B=R/'build/crow-ccd-contact-time-build-01';A=R/'artifacts/crow-rl-ccd-contact-time-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert not subprocess.check_output(['git','status','--porcelain'])
plan=json.loads((A/'plan.json').read_text())
for p,v in plan['inputs'].items():assert sha(R/p)==v['sha256'],p
for p,h in json.loads((B/'unchanged-air-identities.json').read_text()).items():assert sha(Path(p))==h,p
commands=[]
def run(name,cmd,env=None,expected=0):
    start=time.monotonic()
    with (B/f'{name}.stdout').open('xb') as out,(B/f'{name}.stderr').open('xb') as err:
        code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=180).returncode
    commands.append({'name':name,'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start});write(B/'execution.json',commands)
    print(json.dumps(commands[-1]),flush=True)
    if expected is not None:assert code==expected,(name,code)
    return code
run('compile',['xcrun','-sdk','macosx','metal','-std=metal4.0','-O3','-fno-fast-math','-I',R/'include','-c',B/'Collision.candidate.metal','-o',B/'candidate.air'])
run('link',json.loads((B/'link-command.json').read_text()))
for name,shader in [('baseline','baseline.metallib'),('candidate','candidate.metallib'),('repeat','candidate.metallib')]:
    run(name,[B/'cases-probe',B/shader,B/'cases.bin',B/(name+'.bin')])
old=(B/'baseline.bin').read_bytes();new=(B/'candidate.bin').read_bytes();oi=np.frombuffer(old,dtype='<u4').reshape(-1,16);ni=np.frombuffer(new,dtype='<u4').reshape(-1,16);nf=ni.view('<f4')
groups=json.loads((B/'groups.json').read_text());pred=json.loads((B/'oracle.json').read_text());n=len(groups)
raw=(B/'cases.bin').read_bytes();assert struct.unpack_from('<I',raw)[0]==n
c=np.frombuffer(raw,dtype='<f4',offset=4,count=n*40).reshape(n,2,5,4);f=np.frombuffer(raw,dtype='<f4',offset=4+n*160,count=n*40).reshape(n,2,5,4)
wrong=[];gaps=[];unsafe=[];impacts=0
for i,g in enumerate(groups):
    for order in range(2):
        j=i*2+order;before=old[j*64:(j+1)*64];after=new[j*64:(j+1)*64];hit=g!='rotating' and bool(oi[j,14]&4)
        if hit:
            impacts+=1
            if before[:16]!=after[:16] or before[20:]!=after[20:] or ni[j,4]!=oi[j,5]:wrong.append(j)
            p,d=local_segment(c[i],f[i]);alpha=float(nf[j,4])/DT
            gap=float(np.linalg.norm(np.maximum(np.abs(p+alpha*d)-HALF,0.)))-RADIUS
            if abs(gap-TOL)>5e-7:gaps.append({'record':j,'gap_m':gap,'alpha':alpha})
        elif before!=after:wrong.append(j)
        if g!='rotating' and pred[i]['first_alpha'] is not None and pred[i]['minimum_unsigned_box_distance_minus_radius_m']<TOL-5e-7 and ni[j,15]==0 and not ni[j,14]&14:unsafe.append(j)
checks={'only_intended_lower_fields_change':not wrong,'impact_selected_FP64_gap_bound':not gaps,'zero_unsafe_no_impact':not unsafe,
 'repeat_exact':new==(B/'repeat.bin').read_bytes(),'at_least_one_fixed_box_impact_tested':impacts>0}
captures={}
for name in ('clamp','refinement'):
    for variant,shader in [('baseline','baseline.metallib'),('candidate','candidate.metallib')]:
        run(name+'-'+variant,[B/'sweep-probe',B/shader,B/'query.metallib',B/(name+'.ccdsweep'),B/(name+'-'+variant+'.bin')])
    before=(B/(name+'-baseline.bin')).read_bytes();after=(B/(name+'-candidate.bin')).read_bytes()
    a=np.frombuffer(after[:128],dtype='<u4').reshape(2,16);b=np.frombuffer(before[:128],dtype='<u4').reshape(2,16)
    captures[name]={'query_bytes_exact':before[128:]==after[128:],
      'only_lower_time_changes':all(before[j*64:j*64+16]==after[j*64:j*64+16] and before[j*64+20:(j+1)*64]==after[j*64+20:(j+1)*64] and a[j,4]==b[j,5] for j in range(2)),
      'old_lower_and_new_time':[[float(b.view('<f4')[j,4]),float(a.view('<f4')[j,4])] for j in range(2)]}
checks['retained_captures_only_lower_time_changes']=all(x['query_bytes_exact'] and x['only_lower_time_changes'] for x in captures.values())
report={'checks':checks,'impacts':impacts,'wrong_records':wrong,'out_of_bound_selected_FP64_gaps':gaps,'unsafe_no_impact_records':unsafe,'captures':captures,'grid_component_passed':all(checks.values()),
 'physics_capability_qualified':False,'production_replaced':False,'commands':commands}
write(A/'results.json',report)
if not all(checks.values()):print(json.dumps(report));raise SystemExit('Candidate contradicted; no physical trial or production promotion')
# Use the exact candidate shader in the owning library-relative layout.
runtime=B/'runtime';(runtime/'metalrobo').mkdir(parents=True,exist_ok=False)
native=R/'.numi/retained-runtime/ccd-event-capture-20261108/candidate/libmetalrobo.dylib'
shutil.copy2(native,runtime/'libmetalrobo.dylib');shutil.copy2(B/'candidate.metallib',runtime/'metalrobo/MetalRobo.metallib')
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(runtime),DYLD_PRINT_LIBRARIES='1')
physical={}
for name,substeps in [('clamp',32),('refinement',64)]:
    env['METALROBO_CCD_FAILURE_CAPTURE']=str(B/(name+'-physical.ccdsweep'))
    code=run(name+'-physical',[B/'probe',B/(name+'-physical'),'0','1',str(substeps)],env,None)
    loaded=[s for s in (B/(name+'-physical.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
    assert loaded and all(s.endswith(str(runtime/'libmetalrobo.dylib')) for s in loaded)
    physical[name]={'exit_code':code,'actual_loader':loaded}
    for n in ('failure.json','summary.json'):
        p=B/(name+'-physical')/n
        if p.exists():physical[name][n]=json.loads(p.read_text())
report.update(physical_development_replication=physical,all_development_physics_accepted=all(x['exit_code']==0 for x in physical.values()),commands=commands,
 candidate_shader_sha256=sha(B/'candidate.metallib'),candidate_source_sha256=sha(B/'Collision.candidate.metal'))
write(A/'results.json',report)
print(json.dumps({'checks':checks,'physical_development_replication':physical}),flush=True)
