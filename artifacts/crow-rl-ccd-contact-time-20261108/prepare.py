"""Unused conditions for a translating-primitive CCD event-time revision."""
from pathlib import Path
import hashlib,json,shutil,shlex,struct,subprocess
import numpy as np
from oracle import HALF,rotate,local_segment,cast,calibrate
R=Path.cwd();B=R/'build/crow-ccd-contact-time-build-01';A=R/'artifacts/crow-rl-ccd-contact-time-20261108'
P=R/'build/crow-ccd-event-capture-build-01';T=R/'.numi/retained-runtime/ccd-event-capture-20261108/candidate'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert not subprocess.check_output(['git','status','--porcelain'])
B.mkdir(exist_ok=True);A.mkdir(exist_ok=False)
source=(R/'src/metal/Collision.metal').read_text();needle='''            record.intervalAndDistance.x =
                previousAlpha * timestep;'''
assert source.count(needle)==2
source=source.replace(needle,'''            // Each translating primitive advancement is a supporting-plane
            // lower bound on contact. The new query verifies its endpoint
            // has reached the existing distance tolerance; the old iterate
            // can be arbitrarily far back (including zero).
            record.intervalAndDistance.x =
                (translatingSphereBox ? alpha : previousAlpha) * timestep;''',1)
(B/'Collision.candidate.metal').write_text(source)
for src,dst in [(P/'probe','probe'),(P/'cases-probe','cases-probe'),(T/'sweep-probe','sweep-probe'),(T/'metalrobo/MetalRobo.metallib','baseline.metallib'),(T/'CCDSweepProbe.metallib','query.metallib')]:shutil.copy2(src,B/dst)
for name in ('clamp','refinement'):shutil.copy2(P/(name+'.ccdsweep'),B/(name+'.ccdsweep'))
# Reuse every unchanged AIR module from the actual full Release build. The
# candidate changes only Collision, and does not replace the production shader.
tokens=shlex.split(subprocess.check_output(['ninja','-C','build','-t','commands','shaders/MetalRobo.metallib'],text=True))
index=next(i for i,t in enumerate(tokens) if t=='metallib');link=tokens[index-3:]
assert link[:4]==['xcrun','-sdk','macosx','metallib'] and '&&' not in link
air=R/'build/shaders/MetalRoboCollision.air';assert link.count(str(air))==1
link=[str(B/'candidate.air') if x==str(air) else x for x in link]
assert link[-2]=='-o';link[-1]=str(B/'candidate.metallib')
write(B/'link-command.json',link)
write(B/'unchanged-air-identities.json',{p:sha(Path(p)) for p in link if p.endswith('.air') and p!=str(B/'candidate.air')})
rng=np.random.default_rng(2651522901);current=[];future=[];groups=[];pred=[]
for group,count in [('stationary',128),('overlap',128),('rotating',128),('away',256),('grazing',768),('crossing',256),('common-translation',256),('sphere-spin',128)]:
 for j in range(count):
    q=rng.normal(size=4);q=(q/np.linalg.norm(q)).astype(np.float32);center=rng.uniform(-.3,.3,3).astype(np.float32)
    axes=rng.choice(3,size=1+j%3,replace=False);n=np.zeros(3);n[axes]=rng.uniform(.25,1,len(axes))*rng.choice([-1,1],len(axes));n/=np.linalg.norm(n)
    support=rng.uniform(-.8,.8,3)*HALF;support[axes]=np.sign(n[axes])*HALF[axes]
    gap=rng.uniform(11e-6,200e-6) if group!='overlap' else -rng.uniform(.0001,.001)
    p=support+n*(float(np.float32(.008))+gap);tangent=rng.normal(size=3);tangent-=n*(tangent@n);tangent/=np.linalg.norm(tangent);delta=np.zeros(3)
    if group=='away':delta=n*rng.uniform(0,.01)+tangent*rng.uniform(.001,.1)
    if group=='grazing':delta=-n*rng.uniform(1e-5,.005)+tangent*rng.uniform(.001,.15)
    if group=='crossing':delta=-2*p
    if group=='common-translation':delta=-n*rng.uniform(0,.002)+tangent*rng.uniform(0,.01)
    if group=='sphere-spin':delta=n*rng.uniform(-.0001,.0001)+tangent*rng.uniform(0,.002)
    c=np.zeros((2,5,4),dtype='<f4');c[:,2,3]=1;c[0,2]=q;c[0,1,:3]=center;c[0,1,3]=.0463680923;c[1,1,:3]=center+rotate(q.astype(float),p);c[1,1,3]=.008
    f=c.copy();f[1,1,:3]=c[1,1,:3]+rotate(q.astype(float),delta)
    if group=='common-translation':f[:,1,:3]+=rng.uniform(-.05,.05,3)
    if group=='sphere-spin':
        sq=rng.normal(size=4);f[1,2]=(sq/np.linalg.norm(sq)).astype(np.float32);c[1,0,2]=np.asarray([rng.uniform(.001,.01)],dtype='<f4').view('<u4')[0].view('<f4')
    if group=='rotating':
        if j%2:f[0,2,0]=np.nextafter(f[0,2,0],np.float32(np.inf))
        else:
            v=rng.normal(size=4);f[0,2]=(v/np.linalg.norm(v)).astype(np.float32)
    current.append(c);future.append(f);groups.append(group);pred.append(None if group=='rotating' else cast(*local_segment(c,f)))
c=np.asarray(current,dtype='<f4');f=np.asarray(future,dtype='<f4')
with (B/'cases.bin').open('xb') as out:out.write(struct.pack('<I',len(groups)));out.write(c.tobytes());out.write(f.tobytes())
write(B/'groups.json',groups);write(B/'oracle.json',pred);write(A/'calibration.json',calibrate())
inputs=[p for p in B.rglob('*') if p.is_file() and p.suffix not in ('.stdout','.stderr') and '__pycache__' not in p.parts]
write(A/'plan.json',{'question':'Can the supporting-plane certified primitive endpoint eliminate zero-time replay without losing or advancing through contact?',
 'source_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'parent':'artifacts/crow-rl-ccd-event-capture-20261108',
 'revision':'Only for fixed-orientation translating sphere-box pairs, return the current supporting-plane advancement alpha after its query reaches the existing tolerance, rather than the previous iterate. Keep identical upper endpoint, distance queries, flags, iteration count, controls, contact tolerance and event budgets. Other geometries unchanged.',
 'seed':2651522901,'unused_next_seed':2651523901,'units':len(groups),'orders_per_unit':2,
 'prediction':{'only_lower_field_changes_on_fixed_box_impacts':True,'all_nonimpacts_rotating_and_initial_overlap_exact':True,
   'fixed_box_impact_lower_equals_unchanged_upper':True,'FP64_gap_at_selected_time_within_existing_tolerance_plusminus_m':5e-7,
   'unsafe_successful_no_impact_count':0,'repeat_exact':True,'all257_query_bytes_on_each_retained_event_capture_exact':True,
   'development_physical_replication':'Both clamp32 and refinement64 reach all160ticks without CCD event-budget rejection. If not, retain failure and classify next cause; component evidence alone does not qualify beak handling.'},
 'controls':['Unused stationary, initial overlap, literal rotating endpoints, away, grazing, crossing, common translation and sphere spin.','Native baseline/candidate/repeat; FP64 independent continuous reference.','Old motor conditions are development replication, not heldout confirmation.'],
 'inputs':{str(p.relative_to(R)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs},
 'budget':{'compile':1,'link':1,'grid_executions':3,'pair_replays':4,'physical_executions':2,'timeout_seconds_each':180,'new_raw_max_bytes':33554432,
  'stop':'One execution per predeclared condition, no retries, additional budget or policy promotion. Preserve all failed predictions. Production replacement requires explicit integration qualification and unused physical confirmation.'}})
shutil.copy2(__file__,A/'prepare.py');shutil.copy2(B/'run.py',A/'run.py');shutil.copy2(B/'oracle.py',A/'oracle.py')
print(json.dumps({'plan_sha256':sha(A/'plan.json'),'units':len(groups)}),flush=True)
