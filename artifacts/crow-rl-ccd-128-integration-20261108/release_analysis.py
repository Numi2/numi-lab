from pathlib import Path
import csv,hashlib,json,subprocess
import numpy as np
R=Path.cwd();B=R/'build/crow-ccd-articulated-sweep-build-01';A=R/'artifacts/crow-rl-ccd-128-integration-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
inputs={str(B/n/f):sha(B/n/f) for n in ('integrated32','integrated64','admitted128') for f in ('trace.tsv','scene.bin','initial-scene.bin')}
write(A/'release-analysis-plan.json',{'purpose':'Analyze retained traces only: localize cadence disagreement to joint path, release time, free flight or impact. No physical run or parameter changes.','inputs':inputs,'observables':['head pitch and jaw angle at matched ticks','last solved jaw-contact sample then first sustained separation','scene COM velocities, angular velocities and mechanical kinetic energy','free-flight z acceleration versus -9.81 and energy drift'],'limits':'Control-boundary contact evidence samples only final substep; last sampled contact is not a certified intra-control release time. Scene object inferred uniquely from its initial z near .26685 and inverse mass4000.'})
subprocess.run(['git','add',str(A/'release-analysis-plan.json'),str(A/'release_analysis.py')],check=True)
subprocess.run(['git','commit','-m','Seal retained release trajectory diagnosis'],check=True)
subprocess.run(['git','push','origin','HEAD'],check=True)
allrows={};result={}
for name in ('integrated32','integrated64','admitted128'):
    rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/name/'trace.tsv').open(),delimiter='\t')];allrows[name]=rows
    initial=np.fromfile(B/name/'initial-scene.bin',dtype='<f4').reshape(4,-1,32)
    idx=np.flatnonzero(np.abs(initial[0,:,11]-4000)<.01);assert len(idx)==1
    scene=np.fromfile(B/name/'scene.bin',dtype='<f4').reshape(160,4,-1,32)[:,:,idx[0]]
    assert np.allclose(scene[:,:,:3],np.array([[r[k] for k in 'xyz'] for r in rows]).reshape(160,4,3),atol=1e-11)
    lanes=[]
    for lane in range(4):
        q=[r for r in rows if r['lane']==lane];s=scene[:,lane]
        contacts=[int(r['tick']) for r in q if r['tick']>=96 and r['upper_Jn']+r['lower_Jn']>1e-9]
        separated=[int(r['tick']) for r in q if r['tick']>=96 and min(r['upper_gap_m'],r['lower_gap_m'])>.0001]
        t=next(t for t in separated if all(min(q[j]['upper_gap_m'],q[j]['lower_gap_m'])>.0001 for j in range(t,160)))
        ground=next(j for j in range(t,160) if s[j,2]<.006)
        # Gravity plus light authored damping are judged separately. No fit is
        # used to change the physical qualification gate.
        flight=range(t+1,ground)
        vz_acc=[float((s[j,10]-s[j-1,10])/.02) for j in flight]
        lanes.append({'lane':lane,'last_sampled_jaw_impulse_tick':max(contacts) if contacts else None,'first_sustained_0_1mm_separation_tick':t,'first_ground_near_tick':ground,'separation_state':{'position':s[t,:3].tolist(),'velocity':s[t,8:11].tolist(),'angular_velocity':s[t,12:15].tolist(),'pitch':q[t]['pitch'],'jaw':q[t]['jaw']},'free_flight_vertical_acceleration_range_m_s2':[min(vz_acc),max(vz_acc)] if vz_acc else None,'release_trace':[{k:q[j][k] for k in ('tick','x','z','pitch','jaw','upper_Jn','lower_Jn','lower_gap_m')}|{'vx':float(s[j,8]),'vz':float(s[j,10]),'wy':float(s[j,13])} for j in range(96,ground+1,2)]})
    result[name]={'lanes':lanes}
comparisons={}
for c,f in (('integrated32','integrated64'),('integrated64','admitted128')):
    a=allrows[c];b=allrows[f];comparisons[c+'-'+f]={}
    for lo,hi in ((0,31),(32,63),(64,95),(96,127),(128,159)):
        pairs=[(r,s) for r,s in zip(a,b) if lo<=r['tick']<=hi]
        comparisons[c+'-'+f][str((lo,hi))]={k:max(abs(r[k]-s[k]) for r,s in pairs) for k in ('pitch','jaw','x','z')}
write(A/'release-analysis.json',{'arms':result,'matched_phase_component_max_errors':comparisons})
print(json.dumps({'events':{k:[{a:v for a,v in l.items() if a not in ('release_trace','separation_state')} for l in m['lanes']] for k,m in result.items()},'matched_phase_component_max_errors':comparisons},indent=2))
