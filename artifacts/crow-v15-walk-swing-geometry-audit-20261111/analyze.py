from pathlib import Path
import gzip,hashlib,json,math
R=Path.cwd();D=R/'artifacts/crow-v15-walk-swing-geometry-audit-20261111';S=R/'artifacts/crow-v15-toe-launch-control-20261110'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
with gzip.open(S/'control.replay.json.gz','rt') as f:p=json.load(f)['payload']
frames={f['step']:f for f in p['frames'] if 100<=f['step']<=200};assert len(frames)==101 and p['environment']==1
samples={}
with gzip.open(S/'control.stderr.gz','rt') as f:
 for line in f:
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  d=json.loads(line.split(' ',1)[1])
  if d['lane']==1 and 101<=d['step']<=201:samples[d['step']-1]=d
assert len(samples)==101
maxq=max(abs(a-b) for t in frames for a,b in zip(frames[t]['q'],samples[t]['q']));assert maxq<1e-6
names=p['body_names'];ground=names.index('locomotion_ground');res={s:[] for s in ('left','right')}
for t,f in sorted(frames.items()):
 st=f['body_states'];z0=st[ground*13+2];loads={'left':0.,'right':0.}
 for c in samples[t]['contacts']:
  an,bn=c['a_name'],c['b_name'];body=bn if an=='locomotion_ground' else an if bn=='locomotion_ground' else ''
  for side in loads:
   if body.startswith('crow_'+side+'_') and ('foot' in body or '_toe' in body):loads[side]+=max(0.,c['impulses'][0])
 for side in res:
  gaps=[]
  for name,h in [('crow_'+side+'_foot',(.010,.01725,.0092))]+[('crow_'+side+'_'+n,(.018,.0025,.0092)) for n in ('front_toe_1','front_toe_2','front_toe_3','rear_toe')]:
   i=names.index(name)*13; x,y,z,w=st[i+3:i+7];row=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));gaps.append(st[i+2]-z0-sum(abs(a)*b for a,b in zip(row,h)))
  opposite='right' if side=='left' else 'left';res[side].append({'step':t,'minimum_foot_plane_gap_m':min(gaps),'opposite_last_substep_normal_impulse_Ns':loads[opposite]})
def longest(a):
 n=best=0
 for x in a:n=n+1 if x else 0;best=max(best,n)
 return best
out={'source_hashes':{f:sha(S/f) for f in ('control.replay.json.gz','control.stderr.gz')},'replay_environment':1,'observer_lane':1,'observer_step_offset':1,'max_q_mismatch':maxq,'window':'replay steps 100 through 200 inclusive; observer steps 101 through 201 inclusive','method':'post-hoc bottom-most rotated box point relative to z=0 ground plane; last-substep support sample only','sides':{}}
for side,v in res.items():
 clear=[r['minimum_foot_plane_gap_m']>.001 for r in v];qualified=[r['minimum_foot_plane_gap_m']>.001 and r['opposite_last_substep_normal_impulse_Ns']>.001 for r in v];out['sides'][side]={'maximum_minimum_foot_gap_m':max(r['minimum_foot_plane_gap_m'] for r in v),'clear_over_1mm_samples':sum(clear),'longest_clear_run':longest(clear),'clear_with_opposite_sampled_load_count':sum(qualified),'longest_clear_with_opposite_sampled_load':longest(qualified)}
out['limitations']='Single retained development lane, accepted poses and last-substep contacts only; does not measure continuous clearance, contact-point slip or all-substep support.';(D/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
