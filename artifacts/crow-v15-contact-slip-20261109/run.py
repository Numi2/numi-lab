from pathlib import Path
import difflib,gzip,hashlib,json,math,os,shlex,statistics,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-contact-slip-20261109';B=R/'build/crow-v15-contact-slip-20261109';O=R/'artifacts/crow-v15-ground-contact-20261109';base=(R/'build/crow-v15-ground-contact-20261109/c_api.cpp').read_text()
q=chr(92)+chr(34);old=q+'impulses'+q+':[%.9g,%.9g,%.9g],'+q+'separation'+q+':%.9g}';new=q+'impulses'+q+':[%.9g,%.9g,%.9g],'+q+'point'+q+':[%.9g,%.9g,%.9g],'+q+'separation'+q+':%.9g}';assert base.count(old)==1,base.count(old);s=base.replace(old,new)
old='c.impulses.x,c.impulses.y,c.impulses.z,c.pointAndSeparation.w);';new='c.impulses.x,c.impulses.y,c.impulses.z,c.pointAndSeparation.x,c.pointAndSeparation.y,c.pointAndSeparation.z,c.pointAndSeparation.w);';assert s.count(old)==1;s=s.replace(old,new)
src=B/'c_api.cpp';src.write_text(s);original=(R/'src/c_api.cpp').read_text();(A/'observer.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/c_api.cpp',tofile='b/contact-point-observer.cpp')))
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/c_api.cpp') in x);args=shlex.split(line);obj=B/'c_api.cpp.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'artifacts/crow-clearance-observer-20261109'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=B/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cmd=json.loads((O/'feedback_live-bindings.json').read_text())['argv']
for k,v in [('--state-trace',str(B/'trace.tsv')),('--crow-replay-pack',str(B/'replay.json'))]:cmd[cmd.index(k)+1]=v
(A/'bindings.json').write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'plan.md'),'observer_sha256':h(lib),'observer_patch_sha256':h(A/'observer.patch'),'previous_binding':'../crow-v15-ground-contact-20261109/feedback_live-bindings.json'},indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/'rollout.stdout').write_text(p.stdout)
with (A/'rollout.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-3000:])
payload=json.loads((B/'replay.json').read_bytes())['payload'];old=json.load(gzip.open(O/'feedback_live.replay.json.gz','rt'))['payload'];assert payload['frames']==old['frames'],'Observer failed exact replay parity'
frames={f['step']:f for f in payload['frames']};stats={n:{'speeds':[],'impulse_ratios':[]} for n in ('crow_left_foot','crow_right_foot')}
for line in p.stderr.splitlines():
 if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
 sample=json.loads(line.split(' ',1)[1]);step=sample['step']
 if sample['lane']!=1 or not 101<=step<=250:continue
 f=frames[step];stride=payload['body_state_stride']
 for c in sample['contacts']:
  a,b=c['a_name'],c['b_name'];name=next((v for v in (a,b) if v in stats),None)
  if not name or 'locomotion_ground' not in (a,b):continue
  normal=c['normal'];imp=c['impulses'];up=(-1 if a==name else 1)*normal[2]*imp[0]
  if up<=0:continue
  idx=payload['body_names'].index(name);st=f['body_states'][idx*stride:(idx+1)*stride];pos=st[:3];v=st[7:10];w=st[10:13];r=[c['point'][i]-pos[i] for i in range(3)];cross=[w[1]*r[2]-w[2]*r[1],w[2]*r[0]-w[0]*r[2],w[0]*r[1]-w[1]*r[0]];vc=[v[i]+cross[i] for i in range(3)];n2=sum(x*x for x in normal);dot=sum(vc[i]*normal[i] for i in range(3))/n2;speed=math.sqrt(sum((vc[i]-dot*normal[i])**2 for i in range(3)));stats[name]['speeds'].append(speed)
  if imp[0]>1e-6:stats[name]['impulse_ratios'].append(math.hypot(imp[1],imp[2])/imp[0])
def dist(xs):
 xs=sorted(xs);return {'n':len(xs),'median':statistics.median(xs) if xs else None,'p95':xs[math.ceil(.95*len(xs))-1] if xs else None,'maximum':max(xs) if xs else None}
report={'full_replay_parity':True,'frames':len(frames),'foot_contact_point_tangential_speed_m_s':{n:dist(x['speeds']) for n,x in stats.items()},'tangential_to_normal_impulse_ratio':{n:dist(x['impulse_ratios']) for n,x in stats.items()},'scope':'endpoint estimate, not integrated slip; contact point and body velocities may have substep timing difference'};(A/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
