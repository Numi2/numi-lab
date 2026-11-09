from pathlib import Path
import difflib,gzip,hashlib,json,os,shlex,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-contact-slip-20261109';B=R/'build/crow-v15-contact-slip-20261109';C=B/'materialized';C.mkdir(exist_ok=True);original=(R/'src/metal/MetalWorld.mm').read_text();start=original.index('bool encodeDevicePhysicsBodyVelocities(');end=original.index('bool encodeBorrowedCoupledCandidate(',start);func=original[start:end];old='''    const std::size_t environmentCount
) {''';assert func.count(old)==1;func=func.replace(old,'''    const std::size_t environmentCount,
    const std::size_t destinationBodies = kCurrentBodies
) {''');old='context.buffers[kCurrentBodies]';assert func.count(old)==1;func=func.replace(old,'context.buffers[destinationBodies]');s=original[:start]+func+original[end:]
old='''                         !encodeTaskComplete(
                             *selectedState,''';new='''                         !encodeDevicePhysicsBodyVelocities(
                             *selectedState, commandBuffer, diagnostics.layout,
                             sourceQ, sourceV, batch.environmentCount,
                             kCandidateBodies
                         ) ||
                         !encodeTaskComplete(
                             *selectedState,''';assert s.count(old)==1;s=s.replace(old,new)
(A/'materialized-velocity-host.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='a/src/metal/MetalWorld.mm',tofile='b/experimental-MetalWorld.mm')));src=C/'MetalWorld.mm';src.write_text(s)
lines=subprocess.check_output(['ninja','-C',str(R/'build'),'-t','commands'],text=True).splitlines();line=next(x for x in lines if ' -c '+str(R/'src/metal/MetalWorld.mm') in x);args=shlex.split(line);obj=C/'MetalWorld.mm.o'
for k,v in [('-o',str(obj)),('-MF',str(obj)+'.d'),('-MT',str(obj))]:args[args.index(k)+1]=v
args[-1]=str(src);args.insert(1,'-I'+str(R/'src/metal'));subprocess.run(args,cwd=R/'build',check=True)
line=next(x for x in lines if x.startswith(': && /usr/bin/c++') and '-dynamiclib' in x and ' -o lib/libmetalrobo.dylib ' in x);args=shlex.split(line[len(': && '):-len(' && :')]);lib=C/'libmetalrobo.dylib';args[args.index('-o')+1]=str(lib);args[args.index('CMakeFiles/metalrobo.dir/src/c_api.cpp.o')]=str(B/'readout/c_api.cpp.o');args[args.index('CMakeFiles/metalrobo.dir/src/metal/MetalWorld.mm.o')]=str(obj);subprocess.run(args,cwd=R/'build',check=True)
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cmd=json.loads((A/'body-velocity-source-bindings.json').read_text())['argv']
for k,v in [('--state-trace',str(C/'trace.tsv')),('--crow-replay-pack',str(C/'replay.json'))]:cmd[cmd.index(k)+1]=v
(A/'materialized-velocity-bindings.json').write_text(json.dumps({'argv':cmd,'plan_sha256':h(A/'materialized-velocity-plan.md'),'host_patch_sha256':h(A/'materialized-velocity-host.patch'),'library_sha256':h(lib),'observer_patch_sha256':h(A/'feedback-readout-observer.patch')},indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(C),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240);(A/'materialized-velocity.stdout').write_text(p.stdout)
with (A/'materialized-velocity.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-3000:])
raw=(C/'replay.json').read_bytes()
with (A/'materialized-velocity.replay.json.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(raw)
payload=json.loads(raw)['payload'];frames={f['step']:f for f in payload['frames']};prior=json.load(gzip.open(A/'body-velocity-source.replay.json.gz','rt'))['payload'];same=all(all(a[k]==b[k] for k in ('q','v','body_states')) for a,b in zip(payload['frames'],prior['frames']));values=[]
for line in p.stderr.splitlines():
 if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
 sample=json.loads(line.split(' ',1)[1]);t=sample['step']
 if sample['lane']!=1 or not 101<=t<=250:continue
 f=frames[t-1];idx=payload['body_names'].index('crow_left_foot')*payload['body_state_stride'];values.append({'trace_step':t,'task_foot_x':sample['diagnostic_foot_velocity'][0],'replay_foot_x':f['body_states'][idx+7],'task_root_x':sample['diagnostic_foot_velocity'][1],'replay_root_x':f['v'][0]})
data=json.loads(p.stdout);report={'same_physical_states_as_original_library':same,'samples':len(values),'task_foot_x_range':[min(x['task_foot_x'] for x in values),max(x['task_foot_x'] for x in values)],'nonzero_task_foot_samples':sum(abs(x['task_foot_x'])>1e-8 for x in values),'max_foot_velocity_disagreement_m_s':max(abs(x['task_foot_x']-x['replay_foot_x']) for x in values),'max_root_velocity_disagreement_m_s':max(abs(x['task_root_x']-x['replay_root_x']) for x in values),'failed_environment_steps':data['failed_environment_steps'],'physical_failure_count':data['first_episode_navigation']['physical_failure_count'],'scope':'isolated host variant; task-visible signals may change, no production qualification'};(A/'materialized-velocity-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
